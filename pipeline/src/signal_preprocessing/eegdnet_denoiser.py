from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pywt

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None


@dataclass
class EEGDNetConfig:
    """Configuration for EEGDNet-inspired denoising with fuzzy strength weighting."""

    enabled: bool = True
    checkpoint_path: str | None = None
    device: str | None = None
    window_size: int = 512
    hop_size: int = 256
    segment_length: int = 64
    depths: int = 6
    heads: int = 1
    ff_multiplier: float = 2.0
    dropout: float = 0.0
    min_denoise_strength: float = 0.12
    max_denoise_strength: float = 0.74
    fuzzy_low_center: float = 0.10
    fuzzy_mid_center: float = 0.30
    fuzzy_high_center: float = 0.60
    fuzzy_low_weight: float = 0.25
    fuzzy_mid_weight: float = 0.60
    fuzzy_high_weight: float = 0.90
    wavelet: str = "db4"
    wavelet_level: int = 4


class _EEGDNetBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int, ff_multiplier: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(max(embed_dim, round(embed_dim * ff_multiplier)))
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class _EEGDNet(nn.Module):
    def __init__(
        self,
        signal_len: int,
        segment_len: int,
        depths: int,
        heads: int,
        ff_multiplier: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if signal_len % segment_len != 0:
            raise ValueError("signal_len must be divisible by segment_len")

        self.signal_len = int(signal_len)
        self.segment_len = int(segment_len)
        self.num_segments = int(signal_len // segment_len)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_segments, self.segment_len))
        self.blocks = nn.ModuleList(
            [
                _EEGDNetBlock(
                    embed_dim=self.segment_len,
                    heads=max(1, int(heads)),
                    ff_multiplier=float(ff_multiplier),
                    dropout=float(dropout),
                )
                for _ in range(max(1, int(depths)))
            ]
        )
        self.final_norm = nn.LayerNorm(self.segment_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, signal_len]
        x = x.view(x.shape[0], self.num_segments, self.segment_len)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x.reshape(x.shape[0], self.signal_len)


_MODEL_CACHE: dict[str, _EEGDNet] = {}


def _picks_eeg_or_all(raw: mne.io.Raw) -> np.ndarray | None:
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False)
    if eeg_picks.size > 0:
        return eeg_picks
    return None


def _band_energy_ratio(spec: np.ndarray, freqs: np.ndarray, num_band: tuple[float, float], den_band: tuple[float, float]) -> float:
    num_mask = (freqs >= num_band[0]) & (freqs <= num_band[1])
    den_mask = (freqs >= den_band[0]) & (freqs <= den_band[1])
    if not np.any(num_mask) or not np.any(den_mask):
        return 0.0
    num = float(np.mean(spec[num_mask]))
    den = float(np.mean(spec[den_mask]))
    return num / (den + 1e-12)


def _artifact_score(window: np.ndarray, sfreq: float) -> float:
    x = np.asarray(window, dtype=float)
    if x.size < 64 or sfreq <= 0.0:
        return 0.0

    x = x - float(np.mean(x))
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sfreq)

    drift_ratio = _band_energy_ratio(spec, freqs, num_band=(0.1, 1.0), den_band=(1.0, 30.0))
    emg_ratio = _band_energy_ratio(spec, freqs, num_band=(30.0, 90.0), den_band=(1.0, 30.0))
    return float(max(drift_ratio, emg_ratio))


def _triangular_membership(x: float, center: float, width: float) -> float:
    distance = abs(float(x) - float(center))
    return float(max(0.0, 1.0 - (distance / (width + 1e-12))))


def _fuzzy_weight(score: float, cfg: EEGDNetConfig) -> float:
    low = _triangular_membership(score, cfg.fuzzy_low_center, width=max(0.05, cfg.fuzzy_mid_center - cfg.fuzzy_low_center))
    mid = _triangular_membership(score, cfg.fuzzy_mid_center, width=max(0.05, cfg.fuzzy_high_center - cfg.fuzzy_low_center))
    high = _triangular_membership(score, cfg.fuzzy_high_center, width=max(0.05, cfg.fuzzy_high_center - cfg.fuzzy_mid_center))

    total = low + mid + high
    if total <= 1e-12:
        return float(cfg.fuzzy_mid_weight)

    weighted = (
        (low * float(cfg.fuzzy_low_weight))
        + (mid * float(cfg.fuzzy_mid_weight))
        + (high * float(cfg.fuzzy_high_weight))
    ) / total
    return float(np.clip(weighted, 0.0, 1.0))


def _wavelet_denoise(window: np.ndarray, strength: float, cfg: EEGDNetConfig) -> np.ndarray:
    x = np.asarray(window, dtype=float)
    if x.size < 16:
        return x.copy()

    try:
        wavelet = pywt.Wavelet(str(cfg.wavelet))
    except Exception:
        wavelet = pywt.Wavelet("db4")

    max_level = pywt.dwt_max_level(x.size, wavelet.dec_len)
    level = int(max(1, min(int(cfg.wavelet_level), max_level)))

    coeffs = pywt.wavedec(x, wavelet=wavelet, mode="symmetric", level=level)
    sigma = float(np.median(np.abs(coeffs[-1])) / 0.6745) + 1e-12
    base_thresh = sigma * np.sqrt(2.0 * np.log(float(x.size)))
    scaled_thresh = base_thresh * float(np.clip(strength, 0.0, 1.0))

    denoised_coeffs = [coeffs[0]]
    denoised_coeffs.extend(pywt.threshold(c, value=scaled_thresh, mode="soft") for c in coeffs[1:])

    reconstructed = pywt.waverec(denoised_coeffs, wavelet=wavelet, mode="symmetric")
    return np.asarray(reconstructed[: x.size], dtype=float)


def _resolved_segment_length(window_size: int, segment_length: int) -> int:
    q = int(max(4, min(segment_length, window_size)))
    while q > 4 and (window_size % q) != 0:
        q -= 1
    if window_size % q != 0:
        return 0
    return q


def _resolve_device(device: str | None) -> str:
    if torch is None:
        return "cpu"
    if device:
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(cfg: EEGDNetConfig, window_size: int) -> _EEGDNet | None:
    if torch is None or nn is None or not cfg.checkpoint_path:
        return None

    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        return None

    device_name = _resolve_device(cfg.device)
    state = torch.load(str(checkpoint_path), map_location=device_name)
    checkpoint_cfg: dict[str, float | int | str] = {}
    if isinstance(state, dict) and isinstance(state.get("model_config"), dict):
        checkpoint_cfg = state["model_config"]

    model_window_size = int(checkpoint_cfg.get("window_size", window_size))
    if model_window_size != int(window_size):
        return None

    segment_length = int(checkpoint_cfg.get("segment_length", cfg.segment_length))
    segment_len = _resolved_segment_length(model_window_size, segment_length)
    if segment_len <= 0:
        return None

    depths = int(checkpoint_cfg.get("depths", cfg.depths))
    heads = int(checkpoint_cfg.get("heads", cfg.heads))
    ff_multiplier = float(checkpoint_cfg.get("ff_multiplier", cfg.ff_multiplier))
    dropout = float(checkpoint_cfg.get("dropout", cfg.dropout))

    cache_key = (
        f"{checkpoint_path.resolve()}|{model_window_size}|{segment_len}|{depths}|"
        f"{heads}|{ff_multiplier}|{dropout}|{device_name}"
    )
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model = _EEGDNet(
        signal_len=model_window_size,
        segment_len=int(segment_len),
        depths=depths,
        heads=heads,
        ff_multiplier=ff_multiplier,
        dropout=dropout,
    )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        model.load_state_dict(state, strict=False)
    else:
        return None

    model.to(device_name)
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _run_model_window(model: _EEGDNet, window: np.ndarray, cfg: EEGDNetConfig) -> np.ndarray:
    if torch is None:
        return np.asarray(window, dtype=float)

    device_name = _resolve_device(cfg.device)
    signal = np.asarray(window, dtype=np.float32)
    std = float(np.std(signal)) + 1e-12
    signal_norm = signal / std

    tensor = torch.from_numpy(signal_norm[None, :]).to(device_name)
    with torch.inference_mode():
        out = model(tensor).detach().cpu().numpy()[0]
    return np.asarray(out * std, dtype=float)


def denoise_with_eegdnet(raw: mne.io.Raw, config: EEGDNetConfig | None = None) -> tuple[mne.io.Raw, dict[str, float | str | bool]]:
    """Apply EEGDNet-style denoising with fuzzy-weighted denoise strength."""
    cfg = config or EEGDNetConfig()
    if not cfg.enabled:
        return raw, {"enabled": False, "backend": "disabled"}

    picks = _picks_eeg_or_all(raw)
    data = raw.get_data(picks=picks)
    if data.size == 0:
        return raw, {"enabled": True, "backend": "no_data"}

    sfreq = float(raw.info["sfreq"])
    n_times = int(data.shape[1])

    window_size = int(max(64, min(int(cfg.window_size), n_times)))
    hop_size = int(max(1, min(int(cfg.hop_size), window_size)))

    model = _load_model(cfg, window_size=window_size)
    backend = "checkpoint_model" if model is not None else "wavelet_fallback"

    analysis_win = np.hanning(window_size)
    if float(np.sum(analysis_win)) <= 1e-12:
        analysis_win = np.ones(window_size, dtype=float)

    cleaned = np.zeros_like(data, dtype=float)
    score_acc = 0.0
    score_count = 0

    for ch_idx in range(data.shape[0]):
        ch = np.asarray(data[ch_idx], dtype=float)
        ch_out = np.zeros_like(ch)
        ch_weight = np.zeros_like(ch)

        for start in range(0, n_times, hop_size):
            end = min(start + window_size, n_times)
            chunk = ch[start:end]
            if chunk.size <= 1:
                continue

            if chunk.size < window_size:
                pad_width = window_size - chunk.size
                chunk_padded = np.pad(chunk, (0, pad_width), mode="edge")
            else:
                chunk_padded = chunk

            score = _artifact_score(chunk_padded, sfreq=sfreq)
            fuzzy = _fuzzy_weight(score, cfg)
            strength = float(
                np.clip(
                    cfg.min_denoise_strength + fuzzy * (cfg.max_denoise_strength - cfg.min_denoise_strength),
                    0.0,
                    1.0,
                )
            )

            if model is not None:
                denoised_chunk = _run_model_window(model, chunk_padded, cfg)
            else:
                denoised_chunk = _wavelet_denoise(chunk_padded, strength=strength, cfg=cfg)

            mixed_chunk = (1.0 - strength) * chunk_padded + strength * denoised_chunk
            mixed = mixed_chunk[: chunk.size]

            window_slice = analysis_win[: chunk.size]
            ch_out[start:end] += mixed * window_slice
            ch_weight[start:end] += window_slice

            score_acc += float(score)
            score_count += 1

        cleaned[ch_idx] = ch_out / (ch_weight + 1e-12)

    full_data = raw.get_data()
    if picks is None:
        full_data = cleaned
    else:
        full_data[picks, :] = cleaned
    raw._data = full_data

    mean_score = float(score_acc / max(1, score_count))
    return raw, {
        "enabled": True,
        "backend": backend,
        "avg_artifact_score": mean_score,
        "window_size": float(window_size),
        "hop_size": float(hop_size),
        "used_checkpoint": bool(model is not None),
    }
