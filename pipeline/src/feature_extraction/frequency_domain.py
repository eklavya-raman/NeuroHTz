from __future__ import annotations

import mne
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew


DEFAULT_EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _to_epochs_array(data: mne.io.BaseRaw | mne.Epochs | np.ndarray) -> tuple[np.ndarray, float | None, list[str] | None]:
    """Convert input to shape (n_epochs, n_channels, n_times)."""
    if isinstance(data, mne.io.BaseRaw):
        return data.get_data()[np.newaxis, :, :], float(data.info["sfreq"]), list(data.ch_names)
    if isinstance(data, mne.Epochs):
        return data.get_data(), float(data.info["sfreq"]), list(data.ch_names)

    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError("Input must be Raw, Epochs, (channels, times), or (epochs, channels, times).")
    return arr, None, None


def _safe(v: float | np.ndarray) -> float:
    val = float(np.asarray(v).reshape(-1)[0])
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return val


def _band_stats(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> dict[str, float | np.ndarray]:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        zeros = np.zeros(psd.shape[0], dtype=float)
        return {
            "absolute_power_per_channel": zeros,
            "relative_power_per_channel": zeros,
            "mean_psd": 0.0,
            "std_psd": 0.0,
            "variance_psd": 0.0,
            "skewness_psd": 0.0,
            "kurtosis_psd": 0.0,
            "absolute_band_power": 0.0,
            "relative_band_power": 0.0,
        }

    band_psd = psd[:, mask]
    abs_power_ch = np.trapezoid(band_psd, freqs[mask], axis=1)
    total_power_ch = np.trapezoid(psd, freqs, axis=1) + 1e-12
    rel_power_ch = abs_power_ch / total_power_ch
    flat = band_psd.reshape(-1)

    return {
        "absolute_power_per_channel": abs_power_ch,
        "relative_power_per_channel": rel_power_ch,
        "mean_psd": _safe(np.mean(flat)),
        "std_psd": _safe(np.std(flat)),
        "variance_psd": _safe(np.var(flat)),
        "skewness_psd": _safe(skew(flat, bias=False)),
        "kurtosis_psd": _safe(kurtosis(flat, fisher=True, bias=False)),
        "absolute_band_power": _safe(np.mean(abs_power_ch)),
        "relative_band_power": _safe(np.mean(rel_power_ch)),
    }


def compute_frequency_domain_features(
    data: mne.io.BaseRaw | mne.Epochs | np.ndarray,
    sfreq: float | None = None,
    bands: dict[str, tuple[float, float]] | None = None,
    nperseg: int = 256,
) -> dict[str, object]:
    """Compute robust frequency-domain features across EEG bands."""
    epochs, sf_from_data, ch_names = _to_epochs_array(data)
    fs = float(sfreq if sfreq is not None else (sf_from_data or 0.0))
    if fs <= 0:
        raise ValueError("Sampling frequency is required for frequency-domain features.")

    selected_bands = bands or DEFAULT_EEG_BANDS
    n_epochs, n_channels, n_times = epochs.shape
    all_data = epochs.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_times)

    seg = min(max(8, int(nperseg)), all_data.shape[-1])
    freqs, psd = welch(all_data, fs=fs, nperseg=seg, axis=-1)

    out_bands: dict[str, dict[str, float | np.ndarray]] = {}
    for band_name, (fmin, fmax) in selected_bands.items():
        if not (0 <= fmin < fmax <= fs / 2):
            raise ValueError(f"Invalid band range for {band_name}: ({fmin}, {fmax})")
        out_bands[band_name] = _band_stats(psd, freqs, fmin, fmax)

    return {
        "meta": {
            "n_epochs": int(n_epochs),
            "n_channels": int(n_channels),
            "n_times": int(n_times),
            "sfreq": fs,
            "channel_names": ch_names,
        },
        "bands": out_bands,
    }


# Backward-compatible helpers
def power_band(raw: mne.io.BaseRaw, band: tuple[float, float]) -> np.ndarray:
    features = compute_frequency_domain_features(raw, bands={"custom": band})
    return np.asarray(features["bands"]["custom"]["absolute_power_per_channel"], dtype=float)


def compute_psd_features(raw: mne.io.BaseRaw, band: tuple[float, float]) -> dict[str, float | np.ndarray]:
    features = compute_frequency_domain_features(raw, bands={"custom": band})
    return dict(features["bands"]["custom"])


def compute_all_bands(raw: mne.io.BaseRaw) -> dict[str, dict[str, float | np.ndarray]]:
    features = compute_frequency_domain_features(raw, bands=DEFAULT_EEG_BANDS)
    return dict(features["bands"])



