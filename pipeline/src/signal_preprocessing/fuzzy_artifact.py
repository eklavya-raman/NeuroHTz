from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import mne
import numpy as np
import pywt


@dataclass
class FuzzyArtifactConfig:
    """Configuration for fuzzy artifact detection/removal."""

    epoch_seconds: float = 1.0
    swt_level: int = 5
    swt_wavelet: str = "haar"
    cmse_scale: int = 3
    sample_entropy_m: int = 2
    sample_entropy_r_ratio: float = 0.2


def perform_dwt(signal: np.ndarray, wavelet: str = "haar", level: int = 5) -> list[np.ndarray]:
    """Perform DWT decomposition using PyWavelets."""
    x = np.asarray(signal, dtype=float)
    return pywt.wavedec(x, wavelet=wavelet, level=level)


def inverse_dwt(coeffs: list[np.ndarray], wavelet: str = "haar", length: int | None = None) -> np.ndarray:
    """Reconstruct signal from DWT coefficients using PyWavelets."""
    rec = pywt.waverec(coeffs, wavelet=wavelet)
    if length is not None:
        rec = rec[:length]
    return np.asarray(rec, dtype=float)


def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership function."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a) if b > a else 0.0
    return (d - x) / (d - c) if d > c else 0.0


def _sample_entropy(signal: np.ndarray, m: int, r: float) -> float:
    """Compute sample entropy with Chebyshev distance and non-self matches."""
    x = np.asarray(signal, dtype=float)
    n = x.size
    if n <= m + 1:
        return 0.0

    def _phi(order: int) -> float:
        count = 0
        templates = n - order + 1
        if templates <= 1:
            return 0.0
        for i in range(templates - 1):
            ref = x[i : i + order]
            for j in range(i + 1, templates):
                cmp = x[j : j + order]
                if np.max(np.abs(ref - cmp)) <= r:
                    count += 1
        denom = templates * (templates - 1) / 2.0
        return count / denom if denom > 0 else 0.0

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m <= 0 or phi_m1 <= 0:
        return 0.0
    return float(-np.log(phi_m1 / phi_m))


def composite_multiscale_entropy(
    signal: np.ndarray,
    scale: int = 3,
    m: int = 2,
    r_ratio: float = 0.2,
) -> float:
    """Composite multiscale entropy (CMSE) approximation used by fuzzy rules."""
    x = np.asarray(signal, dtype=float)
    if x.size < max(16, scale * (m + 2)):
        return 0.0

    std = np.std(x)
    r = r_ratio * std if std > 0 else r_ratio
    entropies: list[float] = []

    for k in range(scale):
        shifted = x[k:]
        blocks = shifted.size // scale
        if blocks <= m + 1:
            continue
        coarse = shifted[: blocks * scale].reshape(blocks, scale).mean(axis=1)
        entropies.append(_sample_entropy(coarse, m, r))

    if not entropies:
        return 0.0
    return float(np.mean(entropies))


def _moment_skewness(signal: np.ndarray) -> float:
    x = np.asarray(signal, dtype=float)
    if x.size < 3:
        return 0.0
    c = x - x.mean()
    v = np.mean(c**2)
    if v <= 0:
        return 0.0
    return float(np.mean(c**3) / (v ** 1.5))


def _moment_kurtosis_excess(signal: np.ndarray) -> float:
    x = np.asarray(signal, dtype=float)
    if x.size < 4:
        return 0.0
    c = x - x.mean()
    v = np.mean(c**2)
    if v <= 0:
        return 0.0
    return float(np.mean(c**4) / (v**2) - 3.0)


def extract_statistical_features(
    signal: np.ndarray,
    config: FuzzyArtifactConfig,
) -> dict[str, float]:
    """Extract CMSE, skewness, and kurtosis features."""
    return {
        "cmse": composite_multiscale_entropy(
            signal,
            scale=config.cmse_scale,
            m=config.sample_entropy_m,
            r_ratio=config.sample_entropy_r_ratio,
        ),
        "skewness": _moment_skewness(signal),
        "kurtosis": _moment_kurtosis_excess(signal),
    }


def _artifact_score_from_features(cmse: float, skewness: float, kurtosis: float) -> float:
    """Fuzzy score in [0, 1], inspired by rule base in the paper."""
    cmse_low = _trapmf(cmse, 0.0, 0.0, 0.4, 0.8)
    cmse_high = _trapmf(cmse, 0.6, 1.0, 5.0, 6.0)

    skew_abs = abs(skewness)
    skew_low = _trapmf(skew_abs, 0.0, 0.0, 0.25, 0.6)
    skew_high = _trapmf(skew_abs, 0.45, 0.8, 6.0, 7.0)

    k_abs = abs(kurtosis)
    kurt_low_high = max(_trapmf(k_abs, 0.0, 0.0, 0.6, 1.4), _trapmf(k_abs, 4.0, 5.0, 20.0, 25.0))
    kurt_medium = _trapmf(k_abs, 1.0, 2.0, 4.0, 5.0)

    r1 = min(cmse_high, kurt_medium, skew_low)
    r2 = min(cmse_high, kurt_medium, skew_high)
    r3 = min(cmse_high, kurt_low_high, skew_low)
    r4 = min(cmse_high, kurt_low_high, skew_high)
    r5 = min(cmse_low, kurt_medium, skew_low)
    r6 = min(cmse_low, kurt_medium, skew_high)
    r7 = min(cmse_low, kurt_low_high, skew_low)
    r8 = min(cmse_low, kurt_low_high, skew_high)

    artifact_free_score = max(r1, r2, r3, r5)
    artifact_score = max(r4, r6, r7, r8)
    denom = artifact_score + artifact_free_score + 1e-12
    return float(artifact_score / denom)


def fuzzy_is_artifact_epoch(features: dict[str, float], threshold: float = 0.5) -> bool:
    score = _artifact_score_from_features(
        cmse=features["cmse"],
        skewness=features["skewness"],
        kurtosis=features["kurtosis"],
    )
    return score >= threshold


def _next_valid_swt_length(length: int, level: int) -> int:
    block = 2**level
    return ((length + block - 1) // block) * block


def _pad_for_swt(x: np.ndarray, level: int) -> tuple[np.ndarray, int]:
    target = _next_valid_swt_length(x.size, level)
    if target == x.size:
        return x, 0
    pad = target - x.size
    xp = np.pad(x, (0, pad), mode="reflect")
    return xp, pad


def _nn_garrote_threshold(coeff: np.ndarray) -> np.ndarray:
    """Non-negative garrote with universal threshold."""
    c = np.asarray(coeff, dtype=float)
    if c.size == 0:
        return c
    sigma = np.median(np.abs(c)) / 0.6745
    thr = sigma * np.sqrt(2.0 * np.log(max(c.size, 2)))
    abs_c = np.abs(c)
    out = np.zeros_like(c)
    mask = abs_c > thr
    out[mask] = c[mask] - (thr**2 / c[mask])
    return out


def _denoise_artifactual_epoch(epoch: np.ndarray, config: FuzzyArtifactConfig) -> np.ndarray:
    x, pad = _pad_for_swt(np.asarray(epoch, dtype=float), config.swt_level)
    coeffs = pywt.swt(x, wavelet=config.swt_wavelet, level=config.swt_level)

    denoised_coeffs: list[tuple[np.ndarray, np.ndarray]] = []
    for c_a, c_d in coeffs:
        feat_a = extract_statistical_features(c_a, config)
        feat_d = extract_statistical_features(c_d, config)

        c_a_new = _nn_garrote_threshold(c_a) if fuzzy_is_artifact_epoch(feat_a) else c_a
        c_d_new = _nn_garrote_threshold(c_d) if fuzzy_is_artifact_epoch(feat_d) else c_d
        denoised_coeffs.append((c_a_new, c_d_new))

    rec = pywt.iswt(denoised_coeffs, wavelet=config.swt_wavelet)
    if pad > 0:
        rec = rec[:-pad]
    return np.asarray(rec, dtype=float)


def detect_artifactual_epochs(
    signal: np.ndarray,
    sfreq: float,
    config: FuzzyArtifactConfig | None = None,
) -> np.ndarray:
    """Return a boolean mask (one value per epoch) for artifactual epochs."""
    cfg = config or FuzzyArtifactConfig()
    epoch_len = max(1, int(round(cfg.epoch_seconds * sfreq)))
    n_epochs = int(np.ceil(signal.size / epoch_len))
    mask = np.zeros(n_epochs, dtype=bool)

    for i in range(n_epochs):
        s = i * epoch_len
        e = min((i + 1) * epoch_len, signal.size)
        if e - s < max(8, cfg.sample_entropy_m + 2):
            continue
        features = extract_statistical_features(signal[s:e], cfg)
        mask[i] = fuzzy_is_artifact_epoch(features)
    return mask


def remove_fuzzy_artifacts(
    signal: np.ndarray,
    sfreq: float,
    config: FuzzyArtifactConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and remove artifacts from a single EEG channel.

    Returns
    -------
    cleaned_signal:
        Signal after fuzzy-guided SWT artifact suppression.
    epoch_artifact_mask:
        Boolean mask where True means the epoch was marked artifactual.
    """
    cfg = config or FuzzyArtifactConfig()
    x = np.asarray(signal, dtype=float)
    epoch_len = max(1, int(round(cfg.epoch_seconds * sfreq)))
    n_epochs = int(np.ceil(x.size / epoch_len))
    mask = np.zeros(n_epochs, dtype=bool)
    cleaned = x.copy()

    for i in range(n_epochs):
        s = i * epoch_len
        e = min((i + 1) * epoch_len, x.size)
        epoch = x[s:e]
        if epoch.size < max(8, cfg.sample_entropy_m + 2):
            continue

        features = extract_statistical_features(epoch, cfg)
        is_art = fuzzy_is_artifact_epoch(features)
        mask[i] = is_art
        if is_art:
            cleaned[s:e] = _denoise_artifactual_epoch(epoch, cfg)

    return cleaned, mask


def remove_fuzzy_artifacts_raw(
    raw: mne.io.Raw,
    picks: Iterable[int] | None = None,
    config: FuzzyArtifactConfig | None = None,
) -> tuple[mne.io.Raw, dict[str, np.ndarray]]:
    """Apply fuzzy artifact removal channel-wise on an MNE Raw object."""
    cfg = config or FuzzyArtifactConfig()
    sfreq = float(raw.info["sfreq"])

    if picks is None:
        picks_arr = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False)
    else:
        picks_arr = np.asarray(list(picks), dtype=int)

    raw_clean = raw.copy()
    data = raw_clean.get_data()
    epoch_mask_by_channel: dict[str, np.ndarray] = {}

    for ch_idx in picks_arr:
        cleaned, mask = remove_fuzzy_artifacts(data[ch_idx], sfreq, cfg)
        data[ch_idx] = cleaned
        epoch_mask_by_channel[raw_clean.ch_names[ch_idx]] = mask

    raw_clean._data = data
    return raw_clean, epoch_mask_by_channel


def detect_fuzzy_artifacts(
    raw: mne.io.Raw,
    threshold: float | None = None,
    config: FuzzyArtifactConfig | None = None,
) -> list[str]:
    """Backwards-compatible helper that flags channels by artifact epoch ratio."""
    cfg = config or FuzzyArtifactConfig()
    ratio_threshold = 0.3 if threshold is None else float(threshold)

    sfreq = float(raw.info["sfreq"])
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False)

    artifact_channels: list[str] = []
    for ch_idx in eeg_picks:
        mask = detect_artifactual_epochs(raw.get_data(picks=[ch_idx]).ravel(), sfreq, cfg)
        if mask.size > 0 and float(mask.mean()) >= ratio_threshold:
            artifact_channels.append(raw.ch_names[ch_idx])
    return artifact_channels


def mark_fuzzy_artifacts(raw: mne.io.Raw, artifact_channels: list[str]) -> mne.io.Raw:
    raw_marked = raw.copy()
    raw_marked.info["bads"] = sorted(set(raw_marked.info["bads"]).union(artifact_channels))
    return raw_marked

