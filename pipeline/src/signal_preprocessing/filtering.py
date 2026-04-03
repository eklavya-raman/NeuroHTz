from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np


@dataclass
class FilterConfig:
    """Configuration for adaptive filtering."""

    iir_order: int = 4
    auto_notch: bool = True
    notch_harmonics: int = 2
    mains_peak_halfwidth_hz: float = 0.5
    mains_context_halfwidth_hz: float = 2.0
    mains_detection_ratio: float = 1.15
    adaptive_cutoffs: bool = True
    drift_ratio_threshold: float = 0.2
    emg_ratio_threshold: float = 0.12
    drift_hpf_hz: float = 1.0
    emg_lpf_hz: float = 40.0
    drift_softness: float = 0.25
    emg_softness: float = 0.25
    severe_mode_enabled: bool = True
    severe_drift_ratio_threshold: float = 3.6
    severe_emg_ratio_threshold: float = 0.4
    severe_ocular_hpf_hz: float = 6.0
    severe_myogenic_hpf_hz: float = 11.0
    severe_myogenic_lpf_hz: float = 48.0


def _picks_eeg_or_all(raw: mne.io.Raw) -> np.ndarray | None:
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False)
    if eeg_picks.size > 0:
        return eeg_picks
    return None


def _safe_band_edges(raw: mne.io.Raw, l_freq: float, h_freq: float) -> tuple[float, float]:
    sfreq = float(raw.info["sfreq"])
    nyquist = sfreq / 2.0
    l = max(0.01, float(l_freq))
    h = min(float(h_freq), max(l + 0.5, nyquist - 1.0))
    if h <= l:
        h = min(nyquist - 1.0, l + 1.0)
    return l, h


def _band_energy_ratio(spec: np.ndarray, freqs: np.ndarray, num_band: tuple[float, float], den_band: tuple[float, float]) -> float:
    num_mask = (freqs >= num_band[0]) & (freqs <= num_band[1])
    den_mask = (freqs >= den_band[0]) & (freqs <= den_band[1])
    if not np.any(num_mask) or not np.any(den_mask):
        return 0.0
    num = float(np.mean(spec[num_mask]))
    den = float(np.mean(spec[den_mask]))
    return num / (den + 1e-12)


def _spectral_quality_ratios(raw: mne.io.Raw) -> tuple[float, float]:
    sfreq = float(raw.info["sfreq"])
    picks = _picks_eeg_or_all(raw)
    data = raw.get_data(picks=picks)
    if data.size == 0:
        return 0.0, 0.0

    max_samples = int(min(data.shape[1], max(256, round(20.0 * sfreq))))
    x = data[:, :max_samples].reshape(-1)
    if x.size < 64:
        return 0.0, 0.0

    x = x - float(np.mean(x))
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sfreq)
    if spec.size == 0 or freqs.size == 0:
        return 0.0, 0.0

    drift_ratio = _band_energy_ratio(spec, freqs, num_band=(0.1, 1.0), den_band=(1.0, 30.0))
    emg_ratio = _band_energy_ratio(spec, freqs, num_band=(30.0, 90.0), den_band=(1.0, 30.0))
    return drift_ratio, emg_ratio


def _adaptive_band_edges(raw: mne.io.Raw, l_freq: float, h_freq: float, config: FilterConfig) -> tuple[float, float]:
    if not config.adaptive_cutoffs:
        return _safe_band_edges(raw, l_freq, h_freq)

    drift_ratio, emg_ratio = _spectral_quality_ratios(raw)
    l_adj = float(l_freq)
    h_adj = float(h_freq)

    drift_th = float(config.drift_ratio_threshold)
    emg_th = float(config.emg_ratio_threshold)

    if drift_ratio >= drift_th:
        target_l = max(l_adj, float(config.drift_hpf_hz))
        drift_excess = max(0.0, drift_ratio - drift_th) / (drift_th + 1e-12)
        drift_soft = max(1e-6, float(config.drift_softness))
        drift_frac = drift_excess / (drift_excess + drift_soft)
        l_adj = l_adj + drift_frac * (target_l - l_adj)

    if emg_ratio >= emg_th:
        target_h = min(h_adj, float(config.emg_lpf_hz))
        emg_excess = max(0.0, emg_ratio - emg_th) / (emg_th + 1e-12)
        emg_soft = max(1e-6, float(config.emg_softness))
        emg_frac = emg_excess / (emg_excess + emg_soft)
        h_adj = h_adj - emg_frac * (h_adj - target_h)

    # Severe contamination fallback: move toward paper-style filtering only when ratios are clearly extreme.
    if bool(config.severe_mode_enabled):
        severe_drift_th = float(config.severe_drift_ratio_threshold)
        severe_emg_th = float(config.severe_emg_ratio_threshold)

        if drift_ratio >= severe_drift_th and emg_ratio < severe_emg_th:
            # Ocular-like severe contamination -> high-pass around 12 Hz.
            severe_target_l = max(l_adj, float(config.severe_ocular_hpf_hz))
            sev_excess = max(0.0, drift_ratio - severe_drift_th) / (severe_drift_th + 1e-12)
            sev_frac = sev_excess / (sev_excess + 0.2)
            l_adj = l_adj + sev_frac * (severe_target_l - l_adj)

        if emg_ratio >= severe_emg_th:
            # Myogenic-like severe contamination -> 12-40Hz style band-pass.
            severe_target_l = max(l_adj, float(config.severe_myogenic_hpf_hz))
            severe_target_h = min(h_adj, float(config.severe_myogenic_lpf_hz))
            sev_excess = max(0.0, emg_ratio - severe_emg_th) / (severe_emg_th + 1e-12)
            sev_frac = sev_excess / (sev_excess + 0.2)
            l_adj = l_adj + sev_frac * (severe_target_l - l_adj)
            h_adj = h_adj - sev_frac * (h_adj - severe_target_h)

    return _safe_band_edges(raw, l_adj, h_adj)


def inspect_filter_plan(
    raw: mne.io.Raw,
    l_freq: float,
    h_freq: float,
    freqs: list[float],
    config: FilterConfig | None = None,
) -> dict[str, float | bool | list[float]]:
    """Return adaptive filtering decisions without modifying raw data."""
    cfg = config or FilterConfig()
    drift_ratio, emg_ratio = _spectral_quality_ratios(raw)
    selected_l, selected_h = _adaptive_band_edges(raw, l_freq=l_freq, h_freq=h_freq, config=cfg)
    requested = sorted(set(float(f) for f in freqs if float(f) > 0.0))
    if cfg.auto_notch:
        selected_notch = _detect_mains_frequencies(raw, requested, cfg)
    else:
        selected_notch = requested

    return {
        "drift_ratio": float(drift_ratio),
        "emg_ratio": float(emg_ratio),
        "selected_l_freq": float(selected_l),
        "selected_h_freq": float(selected_h),
        "requested_notch_freqs": requested,
        "selected_notch_freqs": selected_notch,
        "adaptive_l_raised": bool(selected_l > (float(l_freq) + 1e-9)),
        "adaptive_h_lowered": bool(selected_h < (float(h_freq) - 1e-9)),
        "severe_ocular_mode": bool(drift_ratio >= float(cfg.severe_drift_ratio_threshold) and emg_ratio < float(cfg.severe_emg_ratio_threshold)),
        "severe_myogenic_mode": bool(emg_ratio >= float(cfg.severe_emg_ratio_threshold)),
    }


def _band_power(spec: np.ndarray, freqs: np.ndarray, f0: float, halfwidth: float) -> float:
    mask = (freqs >= (f0 - halfwidth)) & (freqs <= (f0 + halfwidth))
    if not np.any(mask):
        return 0.0
    return float(np.mean(spec[mask]))


def _detect_mains_frequencies(
    raw: mne.io.Raw,
    candidates_hz: list[float],
    config: FilterConfig,
) -> list[float]:
    sfreq = float(raw.info["sfreq"])
    nyquist = sfreq / 2.0
    picks = _picks_eeg_or_all(raw)
    data = raw.get_data(picks=picks)
    if data.size == 0:
        return []

    max_samples = int(min(data.shape[1], max(256, round(20.0 * sfreq))))
    segment = data[:, :max_samples]
    x = segment.reshape(-1)
    if x.size < 64:
        return []

    x = x - float(np.mean(x))
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sfreq)
    if spec.size == 0 or freqs.size == 0:
        return []

    detected: list[float] = []
    for base in sorted(set(float(f) for f in candidates_hz if f > 0.0)):
        for k in range(1, int(config.notch_harmonics) + 1):
            target = base * k
            if target >= (nyquist - 0.5):
                break

            peak = _band_power(spec, freqs, target, config.mains_peak_halfwidth_hz)
            context_left = _band_power(spec, freqs, target - config.mains_context_halfwidth_hz, config.mains_peak_halfwidth_hz)
            context_right = _band_power(spec, freqs, target + config.mains_context_halfwidth_hz, config.mains_peak_halfwidth_hz)
            context = (context_left + context_right) / 2.0
            ratio = peak / (context + 1e-12)
            if ratio >= float(config.mains_detection_ratio):
                detected.append(float(target))

    return sorted(set(detected))


def bandpass_filter(raw: mne.io.Raw, l_freq: float, h_freq: float, config: FilterConfig | None = None) -> mne.io.Raw:
    """Apply robust IIR Butterworth bandpass filtering in-place and return raw."""
    cfg = config or FilterConfig()
    l_safe, h_safe = _adaptive_band_edges(raw, l_freq=l_freq, h_freq=h_freq, config=cfg)
    raw.filter(
        l_freq=l_safe,
        h_freq=h_safe,
        method="iir",
        iir_params={"order": int(cfg.iir_order), "ftype": "butter"},
        picks=_picks_eeg_or_all(raw),
        verbose="ERROR",
    )
    return raw


def notch_filter(raw: mne.io.Raw, freqs: list[float], config: FilterConfig | None = None) -> mne.io.Raw:
    """Apply selective notch filtering in-place and return raw.

    MNE supports one stop-band per call for IIR, so apply each frequency separately.
    """
    cfg = config or FilterConfig()
    selected = sorted(set(float(f) for f in freqs if float(f) > 0.0))
    if cfg.auto_notch:
        selected = _detect_mains_frequencies(raw, selected, cfg)

    for f in selected:
        raw.notch_filter(
            freqs=[float(f)],
            method="iir",
            iir_params={"order": int(cfg.iir_order), "ftype": "butter"},
            picks=_picks_eeg_or_all(raw),
            verbose="ERROR",
        )
    return raw


def apply_filters(
    raw: mne.io.Raw,
    l_freq: float,
    h_freq: float,
    freqs: list[float],
    config: FilterConfig | None = None,
) -> mne.io.Raw:
    """Apply adaptive bandpass then selective notch filtering."""
    cfg = config or FilterConfig()
    raw = bandpass_filter(raw, l_freq, h_freq, config=cfg)
    raw = notch_filter(raw, freqs, config=cfg)
    return raw

