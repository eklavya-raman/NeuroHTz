from __future__ import annotations

import mne
import numpy as np
from scipy.stats import kurtosis, skew


def _to_epochs_array(data: mne.io.BaseRaw | mne.Epochs | np.ndarray) -> tuple[np.ndarray, list[str] | None]:
    """Convert input to shape (n_epochs, n_channels, n_times)."""
    if isinstance(data, mne.io.BaseRaw):
        return data.get_data()[np.newaxis, :, :], list(data.ch_names)
    if isinstance(data, mne.Epochs):
        return data.get_data(), list(data.ch_names)

    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError("Input must be Raw, Epochs, (channels, times), or (epochs, channels, times).")
    return arr, None


def _safe(v: float | np.ndarray) -> float:
    val = float(np.asarray(v).reshape(-1)[0])
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return val


def _hjorth_parameters(x: np.ndarray) -> tuple[float, float]:
    """Return Hjorth mobility and complexity for 1D signal."""
    if x.size < 3:
        return 0.0, 0.0
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x) + 1e-12
    var_dx = np.var(dx) + 1e-12
    var_ddx = np.var(ddx) + 1e-12
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / (mobility + 1e-12)
    return float(mobility), float(complexity)


def compute_time_domain_features(data: mne.io.BaseRaw | mne.Epochs | np.ndarray) -> dict[str, object]:
    """Compute robust time-domain features per channel and global summary."""
    epochs, ch_names = _to_epochs_array(data)
    n_epochs, n_channels, _ = epochs.shape

    concatenated = epochs.transpose(1, 0, 2).reshape(n_channels, -1)
    channel_features: list[dict[str, float]] = []

    for c in range(n_channels):
        x = concatenated[c]
        mob, comp = _hjorth_parameters(x)
        channel_features.append(
            {
                "mean": _safe(np.mean(x)),
                "std": _safe(np.std(x)),
                "var": _safe(np.var(x)),
                "rms": _safe(np.sqrt(np.mean(x**2))),
                "max": _safe(np.max(x)),
                "min": _safe(np.min(x)),
                "peak_to_peak": _safe(np.ptp(x)),
                "median": _safe(np.median(x)),
                "iqr": _safe(np.percentile(x, 75) - np.percentile(x, 25)),
                "zero_crossing_rate": _safe(np.mean(np.diff(np.signbit(x)) != 0)),
                "line_length": _safe(np.sum(np.abs(np.diff(x)))),
                "skewness": _safe(skew(x, bias=False)),
                "kurtosis": _safe(kurtosis(x, fisher=True, bias=False)),
                "hjorth_mobility": _safe(mob),
                "hjorth_complexity": _safe(comp),
            }
        )

    # Global means across channels for compact downstream usage.
    keys = list(channel_features[0].keys()) if channel_features else []
    global_mean = {k: _safe(np.mean([cf[k] for cf in channel_features])) for k in keys}

    return {
        "meta": {
            "n_epochs": int(n_epochs),
            "n_channels": int(n_channels),
            "channel_names": ch_names,
        },
        "per_channel": channel_features,
        "global_mean": global_mean,
    }


# Backward-compatible API name
def extract_time_domain_features(raw: mne.io.BaseRaw | mne.Epochs | np.ndarray, epoch_length: float | None = None) -> dict[str, object]:
    return compute_time_domain_features(raw)

