import mne


def bandpass_filter(raw: mne.io.Raw, l_freq: float, h_freq: float) -> mne.io.Raw:
    """Apply IIR Butterworth bandpass filtering in-place and return raw."""
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="iir", iir_params={"order": 4, "ftype": "butter"})
    return raw


def notch_filter(raw: mne.io.Raw, freqs: list[float]) -> mne.io.Raw:
    """Apply IIR notch filtering in-place and return raw.

    MNE supports one stop-band per call for IIR, so apply each frequency separately.
    """
    for f in freqs:
        raw.notch_filter(freqs=[float(f)], method="iir", iir_params={"order": 4, "ftype": "butter"})
    return raw


def apply_filters(raw: mne.io.Raw, l_freq: float, h_freq: float, freqs: list[float]) -> mne.io.Raw:
    """Apply bandpass then notch filtering."""
    raw = bandpass_filter(raw, l_freq, h_freq)
    raw = notch_filter(raw, freqs)
    return raw

