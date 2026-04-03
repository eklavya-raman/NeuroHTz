import mne


def read_signal(file_path: str) -> mne.io.Raw:
    """Load a FIF signal as an MNE Raw object."""
    return mne.io.read_raw_fif(file_path, preload=True)
import mne
# Signal reader
def read_signal(file_path: str) -> mne.io.Raw:
    raw = mne.io.read_raw_fif(file_path, preload=True)
    return raw
