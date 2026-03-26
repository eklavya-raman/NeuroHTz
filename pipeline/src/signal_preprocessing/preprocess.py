# Run the preprocessing pipeline
from __future__ import annotations

import os

import mne
import numpy as np

from signal_preprocessing.CWT import perform_cwt
from signal_preprocessing.epoching import make_epochs
from signal_preprocessing.filtering import apply_filters
from signal_preprocessing.fuzzy_artifact import remove_fuzzy_artifacts_raw
from signal_preprocessing.scalogram import generate_scalogram
from signal_preprocessing.signal_reader import read_signal


def clean_raw_signal(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply filtering and fuzzy artifact suppression to raw EEG."""
    raw_signal = raw.copy()
    raw_signal = apply_filters(raw_signal, l_freq=0.5, h_freq=100.0, freqs=[50.0, 60.0])
    raw_signal, _ = remove_fuzzy_artifacts_raw(raw_signal)
    return raw_signal


def process(
    raw: mne.io.Raw,
    event_id: dict[str, int] | None = None,
    tmin: float = -0.2,
    tmax: float = 0.8,
    cwt_wavelet: str = "morl",
    output_dir: str | None = None,
    file_label: str | None = None,
) -> tuple[mne.Epochs, list, list[str]]:
    """Clean raw, epoch it, and generate scalograms from the first EEG channel."""
    cleaned = clean_raw_signal(raw)
    epochs = make_epochs(cleaned, event_id=event_id or {"stimulus": 1}, tmin=tmin, tmax=tmax)

    scalograms = []
    scalogram_paths: list[str] = []
    # Use first EEG channel from each epoch for time-frequency map generation.
    for epoch_idx, epoch in enumerate(epochs):
        signal_1d = np.asarray(epoch[0], dtype=float)
        scales = np.arange(1, max(2, signal_1d.size // 4))
        cwt_matrix, frequencies = perform_cwt(signal_1d, scales=scales, wavelet=cwt_wavelet)
        label_prefix = file_label or "eeg"
        label = f"{label_prefix}_epoch_{epoch_idx}"
        image, saved_path = generate_scalogram(cwt_matrix, frequencies, output_dir=output_dir, label=label)
        scalograms.append(image)
        scalogram_paths.append(saved_path)

    return epochs, scalograms, scalogram_paths


def process_file(
    file_path: str,
    event_id: dict[str, int] | None = None,
    tmin: float = -0.2,
    tmax: float = 0.8,
    output_dir: str | None = None,
) -> tuple[mne.Epochs, list, list[str]]:
    """Load a FIF file and run preprocessing."""
    raw = read_signal(file_path)
    file_label = os.path.splitext(os.path.basename(file_path))[0]
    return process(raw, event_id=event_id, tmin=tmin, tmax=tmax, output_dir=output_dir, file_label=file_label)


def full_folder_process(folder_path: str, output_dir: str | None = None) -> list[tuple[mne.Epochs, list, list[str]]]:
    """Run preprocessing for all FIF files in a folder."""
    processed_results: list[tuple[mne.Epochs, list]] = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".fif"):
            file_path = os.path.join(folder_path, file_name)
            processed_results.append(process_file(file_path, output_dir=output_dir))
    return processed_results


            

