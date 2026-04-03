# Run the preprocessing pipeline
from __future__ import annotations

<<<<<<< HEAD
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import mne
import numpy as np
import pywt

from signal_preprocessing.CWT import perform_cwt_batch
from signal_preprocessing.eegdnet_denoiser import EEGDNetConfig, denoise_with_eegdnet
from signal_preprocessing.epoching import make_epochs
from signal_preprocessing.filtering import FilterConfig, apply_filters
=======
import os

import mne
import numpy as np

from signal_preprocessing.CWT import perform_cwt
from signal_preprocessing.epoching import make_epochs
from signal_preprocessing.filtering import apply_filters
from signal_preprocessing.fuzzy_artifact import remove_fuzzy_artifacts_raw
>>>>>>> c440577402fc029764c39a025d1a146cbf8c3176
from signal_preprocessing.scalogram import generate_scalogram
from signal_preprocessing.signal_reader import read_signal


<<<<<<< HEAD
STAGE_ORDER = ("Read Signal", "Clean Signal", "Create Epochs", "Generate Scalograms")
TOTAL_STAGE_UNITS = len(STAGE_ORDER)
CWT_FMIN_HZ = 1.0
CWT_FMAX_HZ = 45.0
CWT_NUM_SCALES = 320
CWT_BATCH_SIZE = 24
_CWT_SCALE_CACHE: dict[tuple[float, str, float, float, int], np.ndarray] = {}


def _build_fixed_frequency_scales(
    sfreq: float,
    wavelet: str,
    fmin_hz: float = CWT_FMIN_HZ,
    fmax_hz: float = CWT_FMAX_HZ,
    num_scales: int = CWT_NUM_SCALES,
) -> np.ndarray:
    """Build scales from a fixed frequency band so scalograms are comparable across files."""
    if sfreq <= 0:
        raise ValueError("sfreq must be > 0")
    if fmin_hz <= 0 or fmax_hz <= 0 or fmax_hz <= fmin_hz:
        raise ValueError("Require 0 < fmin_hz < fmax_hz")
    if num_scales < 2:
        raise ValueError("num_scales must be >= 2")

    sampling_period = 1.0 / sfreq
    target_freqs = np.geomspace(fmax_hz, fmin_hz, num=num_scales, dtype=np.float32)
    center_freq = float(pywt.central_frequency(wavelet))
    scales = center_freq / (target_freqs * sampling_period)

    return np.clip(scales.astype(np.float32), 1e-3, None)


def _get_cached_cwt_scales(
    sfreq: float,
    wavelet: str,
    fmin_hz: float = CWT_FMIN_HZ,
    fmax_hz: float = CWT_FMAX_HZ,
    num_scales: int = CWT_NUM_SCALES,
) -> np.ndarray:
    key = (round(float(sfreq), 6), str(wavelet), float(fmin_hz), float(fmax_hz), int(num_scales))
    cached = _CWT_SCALE_CACHE.get(key)
    if cached is None:
        cached = _build_fixed_frequency_scales(
            sfreq=float(sfreq),
            wavelet=wavelet,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            num_scales=num_scales,
        )
        _CWT_SCALE_CACHE[key] = cached
    return cached


def _configure_logger(log_path: Path) -> logging.Logger:
    """Configure logger for console and file output."""
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_log_path = log_path
    try:
        file_handler = logging.FileHandler(resolved_log_path, mode="a", encoding="utf-8")
    except PermissionError:
        # Another process may keep the default log file locked on Windows.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_name = f"{log_path.stem}_{timestamp}_{os.getpid()}{log_path.suffix}"
        resolved_log_path = log_path.with_name(fallback_name)
        file_handler = logging.FileHandler(resolved_log_path, mode="a", encoding="utf-8")
        stream_handler.stream.write(
            f"Warning: could not write to {log_path}. Using {resolved_log_path} instead.\n"
        )
        stream_handler.flush()

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging to file: %s", resolved_log_path)
    return logger


def _format_progress_bar(percent: float, width: int = 28) -> str:
    bounded = max(0.0, min(percent, 100.0))
    filled = int((bounded / 100.0) * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _log_progress(
    logger: logging.Logger,
    *,
    file_idx: int,
    total_files: int,
    stage_idx: int,
    stage_name: str,
    stage_current: int,
    stage_total: int,
    step_name: str,
) -> None:
    safe_stage_total = max(1, stage_total)
    safe_stage_current = max(0, min(stage_current, safe_stage_total))
    stage_pct = (safe_stage_current / safe_stage_total) * 100.0

    completed_units = ((file_idx - 1) * TOTAL_STAGE_UNITS) + stage_idx + (safe_stage_current / safe_stage_total)
    total_units = max(1, total_files * TOTAL_STAGE_UNITS)
    overall_pct = (completed_units / total_units) * 100.0

    logger.info(
        "Progress | File %d/%d | Stage: %s %6.2f%% %s | Overall: %6.2f%% %s | Step %d/%d: %s",
        file_idx,
        total_files,
        stage_name,
        stage_pct,
        _format_progress_bar(stage_pct),
        overall_pct,
        _format_progress_bar(overall_pct),
        safe_stage_current,
        safe_stage_total,
        step_name,
    )


def _eegdnet_config_for_mode(mode: str) -> EEGDNetConfig:
    """Build EEGDNet denoising config for the selected runtime mode."""
    selected = mode.strip().lower()
    if selected == "normal":
        return EEGDNetConfig(
            window_size=512,
            hop_size=192,
            min_denoise_strength=0.10,
            max_denoise_strength=0.76,
            depths=6,
            heads=1,
        )
    return EEGDNetConfig()


def clean_raw_signal(
    raw: mne.io.Raw,
    step_callback: Callable[[int, int, str], None] | None = None,
    eegdnet_config: EEGDNetConfig | None = None,
    fuzzy_config: object | None = None,
    asr_config: object | None = None,
    filter_config: FilterConfig | None = None,
) -> mne.io.Raw:
    """Apply bandpass/notch filtering then EEGDNet fuzzy-weighted denoising."""

    def report(step_current: int, step_total: int, step_name: str) -> None:
        if step_callback is not None:
            step_callback(step_current, step_total, step_name)

    report(1, 3, "Copying raw signal")
    raw_signal = raw.copy()

    report(2, 3, "Applying bandpass and notch filters")
    raw_signal = apply_filters(raw_signal, l_freq=0.5, h_freq=100.0, freqs=[50.0, 60.0], config=filter_config)

    report(3, 3, "Applying EEGDNet fuzzy-weighted denoising")
    raw_signal, _ = denoise_with_eegdnet(raw_signal, config=eegdnet_config)
=======
def clean_raw_signal(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply filtering and fuzzy artifact suppression to raw EEG."""
    raw_signal = raw.copy()
    raw_signal = apply_filters(raw_signal, l_freq=0.5, h_freq=100.0, freqs=[50.0, 60.0])
    raw_signal, _ = remove_fuzzy_artifacts_raw(raw_signal)
>>>>>>> c440577402fc029764c39a025d1a146cbf8c3176
    return raw_signal


def process(
    raw: mne.io.Raw,
    event_id: dict[str, int] | None = None,
    tmin: float = -0.2,
    tmax: float = 0.8,
    cwt_wavelet: str = "morl",
    output_dir: str | None = None,
    file_label: str | None = None,
<<<<<<< HEAD
    logger: logging.Logger | None = None,
    file_idx: int = 1,
    total_files: int = 1,
    eegdnet_config: EEGDNetConfig | None = None,
) -> tuple[mne.Epochs, list, list[str]]:
    """Clean raw, epoch it, and generate scalograms from the first EEG channel."""

    if logger:
        logger.info("Stage 2/4 | Clean Signal | START")

    def clean_stage_progress(step_current: int, step_total: int, step_name: str) -> None:
        if logger:
            _log_progress(
                logger,
                file_idx=file_idx,
                total_files=total_files,
                stage_idx=1,
                stage_name="Clean Signal",
                stage_current=step_current,
                stage_total=step_total,
                step_name=step_name,
            )

    cleaned = clean_raw_signal(raw, step_callback=clean_stage_progress, eegdnet_config=eegdnet_config)
    if logger:
        logger.info("Stage 2/4 | Clean Signal | END")

    if logger:
        logger.info("Stage 3/4 | Create Epochs | START")
        _log_progress(
            logger,
            file_idx=file_idx,
            total_files=total_files,
            stage_idx=2,
            stage_name="Create Epochs",
            stage_current=1,
            stage_total=2,
            step_name="Building event list",
        )

    epochs = make_epochs(cleaned, event_id=event_id or {"stimulus": 1}, tmin=tmin, tmax=tmax)

    epoch_count = int(len(epochs.events))

    if logger:
        _log_progress(
            logger,
            file_idx=file_idx,
            total_files=total_files,
            stage_idx=2,
            stage_name="Create Epochs",
            stage_current=2,
            stage_total=2,
            step_name=f"Epoch objects created ({epoch_count} epochs)",
        )
        logger.info("Stage 3/4 | Create Epochs | END")

    scalograms = []
    scalogram_paths: list[str] = []
    total_epochs = epoch_count
    if logger:
        logger.info("Stage 4/4 | Generate Scalograms | START (%d epochs)", total_epochs)

    if total_epochs == 0:
        if logger:
            _log_progress(
                logger,
                file_idx=file_idx,
                total_files=total_files,
                stage_idx=3,
                stage_name="Generate Scalograms",
                stage_current=1,
                stage_total=1,
                step_name="No epochs available; skipping scalogram generation",
            )
            logger.info("Stage 4/4 | Generate Scalograms | END")
        return epochs, scalograms, scalogram_paths

    # Use first EEG channel from each epoch for time-frequency map generation.
    sampling_period = 1.0 / float(cleaned.info["sfreq"]) if float(cleaned.info["sfreq"]) > 0 else 1.0
    scales = _get_cached_cwt_scales(
        sfreq=float(cleaned.info["sfreq"]),
        wavelet=cwt_wavelet,
        fmin_hz=CWT_FMIN_HZ,
        fmax_hz=CWT_FMAX_HZ,
        num_scales=CWT_NUM_SCALES,
    )

    if logger:
        logger.info(
            "Stage 4/4 | CWT config | wavelet=%s, fmin=%.2f Hz, fmax=%.2f Hz, scales=%d, batch_size=%d",
            cwt_wavelet,
            CWT_FMIN_HZ,
            CWT_FMAX_HZ,
            int(scales.size),
            CWT_BATCH_SIZE,
        )

    try:
        epoch_data = epochs.get_data(copy=False)
    except TypeError:
        epoch_data = epochs.get_data()
    channel_signals = np.asarray(epoch_data[:, 0, :], dtype=np.float32)

    for batch_start in range(0, total_epochs, CWT_BATCH_SIZE):
        batch_end = min(total_epochs, batch_start + CWT_BATCH_SIZE)
        batch_signals = channel_signals[batch_start:batch_end]
        batch_cwt, frequencies = perform_cwt_batch(
            batch_signals,
            scales=scales,
            wavelet=cwt_wavelet,
            sampling_period=sampling_period,
        )

        for local_idx, cwt_matrix in enumerate(batch_cwt):
            epoch_idx = batch_start + local_idx
            if logger:
                logger.info("Stage 4/4 | Epoch %d/%d | START", epoch_idx + 1, total_epochs)

            label_prefix = file_label or "eeg"
            label = f"{label_prefix}_epoch_{epoch_idx}"
            image, saved_path = generate_scalogram(cwt_matrix, frequencies, output_dir=output_dir, label=label)
            scalograms.append(image)
            scalogram_paths.append(saved_path)

            if logger:
                _log_progress(
                    logger,
                    file_idx=file_idx,
                    total_files=total_files,
                    stage_idx=3,
                    stage_name="Generate Scalograms",
                    stage_current=epoch_idx + 1,
                    stage_total=total_epochs,
                    step_name=f"Saved scalogram: {saved_path}",
                )
                logger.info("Stage 4/4 | Epoch %d/%d | END", epoch_idx + 1, total_epochs)

    if logger:
        logger.info("Stage 4/4 | Generate Scalograms | END")
=======
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
>>>>>>> c440577402fc029764c39a025d1a146cbf8c3176

    return epochs, scalograms, scalogram_paths


def process_file(
    file_path: str,
    event_id: dict[str, int] | None = None,
    tmin: float = -0.2,
    tmax: float = 0.8,
    output_dir: str | None = None,
<<<<<<< HEAD
    logger: logging.Logger | None = None,
    file_idx: int = 1,
    total_files: int = 1,
    eegdnet_config: EEGDNetConfig | None = None,
) -> tuple[mne.Epochs, list, list[str]]:
    """Load a FIF file and run preprocessing."""
    if logger:
        logger.info("============================================================")
        logger.info("File %d/%d | START | %s", file_idx, total_files, file_path)
        logger.info("Stage 1/4 | Read Signal | START")
        _log_progress(
            logger,
            file_idx=file_idx,
            total_files=total_files,
            stage_idx=0,
            stage_name="Read Signal",
            stage_current=1,
            stage_total=2,
            step_name="Opening FIF file",
        )

    raw = read_signal(file_path)
    file_label = os.path.splitext(os.path.basename(file_path))[0]

    if logger:
        duration_seconds = raw.n_times / raw.info["sfreq"] if raw.info["sfreq"] else 0.0
        _log_progress(
            logger,
            file_idx=file_idx,
            total_files=total_files,
            stage_idx=0,
            stage_name="Read Signal",
            stage_current=2,
            stage_total=2,
            step_name=(
                f"Loaded signal ({len(raw.ch_names)} channels, "
                f"sfreq={raw.info['sfreq']:.2f} Hz, duration={duration_seconds:.2f} s)"
            ),
        )
        logger.info("Stage 1/4 | Read Signal | END")

    return process(
        raw,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        output_dir=output_dir,
        file_label=file_label,
        logger=logger,
        file_idx=file_idx,
        total_files=total_files,
        eegdnet_config=eegdnet_config,
    )


def full_folder_process(
    folder_path: str,
    output_dir: str | None = None,
    logger: logging.Logger | None = None,
    eegdnet_config: EEGDNetConfig | None = None,
) -> list[tuple[mne.Epochs, list, list[str]]]:
    """Run preprocessing for all FIF files in a folder and its subfolders."""
    processed_results: list[tuple[mne.Epochs, list, list[str]]] = []
    fif_files: list[str] = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(".fif"):
                fif_files.append(os.path.join(root, file_name))

    fif_files.sort()
    total_files = len(fif_files)
    if logger:
        logger.info("Found %d FIF files under %s", total_files, folder_path)

    if total_files == 0:
        if logger:
            logger.info("No FIF files found; nothing to process.")
        return processed_results

    for file_idx, file_path in enumerate(fif_files, start=1):
        if logger:
            logger.info("Queue status | Next file: %d/%d", file_idx, total_files)
        try:
            processed_results.append(
                process_file(
                    file_path,
                    output_dir=output_dir,
                    logger=logger,
                    file_idx=file_idx,
                    total_files=total_files,
                    eegdnet_config=eegdnet_config,
                )
            )
            if logger:
                logger.info("File %d/%d | END | SUCCESS", file_idx, total_files)
        except Exception as exc:
            if logger:
                logger.exception("Failed processing %s: %s", file_path, exc)
                logger.info("File %d/%d | END | FAILED", file_idx, total_files)

    if logger:
        logger.info("Finished folder processing: %d successful files", len(processed_results))
    return processed_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EEG preprocessing across the pipeline data folder.")
    parser.add_argument(
        "--mode",
        choices=("fast", "normal"),
        default="fast",
        help="Preprocessing mode. 'fast' prioritizes runtime; 'normal' uses stronger EEGDNet fuzzy-weighted denoising.",
    )
    parser.add_argument(
        "--eegdnet-checkpoint",
        default=None,
        help="Optional path to a trained EEGDNet checkpoint (.pt/.pth).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    denoise_mode = args.mode.lower()
    eegdnet_config = _eegdnet_config_for_mode(denoise_mode)
    if args.eegdnet_checkpoint:
        eegdnet_config.checkpoint_path = str(args.eegdnet_checkpoint)

    pipeline_root = Path(__file__).resolve().parents[2]
    data_dir = pipeline_root / "data"
    scalogram_dir = pipeline_root / "scalograms"
    log_file = pipeline_root / "preprocess_run.log"
    logger = _configure_logger(log_file)
    logger.info("Starting preprocessing run")
    logger.info("Data directory: %s", data_dir)
    logger.info("Scalogram output directory: %s", scalogram_dir)
    logger.info("Log file: %s", log_file)
    logger.info("EEGDNet denoise mode: %s", denoise_mode)
    logger.info(
        "EEGDNet config: window_size=%d, hop_size=%d, depths=%d, heads=%d, min_strength=%.3f, max_strength=%.3f",
        eegdnet_config.window_size,
        eegdnet_config.hop_size,
        eegdnet_config.depths,
        eegdnet_config.heads,
        eegdnet_config.min_denoise_strength,
        eegdnet_config.max_denoise_strength,
    )
    logger.info("EEGDNet checkpoint: %s", eegdnet_config.checkpoint_path or "<none, wavelet fallback>")
    full_folder_process(
        str(data_dir),
        output_dir=str(scalogram_dir),
        logger=logger,
        eegdnet_config=eegdnet_config,
    )
    logger.info("Preprocessing run complete")


if __name__ == "__main__":
    main()
=======
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


            
>>>>>>> c440577402fc029764c39a025d1a146cbf8c3176

