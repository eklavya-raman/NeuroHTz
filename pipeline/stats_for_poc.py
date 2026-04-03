from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

_fx_time = importlib.import_module("feature_extraction.time_domain")
_fx_freq = importlib.import_module("feature_extraction.frequency_domain")
_fx_conn = importlib.import_module("feature_extraction.connectivity")
_fx_fuzzy = importlib.import_module("feature_extraction.fuzzy_features")
_sp_cwt = importlib.import_module("signal_preprocessing.CWT")
_sp_fuzzy = importlib.import_module("signal_preprocessing.fuzzy_artifact")
_sp_pre = importlib.import_module("signal_preprocessing.preprocess")
_sp_reader = importlib.import_module("signal_preprocessing.signal_reader")

DEFAULT_EEG_BANDS = _fx_freq.DEFAULT_EEG_BANDS
compute_time_domain_features = _fx_time.compute_time_domain_features
compute_frequency_domain_features = _fx_freq.compute_frequency_domain_features
compute_connectivity_features = _fx_conn.compute_connectivity_features
extract_fuzzy_features = _fx_fuzzy.extract_fuzzy_features
perform_cwt = _sp_cwt.perform_cwt
FuzzyArtifactConfig = _sp_fuzzy.FuzzyArtifactConfig
clean_raw_signal = _sp_pre.clean_raw_signal
read_signal = _sp_reader.read_signal


DEFAULT_INPUT = ROOT / "data" / "derivatives" / "sub-001" / "eeg" / "sub-001_task-eyesclosed_eeg_raw.fif"
DEFAULT_OUT_DIR = ROOT / "comparisons"
FEATURE_WINDOW_SECONDS = 60.0
FEATURE_RESAMPLE_SFREQ = 128.0
FEATURE_MAX_CHANNELS = 8


def _save_json(path: Path, obj: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(obj, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)


def _robust_normalize(arr: np.ndarray) -> np.ndarray:
	x = np.asarray(arr, dtype=float)
	lo = float(np.percentile(x, 2.0))
	hi = float(np.percentile(x, 98.0))
	if hi - lo <= 1e-12:
		lo = float(np.min(x))
		hi = float(np.max(x))
	denom = max(hi - lo, 1e-12)
	out = (x - lo) / denom
	return np.clip(out, 0.0, 1.0)


def _plot_time_comparison(raw: mne.io.BaseRaw, cleaned: mne.io.BaseRaw, out_path: Path, seconds: float = 12.0) -> None:
	sfreq = float(raw.info["sfreq"])
	n = int(min(raw.n_times, seconds * sfreq))

	ch_names = list(raw.ch_names)
	if "Fp1" in ch_names:
		ch_idx = ch_names.index("Fp1")
	else:
		ch_idx = 0

	t = np.arange(n) / sfreq
	x_raw = raw.get_data(picks=[ch_idx])[0, :n]
	x_clean = cleaned.get_data(picks=[ch_idx])[0, :n]

	fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
	fig.suptitle(f"Raw vs Fully Cleaned EEG ({raw.ch_names[ch_idx]})", fontsize=15)

	axes[0].plot(t, x_raw, color="#ab2f2f", linewidth=1.0)
	axes[0].set_ylabel("Amplitude")
	axes[0].set_title("Raw")
	axes[0].grid(alpha=0.25)

	axes[1].plot(t, x_clean, color="#1f7a4f", linewidth=1.0)
	axes[1].set_xlabel("Time (s)")
	axes[1].set_ylabel("Amplitude")
	axes[1].set_title("Fully Cleaned")
	axes[1].grid(alpha=0.25)

	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=220, bbox_inches="tight")
	plt.close(fig)


def _plot_psd_comparison(raw: mne.io.BaseRaw, cleaned: mne.io.BaseRaw, out_path: Path) -> None:
	sfreq = float(raw.info["sfreq"])
	raw_data = raw.get_data()
	clean_data = cleaned.get_data()

	from scipy.signal import welch

	nperseg = min(2048, raw_data.shape[1])
	freqs, psd_raw = welch(raw_data, fs=sfreq, nperseg=nperseg, axis=-1)
	_, psd_clean = welch(clean_data, fs=sfreq, nperseg=nperseg, axis=-1)

	mean_raw = np.mean(psd_raw, axis=0)
	mean_clean = np.mean(psd_clean, axis=0)
	mask = (freqs >= 0.5) & (freqs <= 45)

	fig, ax = plt.subplots(figsize=(12, 5.5))
	ax.semilogy(freqs[mask], mean_raw[mask] + 1e-12, color="#c44e52", label="Raw", linewidth=1.8)
	ax.semilogy(freqs[mask], mean_clean[mask] + 1e-12, color="#2c7fb8", label="Fully Cleaned", linewidth=1.8)
	ax.set_title("Power Spectral Density Comparison (Channel-Averaged)")
	ax.set_xlabel("Frequency (Hz)")
	ax.set_ylabel("PSD")
	ax.grid(alpha=0.25)
	ax.legend()

	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=220, bbox_inches="tight")
	plt.close(fig)


def _compute_pretty_scalogram_matrix(signal_1d: np.ndarray, sfreq: float, wavelet: str = "morl") -> tuple[np.ndarray, np.ndarray]:
	import pywt

	fmin, fmax = 1.0, 45.0
	n_scales = 320
	sampling_period = 1.0 / sfreq
	target_freqs = np.geomspace(fmax, fmin, num=n_scales, dtype=np.float32)
	scales = pywt.central_frequency(wavelet) / (target_freqs * sampling_period)
	cwt_matrix, freqs = perform_cwt(signal_1d, scales=np.asarray(scales, dtype=np.float32), wavelet=wavelet, sampling_period=sampling_period)
	return cwt_matrix, np.asarray(freqs)


def _plot_scalograms(raw: mne.io.BaseRaw, cleaned: mne.io.BaseRaw, out_path: Path, seconds: float = 10.0) -> None:
	sfreq = float(raw.info["sfreq"])
	n = int(min(raw.n_times, seconds * sfreq))

	raw_sig = raw.get_data(picks=[0])[0, :n]
	clean_sig = cleaned.get_data(picks=[0])[0, :n]

	cwt_raw, f_raw = _compute_pretty_scalogram_matrix(raw_sig, sfreq=sfreq)
	cwt_clean, f_clean = _compute_pretty_scalogram_matrix(clean_sig, sfreq=sfreq)

	# Downsample for figure rendering speed while keeping visual interpretability.
	raw_vis_matrix = np.abs(cwt_raw)[::2, ::2]
	clean_vis_matrix = np.abs(cwt_clean)[::2, ::2]
	f_raw_vis = f_raw[::2]
	f_clean_vis = f_clean[::2]

	vis_raw = _robust_normalize(raw_vis_matrix)
	vis_clean = _robust_normalize(clean_vis_matrix)

	tmax = n / sfreq
	extent_raw = [0.0, tmax, float(np.min(f_raw_vis)), float(np.max(f_raw_vis))]
	extent_clean = [0.0, tmax, float(np.min(f_clean_vis)), float(np.max(f_clean_vis))]

	fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
	fig.suptitle("Scalogram Comparison (Raw vs Fully Cleaned)", fontsize=16)

	im0 = axes[0].imshow(
		vis_raw,
		aspect="auto",
		cmap="turbo",
		origin="lower",
		extent=extent_raw,
		interpolation="bicubic",
	)
	axes[0].set_title("Raw")
	axes[0].set_xlabel("Time (s)")
	axes[0].set_ylabel("Frequency (Hz)")

	im1 = axes[1].imshow(
		vis_clean,
		aspect="auto",
		cmap="turbo",
		origin="lower",
		extent=extent_clean,
		interpolation="bicubic",
	)
	axes[1].set_title("Fully Cleaned")
	axes[1].set_xlabel("Time (s)")

	cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
	cbar.set_label("Normalized CWT Magnitude")

	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=180, bbox_inches="tight")
	plt.close(fig)


def _summarize_features(features: dict[str, object]) -> dict[str, object]:
	time_g = features["time"]["global_mean"]
	fuzzy = features["fuzzy"]
	frequency_bands = features["frequency"]["bands"]
	connectivity = features["connectivity"]

	return {
		"time_global_mean": {
			"std": float(time_g.get("std", 0.0)),
			"rms": float(time_g.get("rms", 0.0)),
			"line_length": float(time_g.get("line_length", 0.0)),
			"hjorth_mobility": float(time_g.get("hjorth_mobility", 0.0)),
			"hjorth_complexity": float(time_g.get("hjorth_complexity", 0.0)),
		},
		"frequency_relative_band_power": {
			band: float(vals.get("relative_band_power", 0.0)) for band, vals in frequency_bands.items()
		},
		"connectivity_means": {
			band: {
				"coherence_mean": float(vals.get("coherence_mean", 0.0)),
				"plv_mean": float(vals.get("plv_mean", 0.0)),
			}
			for band, vals in connectivity.items()
		},
		"fuzzy_summary": {
			"katz_fd_global_mean": float(fuzzy.get("katz_fd_global_mean", 0.0)),
			"fuzzy_entropy_global_mean": float(fuzzy.get("fuzzy_entropy_global_mean", 0.0)),
			"fuzzy_connectivity_mean": float(fuzzy.get("fuzzy_connectivity_mean", 0.0)),
		},
	}


def _prepare_feature_raw(raw: mne.io.BaseRaw, window_seconds: float = FEATURE_WINDOW_SECONDS, sfreq: float = FEATURE_RESAMPLE_SFREQ) -> mne.io.BaseRaw:
	"""Prepare a representative segment for faster documentation feature extraction."""
	prepared = raw.copy()
	duration = prepared.n_times / float(prepared.info["sfreq"])
	if duration > window_seconds:
		prepared.crop(tmin=0.0, tmax=window_seconds)
	current_sfreq = float(prepared.info["sfreq"])
	if current_sfreq > sfreq:
		prepared.resample(sfreq, npad="auto")
	if len(prepared.ch_names) > FEATURE_MAX_CHANNELS:
		prepared.pick(prepared.ch_names[:FEATURE_MAX_CHANNELS])
	return prepared


def _compute_feature_bundle(raw: mne.io.BaseRaw) -> dict[str, object]:
	"""Compute a documentation-focused feature bundle using extraction modules directly."""
	time_features = compute_time_domain_features(raw)
	frequency_features = compute_frequency_domain_features(
		data=raw,
		sfreq=float(raw.info["sfreq"]),
		bands=DEFAULT_EEG_BANDS,
		nperseg=128,
	)
	alpha_connectivity = compute_connectivity_features(
		data=raw,
		sfreq=float(raw.info["sfreq"]),
		fmin=8.0,
		fmax=13.0,
	)
	fuzzy_features = extract_fuzzy_features(
		data=raw,
		sfreq=float(raw.info["sfreq"]),
		entropy_m=2,
		entropy_r=0.2,
		entropy_n=2,
		connectivity_sigma=0.5,
		connectivity_fmin=8.0,
		connectivity_fmax=13.0,
	)

	return {
		"time": time_features,
		"frequency": frequency_features,
		"connectivity": {"alpha": alpha_connectivity},
		"fuzzy": fuzzy_features,
	}


def run_report(input_fif: Path, output_dir: Path, plots_only: bool = False) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	raw = read_signal(str(input_fif))

	strict_fuzzy_cfg = FuzzyArtifactConfig(
		swt_level=5,
		cmse_scale=3,
		max_entropy_points=0,
		prescreen_enabled=False,
		quick_prescreen_enabled=False,
		fuzzy_per_coeff_denoise=True,
		parallel_channels=True,
	)
	cleaned = clean_raw_signal(raw, fuzzy_config=strict_fuzzy_cfg)

	cleaned_fif = output_dir / f"{input_fif.stem}_fully_cleaned_raw.fif"
	cleaned.save(str(cleaned_fif), overwrite=True)

	_plot_time_comparison(raw, cleaned, output_dir / "comparison_time_domain.png")
	_plot_psd_comparison(raw, cleaned, output_dir / "comparison_psd.png")
	_plot_scalograms(raw, cleaned, output_dir / "comparison_scalograms_pretty.png", seconds=6.0)

	if plots_only:
		print(f"Saved cleaned FIF: {cleaned_fif}")
		print(f"Saved plots in: {output_dir}")
		print("Skipped feature extraction (--plots-only).")
		return

	# Extract module stats separately for raw and fully cleaned data.
	raw_for_features = _prepare_feature_raw(raw)
	clean_for_features = _prepare_feature_raw(cleaned)
	raw_features = _compute_feature_bundle(raw_for_features)
	clean_features = _compute_feature_bundle(clean_for_features)

	_save_json(output_dir / "features_raw_full.json", raw_features)
	_save_json(output_dir / "features_clean_full.json", clean_features)
	_save_json(
		output_dir / "features_summary_for_docs.json",
		{
			"input_file": str(input_fif),
			"cleaned_file": str(cleaned_fif),
			"feature_stats_window_seconds": FEATURE_WINDOW_SECONDS,
			"feature_stats_resample_sfreq": FEATURE_RESAMPLE_SFREQ,
			"raw_summary": _summarize_features(raw_features),
			"clean_summary": _summarize_features(clean_features),
		},
	)

	print(f"Saved cleaned FIF: {cleaned_fif}")
	print(f"Saved plots in: {output_dir}")
	print(f"Saved feature JSONs in: {output_dir}")


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate documentation-ready raw vs cleaned EEG comparisons.")
	parser.add_argument("--input-fif", type=str, default=str(DEFAULT_INPUT), help="Path to input FIF file.")
	parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output folder for report artifacts.")
	parser.add_argument("--plots-only", action="store_true", help="Generate cleaned file + plots only, skip feature extraction.")
	return parser


def main() -> None:
	parser = _build_arg_parser()
	args = parser.parse_args()

	input_fif = Path(args.input_fif)
	output_dir = Path(args.output_dir)
	if not input_fif.exists():
		raise FileNotFoundError(f"Input FIF not found: {input_fif}")

	run_report(input_fif=input_fif, output_dir=output_dir, plots_only=bool(args.plots_only))


if __name__ == "__main__":
	main()
