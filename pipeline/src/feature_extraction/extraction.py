from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

if __package__ is None or __package__ == "":
	sys.path.append(str(Path(__file__).resolve().parents[1]))
	from feature_extraction.connectivity import (  # type: ignore
		DEFAULT_EEG_BANDS,
		compute_bandwise_connectivity_features,
	)
	from feature_extraction.frequency_domain import compute_frequency_domain_features  # type: ignore
	from feature_extraction.fuzzy_features import extract_fuzzy_features  # type: ignore
	from feature_extraction.time_domain import compute_time_domain_features  # type: ignore
	from signal_preprocessing.preprocess import clean_raw_signal  # type: ignore
	from signal_preprocessing.signal_reader import read_signal  # type: ignore
else:
	from feature_extraction.connectivity import (
		DEFAULT_EEG_BANDS,
		compute_bandwise_connectivity_features,
	)
	from feature_extraction.frequency_domain import compute_frequency_domain_features
	from feature_extraction.fuzzy_features import extract_fuzzy_features
	from feature_extraction.time_domain import compute_time_domain_features
	from signal_preprocessing.preprocess import clean_raw_signal
	from signal_preprocessing.signal_reader import read_signal


@dataclass
class ExtractionConfig:
	"""Configuration for full EEG feature extraction."""

	clean_signal: bool = True
	bands: dict[str, tuple[float, float]] | None = None
	welch_nperseg: int = 256
	connectivity_nperseg: int = 256
	connectivity_filter_order: int = 4
	fuzzy_entropy_m: int = 2
	fuzzy_entropy_r: float = 0.2
	fuzzy_entropy_n: int = 2
	fuzzy_connectivity_sigma: float = 0.5


def _safe_float(x: np.ndarray | float) -> float:
	val = float(np.asarray(x).reshape(-1)[0])
	if np.isnan(val) or np.isinf(val):
		return 0.0
	return val


def _flatten_feature_dict(features: dict, prefix: str = "") -> dict[str, float]:
	"""Recursively flatten nested feature dictionaries for ML pipelines."""
	flat: dict[str, float] = {}
	for key, value in features.items():
		new_key = f"{prefix}.{key}" if prefix else str(key)
		if isinstance(value, dict):
			flat.update(_flatten_feature_dict(value, new_key))
		elif isinstance(value, np.ndarray):
			if value.ndim == 2:
				for i in range(value.shape[0]):
					for j in range(value.shape[1]):
						flat[f"{new_key}[{i},{j}]"] = _safe_float(value[i, j])
			else:
				for idx, v in enumerate(value.reshape(-1)):
					flat[f"{new_key}[{idx}]"] = _safe_float(v)
		else:
			flat[new_key] = _safe_float(value)
	return flat


def extract_features_from_raw(
	raw: mne.io.BaseRaw,
	config: ExtractionConfig | None = None,
) -> dict[str, object]:
	"""Extract full feature set from an MNE Raw object."""
	cfg = config or ExtractionConfig()
	bands = cfg.bands or DEFAULT_EEG_BANDS

	raw_used = clean_raw_signal(raw) if cfg.clean_signal else raw.copy()
	raw_data = raw_used.get_data()

	time_features = compute_time_domain_features(raw_used)
	frequency_features = compute_frequency_domain_features(
		data=raw_used,
		sfreq=float(raw_used.info["sfreq"]),
		bands=bands,
		nperseg=cfg.welch_nperseg,
	)
	connectivity_features = compute_bandwise_connectivity_features(
		data=raw_used,
		sfreq=float(raw_used.info["sfreq"]),
		bands=bands,
		nperseg=cfg.connectivity_nperseg,
		filter_order=cfg.connectivity_filter_order,
	)
	fuzzy_features = extract_fuzzy_features(
		data=raw_used,
		sfreq=float(raw_used.info["sfreq"]),
		entropy_m=cfg.fuzzy_entropy_m,
		entropy_r=cfg.fuzzy_entropy_r,
		entropy_n=cfg.fuzzy_entropy_n,
		connectivity_sigma=cfg.fuzzy_connectivity_sigma,
	)

	structured: dict[str, object] = {
		"meta": {
			"n_channels": int(raw_data.shape[0]),
			"n_times": int(raw_data.shape[1]),
			"sfreq": float(raw_used.info["sfreq"]),
			"channel_names": list(raw_used.ch_names),
		},
		"time": time_features,
		"frequency": frequency_features,
		"connectivity": connectivity_features,
		"fuzzy": fuzzy_features,
	}

	structured["flat"] = _flatten_feature_dict(
		{
			"time": time_features,
			"frequency": frequency_features,
			"connectivity": connectivity_features,
			"fuzzy": fuzzy_features,
		}
	)
	return structured


def extract_features_from_file(
	file_path: str,
	config: ExtractionConfig | None = None,
) -> dict[str, object]:
	"""Read a FIF file and extract features."""
	raw = read_signal(file_path)
	result = extract_features_from_raw(raw, config=config)
	result["meta"] = {
		**result.get("meta", {}),
		"source_file": file_path,
		"source_name": os.path.basename(file_path),
	}
	return result


def extract_features_from_folder(
	folder_path: str,
	config: ExtractionConfig | None = None,
) -> dict[str, dict[str, object]]:
	"""Extract features for all FIF files in a folder."""
	outputs: dict[str, dict[str, object]] = {}
	for file_name in sorted(os.listdir(folder_path)):
		if file_name.lower().endswith(".fif"):
			file_path = os.path.join(folder_path, file_name)
			outputs[file_name] = extract_features_from_file(file_path, config=config)
	return outputs


def _json_converter(obj):
	"""Convert numpy objects to JSON-serializable types."""
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	if isinstance(obj, (np.floating, np.integer)):
		return obj.item()
	raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_feature_result(result: dict[str, object], output_json: str, output_csv: str) -> None:
	"""Save one extraction result to JSON (structured) and CSV (flat row)."""
	os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
	os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

	with open(output_json, "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2, default=_json_converter)

	flat = result.get("flat", {})
	if not isinstance(flat, dict):
		raise ValueError("Expected result['flat'] to be a dictionary.")

	with open(output_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
		writer.writeheader()
		writer.writerow(flat)


def save_feature_results_folder(results: dict[str, dict[str, object]], output_json: str, output_csv: str) -> None:
	"""Save folder extraction results to JSON and CSV table (one row per file)."""
	os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
	os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

	with open(output_json, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=2, default=_json_converter)

	rows: list[dict[str, object]] = []
	for file_name, result in results.items():
		flat = result.get("flat", {})
		if not isinstance(flat, dict):
			continue
		rows.append({"source_name": file_name, **flat})

	all_keys: list[str] = ["source_name"]
	key_set = set(all_keys)
	for row in rows:
		for key in row.keys():
			if key not in key_set:
				key_set.add(key)
				all_keys.append(key)

	with open(output_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=all_keys)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Full EEG feature extraction to JSON and CSV.")
	parser.add_argument("--input-file", type=str, help="Path to a FIF file.")
	parser.add_argument("--input-folder", type=str, help="Path to a folder containing FIF files.")
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(Path(__file__).resolve().parents[2] / "features"),
		help="Directory where output JSON/CSV will be written.",
	)
	parser.add_argument(
		"--skip-cleaning",
		action="store_true",
		help="Skip preprocessing cleanup before feature extraction.",
	)
	return parser


def main() -> None:
	parser = _build_arg_parser()
	args = parser.parse_args()

	if bool(args.input_file) == bool(args.input_folder):
		parser.error("Provide exactly one of --input-file or --input-folder.")

	cfg = ExtractionConfig(clean_signal=not args.skip_cleaning)
	os.makedirs(args.output_dir, exist_ok=True)

	if args.input_file:
		result = extract_features_from_file(args.input_file, config=cfg)
		base = Path(args.input_file).stem
		json_path = os.path.join(args.output_dir, f"{base}_features.json")
		csv_path = os.path.join(args.output_dir, f"{base}_features.csv")
		save_feature_result(result, output_json=json_path, output_csv=csv_path)
		print(f"Saved: {json_path}")
		print(f"Saved: {csv_path}")
		return

	results = extract_features_from_folder(args.input_folder, config=cfg)
	base = Path(args.input_folder).name or "folder"
	json_path = os.path.join(args.output_dir, f"{base}_features.json")
	csv_path = os.path.join(args.output_dir, f"{base}_features.csv")
	save_feature_results_folder(results, output_json=json_path, output_csv=csv_path)
	print(f"Saved: {json_path}")
	print(f"Saved: {csv_path}")


if __name__ == "__main__":
	main()


