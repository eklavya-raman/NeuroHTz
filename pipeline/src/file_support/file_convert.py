from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mne
import numpy as np


DEFAULT_INPUT_ROOT = (Path(__file__).resolve().parents[4] / "ds004504").resolve()
DEFAULT_OUTPUT_ROOT = (Path(__file__).resolve().parents[2] / "data").resolve()


def _load_json(path: Path) -> dict[str, object]:
	if not path.exists():
		return {}
	with path.open("r", encoding="utf-8") as f:
		obj = json.load(f)
	return obj if isinstance(obj, dict) else {}


def _load_channels_tsv(path: Path) -> list[dict[str, str]]:
	if not path.exists():
		return []
	with path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f, delimiter="\t")
		return [
			{k.strip().lower(): (v.strip() if isinstance(v, str) else "") for k, v in row.items()}
			for row in reader
		]


def _set_channel_types_from_tsv(raw: mne.io.BaseRaw, channels_meta: list[dict[str, str]]) -> None:
	if not channels_meta:
		return

	type_map: dict[str, str] = {
		"EEG": "eeg",
		"EOG": "eog",
		"ECG": "ecg",
		"EMG": "emg",
		"MISC": "misc",
	}
	set_types: dict[str, str] = {}
	for row in channels_meta:
		name = row.get("name", "")
		ch_type = type_map.get(row.get("type", "").upper(), "")
		if name and ch_type and name in raw.ch_names:
			set_types[name] = ch_type

	if set_types:
		raw.set_channel_types(set_types, on_unit_change="ignore")


def _apply_sidecar_metadata(raw: mne.io.BaseRaw, eeg_json: Path, channels_tsv: Path) -> None:
	meta = _load_json(eeg_json)
	channels_meta = _load_channels_tsv(channels_tsv)

	_set_channel_types_from_tsv(raw, channels_meta)

	line_freq = meta.get("PowerLineFrequency")
	if isinstance(line_freq, (int, float)):
		raw.info["line_freq"] = float(line_freq)

	reference = meta.get("EEGReference")
	if isinstance(reference, str) and reference.strip():
		refs = [x for x in reference.split() if x in raw.ch_names]
		if refs:
			# Use listed channels as reference when they are present in the recording.
			raw.set_eeg_reference(ref_channels=refs)


def convert_set_to_fif(set_file: Path, output_fif: Path, overwrite: bool = False) -> Path:
	"""Convert EEGLAB .set to .fif and apply available BIDS sidecar metadata."""
	raw = mne.io.read_raw_eeglab(str(set_file), preload=True)

	base_no_suffix = set_file.with_suffix("")
	eeg_json = base_no_suffix.with_suffix(".json")
	channels_tsv = set_file.with_name(f"{base_no_suffix.name.replace('_eeg', '')}_channels.tsv")

	_apply_sidecar_metadata(raw, eeg_json=eeg_json, channels_tsv=channels_tsv)

	output_fif.parent.mkdir(parents=True, exist_ok=True)
	raw.save(str(output_fif), overwrite=overwrite)
	return output_fif


def convert_numeric_tsv_to_fif(
	tsv_file: Path,
	output_fif: Path,
	eeg_json: Path | None = None,
	channels_tsv: Path | None = None,
	overwrite: bool = False,
) -> Path:
	"""Convert a numeric TSV (samples x channels) to FIF.

	Expected TSV shape:
	- first row is header (channel names)
	- each subsequent row is one sample
	- optional first column named time/sample/index is ignored
	"""
	with tsv_file.open("r", encoding="utf-8", newline="") as f:
		reader = csv.reader(f, delimiter="\t")
		rows = list(reader)

	if len(rows) < 2:
		raise ValueError(f"TSV has no data rows: {tsv_file}")

	headers = [h.strip() for h in rows[0]]
	data_rows = rows[1:]

	ignore_cols = {"time", "timestamp", "sample", "index"}
	keep_idx = [i for i, h in enumerate(headers) if h and h.lower() not in ignore_cols]
	if not keep_idx:
		raise ValueError(f"No channel columns found in TSV: {tsv_file}")

	ch_names = [headers[i] for i in keep_idx]
	arr = np.asarray([[float(r[i]) for i in keep_idx] for r in data_rows], dtype=float)
	if arr.ndim != 2:
		raise ValueError(f"Invalid numeric shape in TSV: {tsv_file}")

	meta = _load_json(eeg_json) if eeg_json else {}
	sfreq = meta.get("SamplingFrequency", 0)
	if not isinstance(sfreq, (int, float)) or float(sfreq) <= 0:
		raise ValueError(
			"SamplingFrequency must be provided (>0) via matching *_eeg.json for TSV conversion."
		)

	info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types="eeg")
	raw = mne.io.RawArray(arr.T, info)

	_apply_sidecar_metadata(raw, eeg_json=eeg_json or Path(""), channels_tsv=channels_tsv or Path(""))

	output_fif.parent.mkdir(parents=True, exist_ok=True)
	raw.save(str(output_fif), overwrite=overwrite)
	return output_fif


def _infer_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
	rel = input_file.relative_to(input_root)
	if input_file.suffix.lower() == ".set":
		name = input_file.name.replace("_eeg.set", "_eeg_raw.fif")
	else:
		name = input_file.with_suffix("").name + "_raw.fif"
	return (output_root / rel.parent / name).resolve()


def batch_convert_dataset(input_root: Path, output_root: Path, overwrite: bool = False) -> list[Path]:
	"""Batch-convert all supported EEG files under a BIDS-like dataset root."""
	converted: list[Path] = []

	set_files = sorted(input_root.rglob("*_eeg.set"))
	for set_file in set_files:
		out = _infer_output_path(set_file, input_root=input_root, output_root=output_root)
		converted.append(convert_set_to_fif(set_file, output_fif=out, overwrite=overwrite))

	# Optional fallback for numeric EEG TSV recordings that have no corresponding .set.
	tsv_files = sorted(input_root.rglob("*_eeg.tsv"))
	for tsv_file in tsv_files:
		set_candidate = tsv_file.with_suffix(".set")
		if set_candidate.exists():
			continue

		base = tsv_file.with_suffix("")
		eeg_json = base.with_suffix(".json")
		channels_tsv = tsv_file.with_name(f"{base.name.replace('_eeg', '')}_channels.tsv")
		out = _infer_output_path(tsv_file, input_root=input_root, output_root=output_root)
		converted.append(
			convert_numeric_tsv_to_fif(
				tsv_file,
				output_fif=out,
				eeg_json=eeg_json,
				channels_tsv=channels_tsv,
				overwrite=overwrite,
			)
		)

	return converted


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Convert EEG SET/TSV(+JSON) files to FIF.")
	parser.add_argument(
		"--input-root",
		type=str,
		default=str(DEFAULT_INPUT_ROOT),
		help="Dataset root (default: sibling ds004504 directory).",
	)
	parser.add_argument(
		"--output-root",
		type=str,
		default=str(DEFAULT_OUTPUT_ROOT),
		help="Output folder for FIF files (default: pipeline/data).",
	)
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing FIF files.")
	return parser


def main() -> None:
	parser = _build_parser()
	args = parser.parse_args()

	input_root = Path(args.input_root).resolve()
	output_root = Path(args.output_root).resolve()

	if not input_root.exists():
		raise FileNotFoundError(f"Input root not found: {input_root}")

	converted = batch_convert_dataset(input_root=input_root, output_root=output_root, overwrite=args.overwrite)
	print(f"Converted {len(converted)} file(s).")
	for p in converted:
		print(p)


if __name__ == "__main__":
	main()
