from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
	sys.path.append(str(Path(__file__).resolve().parents[1]))
	from feature_selection.tsne import run_tsne_from_pca_csv, save_tsne_outputs  # type: ignore
else:
	from feature_selection.tsne import run_tsne_from_pca_csv, save_tsne_outputs


@dataclass
class PCAResult:
	"""Container for PCA outputs."""

	n_samples: int
	n_features: int
	n_components: int
	feature_names: list[str]
	explained_variance: np.ndarray
	explained_variance_ratio: np.ndarray
	cumulative_explained_variance_ratio: np.ndarray
	components: np.ndarray
	projected_data: np.ndarray
	mean_: np.ndarray
	scale_: np.ndarray


def _to_float(value: Any, default: float = 0.0) -> float:
	try:
		if value is None:
			return default
		if isinstance(value, (float, int)):
			val = float(value)
		else:
			text = str(value).strip()
			if text == "":
				return default
			val = float(text)
		if np.isnan(val) or np.isinf(val):
			return default
		return val
	except Exception:
		return default


def load_feature_csv(csv_path: str) -> tuple[np.ndarray, list[str], list[dict[str, str]]]:
	"""Load flat feature CSV and return numeric matrix + metadata rows.

	Non-numeric columns (for example source_name) are kept in metadata and
	excluded from the PCA feature matrix.
	"""
	with open(csv_path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		rows = list(reader)

	if not rows:
		raise ValueError("Input CSV is empty.")

	all_fields = reader.fieldnames or []
	meta_fields = {"source_name", "source_file"}
	feature_names = [name for name in all_fields if name not in meta_fields]

	matrix: list[list[float]] = []
	metadata: list[dict[str, str]] = []
	for row in rows:
		metadata.append({k: row.get(k, "") for k in meta_fields if k in row})
		matrix.append([_to_float(row.get(name, 0.0)) for name in feature_names])

	x = np.asarray(matrix, dtype=float)
	return x, feature_names, metadata


def run_pca(
	x: np.ndarray,
	n_components: int | None = None,
	standardize: bool = True,
) -> PCAResult:
	"""Run PCA using NumPy eigendecomposition of covariance matrix."""
	if x.ndim != 2:
		raise ValueError("Input feature matrix must be 2D: (n_samples, n_features).")

	n_samples, n_features = x.shape
	if n_samples < 2:
		raise ValueError("Need at least 2 samples for PCA.")
	if n_features < 1:
		raise ValueError("Need at least 1 feature for PCA.")

	k = min(n_samples, n_features) if n_components is None else int(n_components)
	if k < 1:
		raise ValueError("n_components must be >= 1.")
	k = min(k, n_samples, n_features)

	mean_ = np.mean(x, axis=0)
	centered = x - mean_

	if standardize:
		scale_ = np.std(centered, axis=0, ddof=1)
		scale_[scale_ == 0] = 1.0
		x_proc = centered / scale_
	else:
		scale_ = np.ones(n_features, dtype=float)
		x_proc = centered

	cov = np.cov(x_proc, rowvar=False)
	eigvals, eigvecs = np.linalg.eigh(cov)

	order = np.argsort(eigvals)[::-1]
	eigvals = eigvals[order]
	eigvecs = eigvecs[:, order]

	eigvals_k = eigvals[:k]
	components = eigvecs[:, :k].T
	projected = x_proc @ eigvecs[:, :k]

	total_var = np.sum(np.clip(eigvals, a_min=0.0, a_max=None))
	if total_var <= 0:
		explained_ratio = np.zeros_like(eigvals_k)
	else:
		explained_ratio = np.clip(eigvals_k, a_min=0.0, a_max=None) / total_var
	cumulative_ratio = np.cumsum(explained_ratio)

	return PCAResult(
		n_samples=n_samples,
		n_features=n_features,
		n_components=k,
		feature_names=[],
		explained_variance=eigvals_k,
		explained_variance_ratio=explained_ratio,
		cumulative_explained_variance_ratio=cumulative_ratio,
		components=components,
		projected_data=projected,
		mean_=mean_,
		scale_=scale_,
	)


def run_pca_from_csv(
	csv_path: str,
	n_components: int | None = None,
	standardize: bool = True,
) -> tuple[PCAResult, list[dict[str, str]]]:
	"""Load CSV and run PCA in one step."""
	x, feature_names, metadata = load_feature_csv(csv_path)
	result = run_pca(x, n_components=n_components, standardize=standardize)
	result.feature_names = feature_names
	return result, metadata


def _to_jsonable(result: PCAResult) -> dict[str, Any]:
	return {
		"n_samples": result.n_samples,
		"n_features": result.n_features,
		"n_components": result.n_components,
		"feature_names": result.feature_names,
		"explained_variance": result.explained_variance.tolist(),
		"explained_variance_ratio": result.explained_variance_ratio.tolist(),
		"cumulative_explained_variance_ratio": result.cumulative_explained_variance_ratio.tolist(),
		"components": result.components.tolist(),
		"mean": result.mean_.tolist(),
		"scale": result.scale_.tolist(),
	}


def save_pca_outputs(
	result: PCAResult,
	metadata_rows: list[dict[str, str]],
	output_json: str,
	output_csv: str,
) -> None:
	"""Save PCA summary (JSON) and projected principal components (CSV)."""
	os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
	os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

	with open(output_json, "w", encoding="utf-8") as f:
		json.dump(_to_jsonable(result), f, indent=2)

	fieldnames = []
	if metadata_rows:
		fieldnames.extend(sorted(metadata_rows[0].keys()))
	fieldnames.extend([f"PC{i + 1}" for i in range(result.n_components)])

	with open(output_csv, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(result.n_samples):
			row = dict(metadata_rows[i]) if i < len(metadata_rows) else {}
			for j in range(result.n_components):
				row[f"PC{j + 1}"] = float(result.projected_data[i, j])
			writer.writerow(row)


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Feature-selection driver: PCA first, optional t-SNE second.")
	parser.add_argument("--input-csv", required=True, help="Path to input feature CSV.")
	parser.add_argument("--n-components", type=int, default=None, help="Number of principal components for PCA.")
	parser.add_argument("--no-standardize", action="store_true", help="Disable feature standardization before PCA.")
	parser.add_argument("--output-dir", default="pipeline/features", help="Output directory for PCA files.")
	parser.add_argument("--output-prefix", default="selection", help="Prefix for output files.")
	parser.add_argument("--skip-tsne", action="store_true", help="Run only PCA and skip t-SNE stage.")
	parser.add_argument("--tsne-components", type=int, default=2, help="Number of t-SNE dimensions.")
	parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity.")
	parser.add_argument("--tsne-learning-rate", type=float, default=200.0, help="t-SNE learning rate.")
	parser.add_argument("--tsne-iter", type=int, default=1000, help="t-SNE iterations.")
	parser.add_argument("--tsne-random-state", type=int, default=42, help="t-SNE random seed.")
	return parser


def main() -> None:
	parser = _build_arg_parser()
	args = parser.parse_args()

	result, metadata = run_pca_from_csv(
		csv_path=args.input_csv,
		n_components=args.n_components,
		standardize=not args.no_standardize,
	)

	os.makedirs(args.output_dir, exist_ok=True)
	pca_json = os.path.join(args.output_dir, f"{args.output_prefix}_pca_summary.json")
	pca_csv = os.path.join(args.output_dir, f"{args.output_prefix}_pca_projected.csv")
	save_pca_outputs(result, metadata, pca_json, pca_csv)

	print(f"Saved PCA summary: {pca_json}")
	print(f"Saved PCA projected data: {pca_csv}")
	print("Explained variance ratio:", [round(v, 6) for v in result.explained_variance_ratio.tolist()])

	if args.skip_tsne:
		return

	tsne_result, tsne_meta, tsne_pc_cols = run_tsne_from_pca_csv(
		pca_csv_path=pca_csv,
		n_components=args.tsne_components,
		perplexity=args.tsne_perplexity,
		learning_rate=args.tsne_learning_rate,
		n_iter=args.tsne_iter,
		random_state=args.tsne_random_state,
	)
	tsne_json = os.path.join(args.output_dir, f"{args.output_prefix}_tsne_summary.json")
	tsne_csv = os.path.join(args.output_dir, f"{args.output_prefix}_tsne_embedding.csv")
	save_tsne_outputs(
		result=tsne_result,
		metadata_rows=tsne_meta,
		input_pc_columns=tsne_pc_cols,
		output_json=tsne_json,
		output_csv=tsne_csv,
	)

	print(f"Saved t-SNE summary: {tsne_json}")
	print(f"Saved t-SNE embedding: {tsne_csv}")
	print(f"t-SNE KL divergence: {tsne_result.kl_divergence:.6f}")


if __name__ == "__main__":
	main()
