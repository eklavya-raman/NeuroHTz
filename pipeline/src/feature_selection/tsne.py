from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.manifold import TSNE


@dataclass
class TSNEResult:
	"""Container for t-SNE outputs."""

	n_samples: int
	n_input_features: int
	n_components: int
	perplexity: float
	learning_rate: float
	n_iter: int
	kl_divergence: float
	embedding: np.ndarray


def _load_pca_projected_csv(csv_path: str) -> tuple[np.ndarray, list[str], list[dict[str, str]]]:
	"""Load PCA projected CSV and return matrix from PC columns + metadata rows."""
	with open(csv_path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		rows = list(reader)

	if not rows:
		raise ValueError("Input PCA CSV is empty.")

	fields = reader.fieldnames or []
	pc_columns = [c for c in fields if c.upper().startswith("PC")]
	if not pc_columns:
		raise ValueError("No PCA component columns found. Expected columns like PC1, PC2, ...")

	metadata_cols = [c for c in fields if c not in pc_columns]

	x_list: list[list[float]] = []
	meta_rows: list[dict[str, str]] = []
	for row in rows:
		x_list.append([float(row.get(col, 0.0) or 0.0) for col in pc_columns])
		meta_rows.append({k: row.get(k, "") for k in metadata_cols})

	x = np.asarray(x_list, dtype=float)
	return x, pc_columns, meta_rows


def run_tsne(
	x: np.ndarray,
	n_components: int = 2,
	perplexity: float = 30.0,
	learning_rate: float = 200.0,
	n_iter: int = 1000,
	random_state: int = 42,
) -> TSNEResult:
	"""Run t-SNE on PCA-reduced features.

	Notes
	-----
	This function imports scikit-learn lazily to provide a clearer message if
	the package is missing.
	"""
	if x.ndim != 2:
		raise ValueError("Input must be 2D: (n_samples, n_features).")

	n_samples, n_features = x.shape
	if n_samples < 2:
		raise ValueError("Need at least 2 samples for t-SNE.")

	# sklearn requires perplexity < n_samples.
	max_perplexity = max(1.0, float(n_samples - 1))
	effective_perplexity = float(min(perplexity, max_perplexity))

	model = TSNE(
		n_components=n_components,
		perplexity=effective_perplexity,
		learning_rate=learning_rate,
		max_iter=n_iter,
		init="pca",
		random_state=random_state,
		metric="euclidean",
	)
	embedding = model.fit_transform(x)

	return TSNEResult(
		n_samples=n_samples,
		n_input_features=n_features,
		n_components=n_components,
		perplexity=effective_perplexity,
		learning_rate=float(learning_rate),
		n_iter=int(n_iter),
		kl_divergence=float(getattr(model, "kl_divergence_", np.nan)),
		embedding=np.asarray(embedding, dtype=float),
	)


def run_tsne_from_pca_csv(
	pca_csv_path: str,
	n_components: int = 2,
	perplexity: float = 30.0,
	learning_rate: float = 200.0,
	n_iter: int = 1000,
	random_state: int = 42,
) -> tuple[TSNEResult, list[dict[str, str]], list[str]]:
	"""Load PCA projected CSV and apply t-SNE."""
	x, pc_columns, meta_rows = _load_pca_projected_csv(pca_csv_path)
	result = run_tsne(
		x=x,
		n_components=n_components,
		perplexity=perplexity,
		learning_rate=learning_rate,
		n_iter=n_iter,
		random_state=random_state,
	)
	return result, meta_rows, pc_columns


def _result_to_json(result: TSNEResult, input_pc_columns: list[str]) -> dict[str, Any]:
	return {
		"n_samples": result.n_samples,
		"n_input_features": result.n_input_features,
		"n_components": result.n_components,
		"input_pc_columns": input_pc_columns,
		"perplexity": result.perplexity,
		"learning_rate": result.learning_rate,
		"n_iter": result.n_iter,
		"kl_divergence": result.kl_divergence,
	}


def save_tsne_outputs(
	result: TSNEResult,
	metadata_rows: list[dict[str, str]],
	input_pc_columns: list[str],
	output_json: str,
	output_csv: str,
) -> None:
	"""Save t-SNE summary (JSON) and embedding points (CSV)."""
	os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
	os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

	with open(output_json, "w", encoding="utf-8") as f:
		json.dump(_result_to_json(result, input_pc_columns), f, indent=2)

	fieldnames: list[str] = []
	if metadata_rows:
		fieldnames.extend(sorted(metadata_rows[0].keys()))
	fieldnames.extend([f"TSNE{i + 1}" for i in range(result.n_components)])

	with open(output_csv, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(result.n_samples):
			row = dict(metadata_rows[i]) if i < len(metadata_rows) else {}
			for j in range(result.n_components):
				row[f"TSNE{j + 1}"] = float(result.embedding[i, j])
			writer.writerow(row)


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Apply t-SNE on PCA projected CSV output.")
	parser.add_argument("--input-csv", required=True, help="Path to PCA projected CSV (contains PC columns).")
	parser.add_argument("--n-components", type=int, default=2, help="Number of t-SNE dimensions.")
	parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
	parser.add_argument("--learning-rate", type=float, default=200.0, help="t-SNE learning rate.")
	parser.add_argument("--n-iter", type=int, default=1000, help="t-SNE iterations.")
	parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
	parser.add_argument("--output-dir", default="pipeline/features", help="Output directory for t-SNE files.")
	parser.add_argument("--output-prefix", default="tsne", help="Prefix for output filenames.")
	return parser


def main() -> None:
	parser = _build_parser()
	args = parser.parse_args()

	result, metadata, pc_columns = run_tsne_from_pca_csv(
		pca_csv_path=args.input_csv,
		n_components=args.n_components,
		perplexity=args.perplexity,
		learning_rate=args.learning_rate,
		n_iter=args.n_iter,
		random_state=args.random_state,
	)

	os.makedirs(args.output_dir, exist_ok=True)
	out_json = os.path.join(args.output_dir, f"{args.output_prefix}_summary.json")
	out_csv = os.path.join(args.output_dir, f"{args.output_prefix}_embedding.csv")
	save_tsne_outputs(
		result=result,
		metadata_rows=metadata,
		input_pc_columns=pc_columns,
		output_json=out_json,
		output_csv=out_csv,
	)

	print(f"Saved t-SNE summary: {out_json}")
	print(f"Saved t-SNE embedding: {out_csv}")
	print(f"KL divergence: {result.kl_divergence:.6f}")


if __name__ == "__main__":
	main()
