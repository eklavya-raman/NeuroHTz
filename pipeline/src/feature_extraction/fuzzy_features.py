from __future__ import annotations

import mne
import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def _to_epochs_array(data: mne.io.BaseRaw | mne.Epochs | np.ndarray) -> tuple[np.ndarray, float | None, list[str] | None]:
	"""Convert supported inputs to shape (n_epochs, n_channels, n_times)."""
	if isinstance(data, mne.io.BaseRaw):
		arr = data.get_data()[np.newaxis, :, :]
		return arr, float(data.info["sfreq"]), list(data.ch_names)

	if isinstance(data, mne.Epochs):
		arr = data.get_data()
		return arr, float(data.info["sfreq"]), list(data.ch_names)

	arr = np.asarray(data, dtype=float)
	if arr.ndim == 2:
		arr = arr[np.newaxis, :, :]
	if arr.ndim != 3:
		raise ValueError("Input array must be (channels, times) or (epochs, channels, times).")
	return arr, None, None


def katz_fractal_dimension(signal: np.ndarray) -> float:
	"""Compute Katz fractal dimension of a 1D signal."""
	x = np.asarray(signal, dtype=float).reshape(-1)
	n = x.size
	if n < 2:
		return 0.0

	diffs = np.abs(np.diff(x))
	L = float(np.sum(diffs))
	if L <= 0:
		return 0.0

	d = float(np.max(np.abs(x - x[0])))
	if d <= 0:
		return 0.0

	return float(np.log10(n) / (np.log10(d / L) + np.log10(n)))


def fuzzy_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2, n: int = 2) -> float:
	"""Compute fuzzy entropy (FuzzyEn) of a 1D signal.

	Parameters
	----------
	m : int
		Embedding dimension.
	r : float
		Similarity tolerance as a ratio of the signal standard deviation.
	n : int
		Fuzzy power in exp(-(d^n)/r).
	"""
	x = np.asarray(signal, dtype=float).reshape(-1)
	N = x.size
	if N <= m + 1:
		return 0.0

	sd = np.std(x)
	if sd <= 0:
		return 0.0

	r_abs = r * sd

	def _phi(order: int) -> float:
		M = N - order + 1
		if M <= 1:
			return 0.0

		patterns = np.empty((M, order), dtype=float)
		for i in range(M):
			p = x[i : i + order]
			patterns[i] = p - np.mean(p)

		ssum = 0.0
		count = 0
		for i in range(M - 1):
			d = np.max(np.abs(patterns[i + 1 :] - patterns[i]), axis=1)
			mu = np.exp(-((d**n) / (r_abs + 1e-12)))
			ssum += float(np.sum(mu))
			count += mu.size

		if count == 0:
			return 0.0
		return ssum / count

	phi_m = _phi(m)
	phi_m1 = _phi(m + 1)
	if phi_m <= 0 or phi_m1 <= 0:
		return 0.0
	return float(np.log(phi_m) - np.log(phi_m1))


def _bandpass_epochs(epochs: np.ndarray, sfreq: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
	nyq = sfreq / 2.0
	if not (0 < fmin < fmax < nyq):
		raise ValueError("Require 0 < fmin < fmax < Nyquist frequency.")

	b, a = butter(order, [fmin / nyq, fmax / nyq], btype="bandpass")
	out = np.empty_like(epochs)
	for e in range(epochs.shape[0]):
		for c in range(epochs.shape[1]):
			out[e, c] = filtfilt(b, a, epochs[e, c])
	return out


def compute_fuzzy_connectivity_matrix(
	data: mne.io.BaseRaw | mne.Epochs | np.ndarray,
	sfreq: float | None = None,
	sigma: float = 0.5,
	fmin: float | None = None,
	fmax: float | None = None,
	filter_order: int = 4,
) -> np.ndarray:
	"""Compute fuzzy connectivity matrix from phase similarity.

	Similarity between channels i,j uses:
	mu = exp(-(delta_phase^2)/(2*sigma^2))
	averaged over time and epochs.
	"""
	epochs, sf_from_data, _ = _to_epochs_array(data)
	fs = float(sfreq if sfreq is not None else (sf_from_data or 0.0))

	if (fmin is not None or fmax is not None) and fs <= 0:
		raise ValueError("Sampling frequency is required when using bandpass for fuzzy connectivity.")

	if fmin is not None and fmax is not None:
		epochs = _bandpass_epochs(epochs, fs, fmin=fmin, fmax=fmax, order=filter_order)

	n_epochs, n_channels, _ = epochs.shape
	conn = np.eye(n_channels, dtype=float)

	for i in range(n_channels):
		for j in range(i + 1, n_channels):
			vals: list[float] = []
			for e in range(n_epochs):
				phi_i = np.angle(hilbert(epochs[e, i]))
				phi_j = np.angle(hilbert(epochs[e, j]))
				dphi = np.angle(np.exp(1j * (phi_i - phi_j)))
				mu = np.exp(-(dphi**2) / (2.0 * (sigma**2 + 1e-12)))
				vals.append(float(np.mean(mu)))

			v = float(np.mean(vals)) if vals else 0.0
			conn[i, j] = v
			conn[j, i] = v

	return conn


def extract_fuzzy_features(
	data: mne.io.BaseRaw | mne.Epochs | np.ndarray,
	sfreq: float | None = None,
	entropy_m: int = 2,
	entropy_r: float = 0.2,
	entropy_n: int = 2,
	connectivity_sigma: float = 0.5,
	connectivity_fmin: float | None = 8.0,
	connectivity_fmax: float | None = 13.0,
) -> dict[str, object]:
	"""Extract fuzzy features: Katz FD, fuzzy entropy, and fuzzy connectivity."""
	epochs, sf_from_data, ch_names = _to_epochs_array(data)
	fs = float(sfreq if sfreq is not None else (sf_from_data or 0.0))

	n_epochs, n_channels, _ = epochs.shape
	kfd = np.zeros((n_epochs, n_channels), dtype=float)
	fen = np.zeros((n_epochs, n_channels), dtype=float)

	for e in range(n_epochs):
		for c in range(n_channels):
			sig = epochs[e, c]
			kfd[e, c] = katz_fractal_dimension(sig)
			fen[e, c] = fuzzy_entropy(sig, m=entropy_m, r=entropy_r, n=entropy_n)

	conn = compute_fuzzy_connectivity_matrix(
		data=epochs,
		sfreq=fs if fs > 0 else None,
		sigma=connectivity_sigma,
		fmin=connectivity_fmin,
		fmax=connectivity_fmax,
	)

	triu = np.triu_indices_from(conn, k=1)
	conn_mean = float(np.mean(conn[triu])) if triu[0].size else 0.0

	return {
		"meta": {
			"n_epochs": int(n_epochs),
			"n_channels": int(n_channels),
			"sfreq": fs,
			"channel_names": ch_names,
		},
		"katz_fd_per_epoch_channel": kfd,
		"fuzzy_entropy_per_epoch_channel": fen,
		"katz_fd_channel_mean": np.mean(kfd, axis=0),
		"fuzzy_entropy_channel_mean": np.mean(fen, axis=0),
		"katz_fd_global_mean": float(np.mean(kfd)),
		"fuzzy_entropy_global_mean": float(np.mean(fen)),
		"fuzzy_connectivity_matrix": conn,
		"fuzzy_connectivity_mean": conn_mean,
	}
