import mne
import numpy as np
from scipy.signal import butter, coherence, filtfilt, hilbert


DEFAULT_EEG_BANDS: dict[str, tuple[float, float]] = {
	"delta": (0.5, 4.0),
	"theta": (4.0, 8.0),
	"alpha": (8.0, 13.0),
	"beta": (13.0, 30.0),
	"gamma": (30.0, 45.0),
}


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
		raise ValueError("Input array must have shape (n_channels, n_times) or (n_epochs, n_channels, n_times).")
	return arr, None, None


def _matrix_upper_mean(matrix: np.ndarray) -> float:
	"""Mean of upper-triangular non-diagonal entries for symmetric matrices."""
	triu = np.triu_indices_from(matrix, k=1)
	return float(np.mean(matrix[triu])) if triu[0].size else 0.0


def compute_coherence_matrix(
	data: mne.io.BaseRaw | mne.Epochs | np.ndarray,
	sfreq: float | None = None,
	fmin: float = 8.0,
	fmax: float = 13.0,
	nperseg: int = 256,
) -> np.ndarray:
	"""Compute symmetric pairwise coherence matrix averaged over a frequency band."""
	epochs, sf_from_data, _ = _to_epochs_array(data)
	fs = float(sfreq if sfreq is not None else sf_from_data)
	if fs <= 0:
		raise ValueError("Sampling frequency must be positive.")

	n_epochs, n_channels, _ = epochs.shape
	coh_mat = np.eye(n_channels, dtype=float)

	for i in range(n_channels):
		for j in range(i + 1, n_channels):
			pair_vals: list[float] = []
			for e in range(n_epochs):
				freqs, coh = coherence(epochs[e, i], epochs[e, j], fs=fs, nperseg=min(nperseg, epochs.shape[-1]))
				band_mask = (freqs >= fmin) & (freqs <= fmax)
				if np.any(band_mask):
					pair_vals.append(float(np.mean(coh[band_mask])))
			value = float(np.mean(pair_vals)) if pair_vals else 0.0
			coh_mat[i, j] = value
			coh_mat[j, i] = value

	return coh_mat


def compute_plv_matrix(
	data: mne.io.BaseRaw | mne.Epochs | np.ndarray,
	sfreq: float | None = None,
	fmin: float = 8.0,
	fmax: float = 13.0,
	filter_order: int = 4,
) -> np.ndarray:
	"""Compute symmetric pairwise phase-locking value matrix in a frequency band."""
	epochs, sf_from_data, _ = _to_epochs_array(data)
	fs = float(sfreq if sfreq is not None else sf_from_data)
	if fs <= 0:
		raise ValueError("Sampling frequency must be positive.")

	nyquist = fs / 2.0
	if not (0 < fmin < fmax < nyquist):
		raise ValueError("Require 0 < fmin < fmax < Nyquist frequency.")

	b, a = butter(filter_order, [fmin / nyquist, fmax / nyquist], btype="bandpass")
	n_epochs, n_channels, _ = epochs.shape
	plv_mat = np.eye(n_channels, dtype=float)

	for i in range(n_channels):
		for j in range(i + 1, n_channels):
			epoch_plvs: list[float] = []
			for e in range(n_epochs):
				x_i = filtfilt(b, a, epochs[e, i])
				x_j = filtfilt(b, a, epochs[e, j])

				phase_i = np.angle(hilbert(x_i))
				phase_j = np.angle(hilbert(x_j))
				dphi = phase_i - phase_j
				plv = np.abs(np.mean(np.exp(1j * dphi)))
				epoch_plvs.append(float(plv))

			value = float(np.mean(epoch_plvs)) if epoch_plvs else 0.0
			plv_mat[i, j] = value
			plv_mat[j, i] = value

	return plv_mat


def compute_connectivity_features(
	data: mne.io.BaseRaw | mne.Epochs | np.ndarray,
	sfreq: float | None = None,
	fmin: float = 8.0,
	fmax: float = 13.0,
) -> dict[str, np.ndarray | float]:
	"""Compute both coherence and PLV matrices with global summary values."""
	coh = compute_coherence_matrix(data=data, sfreq=sfreq, fmin=fmin, fmax=fmax)
	plv = compute_plv_matrix(data=data, sfreq=sfreq, fmin=fmin, fmax=fmax)
	coh_mean = _matrix_upper_mean(coh)
	plv_mean = _matrix_upper_mean(plv)

	return {
		"coherence_matrix": coh,
		"plv_matrix": plv,
		"coherence_mean": coh_mean,
		"plv_mean": plv_mean,
	}


def compute_bandwise_connectivity_features(
	data: mne.io.BaseRaw | mne.Epochs | np.ndarray,
	sfreq: float | None = None,
	bands: dict[str, tuple[float, float]] | None = None,
	nperseg: int = 256,
	filter_order: int = 4,
) -> dict[str, dict[str, np.ndarray | float]]:
	"""Compute coherence and PLV matrices for each EEG band.

	Returns
	-------
	dict
		Mapping band name -> {
			"fmin", "fmax", "coherence_matrix", "plv_matrix", "coherence_mean", "plv_mean"
		}
	"""
	selected_bands = bands or DEFAULT_EEG_BANDS
	results: dict[str, dict[str, np.ndarray | float]] = {}

	for band_name, (fmin, fmax) in selected_bands.items():
		coh = compute_coherence_matrix(
			data=data,
			sfreq=sfreq,
			fmin=fmin,
			fmax=fmax,
			nperseg=nperseg,
		)
		plv = compute_plv_matrix(
			data=data,
			sfreq=sfreq,
			fmin=fmin,
			fmax=fmax,
			filter_order=filter_order,
		)

		results[band_name] = {
			"fmin": float(fmin),
			"fmax": float(fmax),
			"coherence_matrix": coh,
			"plv_matrix": plv,
			"coherence_mean": _matrix_upper_mean(coh),
			"plv_mean": _matrix_upper_mean(plv),
		}

	return results

