import numpy as np
import torch
import os

try:
    import ptwt
except ImportError as exc:  # pragma: no cover - import guard for runtime environments
    raise ImportError(
        "ptwt is required for perform_cwt. Install with: pip install ptwt torch"
    ) from exc


def _resolve_device(device: str | None = None) -> torch.device:
    """Resolve compute device for CWT execution."""
    requested_device = device or os.getenv("PTWT_DEVICE")
    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_ptwt_cwt(
    signal_tensor: torch.Tensor,
    scales_np: np.ndarray,
    wavelet: str,
    sampling_period: float,
) -> tuple[torch.Tensor, np.ndarray]:
    """Run ptwt.cwt in inference mode and return raw tensor outputs."""
    with torch.inference_mode():
        coeffs_tensor, frequencies = ptwt.cwt(
            signal_tensor,
            scales_np,
            wavelet,
            sampling_period=float(sampling_period),
        )
    return coeffs_tensor, np.asarray(frequencies)


def _is_mkl_fft_error(exc: RuntimeError) -> bool:
    text = str(exc)
    return "MKL FFT error" in text or "Inconsistent configuration parameters" in text


def perform_cwt_batch(
    signals: np.ndarray,
    scales: np.ndarray,
    wavelet: str = "morl",
    sampling_period: float = 1.0,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform batched CWT on signals shaped [batch, time] or [time]."""
    compute_device = _resolve_device(device)

    signals_np = np.asarray(signals, dtype=np.float32)
    if signals_np.ndim == 1:
        signals_np = signals_np[None, :]
    elif signals_np.ndim != 2:
        raise ValueError("signals must be 1D or 2D with shape [batch, time]")

    scales_np = np.asarray(scales, dtype=np.float32)
    signal_tensor = torch.from_numpy(signals_np).to(compute_device).contiguous()

    try:
        coeffs_tensor, frequencies = _run_ptwt_cwt(
            signal_tensor,
            scales_np,
            wavelet,
            sampling_period,
        )
    except RuntimeError as exc:
        if not _is_mkl_fft_error(exc):
            raise

        # Fallback for known MKL FFT planner instability on some batched shapes:
        # run one signal at a time and stack results.
        per_signal_coeffs: list[np.ndarray] = []
        frequencies = None
        for i in range(signals_np.shape[0]):
            single_tensor = (
                torch.from_numpy(signals_np[i : i + 1])
                .to(compute_device)
                .contiguous()
            )
            coeffs_single, freqs_single = _run_ptwt_cwt(
                single_tensor,
                scales_np,
                wavelet,
                sampling_period,
            )
            per_signal_coeffs.append(coeffs_single[:, 0, :].detach().cpu().numpy())
            if frequencies is None:
                frequencies = freqs_single

        batch_coeffs = np.stack(per_signal_coeffs, axis=0)
        return batch_coeffs, np.asarray(frequencies)

    # ptwt returns [scales, batch, time]; convert to [batch, scales, time].
    batch_coeffs = coeffs_tensor.permute(1, 0, 2).detach().cpu().numpy()
    return batch_coeffs, np.asarray(frequencies)

# Function to perform Continuous Wavelet Transform (CWT) on a signal
def perform_cwt(
    signal: np.ndarray,
    scales: np.ndarray,
    wavelet: str = "morl",
    sampling_period: float = 1.0,
    device: str | None = None,
):
    """
    Perform Continuous Wavelet Transform (CWT) on a signal.

    Parameters:
    signal (array-like): Input signal to be transformed.
    scales (array-like): Scales at which to compute the CWT.
    wavelet (str): Type of wavelet to use (default is 'morl').

    Returns:
    cwt_matrix (ndarray): CWT coefficients matrix.
    frequencies (ndarray): Corresponding frequencies for the scales.
    """
    scales_np = np.asarray(scales, dtype=np.float32)
    batch_coeffs, frequencies = perform_cwt_batch(
        np.asarray(signal, dtype=np.float32),
        scales_np,
        wavelet=wavelet,
        sampling_period=float(sampling_period),
        device=device,
    )
    cwt_matrix = batch_coeffs[0]
    return cwt_matrix, np.asarray(frequencies)

