import pywt
import numpy as np

# Function to perform Continuous Wavelet Transform (CWT) on a signal
def perform_cwt(signal: np.ndarray, scales: np.ndarray, wavelet: str = 'morl'):
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
    # Compute the CWT coefficients
    cwt_matrix, frequencies = pywt.cwt(signal, scales, wavelet)
    
    return cwt_matrix, frequencies

