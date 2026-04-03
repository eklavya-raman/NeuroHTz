from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


def _build_perceptual_lut(size: int = 256) -> np.ndarray:
    """Create a fixed RGB lookup table tuned for discernible scalograms."""
    # Anchor colors (deep navy -> blue -> cyan -> yellow -> orange -> red).
    anchors = np.array(
        [
            [8, 20, 64],
            [18, 70, 150],
            [38, 145, 190],
            [65, 195, 155],
            [160, 220, 90],
            [245, 215, 70],
            [244, 145, 55],
            [220, 62, 48],
        ],
        dtype=np.float32,
    )
    anchor_x = np.linspace(0.0, 1.0, anchors.shape[0], dtype=np.float32)
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)

    lut = np.empty((size, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(x, anchor_x, anchors[:, c]).astype(np.uint8)
    return lut


_SCALOGRAM_LUT = _build_perceptual_lut()
_MIN_OUTPUT_WIDTH = 960
_MIN_OUTPUT_HEIGHT = 360


def _default_scalogram_dir() -> Path:
    """Return default output directory: pipeline/scalograms."""
    return Path(__file__).resolve().parents[2] / "scalograms"


def _unique_scalogram_name(label: str | None = None) -> str:
    base = "scalogram" if not label else "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    return f"{base}_{uuid4().hex[:10]}.png"


# Function to generate scalogram from CWT coefficients
def generate_scalogram(
    cwt_matrix: np.ndarray,
    frequencies: np.ndarray,
    output_dir: str | Path | None = None,
    label: str | None = None,
    figsize: tuple[float, float] = (14, 8),
    dpi: int = 300,
) -> tuple[np.ndarray, str]:
    # Build a fast image directly from normalized coefficients.
    # Keep signature unchanged for compatibility; figsize/dpi/frequencies are accepted but unused.
    _ = frequencies
    _ = figsize
    _ = dpi

    data = np.asarray(cwt_matrix, dtype=np.float32)
    if data.ndim != 2:
        data = np.atleast_2d(data)

    # Clean invalid values and apply robust contrast for sharper visual separation.
    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(data, 2.0))
    hi = float(np.percentile(data, 98.0))

    if hi - lo > 1e-12:
        data = np.clip(data, lo, hi)
    else:
        lo = float(np.min(data))
        hi = float(np.max(data))

    denom = max(hi - lo, 1e-12)
    normalized = (data - lo) / denom
    normalized = np.clip(normalized, 0.0, 1.0)

    # Mild gamma for better mid-range visibility, then map to a fixed perceptual LUT.
    normalized = np.power(normalized, 0.9, dtype=np.float32)

    # Resize before quantization to reduce visible banding/striping in low-resolution maps.
    src_h, src_w = normalized.shape
    target_w = max(src_w, _MIN_OUTPUT_WIDTH)
    target_h = max(src_h, _MIN_OUTPUT_HEIGHT)
    resized = cv2.resize(normalized, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Small blur helps remove staircase artifacts from repeated adjacent scale rows.
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=0.45, sigmaY=0.45)
    resized = cv2.addWeighted(resized, 1.12, cv2.GaussianBlur(resized, (0, 0), sigmaX=1.0, sigmaY=1.0), -0.12, 0)
    resized = np.clip(resized, 0.0, 1.0)

    # Convert to 8-bit and invert vertically to match prior origin='lower' output.
    idx = (resized * 255.0).astype(np.uint8)
    idx = np.flipud(idx)

    # Add tiny ordered dithering to reduce flat same-color bands after 8-bit quantization.
    bayer4 = np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float32,
    ) / 16.0
    tiled = np.tile(bayer4, (idx.shape[0] // 4 + 1, idx.shape[1] // 4 + 1))[: idx.shape[0], : idx.shape[1]]
    idx = np.clip(idx.astype(np.float32) + (tiled - 0.5) * 1.2, 0, 255).astype(np.uint8)

    rgb = _SCALOGRAM_LUT[idx]

    image = rgb

    # Save image with a unique label into the scalograms directory.
    out_dir = Path(output_dir) if output_dir is not None else _default_scalogram_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_path = out_dir / _unique_scalogram_name(label)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(saved_path), image_bgr):
        raise IOError(f"Failed to write scalogram image to {saved_path}")

    return image, str(saved_path)

def open_scalogram_image(file_path: str) -> np.ndarray:
    image_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read scalogram image: {file_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


