from io import BytesIO
from pathlib import Path
from uuid import uuid4

import PIL
import numpy as np
import matplotlib.pyplot as plt


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
) -> tuple[PIL.Image.Image, str]:
    # Normalize the CWT coefficients for better visualization
    cwt_matrix_normalized = (cwt_matrix - np.min(cwt_matrix)) / (np.max(cwt_matrix) - np.min(cwt_matrix))
    
    # Create a scalogram image using matplotlib
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(cwt_matrix_normalized, aspect='auto', cmap='jet', origin='lower', interpolation='nearest')
    plt.colorbar(label='Normalized CWT Coefficients')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.title('Scalogram')
    
    # Save to an in-memory buffer and return as PIL image.
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    image = PIL.Image.open(buffer).copy()
    buffer.close()

    # Save image with a unique label into the scalograms directory.
    out_dir = Path(output_dir) if output_dir is not None else _default_scalogram_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_path = out_dir / _unique_scalogram_name(label)
    image.save(saved_path, format="PNG")

    return image, str(saved_path)

def open_scalogram_image(file_path: str) -> PIL.Image:
    return PIL.Image.open(file_path)


