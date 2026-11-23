
from pathlib import Path


def ensure_model():
    """Return the locally fine-tuned TactiVision AI model."""

    model_path = (
        Path(__file__).resolve().parent
        / "models"
        / "best_trained.pt"
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    if model_path.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Model checkpoint appears incomplete: {model_path}"
        )

    return str(model_path)
