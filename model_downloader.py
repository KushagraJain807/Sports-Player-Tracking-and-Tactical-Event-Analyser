
from pathlib import Path
import os
import requests

MODEL_URL = (
    "https://huggingface.co/KushagraJain807/"
    "TactiVision-AI-Model/resolve/main/best.pt"
)

def ensure_model():
    model_path = Path(__file__).resolve().parent / "models" / "best.pt"

    MIN_MODEL_SIZE = 180 * 1024 * 1024

    if model_path.is_file() and model_path.stat().st_size >= MIN_MODEL_SIZE:
        return str(model_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(".pt.download")

    try:
        with requests.get(MODEL_URL, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

        if temp_path.stat().st_size < MIN_MODEL_SIZE:
            raise RuntimeError("Model download is incomplete.")
        os.replace(temp_path, model_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return str(model_path)
