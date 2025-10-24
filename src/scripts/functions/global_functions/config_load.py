from pathlib import Path
import os
import sys
from dotenv import load_dotenv


def load_config(env_path: str | None = None) -> dict:
    """
    Load environment variables and prepare project paths.

    Args:
        env_path (str | None): Optional custom .env file path.
    
    Returns:
        dict: A dictionary containing resolved Path objects for 
              base, outputs, detections, and image.
    """
    # --- Load environment file ---
    load_dotenv(dotenv_path=env_path)

    # --- Resolve base paths ---
    base_dir = Path(os.getenv("BASE_DIR", ".")).resolve()

    outputs_dir = Path(
        os.getenv("OUTPUTS_DIR")
    ).resolve()

    detections_dir = Path(
        os.getenv("DETECTIONS_DIR")
    ).resolve()

    image_path = Path(
        os.getenv("IMAGE_PATH")
    ).resolve()

    model_path = Path(
        os.getenv("MODEL_PATH")
    ).resolve()

    artifacts_path = Path(
        os.getenv("ARTIFACTS_PATH")
    ).resolve()

    # --- Add module root to sys.path if needed ---
    modules_root = Path(os.getenv("MODULES_ROOT", "..")).resolve()
    if str(modules_root) not in sys.path:
        sys.path.append(str(modules_root))

    # --- Ensure output directories exist ---
    detections_dir.mkdir(parents=True, exist_ok=True)

    # --- Return everything neatly ---
    return {
        "BASE_DIR": base_dir,
        "OUTPUTS_DIR": outputs_dir,
        "DETECTIONS_DIR": detections_dir,
        "IMAGE_PATH": image_path,
        "MODULES_ROOT": modules_root,
        "MODEL_PATH": model_path,
        "ARTIFACTS_PATH": artifacts_path
    }
