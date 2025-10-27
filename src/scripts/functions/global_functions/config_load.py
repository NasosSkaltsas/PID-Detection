from pathlib import Path
import os
import sys
from dotenv import load_dotenv
from typing import Optional, Dict

def _resolve_from_base(val: Optional[str], base: Path, default: Optional[str] = None) -> Path:
    """
    Resolve a path string relative to `base` if it's not absolute.
    If `val` is None, use `default`.
    """
    s = val if val is not None else default
    if s is None:
        raise ValueError("Required path env var is missing and no default was provided.")
    p = Path(s)
    return (base / p).resolve() if not p.is_absolute() else p.resolve()

def load_config(env_path: Optional[str] = None) -> Dict[str, Path]:
    """
    Load environment variables and prepare project paths.
    """
    load_dotenv(dotenv_path=env_path)

    # Base directory (set this in your env to your repo root, or leave "." if you run from the repo root)
    base_dir = Path(os.getenv("BASE_DIR", ".")).resolve()

    outputs_dir    = _resolve_from_base(os.getenv("OUTPUTS_DIR"),    base_dir, default="app")
    detections_dir = _resolve_from_base(os.getenv("DETECTIONS_DIR"), base_dir, default="app/detections")
    artifacts_path = _resolve_from_base(os.getenv("ARTIFACTS_PATH"), base_dir, default="app/artifacts")
    image_path     = _resolve_from_base(os.getenv("IMAGE_PATH"),     base_dir)
    model_path     = _resolve_from_base(os.getenv("MODEL_PATH"),     base_dir)
    modules_root   = _resolve_from_base(os.getenv("MODULES_ROOT"),   base_dir, default="..")

    if str(modules_root) not in sys.path:
        sys.path.append(str(modules_root))

    outputs_dir.mkdir(parents=True, exist_ok=True)
    detections_dir.mkdir(parents=True, exist_ok=True)

    return {
        "BASE_DIR": base_dir,
        "OUTPUTS_DIR": outputs_dir,
        "DETECTIONS_DIR": detections_dir,
        "IMAGE_PATH": image_path,
        "MODULES_ROOT": modules_root,
        "MODEL_PATH": model_path,
        "ARTIFACTS_PATH": artifacts_path,
    }

