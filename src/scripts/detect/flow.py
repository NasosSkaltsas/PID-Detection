import shlex
import subprocess
import sys
from pathlib import Path
import os
import base64
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact,create_table_artifact
import csv
import json
import pandas as pd

import mimetypes

@task(task_run_name="input_image")
def create_input_image_artifact(base_dir: str, artifact_key: str = "input-image") -> None:
    """
    Reads IMAGE_PATH from the environment, resolves it (relative to base_dir if needed),
    and publishes the image as a markdown artifact.
    """
    logger = get_run_logger()

    env_path = os.getenv("IMAGE_PATH", "").strip()
    if not env_path:
        logger.warning("IMAGE_PATH is not set; skipping input image artifact.")
        return

    candidate = Path(env_path)
    if not candidate.is_absolute():
        candidate = Path(base_dir) / candidate
    candidate = candidate.resolve()

    if not candidate.exists():
        logger.warning("IMAGE_PATH points to a non-existent file: %s", candidate)
        return

    # Guess mime (default to png if unknown)
    mime, _ = mimetypes.guess_type(candidate.name)
    if mime is None:
        # fallback based on extension
        ext = candidate.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif ext in {".gif"}:
            mime = "image/gif"
        elif ext in {".webp"}:
            mime = "image/webp"
        else:
            mime = "image/png"

    with open(candidate, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    title = f"### Input Image\n\n"
    markdown = f'{title}![input image](data:{mime};base64,{encoded})'
    create_markdown_artifact(
        key=artifact_key,
        markdown=markdown,
        description=f"Inspected P&ID diagram"
    )

def csv_to_table_artifact(csv_path: str, artifact_name: str = None):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    create_table_artifact(
        key=artifact_name or Path(csv_path).stem,
        table=rows,
        description=f"Table artifact from {csv_path}",
    )

@task(task_run_name="{script_name}")
def run_script(script: str,script_name: str) -> None:
    """Run a Python script as a subprocess.

    Parameters
    ----------
    script : str
        Path to the .py file to run.
    """

    logger = get_run_logger()

    script_path = Path(script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Determine project root and src path
    project_root = script_path.parents[2]  # e.g., .../digitization-pid
    src_dir = project_root / "src"

    # Build command to run script as a module (preferred)
    # or just run it directly with PYTHONPATH set
    cmd = [sys.executable, str(script_path)]

    # Prepare environment with PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")

    logger.info("Running command: %s", shlex.join(cmd))
    logger.info("Working directory: %s", project_root)

    # Run subprocess
    result = subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )

    # Capture and log output
    if result.stdout:
        logger.info("stdout:\n%s", result.stdout.rstrip())
    if result.stderr:
        logger.warning("stderr:\n%s", result.stderr.rstrip())

    result.check_returncode()


@flow(name="PID-Detection")
def main_flow():
    # --- read env variables from Prefect deployment ---
    base_dir       = Path(os.getenv("BASE_DIR", ".")).resolve()
    detections_dir = str(base_dir / os.getenv("DETECTIONS_DIR", "app/detections"))
    artifacts_dir  = str(base_dir / os.getenv("ARTIFACTS_PATH", "app/artifacts"))

    create_input_image_artifact(str(base_dir), artifact_key="input-pid")

    run_script("scripts/detect/symbol_detect.py",script_name="symbol_detection")

    img_path = detections_dir + "/" + "test2_annotated.png"
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    markdown = f"### Detected Symbols\n\n![detected symbols](data:image/png;base64,{encoded})"
    create_markdown_artifact(
        key="symbol-detection",
        markdown=markdown,
        description="Annotated symbol detection output"
    )

    run_script("scripts/detect/text_detect.py",script_name="text_detection")
    csv_to_table_artifact(detections_dir + "/" + "test2-detections.csv",artifact_name="tagged-components")

    run_script("scripts/detect/line_detect.py",script_name="line_detection")

    for img_name in [
        "detect_lines.png",
        "detect_lines_thicker.png",
        "detected_pipeline.png"
    ]:
        img_path = artifacts_dir + "/" + img_name
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        markdown = f"### {img_name.replace('_', '-')}\n\n![{img_name.replace('_', '-')}](data:image/png;base64,{encoded})"
        create_markdown_artifact(key=img_name.split('.')[0].replace('_', '-'), markdown=markdown)

    run_script("scripts/detect/network_detect.py",script_name="network_detection")
    img_path = artifacts_dir + "/" + "detected_network.png"
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    markdown = f"### Detected Network\n\n![detected network](data:image/png;base64,{encoded})"
    create_markdown_artifact(key="network-detection", markdown=markdown)

    edges_path = artifacts_dir + "/" + "network_edges.json"

    with open(edges_path, "r", encoding="utf-8") as f:
        rows = json.load(f) 

    create_table_artifact(
        key="network-edges",
        table=rows,
        description=f"P&ID Detected Network",
)
if __name__ == "__main__":
    main_flow()