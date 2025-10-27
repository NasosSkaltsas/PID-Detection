import shlex
import subprocess
import sys
from pathlib import Path
import os

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_image_artifact,create_table_artifact
import csv

def csv_to_table_artifact(csv_path: str, artifact_name: str = None):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    create_table_artifact(
        key=artifact_name or Path(csv_path).stem,
        table=rows,
        description=f"Table artifact from {csv_path}",
    )

@task(name="{script}")
def run_script(script: str) -> None:
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
    detections_dir = base_dir / os.getenv("DETECTIONS_DIR", "app/detections")
    artifacts_dir  = base_dir / os.getenv("ARTIFACTS_PATH", "app/artifacts")

    # --- your original logic, now using those env vars ---
    run_script("scripts/detect/symbol_detect.py")
    create_image_artifact(str(detections_dir) + "/" + "test2_annotated.png")

    run_script("scripts/detect/text_detect.py")
    csv_to_table_artifact(str(detections_dir) + "/" + "test2-detections.csv")

    run_script("scripts/detect/line_detect.py")
    create_image_artifact(str(detections_dir) + "/" + "detect_lines.png")
    create_image_artifact(str(detections_dir) + "/" + "detect_lines_thicker.png")
    create_image_artifact(str(detections_dir) + "/" + "detected_pipeline.png")

    run_script("scripts/detect/network_detect.py")
    create_image_artifact(str(detections_dir) + "/" + "detected_network.png")


if __name__ == "__main__":
    main_flow()