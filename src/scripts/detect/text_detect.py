import sys
sys.path.append('..')
from functions.OCR_functions.OCR_help import run_component_ocr
from functions.global_functions.config_load import load_config

if __name__ == "__main__":
    cfg = load_config()
    image_path = str(cfg['IMAGE_PATH'])
    detections_path = str(cfg['DETECTIONS_DIR'])

    df_updated = run_component_ocr(detections_path, image_path)