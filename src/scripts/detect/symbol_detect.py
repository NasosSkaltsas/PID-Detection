import sys
sys.path.append('..')
from scripts.functions.global_functions.config_load import load_config
from scripts.functions.yolo_functions.detect_objects import detect_objects
from ultralytics import YOLO

if __name__ == "__main__":
    cfg = load_config()
    image_path = str(cfg['IMAGE_PATH'])
    detections_path = str(cfg['DETECTIONS_DIR'])
    model= YOLO(str(cfg['MODEL_PATH']))  

    df_det = detect_objects(image_path,model, detections_path)