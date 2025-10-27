import os
from pathlib import Path
import cv2
import pandas as pd

def detect_objects(image_path, model, detections_dir, imgsz=640, conf=0.2, verbose=False):
    """
    Run object detection on an image, draw bounding boxes, and save results.

    Args:
        image_path (str or Path): Path to the input image.
        model: Loaded YOLO model instance.
        detections_dir (str or Path): Directory to save annotated images and CSV files.
        imgsz (int, optional): Image size for inference. Default is 640.
        conf (float, optional): Confidence threshold. Default is 0.2.
        verbose (bool, optional): Whether to print model output logs. Default is False.

    Returns:
        pd.DataFrame: DataFrame of detections.
    """
    image_path = Path(image_path)
    detections_dir = Path(detections_dir)
    detections_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(source=img, imgsz=imgsz, conf=conf, verbose=verbose)[0]
    
    rows = []
    for b in results.boxes:
        cls_id = int(b.cls.item())
        conf_val = float(b.conf.item())
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        label = results.names[cls_id]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        rows.append({
            'label': label,
            'conf': conf_val,
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'cx': cx, 'cy': cy
        })

    df_det = pd.DataFrame(rows)

    annot = img.copy()
    for r in rows:
        x1, y1, x2, y2 = map(int, [r['x1'], r['y1'], r['x2'], r['y2']])
        cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annot, f"{r['label']} {r['conf']:.2f}", 
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    base = os.path.basename(image_path).rsplit('.', 1)[0]

    annotated_path = detections_dir / f"{base}_annotated.png"
    csv_path = detections_dir / f"{base}-detections.csv"

    cv2.imwrite(str(annotated_path), annot)
    df_det.to_csv(csv_path, index=False)

    print(f"✅ Saved annotated image to: {annotated_path}")
    print(f"✅ Saved detections CSV to:  {csv_path}")

    return df_det
