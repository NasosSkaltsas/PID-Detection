import cv2
import os
import numpy as np
import pandas as pd
import ast
from pathlib import Path
from PIL import Image
from scripts.functions.global_functions.config_load import load_config
cfg = load_config()
artifacts_path = str(cfg['ARTIFACTS_PATH'])

def clamp_box(x1, y1, x2, y2, w, h):
    """
    Clamp a bounding box to ensure its coordinates lie within image boundaries.

    Parameters
    ----------
    x1, y1, x2, y2 : float or int
        Original bounding box coordinates.
    w, h : int
        Image width and height.

    Returns
    -------
    tuple[int, int, int, int]
        Clamped bounding box (x1, y1, x2, y2), adjusted to stay within image bounds.
        Guarantees x1 <= x2 and y1 <= y2.
    """
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def pad_box(x1, y1, x2, y2, w, h, ratio):
    """
    Expand a bounding box by a proportional padding ratio while keeping it within image bounds.

    Parameters
    ----------
    x1, y1, x2, y2 : float or int
        Original bounding box coordinates.
    w, h : int
        Image width and height.
    ratio : float
        Padding ratio relative to the box width/height. 
        For example, 0.1 adds 10% padding on each side.

    Returns
    -------
    tuple[int, int, int, int]
        Padded and clamped bounding box (x1, y1, x2, y2).
    """
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = int(bw * ratio), int(bh * ratio)
    return clamp_box(x1 - px, y1 - py, x2 + px, y2 + py, w, h)

def parse_ocr_bbox(s):
    """
    Parse an OCR bounding box string into numeric coordinates.

    The input is expected to be a string representation of a 4-tuple, e.g. "(x, y, w, h)".
    Returns None if the input is invalid or cannot be parsed.

    Parameters
    ----------
    s : str or None
        String containing the OCR bounding box, or None.

    Returns
    -------
    tuple[int, int, int, int] or None
        Parsed bounding box (x, y, w, h) as integers, or None if invalid.
    """
    if s is None or s == "None" or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        x, y, w, h = ast.literal_eval(s)
        return int(x), int(y), int(w), int(h)
    except Exception:
        return None



def process_image_cleaning(
    image_path: str,
    detections_csv_path: str,
    pad_ratio_obj: float = 0.25,
    pad_ratio_ocr: float = 0.10,
    dilate_kernel: tuple = (3, 3),
    inpaint_radius_text: int = 2,
    inpaint_radius_obj: int = 1,
    node_radius: int = 6,
    node_thickness: int = -1,
) -> np.ndarray:
    """
    Cleans an image by removing OCR text and component boxes defined in a CSV,
    then draws green centroid nodes (no numbering) on the cleaned image.

    Parameters
    ----------
    image_path : str
        Path to img
    detections_csv_path : str
        Path to CSV file containing bounding boxes with columns (x1, y1, x2, y2, text_bbox).
    pad_ratio_obj : float, optional
        Padding ratio for component boxes.
    pad_ratio_ocr : float, optional
        Padding ratio for OCR text boxes.
    dilate_kernel : tuple, optional
        Kernel size for morphological dilation.
    inpaint_radius_text : int, optional
        Inpainting radius for text removal.
    inpaint_radius_obj : int, optional
        Inpainting radius for object removal.
    node_radius : int, optional
        Radius of centroid nodes (in pixels).
    node_thickness : int, optional
        Thickness of centroid circles (-1 = filled).
    save_dir : str, optional
        Directory where the processed image will be saved.

    Returns
    -------
    np.ndarray
        The cleaned image with centroid nodes drawn.
    """
    image_path = Path(image_path)
    det_dir = Path(detections_csv_path)

    # Build the CSV path based on the image base name
    base = image_path.stem
    detections_csv_path = det_dir / f"{base}-detections.csv"
    img = cv2.imread(image_path)
    df = pd.read_csv(detections_csv_path)



    H, W = img.shape[:2]
    mask_obj = np.zeros((H, W), dtype=np.uint8)
    mask_txt = np.zeros((H, W), dtype=np.uint8)
    component_centroids = []

    # --- Build masks and collect centroids ---
    for _, r in df.iterrows():
        x1, y1, x2, y2 = float(r.x1), float(r.y1), float(r.x2), float(r.y2)
        cx, cy = int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))
        component_centroids.append((cx, cy))

        mx1, my1, mx2, my2 = pad_box(x1, y1, x2, y2, W, H, pad_ratio_obj)
        cv2.rectangle(mask_obj, (mx1, my1), (mx2, my2), 255, -1)

        ob = parse_ocr_bbox(r.get("text_bbox", None))
        if ob:
            tx, ty, tw, th = ob
            px1, py1, px2, py2 = pad_box(tx, ty, tx + tw, ty + th, W, H, pad_ratio_ocr)
            cv2.rectangle(mask_txt, (px1, py1), (px2, py2), 255, -1)

    # --- Dilation to smooth mask edges ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    mask_obj = cv2.dilate(mask_obj, kernel, iterations=1)
    mask_txt = cv2.dilate(mask_txt, kernel, iterations=1)

    # --- Inpaint text & symbols ---
    img_pass1 = cv2.inpaint(img, mask_txt, inpaint_radius_text, cv2.INPAINT_TELEA)
    img_clean = cv2.inpaint(img_pass1, mask_obj, inpaint_radius_obj, cv2.INPAINT_TELEA)

    # --- Draw centroid nodes ---
    COLOR_NODE = (0, 255, 0)  # Green
    clean_img = img_clean.copy()

    for (cx, cy) in component_centroids:
        cv2.circle(clean_img, (cx, cy), node_radius, COLOR_NODE, node_thickness, lineType=cv2.LINE_AA)

    cv2.imwrite(artifacts_path+"/detect_lines.png", clean_img)

    return clean_img

def thicken_edges(
    img: np.ndarray,
    edge_method: str = "canny",
    canny_thresh1: int = 50,
    canny_thresh2: int = 150,
    dilation_kernel: tuple = (3, 3),
    dilation_iters: int = 1,
    overlay_color: tuple = (0, 0, 0)
) -> np.ndarray:
    """
    Detects edges in an image, thickens them using morphological dilation,
    overlays the thickened edges on the original image, and saves the result.

    Parameters
    ----------
    img_path : np.darray
        Image 
    save_dir : str, optional
        Directory where the result will be saved. Default is 'artifacts/line_detection'.
    edge_method : str, optional
        Edge detection method ('canny' or 'adaptive'). Default is 'canny'.
    canny_thresh1 : int, optional
        Lower threshold for Canny edge detection. Default is 50.
    canny_thresh2 : int, optional
        Upper threshold for Canny edge detection. Default is 150.
    dilation_kernel : tuple, optional
        Size of the kernel used to thicken edges. Default is (3, 3).
    dilation_iters : int, optional
        Number of dilation iterations. Larger values produce thicker lines.
    overlay_color : tuple, optional
        BGR color used to overlay edges. Default is black (0, 0, 0).

    Returns
    -------
    np.ndarray
        Processed image (with thickened black edges).
    """
    # === Convert to grayscale ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Detect edges ===
    if edge_method.lower() == "canny":
        edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
    elif edge_method.lower() == "adaptive":
        # Adaptive edge detection using local thresholding
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3
        )
    else:
        raise ValueError("edge_method must be 'canny' or 'adaptive'")

    # === Thicken detected edges ===
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilation_kernel)
    thick_edges = cv2.dilate(edges, kernel, iterations=dilation_iters)

    # === Overlay thickened edges on original image ===
    thickened_img = img.copy()
    thickened_img[thick_edges > 0] = overlay_color 

    # === Save output ===
    save_path = os.path.join(artifacts_path, "detect_lines_thicker.png")
    cv2.imwrite(save_path, thickened_img)

    return thickened_img