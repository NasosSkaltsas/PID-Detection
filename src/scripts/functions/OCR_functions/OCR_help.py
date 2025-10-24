
import pytesseract
from pytesseract import Output
from typing import Optional, Tuple 
import cv2
import numpy as np
import pandas as pd
pytesseract.pytesseract.tesseract_cmd = r"C:\venv\tesseract.exe"
from pathlib import Path


#  Color & preprocessing 
def magenta_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Keep magenta/pink letters in BGR image.
    Returns a binary mask (255 = keep).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = (np.array([140, 60, 60]), np.array([180, 255, 255]))  # magenta
    lower2, upper2 = (np.array([0,   60, 60]), np.array([10, 255, 255]))   # pinkish-red wrap
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.erode(mask, np.ones((1, 1), np.uint8), iterations=10)

    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    return mask


def binarize_for_ocr(bgr_roi: np.ndarray) -> np.ndarray:
    """
    Make a high-contrast, OCR-friendly patch: black text on white background,
    optionally upsampled, with light sharpening.
    """
    mask = magenta_mask(bgr_roi)
    bin_img = 255 - mask

    # Gentle denoise + unsharp for thin strokes
    bin_img = cv2.GaussianBlur(bin_img, (1, 1), 0)
    bin_img = cv2.addWeighted(bin_img, 1, cv2.GaussianBlur(bin_img, (0, 0), 1.0), -0.5, 0)
    return bin_img

#  OCR helpers 
def run_tesseract(img_bin: np.ndarray, config: str) -> pd.DataFrame:
    return pytesseract.image_to_data(img_bin, lang="eng", output_type=Output.DATAFRAME, config=config)


def sanitize_tess_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # Ensure expected columns & types
    for col in ["text", "conf", "block_num", "par_num", "line_num", "left"]:
        if col not in df:
            df[col] = 0
    df["text"] = df["text"].astype(str)
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1)
    return df

def best_line_text_and_bbox(df: pd.DataFrame) -> Tuple[str, float, Optional[Tuple[int, int, int, int]]]:
    """
    Like best_line_text, but also returns the bounding box of the selected line
    in the SAME coordinate system as the Tesseract dataframe (i.e., the ROI passed to Tesseract).
    Returns: (text, mean_conf, (x, y, w, h)) or (text, -1.0, None) if nothing valid.
    """
    df = sanitize_tess_df(df)
    if df.empty:
        return "", -1.0, None

    keep = df["text"].astype(str).str.strip().ne("") & df["text"].str.lower().ne("nan")
    df = df[keep]
    if df.empty:
        return "", -1.0, None

    df["line_id"] = list(zip(df["block_num"], df["par_num"], df["line_num"]))

    best = ""
    best_conf = -1.0
    best_bbox = None

    for _, g in df.groupby("line_id"):
        g = g.sort_values("left")

        text = " ".join(g["text"].tolist()).strip()
        conf = float(g["conf"].mean())

        lefts  = g["left"].astype(int).to_numpy()
        tops   = g["top"].astype(int).to_numpy()
        widths = g["width"].astype(int).to_numpy()
        heights= g["height"].astype(int).to_numpy()
        if len(lefts) == 0:
            continue
        x1 = int(lefts.min())
        y1 = int(tops.min())
        x2 = int((lefts + widths).max())
        y2 = int((tops + heights).max())
        bbox = (x1, y1, x2 - x1, y2 - y1)

        if conf > best_conf:
            best, best_conf, best_bbox = text, conf, bbox

    return (best or "").strip(), best_conf, best_bbox

#  Main: read text near a detection 
#  OCR configs 
TAG_CFG  = r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# # A slightly looser fallback to catch variants/spacing issues
# (WILL BE SKIPPED FOR NOW)
TAG_CFG_ALT = r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def _crop_centered(bgr: np.ndarray, cx: int, cy: int, win_w: int, win_h: int) -> Tuple[np.ndarray, int, int, int, int]:
    """Crop a window centered at (cx, cy); returns (roi, x1, y1, x2, y2) in IMAGE coords."""
    H, W = bgr.shape[:2]
    x1 = max(0, cx - win_w // 2)
    y1 = max(0, cy - win_h // 2)
    x2 = min(W, x1 + win_w)
    y2 = min(H, y1 + win_h)
    if x2 - x1 < win_w:
        x1 = max(0, min(W - win_w, x1 - (win_w - (x2 - x1))))
        x2 = min(W, x1 + win_w)
    if y2 - y1 < win_h:
        y1 = max(0, min(H - win_h, y1 - (win_h - (y2 - y1))))
        y2 = min(H, y1 + win_h)
    roi = bgr[y1:y2, x1:x2]
    return roi, x1, y1, x2, y2


def _ocr_best_of_two(img_bin: np.ndarray, cfg_a: str, cfg_b: str) -> Tuple[str, float, Optional[Tuple[int, int, int, int]]]:
    """
    Run two Tesseract configs and return (text, conf, bbox_roi),
    where bbox_roi is (x, y, w, h) in the ROI coordinate system.
    """
    # # Optional: show image for debugging
    # import matplotlib.pyplot as plt
    # plt.imshow(img_bin, cmap='gray')
    # plt.title("Binary OCR Input Image")
    # plt.axis('off')
    # plt.show()

    dfA = run_tesseract(img_bin, cfg_a)
    tA, cA, bbA = best_line_text_and_bbox(dfA)

    dfB = run_tesseract(img_bin, cfg_b)
    tB, cB, bbB = best_line_text_and_bbox(dfB)

    # Pick by confidence; tie-breaker = longer alnum text
    if cB > cA or (abs(cB - cA) < 1e-6 and sum(ch.isalnum() for ch in tB) > sum(ch.isalnum() for ch in tA)):
        return tB, cB, bbB
    return tA, cA, bbA



def read_tag_around_point(
    img_bgr: np.ndarray,
    cx: int,
    cy: int,
    *,
    offset_x: int = 40,
    offset_y: int = 40,
    win_w: int = 80,
    win_h: int = 80,
    cfg_primary: str = TAG_CFG,
    cfg_alt: str = TAG_CFG_ALT,
) -> Tuple[str, float, Optional[Tuple[int, int, int, int]]]:
    """
    Returns: (best_text, best_conf, best_bbox_img)
    where best_bbox_img is (x, y, w, h) in FULL-IMAGE coordinates for the selected line.
    If nothing qualifies, returns ("", -1.0, None).
    """
    centers = {
        "top":    (cx, cy - offset_y),
        "bottom": (cx, cy + offset_y),
        "left":   (cx - offset_x, cy),
        "right":  (cx + offset_x, cy),
    }

    candidates: list[Tuple[int, float, str, str, Optional[Tuple[int, int, int, int]]]] = []

    for label, (ux, uy) in centers.items():
        roi, x1, y1, x2, y2 = _crop_centered(img_bgr, ux, uy, win_w, win_h)
        if roi.size == 0:
            continue

        roi_bin = binarize_for_ocr(roi)
        text, conf, bbox_roi = _ocr_best_of_two(roi_bin, cfg_primary, cfg_alt)
        text = text.strip()
        bbox_img: Optional[Tuple[int, int, int, int]] = None
        if bbox_roi is not None:
            bx, by, bw, bh = bbox_roi
            bbox_img = (x1 + bx, y1 + by, bw, bh)

        alnum_len = sum(ch.isalnum() for ch in text)
        if alnum_len >=3:
            candidates.append((alnum_len, float(conf), text, label, bbox_img))

    if not candidates:
        return "", -1.0, None

    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    _, best_conf, best_text, best_label, best_bbox_img = candidates[0]
    return best_text, best_conf, best_bbox_img

def first_char(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with name_ocr's first character corrected by rules:
      - if label == 'valve' or 'valve_oneway'      -> first char 'V'
      - if label == 'compensator'                  -> first char 'A'
      - if 'pressure' or 'temperature' in label    -> first char 'S'
    Rules are case-insensitive on `label`. Exact matches take precedence over the
    substring ('pressure'/'temperature') rule.
    """
    if "label" not in df.columns or "name_ocr" not in df.columns:
        raise ValueError("DataFrame must contain 'label' and 'name_ocr' columns.")

    out = df.copy()

    labels = (
        out["label"]
        .fillna("")
        .astype(str)
        .str.lower()
    )
    names = out["name_ocr"].fillna("").astype(str)

    target = pd.Series(np.nan, index=out.index, dtype="object")

    target = target.mask(labels.isin(["valve", "valve_oneway"]), "V")
    target = target.mask(labels.eq("compensator"), "A")

    has_pressure_temp = labels.str.contains("pressure", na=False) | labels.str.contains("temperature", na=False)
    target = target.mask(has_pressure_temp & target.isna(), "S")

    mask = target.notna()
    corrected = names.copy()
    corrected.loc[mask] = target.loc[mask] + names.loc[mask].str[1:]

    out["name_ocr"] = corrected
    return out




def run_component_ocr(df_csv_path, image_path):
    """
    Runs OCR around detected component centers and saves updated detections CSV.

    Args:
        df_csv_path (str or Path): Path to detections CSV (output from model).
        image_path (str or Path): Path to the source image.
        detections_dir (str or Path): Directory to save updated detections CSV.

    Returns:
        pd.DataFrame: Updated detections DataFrame with OCR results.
    """
    image_path = Path(image_path)
    det_dir = Path(df_csv_path)

    base = image_path.stem
    detections_csv_path = det_dir / f"{base}_detections.csv"

    df_det = pd.read_csv(detections_csv_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    rows = df_det.to_dict(orient="records")

    # Run OCR per detection 
    for r in rows:
        if r.get("label") == "pump":
            # Skip OCR for pumps
            r["name_ocr"] = "pump"
            r["ocr_conf"] = -1.0
            r["text_bbox"] = None
            continue

        # Run OCR in 4 regions around the component
        text, conf, bbox = read_tag_around_point(
            img,
            int(r["cx"]),
            int(r["cy"]),
            offset_x=80,
            offset_y=40,
            win_w=150,
            win_h=60,
        )

        # Store results
        r["name_ocr"] = text
        r["ocr_conf"] = conf
        r["text_bbox"] = bbox 

    #  Build new DataFrame 
    df_det = pd.DataFrame(rows)
    df_det["name_ocr"] = (
        df_det["name_ocr"].astype(str).str.replace(r"\.0$", "", regex=True)
    )

    #  Save updated detections 
    base = image_path.stem
    save_path = det_dir / f"{base}_detections.csv"

    first_char(df_det).to_csv(save_path, index=False)

    print(f"✅ OCR-enhanced detections saved to: {save_path}")

    return df_det
