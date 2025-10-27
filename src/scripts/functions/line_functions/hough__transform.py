
import numpy as np
import pandas as pd
from scripts.functions.global_functions.config_load import load_config
cfg = load_config()
artifacts_path = str(cfg['ARTIFACTS_PATH'])


def merge_axis_aligned_lines(segments, axis_tol=6, overlap_tol=6):
    """
    Merge nearly collinear and overlapping axis-aligned segments.
    - axis_tol: how close (in px) two lines must be on the perpendicular axis to be considered the same line.
    - overlap_tol: how much gap (in px) is allowed to still merge intervals.
    """
    horizontals, verticals = [], []

    # Normalize orientation and endpoints
    for x1, y1, x2, y2 in segments:
        if abs(y2 - y1) <= abs(x2 - x1):  # treat as horizontal
            if x2 < x1: x1, x2, y1, y2 = x2, x1, y2, y1
            horizontals.append((x1, int(round((y1 + y2) / 2)), x2))
        else:  # vertical
            if y2 < y1: x1, x2, y1, y2 = x2, x1, y2, y1
            verticals.append((int(round((x1 + x2) / 2)), y1, y2))

    def cluster_and_merge(intervals, key_idx, a_idx, b_idx):
        """
        intervals: list of tuples like (k, a, b) where
          - k is the near-constant coordinate (y for horizontals, x for verticals)
          - [a,b] is the span along the line (x-range for horizontals, y-range for verticals)
        key_idx: index of k in the tuple
        a_idx, b_idx: indices of the span ends
        """
        # sort by key (y or x), then by start of interval
        intervals = sorted(intervals, key=lambda t: (t[key_idx], t[a_idx], t[b_idx]))
        groups = []

        for t in intervals:
            placed = False
            for g in groups:
                # same "rail" if near in the perpendicular axis
                if abs(t[key_idx] - g["key_mean"]) <= axis_tol:
                    g["members"].append(t)
                    g["key_sum"] += t[key_idx]
                    g["key_mean"] = g["key_sum"] / len(g["members"])
                    placed = True
                    break
            if not placed:
                groups.append({"key_sum": t[key_idx], "key_mean": float(t[key_idx]), "members": [t]})

        merged = []
        for g in groups:
            # merge overlapping/close intervals within this rail
            mem = sorted(g["members"], key=lambda t: t[a_idx])
            cur_a, cur_b = mem[0][a_idx], mem[0][b_idx]
            for t in mem[1:]:
                a, b = t[a_idx], t[b_idx]
                if a <= cur_b + overlap_tol: 
                    cur_b = max(cur_b, b)
                else:
                    merged.append((int(round(g["key_mean"])), cur_a, cur_b))
                    cur_a, cur_b = a, b
            merged.append((int(round(g["key_mean"])), cur_a, cur_b))
        return merged

    # Merge within each orientation
    merged_h = cluster_and_merge([(y, x1, x2) for (x1, y, x2) in horizontals],
                                 key_idx=0, a_idx=1, b_idx=2)  
    merged_v = cluster_and_merge([(x, y1, y2) for (x, y1, y2) in verticals],
                                 key_idx=0, a_idx=1, b_idx=2) 

    # Convert back to (x1,y1,x2,y2)
    merged_segments = []
    for (y, x1, x2) in merged_h:
        merged_segments.append((x1, y, x2, y))
    for (x, y1, y2) in merged_v:
        merged_segments.append((x, y1, x, y2))

    return merged_segments

import cv2
import numpy as np
import pandas as pd
import os

def detect_lines_hough(
    img: np.ndarray,
    threshold_value: int = 200,
    canny_thresh1: int = 50,
    canny_thresh2: int = 150,
    hough_threshold: int = 30,
    min_line_length: int = 10,
    max_line_gap: int = 6,
    axis_tolerance_deg: float = 10,
    run_merge: bool = True,
) -> pd.DataFrame:
    """
    Detects straight lines in an image using the Probabilistic Hough Transform,
    keeps only near-horizontal or near-vertical lines, and optionally merges
    axis-aligned segments.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image.
    save_dir : str, optional
        Directory to save visual outputs. Default: "artifacts/line_detection".
    threshold_value : int, optional
        Threshold for binarizing the grayscale image. Default: 200.
    canny_thresh1 : int, optional
        Lower Canny edge detection threshold. Default: 50.
    canny_thresh2 : int, optional
        Upper Canny edge detection threshold. Default: 150.
    hough_threshold : int, optional
        Minimum number of votes for Hough line detection. Default: 30.
    min_line_length : int, optional
        Minimum length of detected line segments (in pixels). Default: 10.
    max_line_gap : int, optional
        Maximum gap between line segments to link them (in pixels). Default: 6.
    axis_tolerance_deg : float, optional
        Angular tolerance (in degrees) to consider a line horizontal or vertical. Default: 10°.
    run_merge : bool, optional
        If True, merges axis-aligned segments using merge_axis_aligned_lines().
        Default: False.

    Returns
    -------
    pandas.DataFrame
        Table containing detected (and optionally merged) line data:
        columns = ["ID", "x1", "y1", "x2", "y2", "Angle (deg)", "Length (px)"].
    """


    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Create binary mask for line-like pixels 
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    #Morphological closing to bridge gaps 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    #  Edge detection 
    edges = cv2.Canny(mask_clean, canny_thresh1, canny_thresh2)

    # Probabilistic Hough Transform 
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    # axis-aligned  
    def is_axis_aligned(x1, y1, x2, y2, tol_deg=axis_tolerance_deg):
        dx, dy = x2 - x1, y2 - y1
        angle = (np.degrees(np.arctan2(dy, dx)) + 180) % 180
        return min(abs(angle - 0), abs(angle - 180)) < tol_deg or abs(angle - 90) < tol_deg

    # Collect and draw accepted lines 
    vis = img.copy()
    accepted = []
    line_data = []

    if lines is not None:
        for i, (x1, y1, x2, y2) in enumerate(lines[:, 0, :]):
            dx, dy = x2 - x1, y2 - y1
            angle = (np.degrees(np.arctan2(dy, dx)) + 180) % 180
            length = float(np.hypot(dx, dy))
            if is_axis_aligned(x1, y1, x2, y2):
                accepted.append((x1, y1, x2, y2))
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                line_data.append([i, x1, y1, x2, y2, round(angle, 2), round(length, 2)])

    #  Optional merging 
    if run_merge:
        try:
            merged_segments = merge_axis_aligned_lines(accepted, axis_tol=axis_tolerance_deg, overlap_tol=0)
            vis_merged = img.copy()
            merged_data = []
            for i, (x1, y1, x2, y2) in enumerate(merged_segments):
                cv2.line(vis_merged, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(vis_merged, (x1, y1), 4, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(vis_merged, (x1, y1), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(vis_merged, (x2, y2), 4, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(vis_merged, (x2, y2), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

                angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180) % 180
                length = float(np.hypot(x2 - x1, y2 - y1))
                merged_data.append([i, x1, y1, x2, y2, round(angle, 2), round(length, 2)])

            merged_df = pd.DataFrame(merged_data, columns=["ID", "x1", "y1", "x2", "y2", "Angle (deg)", "Length (px)"])
            merged_path = os.path.join(artifacts_path, "/detected_hough_lines.png")
            cv2.imwrite(merged_path, vis_merged)
            return merged_df
        except NameError:
            raise NameError("merge_axis_aligned_lines() is not defined yet.")

    # return non-merged detected lines
    df = pd.DataFrame(line_data, columns=["ID", "x1", "y1", "x2", "y2", "Angle (deg)", "Length (px)"])

    df.to_csv(artifacts_path+"\\"+"hough_lines.csv", index=False)

    return df
