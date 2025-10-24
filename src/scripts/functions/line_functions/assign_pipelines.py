import numpy as np
import pandas as pd
from collections import deque
from typing import List, Set
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from functions.global_functions.config_load import load_config
cfg = load_config()
artifacts_path = str(cfg['ARTIFACTS_PATH'])

def assign_pipelines(
    df_lines: pd.DataFrame,
    detections_path: str,
    image_path:str,
    radius: float = 5.0,
    coord_tol: float = 0,
    block_rejects: bool = True,
    close_tol: float = 2.0,   # tolerance for the correction step
) -> pd.DataFrame:
    """
    Pipelines nodes based on the defined traversal logic,
    then applies a correction rule at the end:

    Correction rule (simplified):
      For each node:
        1. Find all nodes within `close_tol`.
        2. If there are exactly TWO nearby nodes,
           and both belong to the same *other* Pipeline (not the current node’s Pipeline),
           then reassign this node and ALL nodes of its current Pipeline
           to that other Pipeline.
        3. Otherwise, do nothing.
    """
    image_path = Path(image_path)
    det_dir = Path(detections_path)

    # df_lines = pd.read_csv(artifacts_path+"/"+"hough_lines.csv")
    # Build the CSV path based on the image base name
    base = image_path.stem
    detections_csv_path = det_dir / f"{base}_detections.csv"
    df_detections = pd.read_csv(detections_csv_path)
    df_components_nodes = df_detections[['name_ocr','cx','cy','label']]
    df_components_nodes = df_components_nodes.rename(columns={'name_ocr': 'ID', 'cx': 'x', 'cy': 'y','label':'Component'})

    df_lines_nodes= df_lines[['ID','x1','y1','x2','y2']]

    df_points = (
        pd.wide_to_long(df_lines_nodes, stubnames=['x', 'y'], i='ID', j='point')
        .reset_index()
        .drop(columns='point')
        .sort_values(by='ID')
    )
    
    df_points['Component'] = "pipe"

    df=pd.concat([df_points,df_components_nodes]).reset_index()

    # ------------------------
    # Validate input
    # -----------------------
    # 
    required = {"ID", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns {required}")

    out = df.copy()
    out["Pipeline"] = pd.NA

    coords = out[["x", "y"]].to_numpy()
    ids = out["ID"].to_numpy()
    index_arr = out.index.to_numpy()

    # Precompute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)

    # ------------------------
    # Helper functions
    # ------------------------
    def seed_of(unassigned: Set[int]) -> int:
        return min(unassigned, key=lambda k: (out.at[k, "x"], out.at[k, "y"]))

    def same_id_unassigned(i: int, unassigned: Set[int]) -> List[int]:
        pos_i = out.index.get_loc(i)
        same_pos = np.where(ids == ids[pos_i])[0]
        js = index_arr[same_pos]
        return [j for j in js if (j != i) and (j in unassigned)]

    def nearby_unassigned(i: int, unassigned: Set[int], blocked: Set[int]) -> List[int]:
        row_idx = out.index.get_loc(i)
        candidates = np.where(dists[row_idx] <= radius)[0]
        js = index_arr[candidates]
        return [j for j in js if (j in unassigned) and (j not in blocked)]

    def share_coord(i: int, js: List[int]) -> List[int]:
        if not js:
            return []
        xi, yi = out.at[i, "x"], out.at[i, "y"]
        return [
            j for j in js
            if (abs(out.at[j, "x"] - xi) <= coord_tol) or (abs(out.at[j, "y"] - yi) <= coord_tol)
        ]

    def closest_to(i: int, js: List[int]) -> int:
        row_i = out.index.get_loc(i)
        return min(js, key=lambda j: dists[row_i, out.index.get_loc(j)])

    # ------------------------
    # MAIN Pipeline
    # ------------------------
    unassigned: Set[int] = set(out.index.tolist())
    Pipeline_id = 0

    while unassigned:
        start = seed_of(unassigned)
        Pipeline_id += 1
        blocked: Set[int] = set()
        q: deque[int] = deque([start])

        out.at[start, "Pipeline"] = Pipeline_id
        unassigned.remove(start)

        # Bring in same-ID of seed
        seed_sid = same_id_unassigned(start, unassigned)
        for j in seed_sid:
            out.at[j, "Pipeline"] = Pipeline_id
            unassigned.remove(j)
        for j in seed_sid:
            q.appendleft(j)

        # BFS expansion
        while q:
            i = q.popleft()

            # same-ID first
            sid = same_id_unassigned(i, unassigned)
            for j in sid:
                out.at[j, "Pipeline"] = Pipeline_id
                unassigned.remove(j)
            for j in sid:
                q.appendleft(j)

            # then neighbors
            nbrs = nearby_unassigned(i, unassigned, blocked)
            chosen: List[int] = []

            if len(nbrs) == 0:
                pass
            elif len(nbrs) in (1, 2):
                chosen = nbrs
            elif len(nbrs) == 3:
                share = share_coord(i, nbrs)
                if share:
                    keep = closest_to(i, share)
                    chosen = [keep]
                    if block_rejects:
                        blocked.update([j for j in nbrs if j != keep])
                else:
                    if block_rejects:
                        blocked.update(nbrs)
            else:
                # >=4 → add all that share one coordinate
                share = share_coord(i, nbrs)
                chosen = share
                if block_rejects:
                    blocked.update([j for j in nbrs if j not in share])

            # assign chosen + enqueue
            for j in chosen:
                if j in unassigned:
                    out.at[j, "Pipeline"] = Pipeline_id
                    unassigned.remove(j)
                    q.append(j)

                    # same-ID companions of chosen
                    sid_j = same_id_unassigned(j, unassigned)
                    for k in sid_j:
                        out.at[k, "Pipeline"] = Pipeline_id
                        unassigned.remove(k)
                    for k in sid_j:
                        q.appendleft(k)

    
    # FINAL CORRECTION STEP 

    Pipeline_series = out["Pipeline"].astype("Int64")
    idx_to_pos = {idx: pos for pos, idx in enumerate(index_arr)}
    changes = []

    # Loop through each node
    for i in out.index:
        pos_i = idx_to_pos[i]
        close_pos = np.where(dists[pos_i] <= close_tol)[0]
        close_idx = index_arr[close_pos]

        # Exclude self
        close_idx = [j for j in close_idx if j != i]

        # Check if exactly 2 nearby nodes
        if len(close_idx) != 2:
            continue

        # Pipelines of the nearby nodes
        g_self = int(Pipeline_series.loc[i])
        g_near = out.loc[close_idx, "Pipeline"].astype(int).tolist()

        # If both belong to the same other Pipeline
        if (g_near[0] == g_near[1]) and (g_near[0] != g_self):
            target_Pipeline = g_near[0]

            # Overwrite all nodes in this node’s Pipeline
            out.loc[out["Pipeline"] == g_self, "Pipeline"] = target_Pipeline
            changes.append((g_self, target_Pipeline, i, close_idx))

    out_final=out.drop(columns=['index'])
    out_final.to_csv(artifacts_path+"/"+"nodes.csv")
    return out_final

def draw_pipelines(df_lines,
                   out, 
                   image_path: str):
    """
    Draws pipeline lines on an image and saves the result.
    
    Parameters:
        df_pipelines (pd.DataFrame): DataFrame containing columns ['x1', 'y1', 'x2', 'y2', 'Pipeline'].
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
    """
    image_path=Path(image_path)
    df_pipelines = df_lines.merge(out, on='ID', how='left')[["ID","x1","y1","x2","y2","Pipeline"]].drop_duplicates()
    # --- Load image ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # --- Define color map for pipelines ---
    unique_pipelines = df_pipelines['Pipeline'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_pipelines) + 2)  # distinct colors

    # --- Draw each line ---
    for _, row in df_pipelines.iterrows():
        pipeline_id = row['Pipeline']
        color_idx = np.where(unique_pipelines == pipeline_id)[0][0]
        color = tuple(int(c * 255) for c in colors(color_idx)[:3])
        pt1 = (int(row['x1']), int(row['y1']))
        pt2 = (int(row['x2']), int(row['y2']))
        cv2.line(img, pt1, pt2, color, thickness=3)
    
    # --- Save the result ---
    cv2.imwrite(artifacts_path+"/"+"detected_pipeline.png", img)

