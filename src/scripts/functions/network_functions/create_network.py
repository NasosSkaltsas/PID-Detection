import cv2
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from typing import Optional, Tuple, Union

from scripts.functions.global_functions.config_load import load_config
cfg = load_config()
artifacts_path = str(cfg['ARTIFACTS_PATH'])

def create_delaunay_graph(
    pipeline_id: int = 1,
) -> nx.Graph:
    """
    Build a NetworkX graph from a DataFrame of node coordinates using Delaunay triangulation.

    Parameters
    ----------
    df_nodes : pd.DataFrame
        DataFrame with at least ['ID', 'x', 'y', 'Pipeline'] columns.
    pipeline_id : int, optional
        Value of the 'Pipeline' column to filter nodes by (default=1).
    plot : bool, optional
        Whether to plot the resulting Delaunay graph.

    Returns
    -------
    G : networkx.Graph
        Graph with nodes (x, y) and weighted edges (Euclidean distances).
    """
    df_nodes= pd.read_csv((artifacts_path+"/"+"nodes.csv"))
    #  Filter nodes by pipeline
    df1 = df_nodes[df_nodes["Pipeline"] == pipeline_id][["ID", "x", "y","Component"]].copy()

    # Reset index to create a unique numeric index for Delaunay
    df2 = df1.reset_index(drop=True).reset_index()
    
    # Compute mean position per ID 
    df = df2.groupby(["ID","Component"], as_index=False)[["x", "y"]].mean()

    # Save as JSON 
    extract_df = df[["ID", "Component"]].rename(columns={"ID": "label"}).reset_index()
    extract_df.to_json(artifacts_path +"/" +"nodes_mapping.json", orient="records", indent=4)
    
    df["index"] = df.index

    # Delaunay triangulation
    pts = df[["x", "y"]].to_numpy()
    ids = df["index"].to_numpy()
    tri = Delaunay(pts)

    # Build graph from triangles
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(int(row["index"]), x=float(row["x"]), y=float(row["y"]))

    for simplex in tri.simplices:
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            u, v = int(ids[simplex[a]]), int(ids[simplex[b]])
            if not G.has_edge(u, v):
                p1, p2 = pts[simplex[a]], pts[simplex[b]]
                w = float(np.hypot(*(p1 - p2)))
                G.add_edge(u, v, weight=w)

    return G


def draw_graph_on_image(
    G: nx.Graph,
    background: Optional[Union[str, np.ndarray]] = None,  
    image_size: Tuple[int, int] = (1000, 1000),        
    bg_color: Tuple[int, int, int] = (255, 255, 255),   
    node_color: Tuple[int, int, int] = (0, 150, 255),  
    edge_color: Tuple[int, int, int] = (0, 0, 0),        
    node_radius: int = 4,
    edge_thickness: int = 1,
    show_labels: bool = False
):
    """
    Draw a NetworkX graph onto an image. Assumes node (x,y) are pixel coords.

    Parameters
    ----------
    G : Graph with node attrs 'x','y' (pixels).
    background : filepath or np.ndarray (HxWx3 or HxW). If None, make blank.
    image_size : (H,W) for blank canvas.
    Returns image (BGR uint8).
    """
    # Get base image
    if isinstance(background, str):
        img = cv2.imread(background, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {background}")
    elif isinstance(background, np.ndarray):
        img = background.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        h, w = image_size
        img = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Draw edges
    for u, v in G.edges():
        x1, y1 = int(G.nodes[u]['x']), int(G.nodes[u]['y'])
        x2, y2 = int(G.nodes[v]['x']), int(G.nodes[v]['y'])
        cv2.line(img, (x1, y1), (x2, y2), edge_color, edge_thickness, lineType=cv2.LINE_AA)

    # Draw nodes (and optional labels)
    for n, d in G.nodes(data=True):
        x, y = int(d['x']), int(d['y'])
        cv2.circle(img, (x, y), node_radius, node_color, -1, lineType=cv2.LINE_AA)
        if show_labels:
            cv2.putText(img, str(n), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)

    savepth = artifacts_path + "\\"+"detected_network.png"
    cv2.imwrite(savepth,img)

    return 
