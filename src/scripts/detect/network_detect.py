import sys
sys.path.append('..')
from scripts.functions.network_functions.filter_edges import prune_long_diagonals,prune_overlapping_edges,prune_nodes_with_multi_edges
from scripts.functions.network_functions.create_network import create_delaunay_graph,draw_graph_on_image
from scripts.functions.global_functions.config_load import load_config

if __name__ == "__main__":
    
    cfg = load_config()
    image_path = str(cfg['IMAGE_PATH'])

    G = create_delaunay_graph( pipeline_id=1,)
    prune_long_diagonals(G, threshold=64, axis_tol=1.0)
    prune_overlapping_edges(G, axis_tol=5, overlap_tol=0.1, len_tol=0.1)
    prune_nodes_with_multi_edges(G, angle_tol_deg=5.0, len_tol=1e-9)
    draw_graph_on_image(G, background=image_path, show_labels=True,)
