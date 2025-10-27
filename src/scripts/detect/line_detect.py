import sys
sys.path.append('..')
from scripts.functions.line_functions.clean_img import process_image_cleaning,thicken_edges
from scripts.functions.global_functions.config_load import load_config
from scripts.functions.line_functions.hough__transform import detect_lines_hough
from scripts.functions.line_functions.assign_pipelines import assign_pipelines,draw_pipelines

if __name__ == "__main__":
    cfg = load_config()
    image_path = str(cfg['IMAGE_PATH'])
    detections_path = str(cfg['DETECTIONS_DIR'])

    clean_img = process_image_cleaning(image_path, detections_path)
    thick_img = thicken_edges(clean_img)
    df_lines = detect_lines_hough(thick_img)
    out = assign_pipelines(df_lines, detections_path, image_path, radius=70, coord_tol=0, close_tol=10)
    draw_pipelines(df_lines, out, image_path)