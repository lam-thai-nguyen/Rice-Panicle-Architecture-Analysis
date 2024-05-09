import os
import matplotlib.pyplot as plt
import cv2
from image_processor.RicePanicle import RicePanicle
from utils.ricepr_manipulate import resize_junction
from utils.evaluation_image_generating import _pruning, generate_y_true, generate_skeleton_main_axis, generate_skeleton_high_order


def pipeline(binary_path: str) -> None:
    # EXTRACT INFORMATION =======================================
    info = binary_path.split('/')
    name = info[-1][:-4]
    species = None if "O. " not in info[-2] else info[-2]
    if species is not None:
        ricepr_path = f"data/original_ricepr/{species}/{name}.ricepr"
    else:
        if os.path.exists(f"data/original_ricepr/O. glaberrima/{name}.ricepr"):
            ricepr_path = f"data/original_ricepr/O. glaberrima/{name}.ricepr"
        else:
            ricepr_path = f"data/original_ricepr/O. sativa/{name}.ricepr"
    # ===========================================================
    
    # ==================ALL JUNCTIONS============================
    # THINNING
    binary_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    skeleton_img = RicePanicle.Thinning.zhang_suen(binary_img)
    skeleton_img_pruned = _pruning(skeleton_img, 10)
    
    # CLUSTERING
    _, y_pred = RicePanicle.Clustering.crossing_number(skeleton_img_pruned, return_pred_=True)
    
    # EVALUATION
    junction_resized = resize_junction(ricepr_path)
    y_true = generate_y_true(junction_resized)
    f1, pr, rc = RicePanicle.Evaluation.f1_score(y_true, y_pred, _return_metrics=True)
    print(f"--------------------------> ALL JUNCTIONS\nf1: {f1}, precision: {pr}, recall: {rc}\n---------------------------------------------\n")
    
    # ===================MAIN AXIS===============================
    # THINNING
    skeleton_img_main_axis = generate_skeleton_main_axis(skeleton_img, ricepr_path)
    
    # CLUSTERING
    _, y_pred_main_axis = RicePanicle.Clustering.crossing_number(skeleton_img_main_axis, return_pred_=True)
    
    # EVALUATION
    y_true_main_axis = generate_y_true(junction_resized, main_axis=True)
    f1, pr, rc = RicePanicle.Evaluation.f1_score(y_true_main_axis, y_pred_main_axis, _return_metrics=True)
    print(f"--------------------------> MAIN AXIS JUNCTIONS\nf1: {f1}, precision: {pr}, recall: {rc}\n---------------------------------------------\n")
    
    # ===================HIGH ORDER===============================
    # THINNING
    skeleton_img_high_order = generate_skeleton_high_order(skeleton_img, ricepr_path)
    
    # CLUSTERING
    _, y_pred_high_order = RicePanicle.Clustering.crossing_number(skeleton_img_high_order, return_pred_=True)
    
    # EVALUATION
    y_true_high_order = generate_y_true(junction_resized, high_order=True)
    f1, pr, rc = RicePanicle.Evaluation.f1_score(y_true_high_order, y_pred_high_order, _return_metrics=True)
    print(f"--------------------------> HIGH ORDER JUNCTIONS\nf1: {f1}, precision: {pr}, recall: {rc}\n---------------------------------------------\n")


if __name__ == "__main__":
    binary_path = "crack_segmentation/transfer-learning-results/run_2/DEEPCRACK/13_2_1_1_1_DSC01478.png"
    pipeline(binary_path)
    