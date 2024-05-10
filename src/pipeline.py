import os
import matplotlib.pyplot as plt
import cv2
from image_processor.RicePanicle import RicePanicle
from utils.ricepr_manipulate import resize_junction
from utils.evaluation_image_generating import _pruning, generate_y_true, generate_skeleton_main_axis


def pipeline(binary_path: str) -> None:
    """
    ## Naming convention:
    - all junctions _1
    - main axis junctions _2
    - high order junctions: _3
    """
    # EXTRACT INFORMATION =======================================
    info = binary_path.split('/')
    name = info[-1][:-4]
    species = None if "O. " not in info[-2] else info[-2]
    if species is not None:
        ricepr_path = f"data/original_ricepr/{species}/{name}.ricepr"
    else:
        if os.path.exists(f"data/original_ricepr/O. glaberrima/{name}.ricepr"):
            ricepr_path = f"data/original_ricepr/O. glaberrima/{name}.ricepr"
            raw_img_512 = plt.imread(f"images/raw_images_512/O. glaberrima/{name}.jpg")  # May be needed for plotting
        else:
            ricepr_path = f"data/original_ricepr/O. sativa/{name}.ricepr"
            raw_img_512 = plt.imread(f"images/raw_images_512/O. sativa/{name}.jpg")  # May be needed for plotting
            
    binary_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    junction_resized = resize_junction(ricepr_path)
    # ===========================================================
    
    # ==================ALL JUNCTIONS============================
    # THINNING
    skeleton_img_1 = RicePanicle.Thinning.zhang_suen(binary_img)
    
    # CLUSTERING
    _, y_pred_1 = RicePanicle.Clustering.crossing_number(skeleton_img_1, return_pred_=True)
    n_pred_1 = len(y_pred_1[y_pred_1 > 0])
    
    # EVALUATION
    y_true_1 = generate_y_true(junction_resized)
    n_true_1 = len(y_true_1[y_true_1 > 0])
    f1, pr, rc = RicePanicle.Evaluation.f1_score(y_true_1, y_pred_1, _return_metrics=True)
    print(f"--------------------------> ALL JUNCTIONS\nf1: {f1}, precision: {pr}, recall: {rc}\n---------------------------------------------\n")
    
    # ===================MAIN AXIS===============================
    # THINNING
    skeleton_img_2 = generate_skeleton_main_axis(skeleton_img_1, ricepr_path)
    
    # CLUSTERING
    _, y_pred_2 = RicePanicle.Clustering.crossing_number(skeleton_img_2, return_pred_=True)
    n_pred_2 = len(y_pred_2[y_pred_2 > 0])
    
    # EVALUATION
    y_true_2 = generate_y_true(junction_resized, main_axis=True)
    n_true_2 = len(y_true_2[y_true_2 > 0])
    f1, pr, rc = RicePanicle.Evaluation.f1_score(y_true_2, y_pred_2, _return_metrics=True)
    print(f"--------------------------> MAIN AXIS JUNCTIONS\nf1: {f1}, precision: {pr}, recall: {rc}\n---------------------------------------------\n")
    
    # ===================HIGH ORDER===============================
    # THINNING
    skeleton_img_3 = skeleton_img_1 - skeleton_img_2
    
    # CLUSTERING
    _, y_pred_3 = RicePanicle.Clustering.crossing_number(skeleton_img_3, return_pred_=True)
    n_pred_3 = len(y_pred_3[y_pred_3 > 0])
    
    # EVALUATION
    y_true_3 = generate_y_true(junction_resized, high_order=True)
    n_true_3 = len(y_true_3[y_true_3 > 0])
    f1, pr, rc = RicePanicle.Evaluation.f1_score(y_true_3, y_pred_3, _return_metrics=True)
    print(f"--------------------------> HIGH ORDER JUNCTIONS\nf1: {f1}, precision: {pr}, recall: {rc}\n---------------------------------------------")
    
    # ==============CHECK IF THE PROCESS IS TRUSTWORTHY===================
    print(f"\t\t\t\t\t\t\t\t\t\t\t There are {n_true_1} TRUE junctions -> {n_true_1} = {n_true_2} + {n_true_3} -> {n_true_1 == n_true_2 + n_true_3}")
    print(f"\t\t\t\t\t\t\t\t\t\t\t We have predicted {n_pred_1} junctions -> {n_pred_1} = {n_pred_2} + {n_pred_3} -> {n_pred_1 == n_pred_2 + n_pred_3}\n")
    if (n_true_1 == n_true_2 + n_true_3) and (n_pred_1 == n_pred_2 + n_pred_3):
        print("\t\t\t\t\t\t\t\t\t\t\t TRUSTWORTHY!\n")
    else:
        print("\t\t\t\t\t\t\t\t\t\t\t NEEDS REVIEWING!\n")


if __name__ == "__main__":
    binary_path = "crack_segmentation/transfer-learning-results/run_2/DEEPCRACK/13_2_1_1_1_DSC01478.png"
    pipeline(binary_path)
    