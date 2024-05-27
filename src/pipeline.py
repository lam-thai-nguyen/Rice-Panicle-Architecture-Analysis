###########################
# Author: Lam Thai Nguyen #
###########################

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from image_processor.RicePanicle import RicePanicle
from image_processor.AccuracyManager import AccuracyManager
from utils.ricepr_manipulate import resize_junction
from utils.evaluation_image_generating import generate_y_true, generate_skeleton_main_axis


def pipeline(binary_path: str) -> RicePanicle.DetectionAccuracy:
    """
    ## Naming convention:
    - all junctions _1
    - main axis junctions _2
    - high order junctions: _3
    
    ## Idea:
    - Remove false junctions, characterized by a cluster of close junctions, by using _merge_pred().
    - Only apply _merge_pred() for high order junctions as main axis junctions can be close to each other.
    - y_pred_1 = y_pred_2 + y_pred_3_merged = y_pred_2 + (y_pred_1 - y_pred_2)merged. So the code will be messy but makes sense.
    
    ## Returns:
    - RicePanicle.DetectionAccuracy object.
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
    
    # CLUSTERING (y_pred_1 = y_pred_2 + y_pred_3_merged = y_pred_2 + (y_pred_1 - y_pred_2)merged)
    
    # --------------------------------------------------------------------------------------------------------------
    # 1. y_pred_1
    _, y_pred_1 = RicePanicle.Clustering.crossing_number(skeleton_img_1, return_pred_=True)
    # 2. y_pred_2
    skeleton_img_2 = generate_skeleton_main_axis(skeleton_img_1, ricepr_path)
    _, y_pred_2 = RicePanicle.Clustering.crossing_number(skeleton_img_2, return_pred_=True)
    # 3. y_pred_3 = (y_pred_1 - y_pred_2)
    skeleton_img_3 = skeleton_img_1 - skeleton_img_2
    y_pred_3 = y_pred_1 - y_pred_2
    # 4. y_pred_3_merged
    _, y_pred_3_merged = _merge_pred(y_pred_3, skeleton_img_3, binary_path)
    # 5. y_pred_1 = y_pred_2 + y_pred_3_merged
    y_pred_1 = y_pred_2 + y_pred_3_merged
    # --------------------------------------------------------------------------------------------------------------
    
    n_pred_1 = len(y_pred_1[y_pred_1 > 0])
    
    # EVALUATION
    y_true_1 = generate_y_true(junction_resized)
    n_true_1 = len(y_true_1[y_true_1 > 0])
    f1_1, pr_1, rc_1 = RicePanicle.Evaluation.f1_score(y_true_1, y_pred_1, _return_metrics=True)
    print(f"--------------------------> ALL JUNCTIONS\nf1: {f1_1}, precision: {pr_1}, recall: {rc_1}\n---------------------------------------------\n")
    
    # ===================MAIN AXIS===============================
    # THINNING (Done Previously)
    
    # CLUSTERING (Done Previously)
    n_pred_2 = len(y_pred_2[y_pred_2 > 0])
    
    # EVALUATION
    y_true_2 = generate_y_true(junction_resized, main_axis=True)
    n_true_2 = len(y_true_2[y_true_2 > 0])
    f1_2, pr_2, rc_2 = RicePanicle.Evaluation.f1_score(y_true_2, y_pred_2, _return_metrics=True)
    print(f"--------------------------> MAIN AXIS JUNCTIONS\nf1: {f1_2}, precision: {pr_2}, recall: {rc_2}\n---------------------------------------------\n")
    
    # ===================HIGH ORDER===============================
    # THINNING (Done Previously)
    
    # CLUSTERING
    y_pred_3 = y_pred_1 - y_pred_2  # == y_pred_3_merged
    n_pred_3 = len(y_pred_3[y_pred_3 > 0])
    
    # EVALUATION
    y_true_3 = generate_y_true(junction_resized, high_order=True)
    n_true_3 = len(y_true_3[y_true_3 > 0])
    f1_3, pr_3, rc_3 = RicePanicle.Evaluation.f1_score(y_true_3, y_pred_3, _return_metrics=True)
    print(f"--------------------------> HIGH ORDER JUNCTIONS\nf1: {f1_3}, precision: {pr_3}, recall: {rc_3}\n---------------------------------------------")
    
    # ==============CHECK IF THE PROCESS IS TRUSTWORTHY===================
    print(f"\t\t\t\t\t\t\t\t\t\t\t There are {n_true_1} TRUE junctions -> {n_true_1} = {n_true_2} + {n_true_3} -> {n_true_1 == n_true_2 + n_true_3}")
    print(f"\t\t\t\t\t\t\t\t\t\t\t We have predicted {n_pred_1} junctions -> {n_pred_1} = {n_pred_2} + {n_pred_3} -> {n_pred_1 == n_pred_2 + n_pred_3}")
    if (n_true_1 == n_true_2 + n_true_3) and (n_pred_1 == n_pred_2 + n_pred_3):
        print("\t\t\t\t\t\t\t\t\t\t\t TRUSTWORTHY!\n")
    else:
        print("\t\t\t\t\t\t\t\t\t\t\t NEEDS REVIEWING!\n")
        
    # Save accuracy
    detection_accuracy = RicePanicle.DetectionAccuracy(
        name=name,
        all_junctions=[f1_1, pr_1, rc_1],
        main_axis=[f1_2, pr_2, rc_2],
        high_order=[f1_3, pr_3, rc_3]
        )
    
    return detection_accuracy


def _merge_pred(y_pred: np.ndarray, skeleton_img: np.ndarray, binary_path: str, _plot=False) -> np.ndarray:
    """
    ## Description
    Merging close predicted junctions into one junction. Should only be applied to high order junctions.

    ## Argument:
    - y_pred np.ndarray
    - skeleton_img: np.ndarray
    - binary_path: str
    - _plot=False
        
    ## Returns:
    - junction_img_merged: np.ndarray
    - y_pred_merged: np.ndarray
    """
    # EXTRACT INFORMATION =======================================
    if binary_path:
        info = binary_path.split('/')
        name = info[-1][:-4]
        species = None if "O. " not in info[-2] else info[-2]
        model = info[-2] if info[-2] in ["U2CRACKNET", "DEEPCRACK", "FCN", "ACS", "RUC_NET", "SEGNET", 'UNET'] else ""
        if species is None:
            if os.path.exists(f"data/original_ricepr/O. glaberrima/{name}.ricepr"):
                species = "O. glaberrima"
            else:
                species = "O. sativa"
    # ===========================================================
    
    junction_img_merged = np.copy(skeleton_img)
    y_pred_merged = np.copy(y_pred)
    
    white_px = np.argwhere(y_pred_merged > 0)
    n_initial = len(white_px)
    
    db = DBSCAN(eps=7, min_samples=2).fit(white_px)
    labels = db.labels_
    
    # Merging
    for label in np.unique(labels):
        if label != -1:
            pts = white_px[labels == label]
            for a, b in pts:
                y_pred_merged[a, b] = 0
                
            x, y = np.mean(pts, axis=0).astype('i')
            y_pred_merged[x, y] = 255
    
    white_px_merged = np.argwhere(y_pred_merged > 0)
    n_merged = len(white_px_merged)
            
    # Visualization
    bg = np.zeros((512, 512))
    mask = labels != -1
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    colors_dict = {i: colors[i] for i in range(10)}
    
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(bg, cmap='gray')
    ax1.scatter(white_px[:, 1], white_px[:, 0], c='w', s=5)
    ax1.set_title("Pre-merged High Order Junctions.")
    ax1.axis('off')
    
    ax2.imshow(bg, cmap='gray')
    cluster_colors = [colors_dict[label % 10] for label in labels[mask]]
    ax2.scatter(white_px[mask, 1], white_px[mask, 0], c=cluster_colors, s=5)
    ax2.scatter(white_px[~mask, 1], white_px[~mask, 0], c='w', s=5)
    ax2.set_title("Clusters to be merged.")
    ax2.axis('off')
    
    ax3.imshow(bg, cmap='gray')
    ax3.scatter(white_px_merged[:, 1], white_px_merged[:, 0], c='w', s=5)
    ax3.set_title("Clusters merged as one junction.")
    ax3.axis('off')

    plt.suptitle(f"Merging High Order Junctions\nPrevious: {n_initial} -> Merged: {n_merged}")
    
    if binary_path:
        plt.savefig(f"images/pipeline/merge_pred/{model + '/' if model else model}{name}.jpg")
    
    if _plot:
        plt.show()
    
    junction_img_merged = cv2.cvtColor(junction_img_merged, cv2.COLOR_GRAY2RGB)
    for i in range(len(white_px_merged)):
        cv2.circle(junction_img_merged, (white_px_merged[i][1], white_px_merged[i][0]), 2, (255, 0, 0), -1)
    
    return junction_img_merged, y_pred_merged


def test_manager():
    manager = AccuracyManager()
    rp1 = RicePanicle.DetectionAccuracy('13_2_1_1_1_DSC01478', (0.67, 0.64, 0.69), (0.81, 0.81, 0.81), (0.62, 0.59, 0.65))
    rp2 = RicePanicle.DetectionAccuracy('13_2_1_1_1_DSC01479', (0.75, 0.70, 0.80), (0.85, 0.85, 0.85), (0.68, 0.66, 0.70), model="U2CRACKNET")
    rp1.show()
    rp2.show()
    manager.add(rp1)
    manager.add(rp2)
    manager.show()
    try:
        manager.save_as_csv("test.csv")
    except:
        print("File already created.")


def main():
    score_manager = AccuracyManager()
    pred_folder = "images/model_predictions/run_2/U2CRACKNET"
    model = pred_folder.split('/')[-1]
    for pred in os.listdir(pred_folder):
        binary_path = pred_folder + '/' + pred
        detection_accuracy = pipeline(binary_path)
        score_manager.add(detection_accuracy)
        
    score_manager.save_as_csv(f"data/junction_detection_result/O. glaberrima/{model}.csv")
    print(f"File saved: data/junction_detection_result/O. glaberrima/{model}.csv")
    

if __name__ == "__main__":
    main()
    