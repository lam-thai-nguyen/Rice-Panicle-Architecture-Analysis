###########################
# Author: Lam Thai Nguyen #
###########################

import os
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from image_processor.RicePanicle import RicePanicle
from image_processor.AccuracyManager import AccuracyManager
from utils.ricepr_manipulate import resize_junction
from utils.evaluation_image_generating import generate_y_true, generate_skeleton_main_axis


def _extract_info(binary_path: str):
    """
    Extract information from binary path.
    
    ## Example binary path:
    images/model_predictions/K_1_4_UNET_UNET/3_2_1_2_3_DSC01275.png

    Args:
        binary_path (str): Path to the binary image.

    Returns:
        tuple: (name, species, ricepr_path, raw_img_512)
    """
    info = binary_path.split("/")
    name = info[-1][:-4]
    species = None if "O. " not in info[-2] else info[-2]
    if species is not None:
        ricepr_path = f"data/original_ricepr/{species}/{name}.ricepr"
        raw_img_512 = plt.imread(f"images/raw_images_512/{species}/{name}.jpg")
    else:
        if os.path.exists(f"data/original_ricepr/O. glaberrima/{name}.ricepr"):
            ricepr_path = f"data/original_ricepr/O. glaberrima/{name}.ricepr"
            raw_img_512 = plt.imread(f"images/raw_images_512/O. glaberrima/{name}.jpg")
        else:
            ricepr_path = f"data/original_ricepr/O. sativa/{name}.ricepr"
            raw_img_512 = plt.imread(f"images/raw_images_512/O. sativa/{name}.jpg")
            
    return name, species, ricepr_path, raw_img_512


def pipeline(binary_path: str, person: str, criterion: int) -> RicePanicle.DetectionAccuracy:
    """
    ## Naming convention:
    - all junctions: 1
    - main axis junctions: 2
    - high order junctions: 3

    ## Idea:
    - Remove false junctions, characterized by a cluster of close junctions, by using _merge_pred().
    - Only apply _merge_pred() for high order junctions as main axis junctions can be close to each other.
    - y_pred_1 = y_pred_2 + y_pred_3_merged = y_pred_2 + (y_pred_1 - y_pred_2)merged. So the code will be messy but makes sense.

    ## Returns:
    - RicePanicle.DetectionAccuracy object.
    """
    name, _, ricepr_path, raw_img_512 = _extract_info(binary_path)  # Extract information
    
    binary_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)  # Read binary image
    junction_resized = resize_junction(ricepr_path)  # Resize junction (original size -> 512x512) ==>> dict[order] = [(x, y), ...]
    
    print("".center(50, "="))
    print(f"==>> processing image: {name} <<==")

    # ==================ALL JUNCTIONS============================
    # THINNING
    skeleton_img_1 = RicePanicle.Thinning.zhang_suen(binary_img)

    # CLUSTERING (y_pred_1_merged = y_pred_2 + y_pred_3_merged = y_pred_2 + (y_pred_1 - y_pred_2)merged)
    _, y_pred_1 = RicePanicle.Clustering.crossing_number(skeleton_img_1, return_pred_=True)  # y_pred_1
    
    skeleton_img_2 = generate_skeleton_main_axis(skeleton_img_1, ricepr_path)
    _, y_pred_2 = RicePanicle.Clustering.crossing_number(skeleton_img_2, return_pred_=True)  # y_pred_2
    
    skeleton_img_3 = skeleton_img_1 - skeleton_img_2
    y_pred_3 = y_pred_1 - y_pred_2  # y_pred_3 = (y_pred_1 - y_pred_2)
    
    _, y_pred_3_merged = _merge_pred(y_pred_3, skeleton_img_3, binary_path, person=person, criterion=criterion)  # y_pred_3_merged
    
    y_pred_1_merged = y_pred_2 + y_pred_3_merged  # y_pred_1_merged = y_pred_2 + y_pred_3_merged

    n_pred_1 = len(y_pred_1_merged[y_pred_1_merged > 0])

    # EVALUATION
    y_true_1 = generate_y_true(junction_resized)
    n_true_1 = len(y_true_1[y_true_1 > 0])
    f1_1, pr_1, rc_1 = RicePanicle.Evaluation.f1_score(y_true_1, y_pred_1_merged, _return_metrics=True)
    print(f"all junction ==>> f1: {f1_1:.4f}, precision: {pr_1:.4f}, recall: {rc_1:.4f}")

    # ===================MAIN AXIS===============================
    # THINNING (Done Previously)

    # CLUSTERING (Done Previously)
    n_pred_2 = len(y_pred_2[y_pred_2 > 0])

    # EVALUATION
    y_true_2 = generate_y_true(junction_resized, main_axis=True)
    n_true_2 = len(y_true_2[y_true_2 > 0])
    f1_2, pr_2, rc_2 = RicePanicle.Evaluation.f1_score(y_true_2, y_pred_2, _return_metrics=True)
    print(f"main axis ==>> f1: {f1_2:.4f}, precision: {pr_2:.4f}, recall: {rc_2:.4f}")

    # ===================HIGH ORDER===============================
    # THINNING (Done Previously)

    # CLUSTERING
    y_pred_3 = y_pred_3_merged
    n_pred_3 = len(y_pred_3[y_pred_3 > 0])

    # EVALUATION
    y_true_3 = generate_y_true(junction_resized, high_order=True)
    n_true_3 = len(y_true_3[y_true_3 > 0])
    f1_3, pr_3, rc_3 = RicePanicle.Evaluation.f1_score(y_true_3, y_pred_3, _return_metrics=True)
    print(f"high order ==>> f1: {f1_3:.4f}, precision: {pr_3:.4f}, recall: {rc_3:.4f}")

    # ==============CHECK IF THE PROCESS IS TRUSTWORTHY===================
    print(f"\n\t\t\t There are {n_true_1} TRUE junctions -> {n_true_1} = {n_true_2} main axis + {n_true_3} high order -> {n_true_1 == n_true_2 + n_true_3}")
    print(f"\t\t\t We have predicted {n_pred_1} junctions -> {n_pred_1} = {n_pred_2} main axis + {n_pred_3} high order -> {n_pred_1 == n_pred_2 + n_pred_3}")
    
    if (n_true_1 == n_true_2 + n_true_3) and (n_pred_1 == n_pred_2 + n_pred_3):
        print("\t\t\t TRUSTWORTHY!")
    else:
        print("\t\t\t NEEDS REVIEWING!")

    print("".center(50, "="))

    # Plot pipeline
    plot_pipeline(
        file_name=name,
        raw_img_512=raw_img_512,
        binary_img=binary_img,
        all_junctions=[skeleton_img_1, y_pred_1_merged, y_true_1],
        main_axis=[skeleton_img_2, y_pred_2, y_true_2],
        high_order=[skeleton_img_3, y_pred_3, y_true_3],
        person=person,
        criterion=criterion,
    )

    print(f"Saving: images/pipeline/merge_pred/{person}_{criterion}/{name}.jpg")
    print(f"Saving: images/pipeline/pipeline/{person}_{criterion}/{name}.jpg")

    # Save accuracy
    detection_accuracy = RicePanicle.DetectionAccuracy(
        name=name,
        all_junctions=[f1_1, pr_1, rc_1],
        main_axis=[f1_2, pr_2, rc_2],
        high_order=[f1_3, pr_3, rc_3],
    )

    return detection_accuracy


def _merge_pred(y_pred: np.ndarray, skeleton_img: np.ndarray, binary_path: str, person: str, criterion: int) -> np.ndarray:
    """
    ## Description
    Merging close predicted junctions into one junction. Should only be applied to high order junctions.

    ## Argument:
    - y_pred: np.ndarray
    - skeleton_img: np.ndarray
    - binary_path: str
    - person
    - criterion

    ## Returns:
    - junction_img_merged: np.ndarray
    - y_pred_merged: np.ndarray
    """
    name, _, _, raw_img_512 = _extract_info(binary_path)

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

            x, y = np.mean(pts, axis=0).astype("i")
            y_pred_merged[x, y] = 255

    white_px_merged = np.argwhere(y_pred_merged > 0)
    n_merged = len(white_px_merged)

    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    mask = labels != -1

    ax1.imshow(raw_img_512)
    ax1.scatter(white_px[:, 1], white_px[:, 0], c="r", s=5)
    ax1.set_title("Pre-merged High Order Junctions.")
    ax1.axis("off")

    ax2.imshow(raw_img_512)
    ax2.scatter(white_px[mask, 1], white_px[mask, 0], c="w", s=5)
    ax2.scatter(white_px[~mask, 1], white_px[~mask, 0], c="r", s=5)
    ax2.set_title("Clusters to be merged.")
    ax2.axis("off")

    ax3.imshow(raw_img_512)
    ax3.scatter(white_px_merged[:, 1], white_px_merged[:, 0], c="r", s=5)
    ax3.set_title("Clusters merged as one junction.")
    ax3.axis("off")

    plt.suptitle(f"Merging High Order Junctions\nPrevious: {n_initial} -> Merged: {n_merged}")

    plt.tight_layout()
    plt.savefig(f"images/pipeline/merge_pred/{person}_{criterion}/{name}.jpg")
    plt.close(fig)

    junction_img_merged = cv2.cvtColor(junction_img_merged, cv2.COLOR_GRAY2RGB)
    for i in range(len(white_px_merged)):
        cv2.circle(junction_img_merged, (white_px_merged[i][1], white_px_merged[i][0]), 2, (255, 0, 0), -1)

    return junction_img_merged, y_pred_merged


def plot_pipeline(
    file_name: str,
    raw_img_512: np.ndarray,
    binary_img: np.ndarray,
    all_junctions: list,
    main_axis: list,
    high_order: list,
    person: str,
    criterion: int,
) -> None:
    """
    Plot the whole pipeline

    Args:
        raw_img_512 (np.ndarray)
        binary_img (np.ndarray): segmentation result
        all_junctions (list): [skeleton_img_1, y_pred_1, y_true_1]
        main_axis (list): [skeleton_img_2, y_pred_2, y_true_2]
        high_order (list): [skeleton_img_3, y_pred_3, y_true_3]
    """
    skeleton_img_1, y_pred_1, y_true_1 = all_junctions
    skeleton_img_2, y_pred_2, y_true_2 = main_axis
    skeleton_img_3, y_pred_3, y_true_3 = high_order

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Plotting
    axes[0].imshow(binary_img, cmap="gray")
    axes[0].set_title("Binary Image")
    axes[0].axis("off")

    axes[1].imshow(skeleton_img_1, cmap="gray")
    real_px_1 = np.argwhere(y_true_1 > 0)
    for pts in real_px_1:
        axes[1].scatter(pts[1], pts[0], c="b", s=5, marker="s")
    pred_px_1 = np.argwhere(y_pred_1 > 0)
    for pts in pred_px_1:
        axes[1].scatter(pts[1], pts[0], c="r", s=5)
    axes[1].set_title("All Junctions")
    axes[1].axis("off")

    axes[2].imshow(skeleton_img_2, cmap="gray")
    real_px_2 = np.argwhere(y_true_2 > 0)
    for pts in real_px_2:
        axes[2].scatter(pts[1], pts[0], c="b", s=5, marker="s")
    pred_px_2 = np.argwhere(y_pred_2 > 0)
    for pts in pred_px_2:
        axes[2].scatter(pts[1], pts[0], c="r", s=5)
    axes[2].set_title("Main Axis")
    axes[2].axis("off")

    axes[3].imshow(skeleton_img_3, cmap="gray")
    real_px_3 = np.argwhere(y_true_3 > 0)
    for pts in real_px_3:
        axes[3].scatter(pts[1], pts[0], c="b", s=5, marker="s")
    pred_px_3 = np.argwhere(y_pred_3 > 0)
    for pts in pred_px_3:
        axes[3].scatter(pts[1], pts[0], c="r", s=5)
    axes[3].set_title("High Order")
    axes[3].axis("off")

    axes[4].imshow(skeleton_img_1, cmap="gray")
    axes[4].set_title("Skeleton Image")
    axes[4].axis("off")

    axes[5].imshow(raw_img_512)
    for pts in real_px_1:
        axes[5].scatter(pts[1], pts[0], c="b", s=5, marker="s")
    for pts in pred_px_1:
        axes[5].scatter(pts[1], pts[0], c="r", s=5)
    axes[5].axis("off")

    axes[6].imshow(raw_img_512)
    for pts in real_px_2:
        axes[6].scatter(pts[1], pts[0], c="b", s=5, marker="s")
    for pts in pred_px_2:
        axes[6].scatter(pts[1], pts[0], c="r", s=5)
    axes[6].axis("off")

    axes[7].imshow(raw_img_512)
    for pts in real_px_3:
        axes[7].scatter(pts[1], pts[0], c="b", s=5, marker="s")
    for pts in pred_px_3:
        axes[7].scatter(pts[1], pts[0], c="r", s=5)
    axes[7].axis("off")

    plt.tight_layout()
    plt.savefig(f"images/pipeline/pipeline/{person}_{criterion}/{file_name}.jpg")
    plt.close(fig)


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
        manager.save_as_excel("test.xlsx")
    except:
        print("File already created.")


def main():
    argparser = argparse.ArgumentParser(description="Pipeline script")
    argparser.add_argument("person", type=str, help="T: Thai, K: Kien ==>> version of person", choices=["T", "K"])
    argparser.add_argument("evaluation_criterion", type=int, help="1: O. glaberrima, 2: O. sativa, 3: O. glaberrima and O. sativa", choices=[1, 2, 3])
    argparser.add_argument("-m", "--model", type=str, help="Model name", choices=["U2CRACKNET", "SEGNET", "FCN", "DEEPCRACK", "RUC_NET", "ACS", "UNET"], default="UNET")
    argparser.add_argument("-s", "--save-as-excel", type=bool, choices=[True, False], default=False)
    args = argparser.parse_args()

    person, criterion, model, save_as_excel = args.person, args.evaluation_criterion, args.model, args.save_as_excel
    
    score_manager = AccuracyManager()
    pred_folder = "images/model_predictions"

    for pred in os.listdir(pred_folder):
        if pred.startswith(f"{person}_{criterion}"):
            images_path = pred_folder + "/" + pred
            print(f"==>> images_path: {images_path}")
            break

    for image_name in os.listdir(images_path):
        binary_path = images_path + "/" + image_name
        detection_accuracy = pipeline(binary_path, person, criterion)
        score_manager.add(detection_accuracy)

    if save_as_excel:
        score_manager.save_as_excel(f"data/junction_detection_result/{person}_{criterion}_{model}.xlsx")
        print(f"File saved: data/junction_detection_result/{person}_{criterion}_{model}.xlsx")
    

if __name__ == "__main__":
    main()
