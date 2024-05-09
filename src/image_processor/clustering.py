import numpy as np
import cv2
from sklearn.cluster import DBSCAN


def _crossing_number(skeleton_img: np.ndarray, return_pred_: bool) -> list[np.ndarray]:
    """
    ## Description
    Performs Crossing Number Method to find junctions in a given skeleton image.
    
    ## Arguments
    - skeleton_img: np.ndarray -> the skeleton matrix.
    - return_pred_: bool -> Set to true to return y_pred
    
    ## Returns
    - junction_img: np.ndarray -> the skeleton with junction matrix.
    - y_pred: np.ndarray, if return_pred_
    """
    img = np.copy(skeleton_img)
    
    # White px intensity 255 -> 1
    img[img == 255] = 1
    white_px = np.argwhere(img > 0)
    centers = []

    # Crossing number
    for row, col in white_px:
        row, col = int(row), int(col)

        try:
            P1 = img[row, col + 1].astype("i")
            P2 = img[row - 1, col + 1].astype("i")
            P3 = img[row - 1, col].astype("i")
            P4 = img[row - 1, col - 1].astype("i")
            P5 = img[row, col - 1].astype("i")
            P6 = img[row + 1, col - 1].astype("i")
            P7 = img[row + 1, col].astype("i")
            P8 = img[row + 1, col + 1].astype("i")
        except:
            continue

        crossing_number = abs(P2 - P1) + abs(P3 - P2) + abs(P4 - P3) + abs(P5 - P4) + abs(P6 - P5) + abs(P7 - P6) + abs(P8 - P7) + abs(P1 - P8)
        crossing_number //= 2
        if crossing_number == 3 or crossing_number == 4:
            centers.append([row, col])

    # White px intensity 1 -> 255
    img[img == 1] = 255
    junction_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in range(len(centers)):
        cv2.circle(junction_img, (centers[i][1], centers[i][0]), 2, (255, 0, 0), -1)
    
    results = [junction_img]
    
    # Create y_pred
    if return_pred_:
        y_pred = np.zeros((512, 512))
        for i in range(len(centers)):
            y_pred[centers[i][0], centers[i][1]] = 255
        results.append(y_pred)
        
    return results


def _dbscan(skeleton_img: np.ndarray, return_pred_: bool):
    image = np.copy(skeleton_img)

    data_points = np.column_stack(np.where(image > 0))

    dbscan = DBSCAN(eps=1.5, min_samples=4)
    dbscan.fit(data_points)

    cluster_labels = dbscan.labels_

    unique_labels = np.unique(cluster_labels)
    centers = []
    for label in unique_labels:
        if label != -1:
            cluster_points = data_points[cluster_labels == label]
            center = np.mean(cluster_points, axis=0)
            centers.append(center.astype(int))

    junction_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for center in centers:
        cv2.circle(junction_img, tuple((center[::-1])), 2, (255, 0, 0), -1)

    results = [junction_img]

    if return_pred_:
        y_pred = np.zeros((512, 512))
        for i in range(len(centers)):
            y_pred[centers[i][0], centers[i][1]] = 255
        results.append(y_pred)

    return results
