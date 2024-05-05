import numpy as np
import cv2


def cluster(skeleton: np.ndarray, method: str) -> None:
    """
    method: {cn, ...}
    """
    if method == "cn":
        junction_img = _crossing_number(skeleton)
        
    return junction_img
        
    
def _crossing_number(skeleton: np.array) -> np.ndarray:
    """
    Perform crossing number
    """
    img = np.copy(skeleton)
    
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
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in range(len(centers)):
        cv2.circle(img_out, (centers[i][1], centers[i][0]), 2, (0, 0, 255), -1)
    
    cv2.imshow(f"crossing_number", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img_out
