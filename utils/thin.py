import cv2
import numpy as np
from skimage.morphology import skeletonize


def thin(binary_path: str, method: str) -> np.array:
    """
    method takes 'zhang' or 'gradient'

    ## Returns:

    the skeleton as np.array - shape = (512, 512)
    """
    img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    if method == "zhang":
        _, img_threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        skeleton = skeletonize(img_threshold, method="zhang").astype(np.uint8) * 255
    elif method == "gradient":
        ...
    cv2.imshow("RESULT", skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return skeleton


if __name__ == "__main__":
    thin("dataset/annotated/annotated-K/O. glaberrima/2_2_1_1_3_DSC09839.jpg", "zhang")
