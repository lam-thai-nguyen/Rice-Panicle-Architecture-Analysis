import numpy as np
import cv2
from skimage.morphology import skeletonize


def _zhang_suen(binary_img: np.ndarray) -> np.ndarray:
    """
    ## Description
    Perform Z.S. thinning method on a binary image

    ## Arguments
    binary_img (np.ndarray) -> binary image matrix.

    ## Returns
    skeleton_img (np.ndarray) -> skeleton image matrix.
    """
    # Thresholding
    _, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

    # Extracting skeletons
    skeleton_img = skeletonize(binary_img, method="zhang").astype(np.uint8) * 255
    return skeleton_img


def _gradient_based_optimization(): ...
