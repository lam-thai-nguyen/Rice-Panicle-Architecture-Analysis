import numpy as np
import cv2
from skimage.morphology import skeletonize
import torch
from skeletonize import Skeletonize


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


def _gradient_based_optimization(binary_img: np.ndarray):
    _, thresholded_image = cv2.threshold(binary_img, 120, 255, cv2.THRESH_BINARY)

    img = thresholded_image / 255.0

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    skeletonization_module = Skeletonize(probabilistic=True, beta=0.33, tau=1.0, simple_point_detection='Boolean')
    skeleton_stack = np.zeros_like(img_tensor.squeeze())

    for step in range(1):
        skeleton_stack = skeleton_stack + skeletonization_module(img_tensor).numpy().squeeze()

    skeleton_Gradient = (skeleton_stack / 1).round()

    skeleton_Gradient = skeleton_Gradient.astype(np.uint8) * 255
    # cv2.imshow("result", skeleton_Gradient)
    return skeleton_Gradient
