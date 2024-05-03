import cv2
import numpy as np
from skimage.morphology import skeletonize, binary_erosion


def thin(binary_path: str, method: str) -> np.ndarray:
    """
    ## Arguments:
    method takes 'zhang' or 'gradient'

    ## Example
    binary_path = 'dataset/annotated/annotated-T/O. glaberrima/2_2_1_1_3_DSC09839.jpg'

    ## Returns:

    the skeleton as np.ndarray - shape = (512, 512)
    """
    if method == "zhang":
        skeleton = _zhang_suen(binary_path)
    elif method == "gradient":
        skeleton = _gradient_based_optimization(binary_path)

    cv2.imshow(method, skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return skeleton


def _zhang_suen(binary_path: str) -> np.ndarray:
    img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)

    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    _, img_threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # eroded = binary_erosion(img_threshold, np.ones((2, 2), np.uint8)).astype(np.uint8) * 255

    # cv2.imshow('eroded', eroded)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    skeleton = skeletonize(img_threshold, method="zhang").astype(np.uint8) * 255
    return skeleton


def _gradient_based_optimization(binary_path: str) -> np.ndarray:
    img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    _, img_threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    eroded = (binary_erosion(img_threshold, np.ones((2, 2), np.uint8)).astype(np.uint8) * 255)
    ...

def _preprocess(binary_path: str) -> np.ndarray:
    """
    threshold, erosion, dilation to remove noise
    """
    img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    _, img_threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


if __name__ == "__main__":
    thin("dataset/annotated/annotated-K/O. glaberrima/64_2_1_3_2_DSC01622.jpg", "zhang")

    # Erosion needed examples
    # _zhang_suen("dataset/annotated/annotated-K/O. glaberrima/56_2_1_2_3_DSC01608.jpg")
    # _zhang_suen("dataset/annotated/annotated-K/O. glaberrima/64_2_1_3_2_DSC01622.jpg")

    # Dilation needed examples
    # "dataset/annotated/annotated-K/O. sativa/38_2_1_3_1_DSC09528_.jpg"
    # "dataset/annotated/annotated-K/O. sativa/39_2_1_1_3_DSC09538_.jpg"
