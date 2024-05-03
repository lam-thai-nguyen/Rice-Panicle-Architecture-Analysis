import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, binary_erosion
from thin_plotting import plot_preprocess, plot_skeleton, plot_thin


def thin(binary_path: str, method: str, _plot_bin_img=False, _plot_skeleton=False, _plot_result=False) -> list[np.ndarray]:
    """
    ## Arguments:
    method takes 'zhang' or 'gradient'

    ## Example
    binary_path = 'dataset/annotated/annotated-T/O. glaberrima/2_2_1_1_3_DSC09839.jpg'

    ## Returns:

    the skeleton as np.ndarray - shape = (512, 512)
    """
    raw_bin_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    
    if method == "zhang":
        raw_skeleton, processed_skeleton, processed_bin_img = _zhang_suen(raw_bin_img, _plot_bin_img=_plot_bin_img, _plot_skeleton=_plot_skeleton)
    elif method == "gradient":
        raw_skeleton, processed_skeleton, processed_bin_img = _gradient_based_optimization(raw_bin_img)
    
    if _plot_result:
        plot_thin(raw_bin_img, processed_bin_img, raw_skeleton, processed_skeleton)

    return raw_skeleton, processed_skeleton


def _zhang_suen(bin_img: np.ndarray, _plot_bin_img=False, _plot_skeleton=False) -> list[np.ndarray]:
    # Thresholding
    _, img_threshold = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY)
    
    # Preprocessing binary image 
    processed_bin_img = _preprocess(img_threshold, _plot_bin_img=_plot_bin_img)
    
    # Extracting skeletons
    raw_skeleton = skeletonize(img_threshold, method="zhang").astype(np.uint8) * 255
    processed_skeleton = skeletonize(processed_bin_img, method="zhang").astype(np.uint8) * 255
    
    if _plot_skeleton:
        plot_skeleton(raw_skeleton, processed_skeleton)
        
    return raw_skeleton, processed_skeleton, processed_bin_img


def _gradient_based_optimization(bin_img: np.ndarray, _plot_bin_img=False, _plot_skeleton=False) -> list[np.ndarray]:
    ...

def _preprocess(bin_img: np.ndarray, _plot_bin_img=False) -> np.ndarray:
    """
    erosion, dilation, opening, closing to remove noise
    """
    processed_bin_img = bin_img
    
    if _plot_bin_img:
        plot_preprocess(bin_img, processed_bin_img)
        
    return processed_bin_img


if __name__ == "__main__":
    thin("dataset/annotated/annotated-K/O. glaberrima/64_2_1_3_2_DSC01622.jpg", "zhang", 1, 1, 1)

    # Erosion needed examples
    # _zhang_suen("dataset/annotated/annotated-K/O. glaberrima/56_2_1_2_3_DSC01608.jpg")
    # _zhang_suen("dataset/annotated/annotated-K/O. glaberrima/64_2_1_3_2_DSC01622.jpg")

    # Dilation needed examples
    # "dataset/annotated/annotated-K/O. sativa/38_2_1_3_1_DSC09528_.jpg"
    # "dataset/annotated/annotated-K/O. sativa/39_2_1_1_3_DSC09538_.jpg"
