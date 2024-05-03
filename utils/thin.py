import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, binary_erosion


def thin(binary_path: str, method: str, plot_result=False) -> list[np.ndarray]:
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
        raw_skeleton, processed_skeleton, processed_bin_img = _zhang_suen(raw_bin_img)
    elif method == "gradient":
        raw_skeleton, processed_skeleton, processed_bin_img = _gradient_based_optimization(raw_bin_img)

    # cv2.imshow(method, processed_skeleton)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    if plot_result:
        _plot_thin(raw_bin_img, processed_bin_img, raw_skeleton, processed_skeleton)

    return raw_skeleton, processed_skeleton


def _zhang_suen(bin_img: np.ndarray) -> list[np.ndarray]:
    _, img_threshold = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY)
    processed_bin_img = _preprocess(img_threshold)
    raw_skeleton = skeletonize(img_threshold, method="zhang").astype(np.uint8) * 255
    processed_skeleton = skeletonize(processed_bin_img, method="zhang").astype(np.uint8) * 255
    return raw_skeleton, processed_skeleton, processed_bin_img


def _gradient_based_optimization(bin_img: np.ndarray) -> np.ndarray:
    ...

def _preprocess(bin_img: np.ndarray, plot_result=False) -> np.ndarray:
    """
    erosion, dilation, opening, closing to remove noise
    """
    processed_bin_img = bin_img
    
    if plot_result:
        _plot_preprocess(bin_img, processed_bin_img)
        
    return processed_bin_img


def _plot_preprocess(raw_bin_img: np.ndarray, processed_bin_img: np.ndarray) -> None:
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax1.imshow(raw_bin_img, cmap='gray')
    ax1.set_title("Raw")
    ax1.axis('off')
    
    ax2.imshow(processed_bin_img, cmap='gray')
    ax2.set_title("Processed")
    ax2.axis('off')
    
    plt.suptitle("PREPROCESSING")
    plt.show()


def _plot_thin(raw_bin_img: np.ndarray, processed_bin_img, raw_skeleton: np.ndarray, processed_skeleton: np.ndarray) -> None:
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    (ax1, ax2, ax3, ax4) = axes.flatten()
    
    ax1.imshow(raw_bin_img, cmap='gray')
    ax1.set_title("Raw binary")
    ax1.axis('off')
    
    ax2.imshow(processed_bin_img, cmap='gray')
    ax2.set_title("Processed binary")
    ax2.axis('off')
    
    ax3.imshow(raw_skeleton, cmap='gray')
    ax3.set_title("Raw skeleton")
    ax3.axis('off')

    ax4.imshow(processed_skeleton, cmap='gray')
    ax4.set_title("Processed skeleton")
    ax4.axis('off')
    
    plt.suptitle("THINNING")
    plt.show()


if __name__ == "__main__":
    thin("dataset/annotated/annotated-K/O. glaberrima/64_2_1_3_2_DSC01622.jpg", "zhang", True)

    # Erosion needed examples
    # _zhang_suen("dataset/annotated/annotated-K/O. glaberrima/56_2_1_2_3_DSC01608.jpg")
    # _zhang_suen("dataset/annotated/annotated-K/O. glaberrima/64_2_1_3_2_DSC01622.jpg")

    # Dilation needed examples
    # "dataset/annotated/annotated-K/O. sativa/38_2_1_3_1_DSC09528_.jpg"
    # "dataset/annotated/annotated-K/O. sativa/39_2_1_1_3_DSC09538_.jpg"
