import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    processed = _preprocess(img, 1)
    skeleton = skeletonize(processed, method="zhang").astype(np.uint8) * 255
    return skeleton


def _gradient_based_optimization(binary_path: str) -> np.ndarray:
    img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    processed = _preprocess(img)
    ...

def _preprocess(img: np.ndarray, plot_result=False) -> np.ndarray:
    """
    threshold, erosion, dilation, opening, closing to remove noise
    """
    _, img_threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    processed = img_threshold
    
    if plot_result:
        _plot_preprocess(img, processed)
        
    return processed


def _plot_preprocess(pre_img: np.ndarray, post_img: np.ndarray) -> None:
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax1.imshow(pre_img, cmap='gray')
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2.imshow(post_img, cmap='gray')
    ax2.set_title("Processed")
    ax2.axis('off')
    
    plt.suptitle("PREPROCESSING")
    plt.show()
    return None
    

if __name__ == "__main__":
    thin("dataset/annotated/annotated-K/O. glaberrima/64_2_1_3_2_DSC01622.jpg", "zhang")

    # Erosion needed examples
    # _zhang_suen("dataset/annotated/annotated-K/O. glaberrima/56_2_1_2_3_DSC01608.jpg")
    # _zhang_suen("dataset/annotated/annotated-K/O. glaberrima/64_2_1_3_2_DSC01622.jpg")

    # Dilation needed examples
    # "dataset/annotated/annotated-K/O. sativa/38_2_1_3_1_DSC09528_.jpg"
    # "dataset/annotated/annotated-K/O. sativa/39_2_1_1_3_DSC09538_.jpg"
