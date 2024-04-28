import cv2
import numpy as np
from skimage.morphology import skeletonize


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
    _, img_threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    skeleton = skeletonize(img_threshold, method="zhang").astype(np.uint8) * 255
    return skeleton
    

def _gradient_based_optimization(binary_path: str) -> np.ndarray:
    img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    _, img_threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ...

if __name__ == "__main__":
    thin("dataset/annotated/annotated-K/O. glaberrima/2_2_1_1_3_DSC09839.jpg", "zhang")
