import cv2
import numpy as np
from skimage.morphology import skeletonize
from thin_plotting import plot_preprocess, plot_skeleton, plot_thin, extract_info
from thin_process import pre_process, post_process


def thin(binary_path: str, method: str = 'zhang', _plot_bin_img=False, _plot_skeleton=False, _plot_result=False) -> list[np.ndarray]:
    """
    ## Arguments:
    
    - method takes 'zhang' or 'gradient'

    ## Example
    
    - binary_path = 'dataset/annotated/annotated-T/O. glaberrima/2_2_1_1_3_DSC09839.jpg'

    ## Returns:

    - raw_skeleton, shape = (512, 512)
    - processed_skeleton, shape = (512, 512) after preprocessing
    - raw_bin_img , shape = (512, 512) 
    - processed_bin_img , shape = (512, 512) after preprocessing
    """
    file_name, model = extract_info(binary_path)
    
    raw_bin_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    
    if method == "zhang":
        raw_skeleton, processed_skeleton, processed_bin_img = _zhang_suen(raw_bin_img, info=[file_name, model], _plot_bin_img=_plot_bin_img, _plot_skeleton=_plot_skeleton)
    elif method == "gradient":
        raw_skeleton, processed_skeleton, processed_bin_img = _gradient_based_optimization(raw_bin_img)
    
    if _plot_result:
        plot_thin(raw_bin_img, processed_bin_img, raw_skeleton, processed_skeleton)

    return raw_skeleton, processed_skeleton, raw_bin_img, processed_bin_img


def _zhang_suen(bin_img: np.ndarray, info: list[str], _plot_bin_img=False, _plot_skeleton=False) -> list[np.ndarray]:
    """
    info: [file_name, model]
    """
    # Thresholding
    _, img_threshold = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY)
    
    # Preprocessing binary image 
    processed_bin_img = pre_process(img_threshold, info=info, _plot_bin_img=_plot_bin_img)
    
    # Extracting skeletons
    raw_skeleton = skeletonize(img_threshold, method="zhang").astype(np.uint8) * 255
    processed_skeleton = skeletonize(processed_bin_img, method="zhang").astype(np.uint8) * 255
    
    if _plot_skeleton:
        plot_skeleton(raw_skeleton, processed_skeleton, info=info)
        
    return raw_skeleton, processed_skeleton, processed_bin_img


def _gradient_based_optimization(bin_img: np.ndarray, info: list[str], _plot_bin_img=False, _plot_skeleton=False) -> list[np.ndarray]:
    ...


if __name__ == "__main__":
    _, skeleton, _, _ = thin("dataset/annotated/annotated-K/O. glaberrima/64_2_1_3_2_DSC01622.jpg", "zhang", 0, 0)
    post_process(skeleton, 50)
    
    # thin("crack-segmentation/transfer-learning-results/RUC_NET/38_2_1_3_1_DSC09528_.png", 'zhang', 1, 1, 0)
    # thin("crack-segmentation/transfer-learning-results/RUC_NET/39_2_1_2_3_DSC09544_.png", 'zhang', 1, 1, 0)
    # thin("crack-segmentation/transfer-learning-results/RUC_NET/42_2_1_2_1_DSC01724.png", 'zhang', 1, 1, 0)
    
