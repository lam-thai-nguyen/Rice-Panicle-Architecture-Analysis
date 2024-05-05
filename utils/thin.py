import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from thin_plotting import plot_preprocess, plot_skeleton, plot_thin, extract_info, plot_prune
from thin_process import pre_process, post_process
from CustomExceptions import MissingRequiredArgument


def thin(binary_path: str, method: str, _pre_process: bool, _post_process: bool, **kwargs) -> list[np.ndarray]:
    """
    ## Description
    - An integrated process -> Pre-processing - Skeletonization - Post-processing
    
    ## Arguments
    - binary_path = 'dataset/annotated/annotated-T/O. glaberrima/2_2_1_1_3_DSC09839.jpg'
    - method = {zhang, gradient}
    - _pre_process (bool)
    - _post_process (bool)
    
    ## kwargs
    - When _pre_process is True
        - _plot_bin_img=False
        - _plot_skeleton=False
    - When _post_process is True
        - min_length=0
        - _plot_prune=False
    - When both are True
        - _plot_result=False

    
    ## Returns:
    - skeleton (np.ndarray)
    """
    # ============================================
    if _pre_process:
        _plot_bin_img = kwargs.get("_plot_bin_img", False)
        _plot_skeleton = kwargs.get("_plot_skeleton", False)
        _plot_result = kwargs.get("_plot_result", False)
    if _post_process:
        min_length = kwargs.get("min_length", 0)
        _plot_prune = kwargs.get("_plot_prune", False)
    # ============================================
    
    # EXTRACTING INFORMATION
    file_name, model = extract_info(binary_path)
    
    # THRESHOLDING
    raw_bin_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    _, raw_bin_img = cv2.threshold(raw_bin_img, 127, 255, cv2.THRESH_BINARY)
    
    # PRE-PROCESSING
    if _pre_process:
        pre_processed_bin_img = pre_process(bin_img=raw_bin_img, info=[file_name, model])
    
    # THINNING
    if method == "zhang":
        if not _pre_process:
            skeleton = _zhang_suen(raw_bin_img)
        elif _pre_process:
            raw_skeleton = _zhang_suen(raw_bin_img)
            pre_processed_skeleton = _zhang_suen(pre_processed_bin_img)
            skeleton = pre_processed_skeleton
            
    if method == "gradient":
        ...
    
    # VISUALIZATION (IF SPECIFIED)
    if _pre_process:
        if "_plot_bin_img" in kwargs and _plot_bin_img == True:
            plot_preprocess(raw_bin_img=raw_bin_img, pre_processed_bin_img=pre_processed_bin_img)
        if "_plot_skeleton" in kwargs and _plot_skeleton == True:
            plot_skeleton(raw_skeleton=raw_skeleton, pre_processed_skeleton=pre_processed_skeleton)
            
    # POST-PROCESSING
    if _post_process:
        if 'min_length' in kwargs:
            post_processed_skeleton = post_process(skeleton_img=skeleton, min_length=min_length)
            skeleton = post_processed_skeleton
        else:
            raise MissingRequiredArgument(">> Please specify min_length to perform Post-processing <<")
        if "_plot_prune" in kwargs and _plot_prune == True:
            if pre_processed_skeleton is not None:  # Post-processing on pre-processing skeleton
                plot_prune(raw_skeleton=pre_processed_skeleton, post_processed_skeleton=post_processed_skeleton)
            else:  # Post-processing on raw skeleton
                plot_prune(raw_skeleton=raw_skeleton, post_processed_skeleton=post_processed_skeleton)
        
    if _pre_process and _post_process:
        if "_plot_result" in kwargs and _plot_result == True:
            plot_thin(raw_bin_img=raw_bin_img, pre_processed_bin_img=pre_processed_bin_img, raw_skeleton=raw_skeleton, pre_processed_skeleton=pre_processed_skeleton, post_processed_skeleton=post_processed_skeleton)
            
    return skeleton


def _zhang_suen(bin_img: np.ndarray) -> np.ndarray:
    """
    ## Description
    - Perform Z.S. thinning method on a binary image
    
    ## Arguments
    - bin_img (np.ndarray)
    
    ## Returns 
    - skeleton (np.ndarray)
    """
    # Extracting skeletons
    skeleton = skeletonize(bin_img, method="zhang").astype(np.uint8) * 255
    return skeleton


def _gradient_based_optimization(bin_img: np.ndarray, info: list[str], _plot_bin_img=False, _plot_skeleton=False) -> list[np.ndarray]:
    ...


if __name__ == "__main__":
    # skeleton = thin("dataset/annotated/annotated-K/O. glaberrima/64_2_1_3_2_DSC01622.jpg", "zhang", 1, 1, _plot_bin_img=True, _plot_skeleton=True, _plot_result=False, min_length=0)
    
    skeleton = thin("crack-segmentation/transfer-learning-results/RUC_NET/38_2_1_3_1_DSC09528_.png", "zhang", 1, 1, _plot_bin_img=0, _plot_skeleton=0, _plot_result=1, min_length=10, _plot_prune=0)
    
    # thin("crack-segmentation/transfer-learning-results/RUC_NET/39_2_1_2_3_DSC09544_.png", 'zhang', 1, 1, 0)
    # thin("crack-segmentation/transfer-learning-results/RUC_NET/42_2_1_2_1_DSC01724.png", 'zhang', 1, 1, 0)
    
