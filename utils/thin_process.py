import cv2
import numpy as np
from skimage.morphology import square, binary_erosion, binary_dilation, binary_closing, binary_opening
from thin_plotting import plot_preprocess, plot_skeleton, plot_thin, extract_info


def pre_process(bin_img: np.ndarray, info: list[str], **kwargs) -> np.ndarray:
    """
    ## Description
    - Preprocessing method on binary image based on info

    ## Arguments
    - bin_img: (np.ndarray)
    - info: (list[str]) [file_name, model]

    ## kwargs:
    - _plot_bin_img=False

    ## Returns
    - pre_processed_bin_img
    """
    # ================================================
    _plot_bin_img = kwargs.get("_plot_bin_img", False)
    # ================================================
    
    _, model = info
    print(f"==========METHOD: <{model}> PRE-PROCESSING==========")

    # structuring element
    strel = square(2)

    # Performing preprocessing
    if model == "annotated":
        pre_processed_bin_img = binary_erosion(bin_img, footprint=strel)

    elif model == "ACS":
        ...

    elif model == "DEEPCRACK":
        ...

    elif model == "RUC_NET":
        pre_processed_bin_img = binary_dilation(bin_img, footprint=strel)
        pre_processed_bin_img = binary_dilation(pre_processed_bin_img, footprint=strel)
        pre_processed_bin_img = binary_erosion(pre_processed_bin_img, footprint=strel)
        pre_processed_bin_img = binary_erosion(pre_processed_bin_img, footprint=strel)

    elif model == "SEGNET":
        ...

    elif model == "U2CRACKNET":
        ...

    elif model == "UNET":
        ...

    if _plot_bin_img:
        plot_preprocess(bin_img, pre_processed_bin_img, info=info)

    return pre_processed_bin_img.astype(np.uint8) * 255


def post_process(skeleton_img: np.ndarray, min_length: int) -> np.ndarray:
    """
    ## Description
    - Performs PRUNING postprocessing on a skeleton image to remove all branches with length shorter than min_length

    ## Arguments:
    - skeleton_img: (np.ndarray)
    - min_length: (int) minimum number of pixels allowed that make up a branch

    ## Returns:
    - post_processed_skeleton: All short branches are removed
    """
    raw_skeleton = np.copy(skeleton_img)

    # Initiating end_points
    white_px = np.argwhere(raw_skeleton > 0)
    end_points = []  # Tip of the skeleton
    for i in white_px:
        neighbors_mat = raw_skeleton[i[0]-1:i[0]+2, i[1]-1:i[1]+2]
        neighbors = len(np.argwhere(neighbors_mat > 0)) - 1
        if neighbors == 1:
            end_points.append(tuple(i))

    # Get coordinates of points according to their roles
    parents = {}  # parents[child] = parent
    children = []
    for child in end_points:
        while True:
            children.append(child)
            parent = _get_parent(raw_skeleton, child, children)
            parents[child] = parent
            neighbors_mat = raw_skeleton[parent[0]-1:parent[0]+2, parent[1]-1:parent[1]+2]
            if np.sum(neighbors_mat) > 255 * 3:
                parents[parent] = None  # Assume junctions don't have parents
                break
            child = parent

    # Finding branches' paths
    branches = []
    for end_point in end_points:
        path = [end_point]
        current_point = end_point
        
        while True:
            parent = parents[current_point] 
            if parent is None:
                break
            path.append(parent)
            current_point = parent

        branches.append(path)

    # PRUNING
    for branch in branches:
        if len(branch) < min_length:
            for pts in branch:
                raw_skeleton[pts[0], pts[1]] = 0
    
    post_processed_skeleton = raw_skeleton
    return post_processed_skeleton

def _get_parent(skeleton_img: np.ndarray, end_point: tuple, children: list[tuple]) -> tuple:
    """
    Find the neighbor of end_point
    """
    neighbors_mat = skeleton_img[end_point[0]-1:end_point[0]+2, end_point[1]-1:end_point[1]+2]
    for i in range(3):
        for j in range(3):
            if neighbors_mat[i, j] > 0:
                parent = tuple((end_point[0] + (i - 1), end_point[1] + (j - 1)))
                if parent not in children:
                    return parent
    return None
