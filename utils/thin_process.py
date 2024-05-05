import cv2
import numpy as np
from skimage.morphology import square, binary_erosion, binary_dilation, binary_closing, binary_opening
from thin_plotting import plot_preprocess, plot_skeleton, plot_thin, extract_info

def pre_process(bin_img: np.ndarray, info: list[str], _plot_bin_img=False) -> np.ndarray:
    """
    Preprocessing method: Erosion OR Dilation -> Erosion x 2
    """
    _, model = info
    strel = square(2)
    
    if model == "annotated":
        processed_bin_img = binary_erosion(bin_img, footprint=strel)
    
    elif model == "ACS":
        ...
        
    elif model == "DEEPCRACK":
        ...
        
    elif model == "RUC_NET":
        processed_bin_img = binary_dilation(bin_img, footprint=strel)
        processed_bin_img = binary_dilation(processed_bin_img, footprint=strel)
        processed_bin_img = binary_erosion(processed_bin_img, footprint=strel)
        processed_bin_img = binary_erosion(processed_bin_img, footprint=strel)
        
    elif model == "SEGNET":
        ...
        
    elif model == "U2CRACKNET":
        ...
        
    elif model == "UNET":
        ...
    
    if _plot_bin_img:
        plot_preprocess(bin_img, processed_bin_img, info=info)
        
    return processed_bin_img


def post_process(skeleton_img: np.ndarray, min_length: int) -> np.ndarray:
    """
    Performs postprocessing on skeleton images to remove all branches with length shorter than branch_length
    
    ## Arguments:
    - skeleton_img: np.ndarray
    - min_length: minimum number of pixels allowed that make up a branch 
    
    ## Returns:
    - post_processed_skeleton: All short branches are removed
    """
    raw_skeleton = np.copy(skeleton_img)
    
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
            # Follow parent pointers until reaching an intersection
        while True:
            parent = parents[current_point]  # Get parent of current point
            if parent is None:
                break
            path.append(parent)  # Add parent to the path
            current_point = parent  # Update current point
        
        branches.append(path)
    
    for branch in branches:
        if len(branch) < min_length:
            for pts in branch:
                raw_skeleton[pts[0], pts[1]] = 0
                
    cv2.imshow("a", raw_skeleton)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    
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