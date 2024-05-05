import cv2
import numpy as np
from skimage.morphology import skeletonize, square, binary_erosion, binary_dilation, binary_closing, binary_opening
from thin_plotting import plot_preprocess, plot_skeleton, plot_thin, extract_info
import matplotlib.pyplot as plt


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
    processed_bin_img = _preprocess(img_threshold, info=info, _plot_bin_img=_plot_bin_img)
    
    # Extracting skeletons
    raw_skeleton = skeletonize(img_threshold, method="zhang").astype(np.uint8) * 255
    processed_skeleton = skeletonize(processed_bin_img, method="zhang").astype(np.uint8) * 255
    
    if _plot_skeleton:
        plot_skeleton(raw_skeleton, processed_skeleton, info=info)
        
    return raw_skeleton, processed_skeleton, processed_bin_img


def _gradient_based_optimization(bin_img: np.ndarray, info: list[str], _plot_bin_img=False, _plot_skeleton=False) -> list[np.ndarray]:
    ...

def _preprocess(bin_img: np.ndarray, info: list[str], _plot_bin_img=False) -> np.ndarray:
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


def _post_process(skeleton_img: np.ndarray, min_length: int) -> np.ndarray:
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
            parent = _get_parent(skeleton, child, children)
            parents[child] = parent
            neighbors_mat = skeleton[parent[0]-1:parent[0]+2, parent[1]-1:parent[1]+2]
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

if __name__ == "__main__":
    _, skeleton, _, _ = thin("dataset/annotated/annotated-K/O. glaberrima/64_2_1_3_2_DSC01622.jpg", "zhang", 0, 0)
    _post_process(skeleton, 10)
    
    # plt.figure(figsize=(8, 8))
    # plt.imshow(post_processed_skeleton, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # thin("crack-segmentation/transfer-learning-results/RUC_NET/38_2_1_3_1_DSC09528_.png", 'zhang', 1, 1, 0)
    # thin("crack-segmentation/transfer-learning-results/RUC_NET/39_2_1_2_3_DSC09544_.png", 'zhang', 1, 1, 0)
    # thin("crack-segmentation/transfer-learning-results/RUC_NET/42_2_1_2_1_DSC01724.png", 'zhang', 1, 1, 0)
    
