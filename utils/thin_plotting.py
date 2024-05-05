import os
import numpy as np
import matplotlib.pyplot as plt

def plot_preprocess(raw_bin_img: np.ndarray, pre_processed_bin_img: np.ndarray, **kwargs) -> None:
    """
    ## Description
    - Plot raw_bin_img and processed_bin_img side by side
    
    ## kwargs:
    - info: list[str] [file_name, model] -> specify when you want to save the figure
    """
    # ==============================================
    file_name, model = kwargs.get("info", [None, None])
    # ==============================================
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax1.imshow(raw_bin_img, cmap='gray')
    ax1.set_title("Raw")
    ax1.axis('off')
    
    ax2.imshow(pre_processed_bin_img, cmap='gray')
    ax2.set_title("Pre-processed")
    ax2.axis('off')
    
    plt.suptitle("PREPROCESSING")

    # Save the plot
    if file_name is not None and model is not None:
        directory = f"results/preprocess/{model}/{file_name[:-4]}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, f"binary.jpg"))
    plt.show()
    

def plot_skeleton(raw_skeleton: np.ndarray, pre_processed_skeleton: np.ndarray, **kwargs) -> None:
    """
    ## Description
    - Plot raw_skeleton and preprocessed_skeleton side by side
    
    ## kwargs:
    - info: list[str] [file_name, model] -> specify when you want to save the figure
    """
    # ==============================================
    file_name, model = kwargs.get("info", [None, None])
    # ==============================================
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax1.imshow(raw_skeleton, cmap='gray')
    ax1.set_title("Raw")
    ax1.axis('off')
    
    ax2.imshow(pre_processed_skeleton, cmap='gray')
    ax2.set_title("Pre-processed")
    ax2.axis('off')
    
    plt.suptitle("PREPROCESSING")

    # Save the plot
    if file_name is not None and model is not None:
        directory = f"results/preprocess/{model}/{file_name[:-4]}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, f"skeleton.jpg"))
    plt.show()
    

def plot_thin(raw_bin_img: np.ndarray, pre_processed_bin_img: np.ndarray, raw_skeleton: np.ndarray, pre_processed_skeleton: np.ndarray) -> None:
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    (ax1, ax2, ax3, ax4) = axes.flatten()
    
    ax1.imshow(raw_bin_img, cmap='gray')
    ax1.set_title("Raw binary")
    ax1.axis('off')
    
    ax2.imshow(pre_processed_bin_img, cmap='gray')
    ax2.set_title("Pre-processed binary")
    ax2.axis('off')
    
    ax3.imshow(raw_skeleton, cmap='gray')
    ax3.set_title("Raw skeleton")
    ax3.axis('off')

    ax4.imshow(pre_processed_skeleton, cmap='gray')
    ax4.set_title("Pre-processed skeleton")
    ax4.axis('off')
    
    plt.suptitle("THINNING")
    plt.show()
    
    
def extract_info(binary_path: str) -> list[str]:
    info = binary_path.split('/')
    file_name = info[-1]
    model = "annotated"
    model_list = ['ACS', 'DEEPCRACK', 'RUC_NET', 'SEGNET', 'U2CRACKNET', 'UNET']
    for i in info:
        if i in model_list:
            model = i
            break
    
    return file_name, model
    