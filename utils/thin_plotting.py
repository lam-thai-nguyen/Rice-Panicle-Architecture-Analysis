import os
import numpy as np
import matplotlib.pyplot as plt

def plot_preprocess(raw_bin_img: np.ndarray, processed_bin_img: np.ndarray, info: list[str]) -> None:
    file_name, model = info
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax1.imshow(raw_bin_img, cmap='gray')
    ax1.set_title("Raw")
    ax1.axis('off')
    
    ax2.imshow(processed_bin_img, cmap='gray')
    ax2.set_title("Processed")
    ax2.axis('off')
    
    plt.suptitle("PREPROCESSING")
    # Specify save path
    directory = f"results/preprocess/{model}/{file_name[:-4]}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the plot
    plt.savefig(os.path.join(directory, f"binary_{file_name}"))
    plt.show()
    

def plot_skeleton(raw_skeleton: np.ndarray, processed_skeleton: np.ndarray, info: list[str]) -> None:
    file_name, model = info
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax1.imshow(raw_skeleton, cmap='gray')
    ax1.set_title("Raw")
    ax1.axis('off')
    
    ax2.imshow(processed_skeleton, cmap='gray')
    ax2.set_title("Processed")
    ax2.axis('off')
    
    plt.suptitle("PREPROCESSING")
    # Specify save path
    directory = f"results/preprocess/{model}/{file_name[:-4]}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the plot
    plt.savefig(os.path.join(directory, f"skeleton_{file_name}"))
    plt.show()
    

def plot_thin(raw_bin_img: np.ndarray, processed_bin_img, raw_skeleton: np.ndarray, processed_skeleton: np.ndarray) -> None:
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
    
    
def extract_info(binary_path: str) -> list[str]:
    info = binary_path.split('/')
    file_name = info[-1]
    model = "annotated"
    model_list = ['ACS', 'DEEPCRACK', 'RUC_NET', 'SEGNET', 'U2CRACKNEL', 'UNET']
    for i in info:
        if i in model_list:
            model = i
            break
    
    return file_name, model
    