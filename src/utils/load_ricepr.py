import xml.etree.ElementTree as ET
import cv2
import numpy as np


def load_ricepr(ricepr_path: str) -> dict:
    """
    ## Description
    Reads .ricepr file to extract the true junction coordinate

    ## Argument
    ricepr_path (str)

    ## Returns
    junction_xy (dict) -> dict[order] = [(x, y), ...]
    """
    tree = ET.parse(ricepr_path)
    root = tree.getroot()

    junction_xy = {
        "generating": [],
        "end": [],
        "primary": [],
        "secondary": [],
        "tertiary": [],
        "quaternary": [],
    }

    for vertex in root.iter("vertex"):
        x = int(vertex.attrib["x"])
        y = int(vertex.attrib["y"])
        type_ = vertex.attrib["type"]
        
        if type_ == "Generating":
            junction_xy["generating"].append((x, y))
        elif type_ == "Primary":
            junction_xy["primary"].append((x, y))
        elif type_ == "Seconday":
            junction_xy["secondary"].append((x, y))
        elif type_ == "Tertiary":
            junction_xy["tertiary"].append((x, y))
        elif type_ == "Quaternary":
            junction_xy["quaternary"].append((x, y))
        
    return junction_xy


def resize_xy(ricepr_path: str, dst_size: tuple = (512, 512)) -> dict:
    """
    ## Description
    Resizes junction_xy from src size to dst size

    ## Argument
    - ricepr_path (str)
    - dst_size (tuple) = (dst_height, dst_width)

    ## Returns
    junction_xy_resized (dict) = (512, 512)
    """
    # Extract information ========================
    info = ricepr_path.split('/')
    name = info[-1][:-7]
    species = info[-2]
    # ============================================
    
    # Get src size ===============================
    original_image = cv2.imread(f"../../data/original_images/{species}/{name}.jpg", cv2.IMREAD_GRAYSCALE)
    src_size = original_image.shape
    src_height, src_width = src_size
    # ============================================
    
    dst_height, dst_width = dst_size
    junction_xy = load_ricepr(ricepr_path)
    junction_xy_resized = {_key: [] for _key in junction_xy}
    
    # Conversion =================================
    for key in junction_xy:
        for (src_x, src_y) in junction_xy[key]:
            dst_x, dst_y = round((src_x / src_width) * dst_width), round((src_y / src_height) * dst_height)
            junction_xy_resized[key].append((dst_x, dst_y))
    # ============================================
    return junction_xy_resized


def generate_y_true(junction_xy: dict) -> np.ndarray:
    """
    ## Description
    Generates y_true for evaluation
    
    ## Arguments
    - junction_xy: dict
    
    # Returns
    - y_true: np.ndarray
    """
    junction = []
    for value in junction_xy.values():
        junction.extend(value)
    
    y_true = np.zeros((512, 512))
    
    for y, x in junction:
        y_true[x, y] = 255
        
    return y_true


def generate_y_true_main_axis(junction_xy: dict) -> np.ndarray:
    """
    ## Description
    Generates y_true_main_axis for evaluation
    
    ## Arguments
    - junction_xy: dict
    
    # Returns
    - y_true: np.ndarray
    """
    generating = junction_xy.get('generating')
    primary = junction_xy.get('primary')
    main_axis_junction = generating + primary
    
    y_true = np.zeros((512, 512))
    
    for y, x in main_axis_junction:
        y_true[x, y] = 255
        
    return y_true
