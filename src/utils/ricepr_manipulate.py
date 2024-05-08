import xml.etree.ElementTree as ET
import cv2


def resize_junction(ricepr_path: str, dst_size: tuple = (512, 512)) -> dict:
    """
    ## Description
    Resizes junction from src size to dst size | Junctions are extracted from ricepr file

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
    py_ipynb_distinguish = info[0]
    # ============================================

    # Get src size =============================== 
    if py_ipynb_distinguish == "..":
        original_image = cv2.imread(f"../../data/original_images/{species}/{name}.jpg", cv2.IMREAD_GRAYSCALE)
    else:
        original_image = cv2.imread(f"data/original_images/{species}/{name}.jpg", cv2.IMREAD_GRAYSCALE)
        
    src_size = original_image.shape
    src_height, src_width = src_size
    # ============================================
    
    dst_height, dst_width = dst_size
    junction = _load_ricepr(ricepr_path)
    junction_resized = {_key: [] for _key in junction}
    
    # Conversion =================================
    for key in junction:
        for (src_x, src_y) in junction[key]:
            dst_x, dst_y = round((src_x / src_width) * dst_width), round((src_y / src_height) * dst_height)
            junction_resized[key].append((dst_x, dst_y))
    # ============================================
    return junction_resized


def _load_ricepr(ricepr_path: str) -> dict:
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

    junction = {
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
            junction["generating"].append((x, y))
        elif type_ == "Primary":
            junction["primary"].append((x, y))
        elif type_ == "Seconday":
            junction["secondary"].append((x, y))
        elif type_ == "Tertiary":
            junction["tertiary"].append((x, y))
        elif type_ == "Quaternary":
            junction["quaternary"].append((x, y))
        
    return junction
