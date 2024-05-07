import cv2
import numpy as np
import json


def json2binary(json_path: str) -> None:
    """
    ## Description
    Turning a .json file from labelme to a binary image shape=(512, 512).

    ## Argument
    json_path (str)

    ## Example
    json_path = images/annotated/annotated-T/O. glaberrima/1_2_1_1_1_DSC01251.json

    ## Returns
    None
    """
    # Extract information
    info = json_path.split("/")
    file_name = info[-1][:-5]  # 1_2_1_1_1_DSC01251
    species = info[-2]
    user = info[-3]

    # Load annotation file
    with open(json_path) as f:
        data = json.load(f)

    # Create a black background image
    binary_image = np.zeros((512, 512, 3))

    # Loop through objects in json file
    for shape in data["shapes"]:
        _ = shape["label"]
        points = shape["points"]
        pts = np.array(points, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Fill object with white color
        cv2.fillPoly(binary_image, [pts], (255, 255, 255))

    save_path = f"images/annotated/{user}/{species}/{file_name}.jpg"
    cv2.imwrite(save_path, binary_image)
