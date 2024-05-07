import cv2
import numpy as np
import json


def json2binary(json_path: str) -> None:
    """
    Turns .json file from labelme to a binary image (512, 512)

    Args:
        json_path: dataset/annotated/annotated-T/O. glaberrima/2_2_1_1_3_DSC09839.json
    """
    # Load JSON annotation file
    file_name = json_path.split('/')[-1][:-5]
    with open(json_path) as f:
        data = json.load(f)
        
    # Create a black background image
    background = np.zeros((512, 512, 3))

    # Loop through objects in annotation
    for shape in data['shapes']:
        _ = shape['label']
        points = shape['points']
        pts = np.array(points, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Fill object with white color
        cv2.fillPoly(background, [pts], (255, 255, 255))

    save_path = '/'.join(json_path.split('/')[:-1])
    cv2.imwrite(f'{save_path}/{file_name}.jpg', background)
    
    
if __name__ == "__main__":
    # ======Operating on single file======
    json_path = "dataset/annotated/annotated-T/O. glaberrima/2_2_1_1_3_DSC09839.json"
    json2binary(json_path)
    