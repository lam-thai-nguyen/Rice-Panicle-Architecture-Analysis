import os
import cv2
import numpy as np
import json


def json2binary(json_path, original_512x512_path, save_path="dataset/annotated/annotated-K") -> None:
    # Load JSON annotation file
    with open(json_path) as f:
        data = json.load(f)

    # Load image
    image = cv2.imread(original_512x512_path)

    # Create a black background image
    background = np.zeros_like(image)

    # Loop through objects in annotation
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        pts = np.array(points, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Fill object with white color
        cv2.fillPoly(background, [pts], (255, 255, 255))

    index = len("dataset/original_512x512/")
    cv2.imwrite(f'../{save_path}{original_512x512_path[index+2:]}', background)
    print(f'./{save_path}{original_512x512_path[index+2:]}')
    cv2.imshow("result", background)
    cv2.waitKey()
    
if __name__ == "__main__":
    # ======Operating on single file======
    json_path = "../dataset/annotated/annotated-K/13_2_1_3_1_DSC01484.json"
    original_512x512_path = "../dataset/original_512x512/13_2_1_3_1_DSC01484.jpg"
    json2binary(json_path, original_512x512_path)
    
    # ======Automatic for the whole dataset======
    # json_folder = "dataset/annotated/annotated-T"
    # original_512x512_folder = "dataset/original_512x512"
    # file_names = os.listdir(original_512x512_folder)
    
    # for img in file_names:
    #     img_name = img[:-4]
    #     json2binary(json_path=f"{json_folder}/{img_name}.json", original_512x512_path=f"{original_512x512_folder}/{img}")
    