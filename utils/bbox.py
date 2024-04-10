import os
import numpy as np  
import cv2
import xml.etree.ElementTree as ET


def generate_bbox(original_img_path, vertex_coordinates_path):
    """
    Create bounding boxes for grains and junctions

    Args:
        original_img_path (str): dataset/original/2_2_1_1_3_DSC09839.JPG
        vertex_coordinates_path (str): dataset/vertex_coordinates/2_2_1_1_3_DSC09839.ricepr
    """
    img = cv2.imread(original_img_path)
    
    # ========= Bounding boxes for junctions ===========
    generating, end, primary, secondary, tertiary, quaternary = _get_vertex(vertex_coordinates_path)
    
    for x, y in generating:
        cv2.rectangle(img, pt1=(x - 10, y - 10), pt2=(x + 10, y + 10), color=(255, 0, 0), thickness=2)
    for x, y in primary:
        cv2.rectangle(img, pt1=(x - 10, y - 10), pt2=(x + 10, y + 10), color=(255, 0, 0), thickness=2)
    for x, y in secondary:
        cv2.rectangle(img, pt1=(x - 10, y - 10), pt2=(x + 10, y + 10), color=(255, 0, 0), thickness=2)
    for x, y in tertiary:
        cv2.rectangle(img, pt1=(x - 10, y - 10), pt2=(x + 10, y + 10), color=(255, 0, 0), thickness=2)
    for x, y in quaternary:
        cv2.rectangle(img, pt1=(x - 10, y - 10), pt2=(x + 10, y + 10), color=(255, 0, 0), thickness=2)
    # for x, y in end:
    #     cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
        
    # ========= Bounding boxes for grains =========
    edges = _get_edges(vertex_coordinates_path)
    
    for x1, y1, x2, y2 in edges:
        if [x1, y1] in generating and [x2, y2] in end:
            # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
            if abs(x1 - x2) <= 5:
                _draw_bbox(img, x1, y1, x2, y2, condition=1)
                    
            elif abs(x1 - x2) < 25:
                _draw_bbox(img, x1, y1, x2, y2, condition=2)
                
            if abs(y1 - y2) <= 5:
                _draw_bbox(img, x1, y1, x2, y2, condition=3)
            
            elif abs(y1 - y2) < 25:
                _draw_bbox(img, x1, y1, x2, y2, condition=4) 
            
            if abs(x1 - x2) >= 25 and abs(y1 - y2) >= 25:
                cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
            
        if [x1, y1] in primary and [x2, y2] in end:
            # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
            if abs(x1 - x2) <= 5:
                _draw_bbox(img, x1, y1, x2, y2, condition=1)
                    
            elif abs(x1 - x2) < 25:
                _draw_bbox(img, x1, y1, x2, y2, condition=2)
                
            if abs(y1 - y2) <= 5:
                _draw_bbox(img, x1, y1, x2, y2, condition=3)
            
            elif abs(y1 - y2) < 25:
                _draw_bbox(img, x1, y1, x2, y2, condition=4) 
            
            if abs(x1 - x2) >= 25 and abs(y1 - y2) >= 25:
                cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
            
        if [x1, y1] in secondary and [x2, y2] in end:
            if abs(x1 - x2) <= 5:
                _draw_bbox(img, x1, y1, x2, y2, condition=1)
                    
            elif abs(x1 - x2) < 25:
                _draw_bbox(img, x1, y1, x2, y2, condition=2)
                
            if abs(y1 - y2) <= 5:
                _draw_bbox(img, x1, y1, x2, y2, condition=3)
            
            elif abs(y1 - y2) < 25:
                _draw_bbox(img, x1, y1, x2, y2, condition=4) 
            
            if abs(x1 - x2) >= 25 and abs(y1 - y2) >= 25:
                cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
            
        if [x1, y1] in tertiary and [x2, y2] in end:
            # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
            if abs(x1 - x2) <= 5:
                _draw_bbox(img, x1, y1, x2, y2, condition=1)
                    
            elif abs(x1 - x2) < 25:
                _draw_bbox(img, x1, y1, x2, y2, condition=2)
                
            if abs(y1 - y2) <= 5:
                _draw_bbox(img, x1, y1, x2, y2, condition=3)
            
            elif abs(y1 - y2) < 25:
                _draw_bbox(img, x1, y1, x2, y2, condition=4) 
            
            if abs(x1 - x2) >= 25 and abs(y1 - y2) >= 25:
                cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
    
        if [x1, y1] in quaternary and [x2, y2] in end:
            # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
            if abs(x1 - x2) <= 5:
                _draw_bbox(img, x1, y1, x2, y2, condition=1)
                    
            elif abs(x1 - x2) < 25:
                _draw_bbox(img, x1, y1, x2, y2, condition=2)
                
            if abs(y1 - y2) <= 5:
                _draw_bbox(img, x1, y1, x2, y2, condition=3)
            
            elif abs(y1 - y2) < 25:
                _draw_bbox(img, x1, y1, x2, y2, condition=4) 
            
            if abs(x1 - x2) >= 25 and abs(y1 - y2) >= 25:
                cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)

    save_path = "dataset/bbox"
    index = len("dataset/original/")
    cv2.imwrite(save_path + "/" + original_img_path[index:], img)
    

def _get_vertex(vertex_coordinates_path):
    """
    Returns:
        generating (list[list]): coordinates of all generating vertices
        end: ...
        primary: ...
        secondary: ...
        tertiary: ...
    """
    tree = ET.parse(vertex_coordinates_path)
    root = tree.getroot()
    
    generating = []
    end = []
    primary = []
    secondary = []
    tertiary = []
    quaternary = []
    
    for vertex in root.iter('vertex'):
        x = int(vertex.attrib['x'])
        y = int(vertex.attrib['y'])
        type_ = vertex.attrib['type']
        if type_ == "Generating":
            generating.append([x, y])
        elif type_ == "End":
            end.append([x, y])
        elif type_ == "Primary":
            primary.append([x, y])
        elif type_ == "Seconday":
            secondary.append([x, y])
        elif type_ == "Tertiary":
            tertiary.append([x, y])
            
    return generating, end, primary, secondary, tertiary, quaternary


def _get_edges(vertex_coordinates_path):
    """
    Returns: edges: [x1, y1, x2, y2] with (x_i, y_i) -> vertex_i
    """
    tree = ET.parse(vertex_coordinates_path)
    root = tree.getroot()
    
    edges = []
    for edge in root.iter('edge'):
        vertex1 = edge.attrib['vertex1']
        vertex2 = edge.attrib['vertex2']
        
        x1 = int(vertex1.split('=')[1].split(',')[0])
        y1 = int(vertex1.split('=')[2].split(']')[0])
        
        x2 = int(vertex2.split('=')[1].split(',')[0])
        y2 = int(vertex2.split('=')[2].split(']')[0])
        
        edges.append([x1, y1, x2, y2])

    return edges  


def _draw_bbox(img, x1, y1, x2, y2, condition: int) -> None:
    """
    Draw bounding box based on conditions
        condition = 1 (abs(x1 - x2) <= 5)
                  = 2 (5 < abs(x1 - x2) < 25)
                  = 3  (abs(y1 - y2) <= 5)
                  = 4  (5 < abs(y1 - y2) < 25)
    """  
    if condition == 1:
        if x1 > x2:
            cv2.rectangle(img, pt1=(x1 + 12, y1), pt2=(x2 - 12, y2), color=(0, 0, 255), thickness=2)
        elif x1 <= x2:
            cv2.rectangle(img, pt1=(x1 - 12, y1), pt2=(x2 + 12, y2), color=(0, 0, 255), thickness=2)
            
    if condition == 2:
        if x1 > x2:
            cv2.rectangle(img, pt1=(x1 + 7, y1), pt2=(x2 - 5, y2), color=(0, 0, 255), thickness=2)
        elif x1 <= x2:
            cv2.rectangle(img, pt1=(x1 - 7, y1), pt2=(x2 + 5, y2), color=(0, 0, 255), thickness=2)
     
    if condition == 3:
        if y1 > y2:
            cv2.rectangle(img, pt1=(x1, y1 + 12), pt2=(x2, y2 - 12), color=(0, 0, 255), thickness=2)
        elif y1 <= y2:
            cv2.rectangle(img, pt1=(x1, y1 - 12), pt2=(x2, y2 + 12), color=(0, 0, 255), thickness=2)
         
    if condition == 4:
        if y1 > y2:
            cv2.rectangle(img, pt1=(x1, y1 + 7), pt2=(x2, y2 - 5), color=(0, 0, 255), thickness=2)
        elif y1 <= y2:
            cv2.rectangle(img, pt1=(x1, y1 - 7), pt2=(x2, y2 + 5), color=(0, 0, 255), thickness=2)
        
    return None
    

def bounding_box(img, x1, y1, x2, y2) -> None:
    """
    
    """
    if abs(x1 - x2) <= 5:
        _draw_bbox(img, x1, y1, x2, y2, condition=1)
            
    elif abs(x1 - x2) < 25:
        _draw_bbox(img, x1, y1, x2, y2, condition=2)
        
    if abs(y1 - y2) <= 5:
        _draw_bbox(img, x1, y1, x2, y2, condition=3)
    
    elif abs(y1 - y2) < 25:
        _draw_bbox(img, x1, y1, x2, y2, condition=4) 
    
    if abs(x1 - x2) >= 25 and abs(y1 - y2) >= 25:
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)

    return None

        
if __name__ == "__main__":
    original_folder_path = "dataset/original"
    img_names = os.listdir(original_folder_path)
    
    xml_folder_path = "dataset/vertex_coordinates"
    
    for img in img_names:
        img_path = original_folder_path + "/" + img
        xml_path = xml_folder_path + "/" + img[:-4] + ".ricepr"
        if xml_path == "dataset/vertex_coordinates/42_2_2_2_3_DSC01693.ricepr":
            generate_bbox(img_path, xml_path)
            print(f"\nSUCCESSFUL >>>> {img} <<<<")
            break
    