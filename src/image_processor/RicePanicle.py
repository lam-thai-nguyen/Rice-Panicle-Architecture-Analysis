import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from json2binary import json2binary
from bbox import generate_bbox_grains_junctions, generate_bbox_pb
from thin import thin
from cluster import cluster
from CustomExceptions import MissingRequiredFile


class RicePanicle:
    def __init__(self, user: str, file_path: str) -> None:
        """
        user = T or K
        
        file_path can either be original file path or json file path
        
        example path:
            - dataset/original/O. glaberrima/2_2_1_1_3_DSC09839.jpg
            - dataset/annotated/annotated-T/O. glaberrima/2_2_1_1_3_DSC09839.json
        """
        # ====================================
        self.user = user
        self.original_path = None
        self.json_path = None
        self.xml_path = None
        self.binary_path = None
        # ====================================
        self.code = None
        self.accession_id = None
        self.assay_nb = None
        self.repeat_nb = None
        self.plant_nb = None
        self.panicle_nb = None
        self.image_name = None
        self.species = None
        # ====================================
        self.binary: np.ndarray = None
        self.skeleton: np.ndarray = None
        self.junction_img: np.ndarray = None
        # ====================================
        
        if file_path.endswith('jpg'):
            self._process_image_file(file_path)
        elif file_path.endswith('json'):
            self._process_json_file(file_path)

    def _process_image_file(self, file_path) -> None:
        self.original_path = file_path
        info = self.original_path.split('/')[-1].split('_')
        self._extract_info(info, image_file=True)
        self.code = self.original_path.split('/')[-1][:-4]
        self.json_path = f"dataset/annotated/annotated-{self.user}/{self.species}/{self.code}.json"
        self.xml_path = f"dataset/vertex_coordinates/{self.species}/{self.code}.ricepr"
        self.binary_path = self.json_path[:-4] + "jpg"

    def _process_json_file(self, file_path) -> None:
        self.json_path = file_path
        info = self.json_path.split('/')[-1].split('_')
        self._extract_info(info, json_file=True)
        self.code = self.json_path.split('/')[-1][:-5]
        self.original_path = f"dataset/original/{self.species}/{self.code}.jpg"
        self.xml_path = f"dataset/vertex_coordinates/{self.species}/{self.code}.ricepr"
        self.binary_path = self.json_path[:-4] + "jpg"

    def _extract_info(self, info, image_file=False, json_file=False) -> None:
        if image_file:
            index = 4
        elif json_file:
            index = 5
        self.accession_id = int(info[0])
        self.assay_nb = int(info[1])  # always 2
        self.repeat_nb = int(info[2])
        self.plant_nb = int(info[3])
        self.panicle_nb = int(info[4])
        self.image_name = info[5][:-index]
        df = pd.read_excel("dataset/metadata/InfoAccession_Asian_African.xlsx")
        self.species = df["Origin"][self.accession_id - 1]
        
    def return_info(self) -> None:
        print("=======================")
        print(f"Name: {self.image_name}")
        print(f"Species: {self.species}")
        print(f"Accession ID: {self.accession_id}")
        print(f"Assay #: {self.assay_nb}")
        print(f"Repeat #: {self.repeat_nb}")
        print(f"Panicle #: {self.panicle_nb}")
        print(f"Original path: {self.original_path}")
        print(f"json path: {self.json_path}")
        print(f"XML path: {self.xml_path}")
        print(f"Binary path: {self.binary_path}")
        print("=======================")
        
    def json2binary(self) -> None:
        json2binary(self.json_path)
        
    def generate_bbox_grains_junctions(self) -> None:
        generate_bbox_grains_junctions(self.original_path, self.xml_path)
        
    def generate_bbox_pb(self) -> None:
        generate_bbox_pb(self.original_path, self.xml_path)
        
    def thin(self, method: str, _pre_process: bool, _post_process: bool, **kwargs) -> np.ndarray:
        """
        ## Description
        - Performing <method> thinning method
        
        ## Arguments:
        - method = {zhang, gradient}
        - _pre_process (bool)
        - _post_process (bool)
        
        # kwargs
        - When _pre_process is True
            - _plot_bin_img=False
            - _plot_skeleton=False
        - When _post_process is True
            - min_length=0
            - _plot_prune=False
        - When both are True
            - _plot_result=False        
                
        ## Returns
        - skeleton (np.ndarray)
        """
        try:
            self.skeleton = thin(self.binary_path, method, _pre_process, _post_process, **kwargs)
        except AttributeError:
            raise MissingRequiredFile(">> Binary image not found <<")
        
        return self.skeleton
            
    def cluster(self, method: str) -> np.ndarray:
        """
        ## Description
        - Performing <method> to find the junctions in a skeleton image
        
        ## Arguments
        - method = {cn, }
        
        ## Returns
        - junction_img (np.ndarray)
        """
        if isinstance(self.skeleton, np.ndarray):
            self.junction_img = cluster(self.skeleton, method)
            return self.junction_img
        else:
            raise MissingRequiredFile(">> Skeleton not found <<")
        
    def thin_cluster(self, thin_method: str, _pre_process: bool, _post_process: bool, cluster_method: str, **kwargs) -> list[np.ndarray]:
        """
        ## Description
        - Perform <thin_method> thinning and <cluster_method> method.
        
        ## Arguments
        - thin_method = {zhang, gradient}
        - _pre_process (bool)
        - _post_process (bool)
        - cluster_method = {cn, }
        
        ## kwargs
        - Thinning kwargs:
            - When _pre_process is True
                - _plot_bin_img=False
                - _plot_skeleton=False
            - When _post_process is True
                - min_length=0
                - _plot_prune=False
            - When both are True
                - _plot_result=False        
        - Clustering kwargs: 
            - ...
            
        ## Returns
        - skeleton (np.ndarray)
        - junction_img (np.ndarray)      
        """
        self.skeleton = self.thin(thin_method, _pre_process, _post_process, **kwargs)
        self.junction_img = self.cluster(cluster_method)
        
    def imshow_binary(self):
        if self.binary is None:
            print("ERROR! Processed binary not found.")
            return
        plt.figure(figsize=(7, 7))
        plt.imshow(self.binary, cmap='gray')
        plt.axis('off')
        plt.title("Binary Image")
        plt.show()
        
    def imshow_skeleton(self):
        if self.skeleton is None:
            print("ERROR! Skeleton not found.")
            return
        plt.figure(figsize=(7, 7))
        plt.imshow(self.skeleton, cmap='gray')
        plt.axis('off')
        plt.title("Skeleton Image")
        plt.show()
        

if __name__ == "__main__":
    rice_panicle = RicePanicle(user='T', file_path="dataset/annotated/annotated-K/O. sativa/38_2_1_3_1_DSC09528_.json")
    # rice_panicle.return_info()
    rice_panicle.json2binary()
    # rice_panicle.thin(method='zhang', _pre_process=1, _post_process=1, min_length=10, _plot_result=1)
    # rice_panicle.cluster('cn')
    # rice_panicle.thin_cluster('zhang', 1, 0, 'cn')
    