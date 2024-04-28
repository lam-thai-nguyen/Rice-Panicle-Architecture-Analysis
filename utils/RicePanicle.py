import pandas as pd
from json2binary import json2binary
from bbox import generate_bbox_grains_junctions, generate_bbox_pb
from thin import thin


class RicePanicle:
    def __init__(self, user: str, file_path: str) -> None:
        """
        user = T or K
        
        file_path can either be original file path or json file path
        
        example path:
            - dataset/original/O. glaberrima/2_2_1_1_3_DSC09839.jpg
            - dataset/annotated/annotated-T/O. glaberrima/2_2_1_1_3_DSC09839.json
        """
        self.user = user
        self.original_path = None
        self.json_path = None
        self.xml_path = None
        self.binary_path = None
        self.code = None
        self.accession_id = None
        self.assay_nb = None
        self.repeat_nb = None
        self.plant_nb = None
        self.panicle_nb = None
        self.image_name = None
        self.species = None
        
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
        
    def thin(self, method: str) -> None:
        thin(self.binary_path, method)
        
        

if __name__ == "__main__":
    rice_panicle = RicePanicle('T', "dataset/annotated/annotated-T/O. glaberrima/53_2_1_1_1_DSC01741.jpg")
    # rice_panicle.return_info()
    # rice_panicle.generate_bbox_grains_junctions()
    # rice_panicle.generate_bbox_pb()
    # rice_panicle.json2binary()
    rice_panicle.thin('zhang')
