import pandas as pd
from json2binary import json2binary
from bbox import generate_bbox_grains_junctions, generate_bbox_pb


class RicePanicle:
    def __init__(self, original_file_path: str) -> None:
        info = original_file_path.split('/')[-1].split('_')
        self.accession_id = int(info[0])
        self.assay_nb = int(info[1])  # always 2
        self.repeat_nb = int(info[2])
        self.plant_nb = int(info[3])
        self.panicle_nb = int(info[4])
        self.image_name = info[5][:-4]
        df = pd.read_excel("dataset/metadata/InfoAccession_Asian_African.xlsx")
        self.species = df["Origin"][self.accession_id - 1]
        
        self.file_path = original_file_path
        self.code = original_file_path.split('/')[-1][:-4]
        self.xml_path = f"dataset/vertex_coordinates/{self.species}/{self.code}.ricepr"
        
        
    def return_info(self) -> None:
        print("=======================")
        print(f"Name: {self.image_name}")
        print(f"Species: {self.species}")
        print(f"Accession ID: {self.accession_id}")
        print(f"Assay #: {self.assay_nb}")
        print(f"Repeat #: {self.repeat_nb}")
        print(f"Panicle #: {self.panicle_nb}")
        print("=======================")
        

    def json2binary(self, json_path: str) -> None:
        json2binary(json_path)
        
        
    def generate_bbox_grains_junctions(self):
        generate_bbox_grains_junctions(self.file_path, self.xml_path)
        
    
    def generate_bbox_pb(self):
        generate_bbox_pb(self.file_path, self.xml_path)
        


if __name__ == "__main__":
    rice_panicle = RicePanicle("dataset/original/O. glaberrima/2_2_1_1_3_DSC09839.jpg")
    # rice_panicle.return_info()
    # rice_panicle.generate_bbox_grains_junctions()
    # rice_panicle.generate_bbox_pb()
    # rice_panicle.json2binary("dataset/annotated/annotated-K/O. sativa/13_2_1_1_1_DSC01478.json")
    