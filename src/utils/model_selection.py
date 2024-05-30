###########################
# Author: Lam Thai Nguyen #
###########################

""" 
Goal: Selects no more than two models given 5 xlsx files (5 folds).

Selection criteria:
- 01 model with the best performance (highest average F1 score).
- 01 model with the best record (highest seen F1 score).

Evaluation criteria:
- 1: O. glaberrima
- 2: O. sativa
- 3: O. glaberrima and O. sativa 
"""

""" How to run this code as a module: python -m src.utils.model_selection """

import os
import argparse
from ..image_processor.AccuracyManager import AccuracyManager


def main(person, evaluation_criterion, folder):
    manager = AccuracyManager()

    for f in os.listdir(folder):
        if f.startswith(f"{person}_{evaluation_criterion}"):
            print(f"==>> Reading {f}")
            manager.read_fold(f"{folder}/{f}")
        
    model_A, model_B = manager.model_selection()
    print(f"==>> model_A: {model_A}")
    print(f"==>> model_B: {model_B}")
    

def test_manager_fold_tracker(person, evaluation_criterion, folder):
    manager = AccuracyManager()
    for _ in range(5):
        manager.read_fold(f"{folder}/{person}_{evaluation_criterion}_1.xlsx")
    assert manager.fold_A_tracker is not None
    assert manager.fold_B_tracker is not None
    assert len(manager.fold_A_tracker["DEEPCRACK"]["F1"]) == 5


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Model selection script")
    argparser.add_argument("person", type=str, help="T: Thai, K: Kien ==>> version of person", choices=["T", "K"])
    argparser.add_argument("evaluation_criterion", type=int, help="1: O. glaberrima, 2: O. sativa, 3: O. glaberrima and O. sativa", choices=[1, 2, 3])
    argparser.add_argument("-f", "--folder", type=str, help="Folder of segmentation results ==>> xlsx files", default="data/segmentation_result")
    
    args = argparser.parse_args()
    person, evaluation_criterion, folder = args.person, args.evaluation_criterion, args.folder
    people = {"T": "Thai", "K": "Kien"}
    criteria = {"1": "O. glaberrima", "2": "O. sativa", "3": "O. glaberrima and O. sativa"}
    
    print("".center(50, "="))
    print(f"==>> person: {person}, {people[person]}")
    print(f"==>> evaluation_criterion: {evaluation_criterion}. {criteria[str(evaluation_criterion)]}")
    print(f"==>> folder: {folder}")
    print("".center(50, "="))
    
    main(person, evaluation_criterion, folder)
    