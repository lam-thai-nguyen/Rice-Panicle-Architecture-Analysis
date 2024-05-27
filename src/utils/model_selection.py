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
import pandas as pd
import numpy as np
from ..image_processor.AccuracyManager import AccuracyManager


# Change these hyperparameters accordingly
PERSON = "K"
EVALUATION_CRITERIA = 1
FOLDER = "data/segmentation_result"


def main():
    manager = AccuracyManager()

    for f in os.listdir(FOLDER):
        if f.startswith(f"{PERSON}_{EVALUATION_CRITERIA}"):
            manager.read_fold(f"{FOLDER}/{f}")
        
    model_A, model_B = manager.model_selection()
    print(f"==>> model_A: {model_A}")
    print(f"==>> model_B: {model_B}")


if __name__ == "__main__":
    main()  
    