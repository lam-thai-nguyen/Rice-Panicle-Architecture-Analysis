###########################
# Author: Lam Thai Nguyen #
###########################

import numpy as np
from .thinning import _zhang_suen, _gradient_based_optimization
from .clustering import _crossing_number, _dbscan, _finch
from .evaluation import _f1_score


class RicePanicle:
    class Thinning:
        @staticmethod
        def zhang_suen(binary_image: np.ndarray):
            """
            ## Description
            Perform Z.S. thinning method on a binary image

            ## Arguments
            binary_img (np.ndarray) -> binary image matrix.

            ## Returns
            skeleton_img (np.ndarray) -> skeleton image matrix.
            """
            return _zhang_suen(binary_image)

        @staticmethod
        def gradient_based_optimization(binary_image: np.ndarray):
            return _gradient_based_optimization(binary_image)

    class Clustering:
        @staticmethod
        def crossing_number(skeleton_img: np.ndarray, return_pred_: bool):
            """
            ## Description
            Performs Crossing Number Method to find junctions in a given skeleton image.
            
            ## Arguments
            - skeleton_img: np.ndarray -> the skeleton matrix.
            - return_pred_: bool
            
            ## Returns
            - junction_img: np.ndarray -> the skeleton with junction matrix.
            - y_pred: np.ndarray, if return_pred_
            """
            return _crossing_number(skeleton_img, return_pred_)

        @staticmethod
        def dbscan(skeleton_img: np.ndarray, return_pred_: bool):
            return _dbscan(skeleton_img, return_pred_)
        
        @staticmethod
        def finch(skeleton_img: np.ndarray, return_pred_: bool):
            """
            ## Description
            Performs FINCH Method to find junctions in a given skeleton image.
            
            ## Arguments
            - skeleton_img: np.ndarray -> the skeleton matrix.
            - return_pred_: bool -> Set to true to return y_pred
            
            ## Returns
            - junction_img: np.ndarray -> the skeleton with junction matrix.
            - y_pred: np.ndarray, if return_pred_
            """
            return _finch(skeleton_img, return_pred_)

    class Evaluation:
        @staticmethod
        def f1_score(y_true: np.ndarray, y_pred: np.ndarray, _return_metrics: bool = False):
            """
            ## Description
            Compute the f1-score as the accuracy of rice panicle junction detection.

            ## Arguments
            - y_true (np.ndarray)
            - y_pred (np.ndarray)
            - _return_metrics (bool) = False -> if True, returns precision and recall

            ## Returns:
            f1, (pr, rc) -> f1-score, (precision, recall)
            """
            return _f1_score(y_true, y_pred, _return_metrics)
        
    class DetectionAccuracy:
        def __init__(self, name: str, all_junctions: list[float], main_axis: list[float], high_order: list[float], model: str = None):
            self.name = name
            self.f1_1, self.pr_1, self.rc_1 = all_junctions
            self.f1_2, self.pr_2, self.rc_2 = main_axis
            self.f1_3, self.pr_3, self.rc_3 = high_order
            self.model = model
            
        def show(self):
            print(f"==>> {self.name} <<==")
            print(f"All junctions: f1-score: {self.f1_1:.4f}, precision: {self.pr_1:.4f}, recall: {self.rc_1:.4f}")
            print(f"Main axis junctions: f1-score: {self.f1_2:.4f}, precision: {self.pr_2:.4f}, recall: {self.rc_2:.4f}")
            print(f"High order junctions: f1-score: {self.f1_3:.4f}, precision: {self.pr_3:.4f}, recall: {self.rc_3:.4f}")
            if self.model is not None:
                print(f"Detected by Model: {self.model}")
