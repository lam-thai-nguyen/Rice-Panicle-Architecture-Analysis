import numpy as np
from .thinning import _zhang_suen, _gradient_based_optimization
from .clustering import _crossing_number, _dbscan
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
        def gradient_based_optimization(image):
            return _gradient_based_optimization(image)

    class Clustering:
        @staticmethod
        def crossing_number(skeleton_img: np.ndarray):
            """
            ## Description
            Performs Crossing Number Method to find junctions in a given skeleton image.

            ## Arguments
            skeleton_img: np.ndarray -> the skeleton matrix.

            ## Returns
            junction_img: np.ndarray -> the skeleton with junction matrix.
            """
            return _crossing_number(skeleton_img)

        @staticmethod
        def dbscan(image):
            return _dbscan(image)

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
        