import numpy as np
from .thinning import _zhang_suen, _gradient_based_optimization
from .clustering import _crossing_number, _dbscan


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
        pass
