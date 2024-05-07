import numpy as np
from sklearn.metrics import f1_score


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray, _return_metrics: bool = False) -> float:
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
    