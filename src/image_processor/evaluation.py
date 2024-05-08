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
    y_true_disk, n_true_junctions = _junction2disk(y_true, return_counts=True)

    white_px_true = np.argwhere(y_true_disk > 0)
    white_px_pred = np.argwhere(y_pred > 0)
    n_pred_junctions = len(white_px_pred)
    
    y_true_tuple = tuple(map(tuple, white_px_true))
    y_pred_tuple = tuple(map(tuple, white_px_pred))
    
    true_pos, false_pos, false_neg = (0, 0, 0)
    
    for junction in y_pred_tuple:
        if junction in y_true_tuple:
            true_pos += 1
            
    false_pos = n_pred_junctions - true_pos
    false_neg = n_true_junctions - true_pos
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1_score = 2 * precision * recall / (precision + recall)

    results = [f1_score]
    if _return_metrics:
        results.append(precision)
        results.append(recall)
    
    return results


def _junction2disk(y_true: np.ndarray, return_counts=False) -> np.ndarray:
    """
    ## Description
    Turns junction to a disk of radius R=3
    
    ## Arguments
    - y_true: np.ndarray
    - return_counts=False
    
    ## Returns
    y_true_disk: np.ndarray
    """
    y_true_disk = np.zeros((512, 512))
    
    white_px = np.argwhere(y_true > 0)
    n_junctions = len(white_px)
    
    for x, y in white_px:
        y_true_disk[x-3:x+4, y] = 255
        y_true_disk[x, y-3:y+4] = 255
        y_true_disk[x-2:x+3, y-2:y+3] = 255
        
    results = [y_true_disk]
    
    if return_counts:
        results.append(n_junctions)
    
    return results
    