"""
Adapted from the following references:
[1] https://github.com/JunMa11/NeurIPS-CellSeg/blob/main/baseline/compute_metric.py
[2] https://github.com/stardist/stardist/blob/master/stardist/matching.py

"""
from typing import Optional
import numpy as np
from skimage import segmentation
from scipy.optimize import linear_sum_assignment
from numba import jit
from cellpose.metrics import average_precision

__all__ = ["evaluate_f1_score_cellseg", "evaluate_f1_score"]


def evaluate_f1_score_cellseg(masks_true, masks_pred, threshold=0.5):
    """
    Get confusion elements for cell segmentation results.
    Boundary pixels are not considered during evaluation.
    """

    if np.prod(masks_true.shape) < (5000 * 5000):
        masks_true = _remove_boundary_cells(masks_true.astype(np.int32))
        masks_pred = _remove_boundary_cells(masks_pred.astype(np.int32))

        tp, fp, fn = get_confusion(masks_true, masks_pred, threshold)

    # Compute by Patch-based way for large images
    else:
        H, W = masks_true.shape
        roi_size = 2000

        # Get patch grid by roi_size
        if H % roi_size != 0:
            n_H = H // roi_size + 1
            new_H = roi_size * n_H
        else:
            n_H = H // roi_size
            new_H = H

        if W % roi_size != 0:
            n_W = W // roi_size + 1
            new_W = roi_size * n_W
        else:
            n_W = W // roi_size
            new_W = W

        # Allocate values on the grid
        gt_pad = np.zeros((new_H, new_W), dtype=masks_true.dtype)
        pred_pad = np.zeros((new_H, new_W), dtype=masks_true.dtype)
        gt_pad[:H, :W] = masks_true
        pred_pad[:H, :W] = masks_pred

        tp, fp, fn = 0, 0, 0

        # Calculate confusion elements for each patch
        for i in range(n_H):
            for j in range(n_W):
                gt_roi = _remove_boundary_cells(
                    gt_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]
                )
                pred_roi = _remove_boundary_cells(
                    pred_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]
                )
                tp_i, fp_i, fn_i = get_confusion(gt_roi, pred_roi, threshold)
                tp += tp_i
                fp += fp_i
                fn += fn_i

    # Calculate f1 score
    precision, recall, f1_score = evaluate_f1_score(tp, fp, fn)

    return precision, recall, f1_score


def evaluate_f1_score(tp, fp, fn):
    """Evaluate F1-score for the given confusion elements"""

    # Do not Compute on trivial results
    if tp == 0:
        precision, recall, f1_score = 0, 0, 0

    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def _remove_boundary_cells(mask):
    """Remove cells on the boundary from the mask"""

    # Identify boundary cells
    W, H = mask.shape
    bd = np.ones((W, H))
    bd[2 : W - 2, 2 : H - 2] = 0
    bd_cells = np.unique(mask * bd)

    # Remove cells on the boundary
    for i in bd_cells[1:]:
        mask[mask == i] = 0

    # Allocate labels as sequential manner
    new_label, _, _ = segmentation.relabel_sequential(mask)

    return new_label


def get_confusion(masks_true, masks_pred, threshold=0.5):
    """Calculate confusion matrix elements: (TP, FP, FN)"""
    num_gt_instances = np.max(masks_true)
    num_pred_instances = np.max(masks_pred)

    if num_pred_instances == 0:
        print("No segmentation results!")
        tp, fp, fn = 0, 0, 0

    else:
        # Calculate IoU and exclude background label (0)
        iou = _get_iou(masks_true, masks_pred)
        iou = iou[1:, 1:]

        # Calculate true positives
        tp = _get_true_positive(iou, threshold)
        fp = num_pred_instances - tp
        fn = num_gt_instances - tp

    return tp, fp, fn


def _get_true_positive(iou, threshold=0.5):
    """Get true positive (TP) pixels at the given threshold"""

    # Number of instances to be matched
    num_matched = min(iou.shape[0], iou.shape[1])

    # Find optimal matching by using IoU as tie-breaker
    costs = -(iou >= threshold).astype(np.float32) - iou / (2 * num_matched)
    matched_gt_label, matched_pred_label = linear_sum_assignment(costs)

    # Consider as the same instance only if the IoU is above the threshold
    match_ok = iou[matched_gt_label, matched_pred_label] >= threshold
    tp = match_ok.sum()

    return tp


def _get_iou(masks_true, masks_pred):
    """Get the iou between masks_true and masks_pred"""

    # Get overlap matrix (GT Instances Num, Pred Instance Num)
    overlap = _label_overlap(masks_true, masks_pred)

    # Predicted instance pixels
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)

    # GT instance pixels
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)

    # Calculate intersection of union (IoU)
    union = n_pixels_pred + n_pixels_true - overlap
    iou = overlap / union

    # Ensure numerical values
    iou[np.isnan(iou)] = 0.0

    return iou


@jit(nopython=True)
def _label_overlap(x, y):
    """Get pixel overlaps between two masks

    Parameters
    ------------
    x, y (np array; dtype int): 0=NO masks; 1,2... are mask labels

    Returns
    ------------
    overlap (np array; dtype int): Overlaps of size [x.max()+1, y.max()+1]
    """

    # Make as 1D array
    x, y = x.ravel(), y.ravel()

    # Preallocate a Contact Map matrix
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

    # Calculate the number of shared pixels for each label
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1

    return overlap

def scores_per_threshold(masks_true, masks_pred, thresholds: Optional[np.array] = None):
    """
    Computes F1, Average precision, recall and precision scores at given IoU threshold
    based on `cellpose.metrics.average_precision` 
    
    masks_true, masks_pred: Lists of masks. Masks are 2D np.arrays of type `int`. 
                            0 for background pixel, >0 for instance pixels
                            
    thresholds: np.array of IoU thresholds between 0 and 1. ROIs with IoU < threshold will
                 not be matched.
                 
    Outputs:   
    
    f1, avg_precision, recall, precision: (n_masks x n_thresholds) arrays with one score per 
                                          threshold
    thresholds: (n_thresholds) array with the used IoU thresholds
    (tp,fp,fn): (n_masks x n_thresholds) arrays holding the True Positives, False Positives and
                False Negatives for each mask at each threshold.
    """
    if thresholds is None:
        # floating point arithmetic makes errors, so specify these values by hand
        thresholds = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70,
                               0.75, 0.8, 0.85, 0.9, 0.95])

    masks_true_list = [m for m in masks_true]
    masks_pred_list = [m for m in masks_pred]

    avg_precision, tp, fp, fn = average_precision(masks_true_list, masks_pred_list, threshold=thresholds)
    # Avoid division by zero
    prec_denom = np.maximum((tp + fp), 1.0)
    precision = tp / prec_denom
    recall_denom = np.maximum((tp + fn), 1.0)
    recall = tp / recall_denom
    f1_denom = np.maximum((2 * tp + fn + fp), 1.0)
    f1 = 2 * tp / f1_denom

    return f1, avg_precision, recall, precision, thresholds, (tp, fp, fn)