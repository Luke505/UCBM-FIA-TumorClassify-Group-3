"""
Evaluation metrics for binary classification

This module implements various metrics for evaluating binary classification
performance, including accuracy, sensitivity, specificity, and AUC
"""

from typing import Dict, Tuple

import numpy as np


def confusion_matrix(labels_true: np.ndarray, labels_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Calculate confusion matrix components
    
    For binary classification (classes 2 and 4):
    - True Positive (TP): Correctly predicted malignant (class 4)
    - True Negative (TN): Correctly predicted benign (class 2)
    - False Positive (FP): Predicted malignant but actually benign
    - False Negative (FN): Predicted benign but actually malignant
    
    Parameters:
    -----------
    labels_true : np.ndarray
        True labels
    labels_pred : np.ndarray
        Predicted labels
        
    Returns:
    --------
    Tuple[int, int, int, int]
        (tn, fp, fn, tp)
    """
    # Convert labels to binary (0 for benign/class 2, 1 for malignant/class 4)
    labels_true_binary = (labels_true == 4).astype(int)
    labels_pred_binary = (labels_pred == 4).astype(int)

    # Calculate confusion matrix components
    tp = np.sum((labels_true_binary == 1) & (labels_pred_binary == 1))
    tn = np.sum((labels_true_binary == 0) & (labels_pred_binary == 0))
    fp = np.sum((labels_true_binary == 0) & (labels_pred_binary == 1))
    fn = np.sum((labels_true_binary == 1) & (labels_pred_binary == 0))

    return tn, fp, fn, tp


def accuracy_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    
    Accuracy represents the proportion of correct predictions
    
    Parameters:
    -----------
    labels_true : np.ndarray
        True labels
    labels_pred : np.ndarray
        Predicted labels
        
    Returns:
    --------
    float
        Accuracy score between 0 and 1
    """
    tn, fp, fn, tp = confusion_matrix(labels_true, labels_pred)
    total = tp + tn + fp + fn

    if total == 0:
        return 0.0

    return (tp + tn) / total


def error_rate(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate error rate: (fp + fn) / (tp + tn + fp + fn)
    
    Error rate represents the proportion of incorrect predictions
    Error rate = 1 - Accuracy
    
    Parameters:
    -----------
    labels_true : np.ndarray
        True labels
    labels_pred : np.ndarray
        Predicted labels
        
    Returns:
    --------
    float
        Error rate between 0 and 1
    """
    return 1.0 - accuracy_score(labels_true, labels_pred)


def sensitivity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    # noinspection GrazieInspection
    """
    Calculate sensitivity (recall, true positive rate): tp / (tp + fn)

    Sensitivity measures the proportion of actual positives (malignant)
    that are correctly identified

    Parameters:
    -----------
    labels_true : np.ndarray
        True labels
    labels_pred : np.ndarray
        Predicted labels

    Returns:
    --------
    float
        Sensitivity score between 0 and 1
    """
    tn, fp, fn, tp = confusion_matrix(labels_true, labels_pred)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def specificity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate specificity (true negative rate): tn / (tn + fp)
    
    Specificity measures the proportion of actual negatives (benign)
    that are correctly identified
    
    Parameters:
    -----------
    labels_true : np.ndarray
        True labels
    labels_pred : np.ndarray
        Predicted labels
        
    Returns:
    --------
    float
        Specificity score between 0 and 1
    """
    tn, fp, fn, tp = confusion_matrix(labels_true, labels_pred)

    if tn + fp == 0:
        return 0.0

    return tn / (tn + fp)


def geometric_mean(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate geometric mean: sqrt(sensitivity * specificity)
    
    The geometric mean provides a balanced measure for imbalanced datasets
    
    Parameters:
    -----------
    labels_true : np.ndarray
        True labels
    labels_pred : np.ndarray
        Predicted labels
        
    Returns:
    --------
    float
        Geometric mean between 0 and 1
    """
    sens = sensitivity(labels_true, labels_pred)
    spec = specificity(labels_true, labels_pred)

    return np.sqrt(sens * spec)


def auc_score(labels_true: np.ndarray, labels_proba: np.ndarray) -> float:
    """
    Calculate Area Under the ROC Curve (AUC) using the trapezoidal rule
    
    The ROC curve plots True Positive Rate (sensitivity) vs. False Positive Rate
    (1 - specificity) at various threshold settings
    
    Parameters:
    -----------
    labels_true : np.ndarray
        True labels
    labels_proba : np.ndarray
        Predicted probabilities for the positive class (malignant/class 4)
        If 2D array, assumes the second column contains probabilities for class 4
        
    Returns:
    --------
    float
        AUC score between 0 and 1
    """
    # Handle 2D probability arrays
    if len(labels_proba.shape) == 2:
        if labels_proba.shape[1] > 1:
            labels_proba = labels_proba[:, 1]  # Get probabilities for positive class (class 4)
        else:
            labels_proba = labels_proba[:, 0]  # If only one column, use it

    # Convert labels to binary (0 for benign, 1 for malignant)
    labels_true_binary = (labels_true == 4).astype(int)

    # Sort by predicted probability in descending order
    sorted_indices = np.argsort(-labels_proba)
    labels_true_sorted = labels_true_binary[sorted_indices]

    # Calculate cumulative number of positives and negatives
    n_positive = np.sum(labels_true_binary)
    n_negative = len(labels_true_binary) - n_positive

    if n_positive == 0 or n_negative == 0:
        # If all samples are of one class, AUC is undefined (return 0.5)
        return 0.5

    # Calculate TPR and FPR at each threshold
    tpr_list = []
    fpr_list = []

    tp = 0
    fp = 0

    for label in labels_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_positive
        fpr = fp / n_negative

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Add origin point (0, 0)
    tpr_list = [0] + tpr_list
    fpr_list = [0] + fpr_list

    # Calculate AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(tpr_list)):
        # Area of trapezoid
        width = fpr_list[i] - fpr_list[i - 1]
        height = (tpr_list[i] + tpr_list[i - 1]) / 2
        auc += width * height

    return auc


def calculate_all_metrics(labels_true: np.ndarray, labels_pred: np.ndarray,
    labels_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate all evaluation metrics
    
    Parameters:
    -----------
    labels_true : np.ndarray
        True labels
    labels_pred : np.ndarray
        Predicted labels
    labels_proba : np.ndarray, optional
        Predicted probabilities (required for AUC calculation)
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing all metric scores
    """
    metrics = {
        'accuracy': accuracy_score(labels_true, labels_pred),
        'error_rate': error_rate(labels_true, labels_pred),
        'sensitivity': sensitivity(labels_true, labels_pred),
        'specificity': specificity(labels_true, labels_pred),
        'geometric_mean': geometric_mean(labels_true, labels_pred),
    }

    # Calculate AUC if probabilities are provided
    if labels_proba is not None:
        metrics['auc'] = auc_score(labels_true, labels_proba)

    # Add confusion matrix components
    tn, fp, fn, tp = confusion_matrix(labels_true, labels_pred)
    metrics['confusion_matrix'] = {
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp)
    }

    return metrics
