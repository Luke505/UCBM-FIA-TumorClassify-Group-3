"""
Visualization functions for plotting results and saving outputs

This module provides functions to generate the following plots:
1. Confusion Matrix
2. ROC Curve
3. Metrics Comparison (Bar Chart)
4. Accuracy Distribution Across Folds (Boxplot/Histogram)
5. Accuracy vs. Number of Neighbors (k)
6. Error Rate vs. Number of Neighbors (k)
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_output_directory(base_dir: str = 'results', data_filename: Optional[str] = None,
    use_subfolder: bool = True) -> str:
    """
    Create output directory with optional subfolder organization
    
    Parameters:
    -----------
    base_dir : str
        Base directory for results (default: 'results')
    data_filename : str, optional
        Name of the input data file (used for subfolder naming)
    use_subfolder : bool
        Whether to create a subfolder for this run (default: True)
        
    Returns:
    --------
    str
        Path to the output directory
    """
    if not use_subfolder:
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    # Extract data file name without extension
    if data_filename:
        data_name = os.path.splitext(os.path.basename(data_filename))[0]
    else:
        data_name = 'experiment'

    # Create a subfolder name with data name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder_name = f"{data_name}_{timestamp}"

    output_dir = os.path.join(base_dir, subfolder_name)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def plot_confusion_matrix(confusion_matrix_dict: Dict[str, int], output_path: Optional[str] = None):
    """
    Plot confusion matrix as a heatmap for binary classification
    
    Displays True Positives (TP), True Negatives (TN), 
    False Positives (FP), and False Negatives (FN)
    
    Parameters:
    -----------
    confusion_matrix_dict : Dict[str, int]
        Dictionary with keys 'TN', 'FP', 'FN', 'TP'
    output_path : str, optional
        Path to save the plot. If None, displays interactively
    """
    # Extract values
    tn = confusion_matrix_dict['TN']
    fp = confusion_matrix_dict['FP']
    fn = confusion_matrix_dict['FN']
    tp = confusion_matrix_dict['TP']

    # Create confusion matrix array
    # Layout: [[TN, FP], [FN, TP]]
    cm = np.array([[tn, fp], [fn, tp]])

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Display confusion matrix as an image
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Set class labels
    classes = ['Benign (2)', 'Malignant (4)']
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title='Confusion Matrix - Binary Tumor Classification',
        ylabel='True Class',
        xlabel='Predicted Class'
    )

    # Rotate tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations with values
    thresh = cm.max() / 2.0
    labels = [['TN', 'FP'], ['FN', 'TP']]

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:d}",
                ha="center", va="center", color=text_color, fontsize=12)

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curve(labels_true: np.ndarray, labels_proba: np.ndarray, auc: float,
    output_path: Optional[str] = None):
    """
    Plot ROC (Receiver Operating Characteristic) curve

    Plots True Positive Rate (TPR) vs. False Positive Rate (FPR)
    and includes a diagonal reference line for a random classifier
    
    Parameters:
    -----------
    labels_true : np.ndarray
        True class labels
    labels_proba : np.ndarray
        Predicted probabilities for the positive class (malignant)
    auc : float
        Area Under the Curve value
    output_path : str, optional
        Path to save the plot. If None, displays interactively
    """
    # Handle 2D probability arrays (extract positive class probabilities)
    if len(labels_proba.shape) == 2:
        labels_proba = labels_proba[:, 1]

    # Convert labels to binary (0 for benign, 1 for malignant)
    labels_true_binary = (labels_true == 4).astype(int)

    # Sort by predicted probability in descending order
    sorted_indices = np.argsort(-labels_proba)
    labels_true_sorted = labels_true_binary[sorted_indices]

    # Calculate the number of positive and negative samples
    n_positive = np.sum(labels_true_binary)
    n_negative = len(labels_true_binary) - n_positive

    # Initialize TPR and FPR lists
    tpr_list = [0.0]
    fpr_list = [0.0]

    tp = 0
    fp = 0

    # Calculate TPR and FPR for each threshold
    for label in labels_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_positive if n_positive > 0 else 0
        fpr = fp / n_negative if n_negative > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve
    ax.plot(fpr_list, tpr_list, color='darkorange', lw=2.5,
        label=f'ROC Curve (AUC = {auc:.3f})')

    # Plot the diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
        label='Random Classifier (AUC = 0.500)')

    # Set plot limits and labels
    ax.set_xlim(xmin=0.0, xmax=1.0)
    ax.set_ylim(ymin=0.0, ymax=1.05)
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax.set_title('ROC Curve - Receiver Operating Characteristic', fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_metrics_comparison(metrics_dict: Dict[str, float], output_path: Optional[str] = None):
    """
    Plot bar chart comparing multiple evaluation metrics
    
    Creates a bar chart with 5 metrics: Accuracy, Error Rate, Sensitivity, 
    Specificity, and Geometric Mean
    
    Parameters:
    -----------
    metrics_dict : Dict[str, float]
        Dictionary containing average metrics across folds
    output_path : str, optional
        Path to save the plot. If None, displays interactively
    """
    # Define the metrics to plot and their labels
    metric_keys = ['accuracy', 'error_rate', 'sensitivity', 'specificity', 'geometric_mean']
    metric_labels = ['Accuracy', 'Error Rate', 'Sensitivity', 'Specificity', 'Geometric Mean']

    # Extract values (try both with and without the 'avg_' prefix)
    values = []
    for key in metric_keys:
        # Try with avg_ prefix first, then without
        value = metrics_dict.get(f'avg_{key}', metrics_dict.get(key, 0.0))
        values.append(value)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the bar chart
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    bars = ax.bar(metric_labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
            f'{value:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Set labels and title
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(ymin=0.0, ymax=1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_accuracy_distribution(accuracies: List[float], output_path: Optional[str] = None,
    plot_type: str = 'boxplot'):
    """
    Plot the distribution of accuracy values across multiple folds or experiments
    
    Visualizes the distribution using either a boxplot or histogram to show
    how accuracy varies across different folds/experiments
    
    Parameters:
    -----------
    accuracies : List[float]
        List of accuracy values from different folds/experiments
    output_path : str, optional
        Path to save the plot. If None, displays interactively
    plot_type : str
        Type of plot: 'boxplot' or 'histogram' (default: 'boxplot')
    """
    accuracies = np.array(accuracies)

    fig, ax = plt.subplots(figsize=(8, 6))

    if plot_type == 'boxplot':
        # Create boxplot
        bp = ax.boxplot(accuracies, patch_artist=True,
            widths=0.5, showmeans=True)

        # Customize boxplot appearance
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)

        for mean in bp['means']:
            mean.set_marker('D')
            mean.set_markerfacecolor('green')
            mean.set_markersize(8)

        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Accuracy Distribution Across Folds', fontsize=13)
        ax.set_xticklabels(['Accuracy'])
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics text
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        median_acc = np.median(accuracies)

        stats_text = f'Mean: {mean_acc:.4f}\nMedian: {median_acc:.4f}\nStd: {std_acc:.4f}'
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    else:  # histogram
        # Create histogram
        ax.hist(accuracies, bins=min(10, len(accuracies)),
            color='skyblue', edgecolor='navy', alpha=0.7)

        # Add mean line
        mean_acc = np.mean(accuracies).astype(float)
        ax.axvline(mean_acc, color='red', linestyle='--', linewidth=2,
            label=f'Mean: {mean_acc:.4f}')

        ax.set_xlabel('Accuracy', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Accuracy Distribution Across Folds', fontsize=13)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy distribution plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_accuracy_vs_k(k_values: List[int], accuracies: List[float],
    output_path: Optional[str] = None):
    """
    Plot the relationship between the accuracy and the number of neighbors (k).
    
    Parameters:
    -----------
    k_values : List[int]
        List of k values (number of neighbors)
    accuracies : List[float]
        Corresponding average accuracy for each k value
    output_path : str, optional
        Path to save the plot. If None, displays interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot accuracy vs. k with markers
    ax.plot(k_values, accuracies, marker='o', linestyle='-', linewidth=2,
        markersize=8, color='darkblue', markerfacecolor='lightblue',
        markeredgewidth=2, markeredgecolor='darkblue')

    # Highlight the best k value
    best_idx = np.argmax(accuracies)
    best_k = k_values[best_idx]
    best_acc = accuracies[best_idx]

    ax.plot(best_k, best_acc, marker='*', markersize=20, color='red',
        markeredgewidth=2, markeredgecolor='darkred',
        label=f'Best: k={best_k}, Acc={best_acc:.4f}')

    # Set labels and title
    ax.set_xlabel('Number of Neighbors (k)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs. Number of Neighbors (k)', fontsize=14)
    ax.set_xticks(k_values)
    ax.set_ylim(ymin=min(accuracies) * 0.95, ymax=max(accuracies) * 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy vs. k plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_error_rate_vs_k(k_values: List[int], error_rates: List[float],
    output_path: Optional[str] = None):
    """
    Plot the relationship between the error rate and the number of neighbors (k).
    
    Parameters:
    -----------
    k_values : List[int]
        List of k values (number of neighbors)
    error_rates : List[float]
        Corresponding error rates for each k value
    output_path : str, optional
        Path to save the plot. If None, displays interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot error rate vs. k with markers
    ax.plot(k_values, error_rates, marker='s', linestyle='-', linewidth=2,
        markersize=8, color='darkred', markerfacecolor='lightcoral',
        markeredgewidth=2, markeredgecolor='darkred')

    # Highlight the optimal k value with the lowest error
    best_idx = np.argmin(error_rates)
    best_k = k_values[best_idx]
    best_err = error_rates[best_idx]

    ax.plot(best_k, best_err, marker='*', markersize=20, color='green',
        markeredgewidth=2, markeredgecolor='darkgreen',
        label=f'Best: k={best_k}, Error={best_err:.4f}')

    # Set labels and title
    ax.set_xlabel('Number of Neighbors (k)', fontsize=12)
    ax.set_ylabel('Error Rate (1 - Accuracy)', fontsize=12)
    ax.set_title('Error Rate vs. Number of Neighbors (k)', fontsize=14)
    ax.set_xticks(k_values)
    ax.set_ylim(ymin=0, ymax=max(error_rates) * 1.1)
    ax.grid(visible=True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Error rate vs. k plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def save_results_to_file(results: Dict[str, Any], output_path: str):
    """
    Save the result dictionary to a CSV or Excel file.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Dictionary containing results to save
    output_path : str
        Path to save the file (should end with .csv or .xlsx)
    """
    # Flatten the result dictionary
    flattened_results = {}

    for key, value in results.items():
        if isinstance(value, dict):
            # Flatten nested dictionaries
            for sub_key, sub_value in value.items():
                flattened_results[f"{key}_{sub_key}"] = sub_value
        elif isinstance(value, (list, np.ndarray)):
            # Convert lists/arrays to comma-separated strings
            if isinstance(value, np.ndarray):
                value = value.tolist()
            flattened_results[key] = str(value)
        else:
            flattened_results[key] = value

    # Create DataFrame
    df = pd.DataFrame([flattened_results])

    # Save to file
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.xlsx'):
        df.to_excel(output_path, index=False)
    else:
        # Default to CSV
        output_path += '.csv'
        df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")


def save_fold_results(metrics_per_fold: List[Dict[str, float]], output_path: str):
    """
    Save per-fold metrics to a CSV or Excel file.
    
    Parameters:
    -----------
    metrics_per_fold : List[Dict[str, float]]
        List of metric dictionaries, one per fold
    output_path : str
        Path to save the file
    """
    # Flatten confusion matrices
    rows = []
    for i, fold_metrics in enumerate(metrics_per_fold):
        row = {'fold': i + 1.0}
        for key, value in fold_metrics.items():
            if key == 'confusion_matrix' and isinstance(value, dict):
                for cm_key, cm_value in value.items():
                    row[f'cm_{cm_key}'] = cm_value
            else:
                row[key] = value
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to file
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.xlsx'):
        df.to_excel(output_path, index=False)
    else:
        output_path += '.csv'
        df.to_csv(output_path, index=False)

    print(f"Per-fold results saved to {output_path}")
