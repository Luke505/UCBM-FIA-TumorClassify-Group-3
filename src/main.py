"""
Main entry point for the UCBM-FIA TumorClassify application

This module provides the command-line interface for running the tumor
classification pipeline with various configuration options
"""

import argparse
import json
import os
import sys
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from .classifier.knn_classifier import KNNClassifier
from .evaluation.strategies import ValidationStrategyFactory, BaseValidationStrategy
from .metrics.metrics import calculate_all_metrics
from .utils.data_loader import DataLoaderFactory
from .utils.preprocessing import compute_normalization_params, apply_normalization
from .visualization.plots import (
    create_output_directory,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metrics_comparison,
    plot_accuracy_distribution,
    plot_accuracy_vs_k,
    plot_error_rate_vs_k,
    save_results_to_file,
    save_fold_results
)


def parse_arguments():
    """
    Parse command-line arguments

    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    # noinspection GrazieInspection,SpellCheckingInspection
    parser = argparse.ArgumentParser(
        description='UCBM-FIA TumorClassify - k-NN based tumor classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (holdout validation, k=3)
  python -m src.main --data data/version_1.csv
  
  # Run with K-Fold cross-validation
  python -m src.main --data data/version_1.csv --strategy kfold --k 5 --K 10
  
  # Run with Leave-One-Out
  python -m src.main --data data/version_1.csv --strategy loocv --k 3
  
  # Run with Bootstrap and save results
  python -m src.main --data data/version_1.csv --strategy bootstrap --k 7 --K 50 --output-dir results
        """
    )

    # Required arguments
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to the input data file (CSV, TXT, JSON, XLSX, TSV)',
        default=None
    )

    # noinspection GrazieInspection
    parser.add_argument(
        '--aliases',
        type=str,
        default=None,
        help='Column name aliases as JSON string or key=value pairs (e.g., "id:patient_id,class:diagnosis")'
    )

    # k-NN parameters
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='Number of neighbors for k-NN classifier (default: 3)'
    )

    parser.add_argument(
        '--k-values',
        type=str,
        default=None,
        help='Comma-separated list of k values to compare (e.g., "1,3,5,7,9"). '
             + 'When specified, runs experiments for each k and generates comparison plots. '
             + 'Overrides --k parameter.'
    )

    # Validation strategy
    # noinspection SpellCheckingInspection
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        default='holdout',
        choices=[
            'holdout', 'random_subsampling', 'kfold', 'k-fold',
            'loocv', 'leave-one-out', 'lpocv', 'leave-p-out',
            'stratified', 'stratified_kfold', 'stratified_shuffle', 'bootstrap'
        ],
        help='Validation strategy (default: holdout)'
    )

    # Strategy-specific parameters
    parser.add_argument(
        '--K',
        type=int,
        default=5,
        help='Number of folds/splits/iterations for validation strategy (default: 5)'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.3,
        help='Test set size for holdout/random subsampling (default: 0.3)'
    )

    parser.add_argument(
        '--p',
        type=int,
        default=2,
        help='Number of samples to leave out for the LeavePOut strategy (default: 2)'
    )

    # Metrics
    parser.add_argument(
        '--metrics', '-m',
        type=str,
        nargs='+',
        default=['all'],
        choices=['accuracy', 'error_rate', 'sensitivity', 'specificity', 'geometric_mean', 'auc', 'all'],
        help='Metrics to compute (default: all)'
    )

    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )

    parser.add_argument(
        '--no-subfolder',
        action='store_true',
        help='Do not create a timestamped subfolder for outputs (saves directly to output-dir)'
    )

    # Random seed
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def print_banner():
    """Print application banner"""
    print("=" * 70)
    print("  UCBM-FIA TumorClassify - Group 3")
    print("  k-Nearest Neighbors Tumor Classification")
    print("=" * 70)
    print()


def load_and_preprocess_data(filepath: str, verbose: bool = False, aliases: Dict[str, str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess data from the file

    Invalid samples (duplicate IDs, invalid classes, missing/invalid features)
    are automatically filtered during loading. Feature normalization is NOT
    applied here — it is performed per-fold inside run_evaluation() to avoid
    data leakage (min/max are computed on training data only).

    Parameters:
    -----------
    filepath : str
        Path to the data file
    verbose : bool
        Whether to print verbose output (default: False)
    aliases : Dict[str, str], optional
        Mapping of custom column names to standard feature names

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Features and labels as numpy arrays (not yet normalized)
    """
    if verbose:
        print(f"Loading data from: {filepath}")

    # Load data as TumorDataset (invalid samples automatically filtered)
    dataset = DataLoaderFactory.load_data(filepath, aliases)

    if verbose:
        dist = dataset.get_class_distribution()
        print(f"  Loaded {len(dataset)} valid samples with 9 features")
        print(f"  Class distribution: Benign (2): {dist[2]}, Malignant (4): {dist[4]}")

    # Convert to arrays (normalization is deferred to per-fold evaluation)
    features, labels = dataset.to_arrays()

    return features, labels


def create_validation_strategy(args):
    """
    Create a validation strategy based on arguments

    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments

    Returns:
    --------
    BaseValidationStrategy
        The configured validation strategy
    """
    strategy_params = {'random_state': args.random_state}

    # noinspection SpellCheckingInspection
    if args.strategy in ['holdout']:
        strategy_params['test_size'] = args.test_size
    elif args.strategy in ['random_subsampling', 'kfold', 'k-fold', 'stratified', 'stratified_kfold', 'bootstrap']:
        strategy_params['n_splits'] = args.K
        if args.strategy in ['random_subsampling']:
            strategy_params['test_size'] = args.test_size
    elif args.strategy in ['stratified_shuffle']:
        strategy_params['n_splits'] = args.K
        strategy_params['test_size'] = args.test_size
    elif args.strategy in ['lpocv', 'leave-p-out']:
        strategy_params['p'] = args.p
        strategy_params['max_splits'] = args.K
    elif args.strategy in ['loocv', 'leave-one-out']:
        # LOOCV does not accept any parameters (no random_state, no n_splits)
        strategy_params = {}

    return ValidationStrategyFactory.create_strategy(args.strategy, **strategy_params)


def run_evaluation(features: np.ndarray[tuple[int, int], np.dtype[np.float64]], labels: np.ndarray[tuple[int], np.dtype[np.float64]], classifier: KNNClassifier,
    validation_strategy: BaseValidationStrategy, verbose=False) -> Tuple[
    List[Dict[str, float]],
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int, int], np.dtype[np.float64]]
]:
    """
    Run evaluation using the specified validation strategy

    Feature normalization is performed per-fold: min/max are computed on
    the training set only and applied to both training and test sets.
    This avoids data leakage.

    Parameters:
    -----------
    features : np.ndarray[tuple[int, int], np.dtype[np.float64]]
        Feature matrix (raw, not yet normalized)
    labels : np.ndarray[tuple[int], np.dtype[np.float64]]
        Label vector
    classifier : KNNClassifier
        The k-NN classifier
    validation_strategy : BaseValidationStrategy
        The validation strategy
    verbose : bool
        Whether to print progress

    Returns:
    --------
    Tuple[List[Dict[str, float]], np.ndarray[tuple[int, int], np.dtype[np.float64]], np.ndarray[tuple[int, int], np.dtype[np.float64]], np.ndarray[tuple[int, int], np.dtype[np.float64]]]
        Metrics per fold, all true labels, all predictions, all probabilities
    """
    n_splits = validation_strategy.get_n_splits(features, labels)

    if verbose:
        print(f"Running {n_splits} fold(s)...")

    metrics_per_fold = []
    all_labels_true = []
    all_labels_pred = []
    all_labels_proba = []

    for fold_idx, (train_idx, test_idx) in enumerate(validation_strategy.split(features, labels), 1):
        if verbose:
            print(f"  Fold {fold_idx}/{n_splits}...", end=' ')

        feature_train: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        feature_test: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        label_train: np.ndarray[tuple[int], np.dtype[np.float64]]
        label_test: np.ndarray[tuple[int], np.dtype[np.float64]]

        # Split data
        feature_train, feature_test = features[train_idx], features[test_idx]
        label_train, label_test = labels[train_idx], labels[test_idx]

        # Normalize features per-fold: fit on training data only, apply to both
        # This prevents data leakage (test set min/max do not influence scaling)
        min_vals, max_vals = compute_normalization_params(feature_train)
        feature_train = apply_normalization(feature_train, min_vals, max_vals)
        feature_test = apply_normalization(feature_test, min_vals, max_vals)

        # Train classifier
        classifier.fit(feature_train, label_train)

        # Make predictions
        labels_pred = classifier.predict(feature_test)
        labels_proba = classifier.predict_proba(feature_test)

        # Calculate metrics
        fold_metrics = calculate_all_metrics(label_test, labels_pred, labels_proba)
        metrics_per_fold.append(fold_metrics)

        # Store predictions
        all_labels_true.extend(label_test)
        all_labels_pred.extend(labels_pred)
        all_labels_proba.extend(labels_proba)

        if verbose:
            print(f"Accuracy: {fold_metrics['accuracy']:.4f}")

    return metrics_per_fold, np.array(all_labels_true), np.array(all_labels_pred), np.array(all_labels_proba)


def _filter_metric_names(all_metric_names: List[str], selected_metrics: List[str]) -> List[str]:
    """
    Filter metric names based on user selection

    Parameters:
    -----------
    all_metric_names : List[str]
        All available metric names (excluding 'confusion_matrix')
    selected_metrics : List[str]
        User-selected metrics (e.g., ['accuracy', 'auc'] or ['all'])

    Returns:
    --------
    List[str]
        Filtered list of metric names to display
    """
    if 'all' in selected_metrics:
        return all_metric_names
    return [m for m in all_metric_names if m in selected_metrics]


def print_results(metrics_per_fold, verbose=False, selected_metrics=None):
    """
    Print evaluation results

    Parameters:
    -----------
    metrics_per_fold : List[Dict]
        Metrics for each fold
    verbose : bool
        Whether to print detailed results
    selected_metrics : List[str], optional
        Which metrics to display (default: all). Accepts metric names like
        'accuracy', 'error_rate', 'sensitivity', 'specificity',
        'geometric_mean', 'auc', or 'all' to show everything.
    """
    if selected_metrics is None:
        selected_metrics = ['all']

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # Calculate average metrics
    all_metric_names = [k for k in metrics_per_fold[0].keys() if k not in ['confusion_matrix']]
    display_metrics = _filter_metric_names(all_metric_names, selected_metrics)

    print("\nAverage Metrics Across All Folds:")
    print("-" * 70)

    for metric_name in display_metrics:
        values = [fold[metric_name] for fold in metrics_per_fold]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric_name.replace('_', ' ').title():20s}: {mean_val:.4f} ± {std_val:.4f}")

    # Print confusion matrix (sum across all folds)
    print("\nOverall Confusion Matrix:")
    print("-" * 70)
    total_cm = {
        'TN': sum(fold['confusion_matrix']['TN'] for fold in metrics_per_fold),
        'FP': sum(fold['confusion_matrix']['FP'] for fold in metrics_per_fold),
        'FN': sum(fold['confusion_matrix']['FN'] for fold in metrics_per_fold),
        'TP': sum(fold['confusion_matrix']['TP'] for fold in metrics_per_fold)
    }

    print(f"  True Negatives (TN):  {total_cm['TN']}")
    print(f"  False Positives (FP): {total_cm['FP']}")
    print(f"  False Negatives (FN): {total_cm['FN']}")
    print(f"  True Positives (TP):  {total_cm['TP']}")

    if verbose and len(metrics_per_fold) > 1:
        print("\nPer-Fold Results:")
        print("-" * 70)
        for i, fold_metrics in enumerate(metrics_per_fold, 1):
            print(f"\n  Fold {i}:")
            for metric_name in display_metrics:
                print(f"    {metric_name.replace('_', ' ').title():18s}: {fold_metrics[metric_name]:.4f}")


def save_outputs(args, metrics_per_fold, all_labels_true, all_labels_pred, all_labels_proba,
                 selected_metrics=None):
    """
    Save results and plots

    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    metrics_per_fold : List[Dict]
        Metrics for each fold
    all_labels_true : np.ndarray[tuple[int, int], np.dtype[np.float64]]
        All true labels
    all_labels_pred : np.ndarray[tuple[int, int], np.dtype[np.float64]]
        All predictions
    all_labels_proba : np.ndarray[tuple[int, int], np.dtype[np.float64]]
        All predicted probabilities
    selected_metrics : List[str], optional
        Which metrics to include in output files (default: all)
    """
    if selected_metrics is None:
        selected_metrics = ['all']
    # Create output directory with optional subfolder

    # By default, creates a subfolder with the data name and timestamp
    # Unless --no-subfolder flag is provided
    use_subfolder = not getattr(args, 'no_subfolder', False)
    output_dir = create_output_directory(
        base_dir=args.output_dir,
        data_filename=args.data if hasattr(args, 'data') else None,
        use_subfolder=use_subfolder
    )

    # Base filename for this run
    base_filename = f"{args.strategy}_k{args.k}"

    # Save metrics to CSV
    metrics_file = os.path.join(output_dir, f"{base_filename}_metrics.csv")
    save_fold_results(metrics_per_fold, metrics_file)

    # Save summary results
    summary = {
        'strategy': args.strategy,
        'k': args.k,
        'K': args.K if hasattr(args, 'K') else 1,
        'n_samples': len(all_labels_true),
        'data_file': args.data if hasattr(args, 'data') else 'unknown'
    }

    # Add average metrics (filtered by user selection)
    all_metric_names = [k for k in metrics_per_fold[0].keys() if k not in ['confusion_matrix']]
    display_metrics = _filter_metric_names(all_metric_names, selected_metrics)
    for metric_name in display_metrics:
        values = [fold[metric_name] for fold in metrics_per_fold]
        summary[f'avg_{metric_name}'] = np.mean(values)
        summary[f'std_{metric_name}'] = np.std(values)

    # Add total confusion matrix
    total_cm = {
        'TN': sum(fold['confusion_matrix']['TN'] for fold in metrics_per_fold),
        'FP': sum(fold['confusion_matrix']['FP'] for fold in metrics_per_fold),
        'FN': sum(fold['confusion_matrix']['FN'] for fold in metrics_per_fold),
        'TP': sum(fold['confusion_matrix']['TP'] for fold in metrics_per_fold)
    }
    summary['confusion_matrix'] = total_cm

    summary_file = os.path.join(output_dir, f"{base_filename}_summary.csv")
    save_results_to_file(summary, summary_file)

    # Generate plots if not disabled
    if not args.no_plots:
        print("\nGenerating plots...")

        # 1. Confusion matrix
        cm_file = os.path.join(output_dir, f"{base_filename}_confusion_matrix.png")
        plot_confusion_matrix(total_cm, cm_file)

        # 2. ROC curve
        if len(all_labels_proba) > 0:
            auc_values = [fold['auc'] for fold in metrics_per_fold if 'auc' in fold]
            if auc_values:
                avg_auc = np.mean(auc_values).astype(float)
                roc_file = os.path.join(output_dir, f"{base_filename}_roc_curve.png")
                plot_roc_curve(all_labels_true, all_labels_proba, avg_auc, roc_file)

        # 3. Metrics comparison (bar chart with 5 metrics)
        metrics_comp_file = os.path.join(output_dir, f"{base_filename}_metrics_comparison.png")
        plot_metrics_comparison(summary, metrics_comp_file)

        # 4. Accuracy distribution (only if multiple folds)
        if len(metrics_per_fold) > 1:
            accuracies = [fold['accuracy'] for fold in metrics_per_fold]
            acc_dist_file = os.path.join(output_dir, f"{base_filename}_accuracy_distribution.png")
            plot_accuracy_distribution(accuracies, acc_dist_file, plot_type='boxplot')

    print(f"\nAll results saved to: {output_dir}/")


def run_k_comparison(args: argparse.Namespace, features: np.ndarray[tuple[int, int], np.dtype[np.float64]], labels: np.ndarray[tuple[int], np.dtype[np.float64]]):
    """
    Run k-NN experiments for multiple k values and generate comparison plots
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    features : np.ndarray[tuple[int, int], np.dtype[np.float64]]
        Feature matrix
    labels : np.ndarray[tuple[int], np.dtype[np.float64]]
        Label vector
    """
    # Parse k values
    k_values = [int(k.strip()) for k in args.k_values.split(',')]
    k_values = sorted(k_values)

    print("\n" + "=" * 70)
    print("  k-NN Performance Comparison Across Different k Values")
    print("=" * 70)
    print(f"\nComparing k values: {k_values}")
    print(f"Validation strategy: {args.strategy}")
    print()

    # Create the base output directory
    use_subfolder = not args.no_subfolder
    base_output_dir = create_output_directory(
        base_dir=args.output_dir,
        data_filename=args.data,
        use_subfolder=use_subfolder
    )

    # Store results for each k
    all_k_results = []

    for k_val in k_values:
        print(f"\n{'=' * 70}")
        print(f"  Running experiments with k={k_val}")
        print(f"{'=' * 70}")

        # Create a subfolder for this k value
        k_output_dir = os.path.join(base_output_dir, f"k{k_val}")
        os.makedirs(k_output_dir, exist_ok=True)

        # Create a classifier with current k
        classifier = KNNClassifier(k=k_val)

        # Create validation strategy
        validation_strategy = create_validation_strategy(args)

        # Run evaluation
        metrics_per_fold, all_labels_true, all_labels_pred, all_labels_proba = run_evaluation(
            features, labels, classifier, validation_strategy, args.verbose
        )

        # Print results for this k (filtered by --metrics selection)
        print_results(metrics_per_fold, verbose=False, selected_metrics=args.metrics)

        # Save outputs to k-specific subfolder
        base_filename = f"{args.strategy}_k{k_val}"

        # Save metrics to CSV
        metrics_file = os.path.join(k_output_dir, f"{base_filename}_metrics.csv")
        save_fold_results(metrics_per_fold, metrics_file)

        # Calculate summary
        summary = {
            'k': k_val,
            'strategy': args.strategy,
            'n_samples': len(all_labels_true)
        }

        # Always compute all metrics internally (needed for comparison plots)
        all_metric_names = [k for k in metrics_per_fold[0].keys() if k not in ['confusion_matrix']]
        for metric_name in all_metric_names:
            values = [fold[metric_name] for fold in metrics_per_fold]
            summary[f'avg_{metric_name}'] = np.mean(values)
            summary[f'std_{metric_name}'] = np.std(values)

        # Add confusion matrix
        total_cm = {
            'TN': sum(fold['confusion_matrix']['TN'] for fold in metrics_per_fold),
            'FP': sum(fold['confusion_matrix']['FP'] for fold in metrics_per_fold),
            'FN': sum(fold['confusion_matrix']['FN'] for fold in metrics_per_fold),
            'TP': sum(fold['confusion_matrix']['TP'] for fold in metrics_per_fold)
        }
        summary['confusion_matrix'] = total_cm

        summary_file = os.path.join(k_output_dir, f"{base_filename}_summary.csv")
        save_results_to_file(summary, summary_file)

        # Generate plots if not disabled
        if not args.no_plots:
            # Confusion matrix
            cm_file = os.path.join(k_output_dir, f"{base_filename}_confusion_matrix.png")
            plot_confusion_matrix(total_cm, cm_file)

            # ROC curve
            if len(all_labels_proba) > 0:
                auc_values = [fold['auc'] for fold in metrics_per_fold if 'auc' in fold]
                if auc_values:
                    avg_auc = np.mean(auc_values).astype(float)
                    roc_file = os.path.join(k_output_dir, f"{base_filename}_roc_curve.png")
                    plot_roc_curve(all_labels_true, all_labels_proba, avg_auc, roc_file)

            # Metrics comparison
            metrics_comp_file = os.path.join(k_output_dir, f"{base_filename}_metrics_comparison.png")
            plot_metrics_comparison(summary, metrics_comp_file)

            # Accuracy distribution (only if multiple folds)
            if len(metrics_per_fold) > 1:
                accuracies = [fold['accuracy'] for fold in metrics_per_fold]
                acc_dist_file = os.path.join(k_output_dir, f"{base_filename}_accuracy_distribution.png")
                plot_accuracy_distribution(accuracies, acc_dist_file, plot_type='boxplot')

        print(f"\nResults for k={k_val} saved to: {k_output_dir}/")

        # Store summary for comparison
        all_k_results.append(summary)

    # Generate comparison plots in the base directory
    print("\n" + "=" * 70)
    print("  Generating k-Values Comparison Plots")
    print("=" * 70)

    accuracies = [r['avg_accuracy'] for r in all_k_results]
    error_rates = [r['avg_error_rate'] for r in all_k_results]

    if not args.no_plots:
        # Accuracy vs. k plot
        acc_plot_path = os.path.join(base_output_dir, 'accuracy_vs_k.png')
        plot_accuracy_vs_k(k_values, accuracies, acc_plot_path)

        # Error rate vs. k plot
        err_plot_path = os.path.join(base_output_dir, 'error_rate_vs_k.png')
        plot_error_rate_vs_k(k_values, error_rates, err_plot_path)

    # Save comparison results to CSV
    comparison_df = pd.DataFrame({
        'k': k_values,
        'accuracy': accuracies,
        'error_rate': error_rates,
        'sensitivity': [r['avg_sensitivity'] for r in all_k_results],
        'specificity': [r['avg_specificity'] for r in all_k_results],
        'geometric_mean': [r['avg_geometric_mean'] for r in all_k_results],
        'auc': [r['avg_auc'] for r in all_k_results]
    })

    csv_path = os.path.join(base_output_dir, 'k_comparison_results.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nComparison results saved to {csv_path}")

    # Print comparison summary
    print("\n" + "=" * 70)
    print("  K-VALUES COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'k':>4} | {'Accuracy':>10} | {'Error Rate':>10} | {'Sensitivity':>11} | {'Specificity':>11}")
    print("-" * 70)
    for k, res in zip(k_values, all_k_results):
        print(f"{k:>4} | {res['avg_accuracy']:>10.4f} | {res['avg_error_rate']:>10.4f} | "
              f"{res['avg_sensitivity']:>11.4f} | {res['avg_specificity']:>11.4f}")

    # Find best k
    best_idx = np.argmax(accuracies)
    best_k = k_values[best_idx]
    best_accuracy = accuracies[best_idx]

    print("\n" + "-" * 70)
    print(f"Best k value: {best_k} (Accuracy: {best_accuracy:.4f})")
    print(f"\nAll results saved to: {base_output_dir}/")


def find_default_data_file():
    """
    Find a default data file in common locations

    Returns:
    --------
    str or None
        Path to a data file if found, None otherwise
    """
    # Check for data files in common locations
    path = 'data'
    extensions = ['.csv', '.txt', '.json', '.xlsx', '.tsv']

    if os.path.exists(path) and os.path.isdir(path):
        for file in os.listdir(path):
            if any(file.endswith(ext) for ext in extensions):
                return os.path.join(path, file)

    return None


def parse_aliases(aliases_str: str) -> Dict[str, str] | None:
    """
    Parse aliases string into dictionary
    
    Supports formats:
    - "key1:value1,key2:value2"
    - JSON string: '{"key1":"value1","key2":"value2"}'
    
    Parameters:
    -----------
    aliases_str : str
        Aliases string
        
    Returns:
    --------
    Dict[str, str]
        Parsed aliases dictionary
    """
    if not aliases_str:
        return None

    # Try to parse as JSON first
    try:
        return json.loads(aliases_str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Parse as 'key:value,key:value' format
    aliases = {}
    for pair in aliases_str.split(','):
        if ':' in pair:
            key, value = pair.split(':', 1)
            aliases[key.strip()] = value.strip()

    return aliases if aliases else None


def main():
    """Main application entry point"""
    print_banner()

    # Parse arguments
    args = parse_arguments()

    # Parse aliases if provided
    aliases = parse_aliases(args.aliases) if hasattr(args, 'aliases') and args.aliases else None

    # Find data file if not specified
    if args.data is None:
        args.data = find_default_data_file()
        if args.data is None:
            print("ERROR: No data file specified and no default file found.")
            print("Please specify a data file using the --data argument.")
            print("\nExample: python -m src.main --data tests_data/version_1.csv")
            sys.exit(1)
        else:
            print(f"Using default data file: {args.data}\n")

    # Verify data file exists
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    # Print configuration
    print("Configuration:")
    print(f"  Data file: {args.data}")
    if args.k_values:
        print(f"  Mode: k-values comparison")
        print(f"  k values: {args.k_values}")
    else:
        print(f"  k (neighbors): {args.k}")
    print(f"  Validation strategy: {args.strategy}")
    # noinspection SpellCheckingInspection
    if args.strategy not in ['loocv', 'leave-one-out']:
        print(f"  K (folds/splits): {args.K}")
    if args.strategy in ['holdout', 'random_subsampling', 'stratified_shuffle']:
        print(f"  Test size: {args.test_size}")
    if 'all' in args.metrics:
        print(f"  Metrics: all")
    else:
        print(f"  Metrics: {', '.join(args.metrics)}")
    print(f"  Random seed: {args.random_state}")
    print()

    try:
        # Load and preprocess data
        features, labels = load_and_preprocess_data(args.data, args.verbose, aliases)

        # Check if k-values comparison mode is enabled
        if args.k_values:
            # Run k-values comparison mode
            run_k_comparison(args, features, labels)
        else:
            # Standard single-k evaluation mode

            # Create classifier
            if args.verbose:
                print(f"Creating k-NN classifier with k={args.k}...")
            classifier = KNNClassifier(k=args.k)

            # Create validation strategy
            if args.verbose:
                print(f"Creating validation strategy: {args.strategy}...")
            validation_strategy = create_validation_strategy(args)

            # Run evaluation
            print()
            metrics_per_fold, all_labels_true, all_labels_pred, all_labels_proba = run_evaluation(
                features, labels, classifier, validation_strategy, args.verbose
            )

            # Print results (filtered by --metrics selection)
            print_results(metrics_per_fold, args.verbose, selected_metrics=args.metrics)

            # Save outputs (filtered by --metrics selection)
            save_outputs(args, metrics_per_fold, all_labels_true, all_labels_pred, all_labels_proba,
                         selected_metrics=args.metrics)

        print("\n" + "=" * 70)
        print("  Evaluation completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
