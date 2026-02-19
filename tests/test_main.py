"""
Unit tests for the main.py module
"""

import os
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main import (
    parse_arguments,
    load_and_preprocess_data,
    create_validation_strategy,
    run_evaluation,
    print_results,
    save_outputs,
    find_default_data_file,
    run_k_comparison,
    parse_aliases,
    _filter_metric_names
)
from src.classifier.knn_classifier import KNNClassifier
from src.evaluation.strategies import (
    HoldoutValidation, KFoldCrossValidation, LeaveOneOutCrossValidation,
    RandomSubsampling, LeavePOutCrossValidation
)


class TestParseArguments(unittest.TestCase):
    """Test command-line argument parsing"""

    def test_parse_arguments_defaults(self):
        """Test parsing with default arguments"""
        with patch('sys.argv', ['main.py', '--data', 'test.csv']):
            args = parse_arguments()
            self.assertEqual(args.data, 'test.csv')
            self.assertEqual(args.k, 3)
            self.assertEqual(args.strategy, 'holdout')
            self.assertEqual(args.K, 5)
            self.assertFalse(args.no_plots)
            self.assertFalse(args.verbose)
            self.assertIsNone(args.aliases)

    def test_parse_arguments_custom_k(self):
        """Test parsing with custom k value"""
        with patch('sys.argv', ['main.py', '--data', 'test.csv', '--k', '7']):
            args = parse_arguments()
            self.assertEqual(args.k, 7)

    def test_parse_arguments_k_values(self):
        """Test parsing with k-values parameter"""
        with patch('sys.argv', ['main.py', '--data', 'test.csv', '--k-values', '1,3,5,7']):
            args = parse_arguments()
            self.assertEqual(args.k_values, '1,3,5,7')

    def test_parse_arguments_aliases(self):
        """Test parsing with aliases parameter"""
        with patch('sys.argv', ['main.py', '--data', 'test.csv', '--aliases', 'id:patient_id']):
            args = parse_arguments()
            self.assertEqual(args.aliases, 'id:patient_id')

    def test_parse_arguments_strategy(self):
        """Test parsing with different validation strategies"""
        # noinspection SpellCheckingInspection
        strategies = ['holdout', 'kfold', 'loocv', 'bootstrap', 'stratified']
        for strategy in strategies:
            with patch('sys.argv', ['main.py', '--data', 'test.csv', '--strategy', strategy]):
                args = parse_arguments()
                self.assertEqual(args.strategy, strategy)

    def test_parse_arguments_flags(self):
        """Test parsing with boolean flags"""
        with patch('sys.argv', ['main.py', '--data', 'test.csv', '--no-plots', '--verbose', '--no-subfolder']):
            args = parse_arguments()
            self.assertTrue(args.no_plots)
            self.assertTrue(args.verbose)
            self.assertTrue(args.no_subfolder)


class TestParseAliases(unittest.TestCase):
    """Test alias parsing functionality"""

    def test_parse_aliases_none(self):
        """Test with None input"""
        self.assertIsNone(parse_aliases(None))

    def test_parse_aliases_empty(self):
        """Test with empty string"""
        self.assertIsNone(parse_aliases(''))

    def test_parse_aliases_colon_format(self):
        """Test colon-separated format"""
        result = parse_aliases('id:patient_id,class:diagnosis')
        self.assertEqual(result, {'id': 'patient_id', 'class': 'diagnosis'})

    def test_parse_aliases_json_format(self):
        """Test JSON format"""
        result = parse_aliases('{"id": "patient_id", "class": "diagnosis"}')
        self.assertEqual(result, {'id': 'patient_id', 'class': 'diagnosis'})


class TestLoadAndPreprocessData(unittest.TestCase):
    """Test data loading and preprocessing"""

    def setUp(self):
        """Create a temporary test data file"""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test_data.csv')

        # Create a simple test CSV with proper column names
        data = ("sample_code_number,clump_thickness,uniformity_cell_size,"
                "uniformity_cell_shape,marginal_adhesion,"
                "single_epithelial_cell_size,bare_nuclei,"
                "bland_chromatin,normal_nucleoli,mitoses,class\n")
        for i in range(20):
            class_label = 2 if i < 10 else 4
            v = i % 10 + 1
            data += f"{i},{v},{v},{v},{v},{v},{v},{v},{v},{v},{class_label}\n"

        with open(self.test_file, 'w') as f:
            f.write(data)

    def tearDown(self):
        """Remove the temporary directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_and_preprocess_basic(self):
        """Test basic data loading and preprocessing"""
        features, labels = load_and_preprocess_data(self.test_file, verbose=False)

        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(features), len(labels))
        self.assertGreater(len(features), 0)
        self.assertEqual(features.shape[1], 9)

    def test_load_and_preprocess_verbose(self):
        """Test loading with verbose output"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            features, labels = load_and_preprocess_data(self.test_file, verbose=True)
            output = fake_out.getvalue()
            self.assertIn('Loading data', output)

    def test_load_and_preprocess_with_aliases(self):
        """Test loading with aliases parameter"""
        features, labels = load_and_preprocess_data(self.test_file, verbose=False, aliases=None)
        self.assertGreater(len(features), 0)

    def test_labels_are_valid(self):
        """Test that loaded labels are only 2 or 4"""
        features, labels = load_and_preprocess_data(self.test_file, verbose=False)
        self.assertTrue(np.all((labels == 2) | (labels == 4)))


class TestCreateValidationStrategy(unittest.TestCase):
    """Test validation strategy creation"""

    def test_create_holdout_strategy(self):
        """Test creating holdout validation strategy"""
        args = MagicMock()
        args.strategy = 'holdout'
        args.test_size = 0.3
        args.random_state = 42
        args.K = 5
        args.p = 2

        strategy = create_validation_strategy(args)
        self.assertIsInstance(strategy, HoldoutValidation)

    # noinspection SpellCheckingInspection
    def test_create_kfold_strategy(self):
        """Test creating k-fold validation strategy"""
        args = MagicMock()
        args.strategy = 'kfold'
        args.test_size = 0.3
        args.random_state = 42
        args.K = 5
        args.p = 2

        strategy = create_validation_strategy(args)
        self.assertIsInstance(strategy, KFoldCrossValidation)

    def test_create_loocv_strategy(self):
        """Test creating LOOCV strategy (should not crash with random_state)"""
        args = MagicMock()
        args.strategy = 'loocv'
        args.test_size = 0.3
        args.random_state = 42
        args.K = 5
        args.p = 2

        strategy = create_validation_strategy(args)
        self.assertIsInstance(strategy, LeaveOneOutCrossValidation)

    def test_create_leave_one_out_alias_strategy(self):
        """Test creating LOOCV strategy using the 'leave-one-out' alias"""
        args = MagicMock()
        args.strategy = 'leave-one-out'
        args.test_size = 0.3
        args.random_state = 42
        args.K = 5
        args.p = 2

        strategy = create_validation_strategy(args)
        self.assertIsInstance(strategy, LeaveOneOutCrossValidation)

    def test_create_random_subsampling_strategy(self):
        """Test creating Random Subsampling (B) strategy"""
        args = MagicMock()
        args.strategy = 'random_subsampling'
        args.test_size = 0.3
        args.random_state = 42
        args.K = 10
        args.p = 2

        strategy = create_validation_strategy(args)
        self.assertIsInstance(strategy, RandomSubsampling)
        self.assertEqual(strategy.n_splits, 10)

    def test_create_lpocv_strategy(self):
        """Test creating Leave-p-Out (C) strategy"""
        args = MagicMock()
        args.strategy = 'lpocv'
        args.test_size = 0.3
        args.random_state = 42
        args.K = 50
        args.p = 3

        strategy = create_validation_strategy(args)
        self.assertIsInstance(strategy, LeavePOutCrossValidation)
        self.assertEqual(strategy.p, 3)

    # noinspection SpellCheckingInspection
    def test_create_strategy_with_different_K(self):
        """Test creating strategy with different K values"""
        for K in [3, 5, 10]:
            args = MagicMock()
            args.strategy = 'kfold'
            args.K = K
            args.random_state = 42
            args.test_size = 0.3
            args.p = 2

            strategy = create_validation_strategy(args)
            self.assertEqual(strategy.n_splits, K)


class TestRunEvaluation(unittest.TestCase):
    """Test evaluation execution"""

    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.features = np.random.rand(50, 9)
        self.labels = np.array([2] * 25 + [4] * 25)
        self.classifier = KNNClassifier(k=3)

    def test_run_evaluation_holdout(self):
        """Test running evaluation with holdout"""
        strategy = HoldoutValidation(test_size=0.3, random_state=42)

        metrics_per_fold, labels_true, labels_pred, labels_proba = run_evaluation(
            self.features, self.labels, self.classifier, strategy, verbose=False
        )

        self.assertEqual(len(metrics_per_fold), 1)
        self.assertIn('accuracy', metrics_per_fold[0])
        self.assertIn('confusion_matrix', metrics_per_fold[0])
        self.assertEqual(len(labels_true), len(labels_pred))

    # noinspection SpellCheckingInspection
    def test_run_evaluation_kfold(self):
        """Test running evaluation with k-fold"""
        strategy = KFoldCrossValidation(n_splits=3, random_state=42)

        metrics_per_fold, labels_true, labels_pred, labels_proba = run_evaluation(
            self.features, self.labels, self.classifier, strategy, verbose=False
        )

        self.assertEqual(len(metrics_per_fold), 3)
        for metrics in metrics_per_fold:
            self.assertIn('accuracy', metrics)
            self.assertGreater(metrics['accuracy'], 0)

    def test_run_evaluation_verbose(self):
        """Test running evaluation with verbose output"""
        strategy = HoldoutValidation(test_size=0.3, random_state=42)

        with patch('sys.stdout', new=StringIO()) as fake_out:
            run_evaluation(self.features, self.labels, self.classifier, strategy, verbose=True)
            output = fake_out.getvalue()
            self.assertIn('Running', output)


class TestPrintResults(unittest.TestCase):
    """Test results printing"""

    def test_print_results_basic(self):
        """Test basic results printing"""
        metrics_per_fold = [
            {
                'accuracy': 0.95,
                'error_rate': 0.05,
                'sensitivity': 0.93,
                'specificity': 0.96,
                'geometric_mean': 0.945,
                'auc': 0.98,
                'confusion_matrix': {'TN': 50, 'FP': 5, 'FN': 3, 'TP': 42}
            }
        ]

        with patch('sys.stdout', new=StringIO()) as fake_out:
            print_results(metrics_per_fold, verbose=False)
            output = fake_out.getvalue()
            self.assertIn('RESULTS', output)
            self.assertIn('Accuracy', output)
            self.assertIn('Confusion Matrix', output)

    def test_print_results_multiple_folds(self):
        """Test printing results for multiple folds"""
        metrics_per_fold = [
            {'accuracy': 0.95, 'error_rate': 0.05, 'confusion_matrix': {'TN': 50, 'FP': 5, 'FN': 3, 'TP': 42}},
            {'accuracy': 0.96, 'error_rate': 0.04, 'confusion_matrix': {'TN': 52, 'FP': 3, 'FN': 4, 'TP': 41}}
        ]

        with patch('sys.stdout', new=StringIO()) as fake_out:
            print_results(metrics_per_fold, verbose=True)
            output = fake_out.getvalue()
            self.assertIn('Per-Fold Results', output)


class TestMetricsFiltering(unittest.TestCase):
    """Test metrics selection and filtering"""

    def setUp(self):
        """Set up common test data"""
        self.metrics_per_fold = [
            {
                'accuracy': 0.95,
                'error_rate': 0.05,
                'sensitivity': 0.93,
                'specificity': 0.96,
                'geometric_mean': 0.945,
                'auc': 0.98,
                'confusion_matrix': {'TN': 50, 'FP': 5, 'FN': 3, 'TP': 42}
            }
        ]

    def test_filter_metric_names_all(self):
        """Test that 'all' returns all metric names"""
        all_names = ['accuracy', 'error_rate', 'sensitivity', 'specificity', 'geometric_mean', 'auc']
        result = _filter_metric_names(all_names, ['all'])
        self.assertEqual(result, all_names)

    def test_filter_metric_names_subset(self):
        """Test filtering to a subset of metrics"""
        all_names = ['accuracy', 'error_rate', 'sensitivity', 'specificity', 'geometric_mean', 'auc']
        result = _filter_metric_names(all_names, ['accuracy', 'auc'])
        self.assertEqual(result, ['accuracy', 'auc'])

    def test_filter_metric_names_single(self):
        """Test filtering to a single metric"""
        all_names = ['accuracy', 'error_rate', 'sensitivity']
        result = _filter_metric_names(all_names, ['sensitivity'])
        self.assertEqual(result, ['sensitivity'])

    def test_print_results_with_selected_metrics(self):
        """Test that print_results only displays selected metrics"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print_results(self.metrics_per_fold, verbose=False, selected_metrics=['accuracy', 'auc'])
            output = fake_out.getvalue()
            self.assertIn('Accuracy', output)
            self.assertIn('Auc', output)
            # These should NOT be present in the metrics section
            self.assertNotIn('Error Rate', output)
            self.assertNotIn('Sensitivity', output)

    def test_print_results_all_metrics(self):
        """Test that selected_metrics=['all'] shows everything"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print_results(self.metrics_per_fold, verbose=False, selected_metrics=['all'])
            output = fake_out.getvalue()
            self.assertIn('Accuracy', output)
            self.assertIn('Sensitivity', output)
            self.assertIn('Specificity', output)

    def test_parse_arguments_metrics_default(self):
        """Test that --metrics defaults to ['all']"""
        with patch('sys.argv', ['main.py', '--data', 'test.csv']):
            args = parse_arguments()
            self.assertEqual(args.metrics, ['all'])

    def test_parse_arguments_metrics_custom(self):
        """Test parsing custom metrics selection"""
        with patch('sys.argv', ['main.py', '--data', 'test.csv', '--metrics', 'accuracy', 'sensitivity', 'auc']):
            args = parse_arguments()
            self.assertEqual(args.metrics, ['accuracy', 'sensitivity', 'auc'])


class TestSaveOutputs(unittest.TestCase):
    """Test output saving"""

    def setUp(self):
        """Set up a test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.args = MagicMock()
        self.args.output_dir = self.test_dir
        self.args.data = 'test_data.csv'
        self.args.strategy = 'kfold'
        self.args.k = 5
        self.args.K = 3
        self.args.no_plots = True
        self.args.no_subfolder = True

    def tearDown(self):
        """Remove the temporary directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_outputs_basic(self):
        """Test basic output saving"""
        metrics_per_fold = [
            {'accuracy': 0.95, 'error_rate': 0.05, 'confusion_matrix': {'TN': 50, 'FP': 5, 'FN': 3, 'TP': 42}}
        ]
        labels_true = np.array([2, 4, 2, 4])
        labels_pred = np.array([2, 4, 2, 4])
        labels_proba = np.array([0.1, 0.9, 0.2, 0.8])

        save_outputs(self.args, metrics_per_fold, labels_true, labels_pred, labels_proba)

        files = os.listdir(self.test_dir)
        self.assertGreater(len(files), 0)
        csv_files = [f for f in files if f.endswith('.csv')]
        self.assertGreater(len(csv_files), 0)


class TestFindDefaultDataFile(unittest.TestCase):
    """Test default data file detection"""

    def setUp(self):
        """Create temporary test data directory"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()

    def tearDown(self):
        """Clean up"""
        os.chdir(self.original_cwd)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_find_default_data_file_csv(self):
        """Test finding the default CSV file"""
        os.chdir(self.test_dir)

        # The function looks in 'data/' directory
        data_dir = os.path.join(self.test_dir, 'data')
        os.makedirs(data_dir)
        test_file = os.path.join(data_dir, 'test.csv')
        with open(test_file, 'w') as f:
            f.write("test,data\n1,2\n")

        result = find_default_data_file()
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith('.csv'))

    def test_find_default_data_file_not_found(self):
        """Test when no default data file exists"""
        os.chdir(self.test_dir)
        result = find_default_data_file()
        self.assertIsNone(result)


class TestRunKComparison(unittest.TestCase):
    """Test k-values comparison functionality"""

    def setUp(self):
        """Set up a test environment"""
        self.test_dir = tempfile.mkdtemp()
        np.random.seed(42)
        self.features = np.random.rand(50, 9)
        self.labels = np.array([2] * 25 + [4] * 25)

        self.args = MagicMock()
        self.args.k_values = '3,5,7'
        self.args.strategy = 'holdout'
        self.args.K = 5
        self.args.test_size = 0.3
        self.args.p = 2
        self.args.random_state = 42
        self.args.output_dir = self.test_dir
        self.args.data = 'test_data.csv'
        self.args.no_plots = True
        self.args.no_subfolder = False
        self.args.verbose = False

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_run_k_comparison_basic(self):
        """Test basic k-values comparison"""
        run_k_comparison(self.args, self.features, self.labels)
        self.assertTrue(os.path.exists(self.test_dir))

    def test_run_k_comparison_output_structure(self):
        """Test output structure of k-values comparison"""
        run_k_comparison(self.args, self.features, self.labels)
        subdirectories = [d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))]
        self.assertGreater(len(subdirectories), 0, "Should create at least one output directory")


class TestMainIntegration(unittest.TestCase):
    """Integration tests for the main module"""

    def setUp(self):
        """Set up a test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test_data.csv')

        data = ("sample_code_number,clump_thickness,uniformity_cell_size,"
                "uniformity_cell_shape,marginal_adhesion,"
                "single_epithelial_cell_size,bare_nuclei,"
                "bland_chromatin,normal_nucleoli,mitoses,class\n")
        for i in range(30):
            class_label = 2 if i < 15 else 4
            v = i % 10 + 1
            data += f"{i},{v},{v},{v},{v},{v},{v},{v},{v},{v},{class_label}\n"

        with open(self.test_file, 'w') as f:
            f.write(data)

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_full_pipeline_holdout(self):
        """Test the complete pipeline with holdout validation"""
        features, labels = load_and_preprocess_data(self.test_file, verbose=False)
        classifier = KNNClassifier(k=3)
        strategy = HoldoutValidation(test_size=0.3, random_state=42)

        metrics_per_fold, labels_true, labels_pred, labels_proba = run_evaluation(
            features, labels, classifier, strategy, verbose=False
        )

        self.assertEqual(len(metrics_per_fold), 1)
        self.assertGreater(metrics_per_fold[0]['accuracy'], 0)

    # noinspection SpellCheckingInspection
    def test_full_pipeline_kfold(self):
        """Test the complete pipeline with k-fold validation"""
        features, labels = load_and_preprocess_data(self.test_file, verbose=False)
        classifier = KNNClassifier(k=5)
        strategy = KFoldCrossValidation(n_splits=3, random_state=42)

        metrics_per_fold, labels_true, labels_pred, labels_proba = run_evaluation(
            features, labels, classifier, strategy, verbose=False
        )

        self.assertEqual(len(metrics_per_fold), 3)
        for metrics in metrics_per_fold:
            self.assertGreater(metrics['accuracy'], 0)


if __name__ == '__main__':
    unittest.main()
