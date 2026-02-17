"""
Unit tests for data loader module

Tests the data loading functionality for various file formats
including CSV, TXT, JSON, XLSX, and TSV
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import TumorDataset
from src.utils.data_loader import (
    BaseDataLoader,
    DataLoaderFactory,
    CSVDataLoader,
    TXTDataLoader,
    TSVDataLoader,
    JSONDataLoader,
    XLSXDataLoader
)


class TestDataLoaderFactory(unittest.TestCase):
    """Test cases for DataLoaderFactory"""

    def test_create_csv_loader(self):
        """Test creation of a CSV loader"""
        loader = DataLoaderFactory.create_loader('test.csv')
        self.assertIsInstance(loader, CSVDataLoader)

    def test_create_txt_loader(self):
        """Test creation of a TXT loader"""
        loader = DataLoaderFactory.create_loader('test.txt')
        self.assertIsInstance(loader, TXTDataLoader)

    def test_create_tsv_loader(self):
        """Test creation of TSV loader"""
        loader = DataLoaderFactory.create_loader('test.tsv')
        self.assertIsInstance(loader, TSVDataLoader)

    def test_create_json_loader(self):
        """Test creation of a JSON loader"""
        loader = DataLoaderFactory.create_loader('test.json')
        self.assertIsInstance(loader, JSONDataLoader)

    def test_create_xlsx_loader(self):
        """Test creation of XLSX loader"""
        loader = DataLoaderFactory.create_loader('test.xlsx')
        self.assertIsInstance(loader, XLSXDataLoader)

    def test_unsupported_format(self):
        """Test that unsupported formats raise ValueError"""
        with self.assertRaises(ValueError):
            DataLoaderFactory.create_loader('test.pdf')


class TestBaseDataLoader(unittest.TestCase):
    """Test cases for base data loader static helpers"""

    def test_clean_value_normal(self):
        """Test cleaning normal numeric values"""
        self.assertEqual(BaseDataLoader._clean_value('5'), 5)
        self.assertEqual(BaseDataLoader._clean_value(10), 10)

    def test_clean_value_rounds_floats(self):
        """Test that floats are rounded to nearest int"""
        self.assertEqual(BaseDataLoader._clean_value('3.5'), 4)
        self.assertEqual(BaseDataLoader._clean_value('3.4'), 3)

    def test_clean_value_comma_decimal(self):
        """Test cleaning values with comma as a decimal separator"""
        self.assertEqual(BaseDataLoader._clean_value('3,5'), 4)

    def test_clean_value_missing(self):
        """Test cleaning missing values returns None"""
        self.assertIsNone(BaseDataLoader._clean_value(''))
        self.assertIsNone(BaseDataLoader._clean_value(None))
        self.assertIsNone(BaseDataLoader._clean_value(np.nan))

    def test_clean_value_out_of_range(self):
        """Test that values outside 1-10 return None"""
        self.assertIsNone(BaseDataLoader._clean_value(0))
        self.assertIsNone(BaseDataLoader._clean_value(11))
        self.assertIsNone(BaseDataLoader._clean_value(-1))

    def test_clean_value_corrupted(self):
        """Test cleaning corrupted string values"""
        self.assertIsNone(BaseDataLoader._clean_value('abc'))
        self.assertIsNone(BaseDataLoader._clean_value('###'))

    def test_normalize_class_label_numeric(self):
        """Test normalizing numeric class labels"""
        self.assertEqual(BaseDataLoader._normalize_class_label(2), 2)
        self.assertEqual(BaseDataLoader._normalize_class_label(4), 4)
        self.assertEqual(BaseDataLoader._normalize_class_label('2'), 2)
        self.assertEqual(BaseDataLoader._normalize_class_label('4'), 4)

    def test_normalize_class_label_text(self):
        """Test normalizing text class labels"""
        self.assertEqual(BaseDataLoader._normalize_class_label('benign'), 2)
        self.assertEqual(BaseDataLoader._normalize_class_label('malignant'), 4)
        # noinspection SpellCheckingInspection
        self.assertEqual(BaseDataLoader._normalize_class_label('maligant'), 4)

    def test_normalize_class_label_invalid(self):
        """Test that invalid class labels return None"""
        self.assertIsNone(BaseDataLoader._normalize_class_label(3))
        self.assertIsNone(BaseDataLoader._normalize_class_label('unknown'))
        self.assertIsNone(BaseDataLoader._normalize_class_label(np.nan))


class TestDataLoadingIntegration(unittest.TestCase):
    """Integration tests for loading actual test data files"""

    @classmethod
    def setUpClass(cls):
        """Set up paths to test data files"""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'tests_data')
        if not os.path.exists(cls.test_data_dir):
            cls.test_data_dir = 'tests_data'

    def _assert_valid_dataset(self, dataset):
        """Helper: verify a loaded dataset is valid"""
        self.assertIsInstance(dataset, TumorDataset)
        self.assertGreater(len(dataset), 0, "Should have at least one sample")

        features, labels = dataset.to_arrays()
        self.assertEqual(features.shape[1], 9, "Should have nine features")
        self.assertEqual(len(features), len(labels))

        # All labels must be 2 or 4
        self.assertTrue(np.all((labels == 2) | (labels == 4)))

        # All features must be integers 1-10
        self.assertTrue(np.all(features >= 1))
        self.assertTrue(np.all(features <= 10))

        # No duplicate IDs
        ids = dataset.get_ids()
        self.assertEqual(len(ids), len(set(ids)), "Should have no duplicate IDs")

    def test_load_csv_file(self):
        """Test loading CSV file (version_1.csv)"""
        filepath = os.path.join(self.test_data_dir, 'version_1.csv')
        if not os.path.exists(filepath):
            self.skipTest(f"Test file not found: {filepath}")

        dataset = DataLoaderFactory.load_data(filepath)
        self._assert_valid_dataset(dataset)

    def test_load_txt_file(self):
        """Test loading TXT file (version_3.txt)"""
        filepath = os.path.join(self.test_data_dir, 'version_3.txt')
        if not os.path.exists(filepath):
            self.skipTest(f"Test file not found: {filepath}")

        dataset = DataLoaderFactory.load_data(filepath)
        self._assert_valid_dataset(dataset)

    def test_load_tsv_file(self):
        """Test loading TSV file (version_5.tsv)"""
        filepath = os.path.join(self.test_data_dir, 'version_5.tsv')
        if not os.path.exists(filepath):
            self.skipTest(f"Test file not found: {filepath}")

        dataset = DataLoaderFactory.load_data(filepath)
        self._assert_valid_dataset(dataset)

    def test_load_json_file(self):
        """Test loading JSON file (version_4.json)"""
        filepath = os.path.join(self.test_data_dir, 'version_4.json')
        if not os.path.exists(filepath):
            self.skipTest(f"Test file not found: {filepath}")

        dataset = DataLoaderFactory.load_data(filepath)
        self._assert_valid_dataset(dataset)

    def test_load_xlsx_file(self):
        """Test loading XLSX file (version_2.xlsx)"""
        filepath = os.path.join(self.test_data_dir, 'version_2.xlsx')
        if not os.path.exists(filepath):
            self.skipTest(f"Test file not found: {filepath}")

        dataset = DataLoaderFactory.load_data(filepath)
        self._assert_valid_dataset(dataset)


if __name__ == '__main__':
    unittest.main()
