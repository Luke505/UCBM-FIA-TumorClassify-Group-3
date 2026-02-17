"""
Data loader module for handling multiple file formats

This module implements the Factory pattern to create appropriate data loaders
for different file formats (CSV, TXT, JSON, XLSX, TSV)
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

import pandas as pd

from src.model import TumorSample, TumorDataset, TumorFeatures


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders
    
    All data loaders must implement the load method that returns a TumorDataset.
    Invalid samples (duplicate IDs, invalid classes, missing features) are automatically filtered out.
    """

    # Mapping of possible column name variations to standard feature names
    # noinspection SpellCheckingInspection
    COLUMN_MAPPINGS = {
        'clump_thickness': ['clump_thickness', 'clump_thickness_ty'],
        'uniformity_cell_size': ['uniformity_cell_size', 'uniformity_of_cell_size', 'uniformity_cellsize_xx', 'uni!'],
        'uniformity_cell_shape': ['uniformity_cell_shape', 'uniformity_of_cell_shape'],
        'marginal_adhesion': ['marginal_adhesion', 'mar!'],
        'single_epithelial_cell_size': ['single_epithelial_cell_size', 'sin!'],
        'bare_nuclei': ['bare_nuclei', 'barenucleix_wrong'],
        'bland_chromatin': ['bland_chromatin'],
        'normal_nucleoli': ['normal_nucleoli'],
        'mitoses': ['mitoses', 'mit!'],
        'class': ['class', 'classtype_v1', 'classtype'],
        'id': ['sample_code_number', 'id', 'sam!']
    }

    # Mapping of column indices to feature positions
    COLUMN_INDEX_MAP = {
        'col_0': 'id',
        'col_1': 'clump_thickness',
        'col_2': 'uniformity_cell_size',
        'col_3': 'uniformity_cell_shape',
        'col_4': 'marginal_adhesion',
        'col_5': 'single_epithelial_cell_size',
        'col_6': 'bare_nuclei',
        'col_7': 'bland_chromatin',
        'col_8': 'normal_nucleoli',
        'col_9': 'mitoses',
        'col_10': 'class'
    }

    @staticmethod
    @abstractmethod
    def load(filepath: str, aliases: Optional[Dict[str, str]] = None) -> TumorDataset:
        """
        Load data from a file
        
        Parameters:
        -----------
        filepath : str
            Path to the data file
        aliases : Optional[Dict[str, str]]
            Optional mapping of custom column names to standard feature names
            
        Returns:
        --------
        TumorDataset
            Dataset containing valid tumor samples
        """
        pass

    @staticmethod
    def _clean_value(value: Any) -> Optional[int]:
        """
        Clean and convert a value to int (1-10), handling various corruptions
        
        Parameters:
        -----------
        value : Any
            The value to clean
            
        Returns:
        --------
        Optional[int]
            The cleaned value (1-10), or None if invalid
        """
        if pd.isna(value) or value == '' or value is None:
            return None

        # Convert to string for processing
        str_value = str(value).strip()

        # Replace comma with dot for decimal separator
        str_value = str_value.replace(',', '.')

        try:
            float_val = float(str_value)
            int_val = int(round(float_val))
            if 1 <= int_val <= 10:
                return int_val
            return None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _normalize_class_label(label: Any) -> Optional[int]:
        """
        Normalize class labels to the standard format (2 for benign, 4 for malignant)
        
        Parameters:
        -----------
        label : Any
            The class label to normalize
            
        Returns:
        --------
        Optional[int]
            2 for benign, 4 for malignant, None if not recognized
        """
        if pd.isna(label):
            return None

        label_str = str(label).strip().lower()

        # Handle text labels
        # noinspection SpellCheckingInspection
        if label_str == 'benign':
            return 2
        elif 'malignant' in label_str or 'maligant' in label_str:  # Handles both "malignant" and "maligant" typo (from version_3.txt)
            return 4

        # Handle numeric labels
        try:
            label_num = float(label_str)
            if label_num in [2, 2.0]:
                return 2
            elif label_num in [4, 4.0]:
                return 4
        except (ValueError, TypeError):
            pass

        return None

    @staticmethod
    def _create_sample_from_row(row: pd.Series, feature_columns: Dict[str, str],
        id_column: str, class_column: str,
        aliases: Optional[Dict[str, str]] = None) -> Optional[TumorSample]:
        """
        Create a TumorSample from a dataframe row
        
        Parameters:
        -----------
        row : pd.Series
            Row from the dataframe
        feature_columns : Dict[str, str]
            Mapping of standard feature names to actual column names
        id_column : str
            Name of the ID column
        class_column : str
            Name of the class column
        aliases : Optional[Dict[str, str]]
            Optional column name aliases
            
        Returns:
        --------
        Optional[TumorSample]
            TumorSample if valid, None otherwise
        """
        try:
            # Extract ID
            sample_id = int(row[id_column])

            # Extract class
            tumor_class = BaseDataLoader._normalize_class_label(row[class_column])
            if tumor_class not in [2, 4]:
                return None

            # Extract features
            feature_values = {}
            for standard_name, col_name in feature_columns.items():
                value = BaseDataLoader._clean_value(row[col_name])
                if value is None:
                    return None  # Missing or invalid feature value
                feature_values[standard_name] = value

            # Create features object
            features = TumorFeatures(**feature_values)

            # Create and return sample
            return TumorSample(id=sample_id, features=features, tumor_class=tumor_class)

        except (ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def _find_column(df: pd.DataFrame, feature: str) -> Optional[str]:
        """
        Find the actual column name in the dataframe for a given feature

        Parameters:
        -----------
        df : pd.DataFrame
            The input dataframe
        feature : str
            The standard feature name

        Returns:
        --------
        Optional[str]
            The actual column name in the dataframe, or None if not found
        """
        possible_names = BaseDataLoader.COLUMN_MAPPINGS.get(feature, [])
        for col in df.columns:
            if col.strip().lower().replace(" ", "_") in possible_names:
                return col
        return None

    @staticmethod
    def _load_from_dataframe(df: pd.DataFrame, aliases: Optional[Dict[str, str]] = None) -> TumorDataset:
        """
        Load tumor samples from a pandas DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            The input dataframe
        aliases : Optional[Dict[str, str]]
            Optional column name aliases
            
        Returns:
        --------
        TumorDataset
            Dataset containing valid samples
        """
        dataset = TumorDataset()

        # Rename col_X positional columns to standard names if present
        col_x_present = any(c in df.columns for c in BaseDataLoader.COLUMN_INDEX_MAP)
        if col_x_present:
            rename_map = {k: v for k, v in BaseDataLoader.COLUMN_INDEX_MAP.items() if k in df.columns}
            df = df.rename(columns=rename_map)

        # Find column mappings
        feature_names = [
            'clump_thickness',
            'uniformity_cell_size',
            'uniformity_cell_shape',
            'marginal_adhesion',
            'single_epithelial_cell_size',
            'bare_nuclei',
            'bland_chromatin',
            'normal_nucleoli',
            'mitoses'
        ]

        feature_columns = {}
        for feature in feature_names:
            col_name = BaseDataLoader._find_column(df, feature)
            if col_name:
                feature_columns[feature] = col_name
            else:
                raise ValueError(f"Could not find column for feature: {feature}")

        id_column = BaseDataLoader._find_column(df, 'id')
        if not id_column:
            raise ValueError("Could not find ID column")

        class_column = BaseDataLoader._find_column(df, 'class')
        if not class_column:
            raise ValueError("Could not find class column")

        # Process each row
        for _, row in df.iterrows():
            sample = BaseDataLoader._create_sample_from_row(
                row, feature_columns, id_column, class_column, aliases
            )
            if sample is not None:
                dataset.add_sample(sample)  # Automatically filters duplicates

        return dataset


class CSVDataLoader(BaseDataLoader):
    """
    Data loader for CSV files
    
    Handles CSV files with various column names and formats,
    extracting the 9 tumor features and class label
    """

    @staticmethod
    def load(filepath: str, aliases: Optional[Dict[str, str]] = None) -> TumorDataset:
        """
        Load data from a CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        aliases : Optional[Dict[str, str]]
            Optional column name aliases
            
        Returns:
        --------
        TumorDataset
            Dataset containing valid tumor samples
        """
        # Read the CSV file
        df = pd.read_csv(filepath)

        # Normalize column names to lowercase
        df.columns = df.columns.str.strip().str.lower()

        # Load samples from dataframe
        return BaseDataLoader._load_from_dataframe(df, aliases)


class TXTDataLoader(BaseDataLoader):
    """
    Data loader for TXT files (tab-separated format)
    
    Handles TXT files with tab delimiters and text-based class labels
    """

    @staticmethod
    def load(filepath: str, aliases: Optional[Dict[str, str]] = None) -> TumorDataset:
        """
        Load data from a TXT file (tab-separated)

        Parameters:
        -----------
        filepath : str
            Path to the TXT file
        aliases : Optional[Dict[str, str]]
            Optional column name aliases

        Returns:
        --------
        TumorDataset
            Dataset containing valid tumor samples
        """
        # Read TXT file as tab-separated
        df = pd.read_csv(filepath, sep='\t', engine='python')

        # Remove empty rows
        df = df.dropna(how='all')

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # Load samples from dataframe
        return BaseDataLoader._load_from_dataframe(df, aliases)


class TSVDataLoader(BaseDataLoader):
    """
    Data loader for TSV files (tab-separated values)
    
    Handles TSV files with indexed columns (col_0, col_1, etc.) and corrupted values
    """

    @staticmethod
    def load(filepath: str, aliases: Optional[Dict[str, str]] = None) -> TumorDataset:
        """
        Load data from a TSV file
        
        Parameters:
        -----------
        filepath : str
            Path to the TSV file
        aliases : Optional[Dict[str, str]]
            Optional column name aliases
            
        Returns:
        --------
        TumorDataset
            Dataset containing valid tumor samples
        """
        # Read the TSV file
        df = pd.read_csv(filepath, sep='\t')

        # Remove completely empty rows
        df = df.dropna(how='all')

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # Rename indexed columns to standard names
        if 'col_0' in df.columns:
            rename_map = {
                'col_0': 'id',
                'col_1': 'clump_thickness',
                'col_2': 'uniformity_cell_size',
                'col_3': 'uniformity_cell_shape',
                'col_4': 'marginal_adhesion',
                'col_5': 'single_epithelial_cell_size',
                'col_6': 'bare_nuclei',
                'col_7': 'bland_chromatin',
                'col_8': 'normal_nucleoli',
                'col_9': 'mitoses',
                'col_10': 'class'
            }
            df = df.rename(columns=rename_map)

        # Load samples from dataframe
        return BaseDataLoader._load_from_dataframe(df, aliases)


class JSONDataLoader(BaseDataLoader):
    """
    Data loader for JSON files
    
    Handles JSON files with various key names and formats
    """

    @staticmethod
    def load(filepath: str, aliases: Optional[Dict[str, str]] = None) -> TumorDataset:
        """
        Load data from a JSON file
        
        Parameters:
        -----------
        filepath : str
            Path to the JSON file
        aliases : Optional[Dict[str, str]]
            Optional column name aliases
            
        Returns:
        --------
        TumorDataset
            Dataset containing valid tumor samples
        """
        # Read the JSON file
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(data)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # Load samples from dataframe
        return BaseDataLoader._load_from_dataframe(df, aliases)


class XLSXDataLoader(BaseDataLoader):
    """
    Data loader for Excel (XLSX) files
    
    Handles XLSX files similar to CSV format
    """

    @staticmethod
    def load(filepath: str, aliases: Optional[Dict[str, str]] = None) -> TumorDataset:
        """
        Load data from an XLSX file
        
        Parameters:
        -----------
        filepath : str
            Path to the XLSX file
        aliases : Optional[Dict[str, str]]
            Optional column name aliases
            
        Returns:
        --------
        TumorDataset
            Dataset containing valid tumor samples
        """
        # Read the Excel file
        df = pd.read_excel(filepath)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # Load samples from dataframe
        return BaseDataLoader._load_from_dataframe(df, aliases)


class DataLoaderFactory:
    """
    Factory class for creating appropriate data loaders based on file extension
    
    This implements the Factory design pattern to provide a unified interface
    for loading data from different file formats
    """

    _loaders = {
        '.csv': CSVDataLoader,
        '.txt': TXTDataLoader,
        '.tsv': TSVDataLoader,
        '.json': JSONDataLoader,
        '.xlsx': XLSXDataLoader,
        '.xls': XLSXDataLoader,
    }

    @classmethod
    def create_loader(cls, filepath: str) -> BaseDataLoader:
        """
        Create an appropriate data loader based on a file extension
        
        Parameters:
        -----------
        filepath : str
            Path to the data file
            
        Returns:
        --------
        BaseDataLoader
            An instance of the appropriate data loader
            
        Raises:
        -------
        ValueError
            If the file extension is not supported
        """
        import os
        _, ext = os.path.splitext(filepath.lower())

        loader_class = cls._loaders.get(ext)
        if loader_class is None:
            raise ValueError(f"Unsupported file format: {ext}")

        return loader_class()

    @classmethod
    def load_data(cls, filepath: str, aliases: Optional[Dict[str, str]] = None) -> TumorDataset:
        """
        Load data from a file using the appropriate loader
        
        This is a convenience method that creates the loader and loads the data
        
        Parameters:
        -----------
        filepath : str
            Path to the data file
        aliases : Optional[Dict[str, str]]
            Optional column name aliases
            
        Returns:
        --------
        TumorDataset
            Dataset containing valid tumor samples
        """
        loader = cls.create_loader(filepath)
        return loader.load(filepath, aliases)
