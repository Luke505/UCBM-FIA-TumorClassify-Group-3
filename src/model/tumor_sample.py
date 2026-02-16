"""
Tumor sample data models

This module defines the core data structures for representing tumor samples,
including features and complete sample objects.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np


@dataclass
class TumorFeatures:
    """
    Represents the 9 tumor features used for classification
    
    All feature values must be integers between 1 and 10 (inclusive).
    
    Attributes:
    -----------
    clump_thickness : int
        Thickness of the cell clump (1-10)
    uniformity_cell_size : int
        Uniformity of cell size (1-10)
    uniformity_cell_shape : int
        Uniformity of cell shape (1-10)
    marginal_adhesion : int
        Marginal adhesion of cells (1-10)
    single_epithelial_cell_size : int
        Size of single epithelial cells (1-10)
    bare_nuclei : int
        Presence of bare nuclei (1-10)
    bland_chromatin : int
        Bland chromatin texture (1-10)
    normal_nucleoli : int
        Normal nucleoli characteristics (1-10)
    mitoses : int
        Mitotic activity (1-10)
    """

    clump_thickness: int
    uniformity_cell_size: int
    uniformity_cell_shape: int
    marginal_adhesion: int
    single_epithelial_cell_size: int
    bare_nuclei: int
    bland_chromatin: int
    normal_nucleoli: int
    mitoses: int

    def __post_init__(self):
        """Validate feature values after initialization"""
        features = [
            self.clump_thickness,
            self.uniformity_cell_size,
            self.uniformity_cell_shape,
            self.marginal_adhesion,
            self.single_epithelial_cell_size,
            self.bare_nuclei,
            self.bland_chromatin,
            self.normal_nucleoli,
            self.mitoses
        ]

        for i, value in enumerate(features):
            if not isinstance(value, (int, float)):
                raise ValueError(f"Feature {i} must be a number, got {type(value)}")
            if not (1 <= value <= 10):
                raise ValueError(f"Feature {i} must be between 1 and 10, got {value}")

    def to_array(self) -> np.ndarray[tuple[int], np.dtype[int]]:
        """
        Convert features to a numpy array
        
        Returns:
        --------
        np.ndarray
            Array of shape (9) containing the feature values
        """
        return np.array([
            self.clump_thickness,
            self.uniformity_cell_size,
            self.uniformity_cell_shape,
            self.marginal_adhesion,
            self.single_epithelial_cell_size,
            self.bare_nuclei,
            self.bland_chromatin,
            self.normal_nucleoli,
            self.mitoses
        ], dtype=int)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TumorFeatures':
        """
        Create TumorFeatures from a numpy array
        
        Parameters:
        -----------
        arr : np.ndarray
            Array of shape (9) containing feature values
            
        Returns:
        --------
        TumorFeatures
            New TumorFeatures instance
        """
        if len(arr) != 9:
            raise ValueError(f"Expected 9 features, got {len(arr)}")

        return cls(
            clump_thickness=int(round(arr[0])),
            uniformity_cell_size=int(round(arr[1])),
            uniformity_cell_shape=int(round(arr[2])),
            marginal_adhesion=int(round(arr[3])),
            single_epithelial_cell_size=int(round(arr[4])),
            bare_nuclei=int(round(arr[5])),
            bland_chromatin=int(round(arr[6])),
            normal_nucleoli=int(round(arr[7])),
            mitoses=int(round(arr[8]))
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], aliases: Optional[Dict[str, str]] = None) -> 'TumorFeatures':
        """
        Create TumorFeatures from a dictionary with flexible key names
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Dictionary containing feature values
        aliases : Optional[Dict[str, str]]
            Optional mapping of custom keys to standard feature names
            Example: {'thickness': 'clump_thickness'}
            
        Returns:
        --------
        TumorFeatures
            New TumorFeatures instance
            
        Raises:
        -------
        ValueError
            If required features are missing or invalid
        """
        # Default to snake_case keys
        if aliases is None:
            aliases = {}

        # Helper function to find value in a dict with various key formats
        def get_value(standard_key: str) -> Any:
            # Check if there's an alias
            if standard_key in aliases:
                custom_key = aliases[standard_key]
                if custom_key in data:
                    return data[custom_key]

            # Try standard snake_case
            if standard_key in data:
                return data[standard_key]

            # Try variations (lowercase, with spaces, etc.)
            for key in data.keys():
                if key.lower().replace(' ', '_').replace('-', '_') == standard_key:
                    return data[key]

            raise ValueError(f"Missing required feature: {standard_key}")

        try:
            return cls(
                clump_thickness=int(get_value('clump_thickness')),
                uniformity_cell_size=int(get_value('uniformity_cell_size')),
                uniformity_cell_shape=int(get_value('uniformity_cell_shape')),
                marginal_adhesion=int(get_value('marginal_adhesion')),
                single_epithelial_cell_size=int(get_value('single_epithelial_cell_size')),
                bare_nuclei=int(get_value('bare_nuclei')),
                bland_chromatin=int(get_value('bland_chromatin')),
                normal_nucleoli=int(get_value('normal_nucleoli')),
                mitoses=int(get_value('mitoses'))
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid feature values: {e}")


@dataclass
class TumorSample:
    """
    Represents a complete tumor sample with ID, features, and classification
    
    Attributes:
    -----------
    id : int
        Unique identifier for the sample
    features : TumorFeatures
        The 9 tumor features
    tumor_class : int
        Classification: 2 for benign, 4 for malignant
    """

    id: int
    features: TumorFeatures
    tumor_class: int

    def __post_init__(self):
        """Validate sample after initialization"""
        if not isinstance(self.id, (int, np.integer)):
            try:
                self.id = int(self.id)
            except (ValueError, TypeError):
                raise ValueError(f"ID must be an integer, got {type(self.id)}")

        if not isinstance(self.features, TumorFeatures):
            raise ValueError(f"Features must be TumorFeatures instance, got {type(self.features)}")

        if self.tumor_class not in [2, 4]:
            raise ValueError(f"Tumor class must be 2 (benign) or 4 (malignant), got {self.tumor_class}")

    @property
    def is_malignant(self) -> bool:
        """Check if the tumor is malignant"""
        return self.tumor_class == 4

    @property
    def is_benign(self) -> bool:
        """Check if the tumor is benign"""
        return self.tumor_class == 2

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert sample to dictionary
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary representation of the sample
        """
        return {
            'id': self.id,
            'clump_thickness': self.features.clump_thickness,
            'uniformity_cell_size': self.features.uniformity_cell_size,
            'uniformity_cell_shape': self.features.uniformity_cell_shape,
            'marginal_adhesion': self.features.marginal_adhesion,
            'single_epithelial_cell_size': self.features.single_epithelial_cell_size,
            'bare_nuclei': self.features.bare_nuclei,
            'bland_chromatin': self.features.bland_chromatin,
            'normal_nucleoli': self.features.normal_nucleoli,
            'mitoses': self.features.mitoses,
            'class': self.tumor_class
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], aliases: Optional[Dict[str, str]] = None) -> Optional['TumorSample']:
        """
        Create TumorSample from a dictionary with validation
        
        This method validates the input and returns None if the sample is invalid
        (duplicate ID handled by TumorDataset, missing/invalid features, or invalid class).
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Dictionary containing sample data
        aliases : Optional[Dict[str, str]]
            Optional mapping of custom keys to standard names
            
        Returns:
        --------
        Optional[TumorSample]
            New TumorSample instance, or None if invalid
        """
        try:
            # Extract ID
            sample_id = None
            for key in ['id', 'sample_code_number', 'sample_id']:
                if key in data:
                    sample_id = int(data[key])
                    break

            if sample_id is None:
                # Try to find ID with aliases
                if aliases and 'id' in aliases:
                    custom_key = aliases['id']
                    if custom_key in data:
                        sample_id = int(data[custom_key])

            if sample_id is None:
                return None  # No valid ID found

            # Extract class
            tumor_class = None
            for key in ['class', 'tumor_class', 'classtype', 'classtype_v1']:
                if key in data:
                    value = data[key]
                    if isinstance(value, str):
                        value = value.lower().strip()
                        if value == 'benign':
                            tumor_class = 2
                        elif value == 'malignant':
                            tumor_class = 4
                    else:
                        try:
                            val = int(value)
                            if val in [2, 4]:
                                tumor_class = val
                        except (ValueError, TypeError):
                            pass
                    if tumor_class is not None:
                        break

            if tumor_class is None:
                # Try with aliases
                if aliases and 'class' in aliases:
                    custom_key = aliases['class']
                    if custom_key in data:
                        try:
                            tumor_class = int(data[custom_key])
                        except (ValueError, TypeError):
                            pass

            if tumor_class not in [2, 4]:
                return None  # Invalid class

            # Extract features
            features = TumorFeatures.from_dict(data, aliases)

            return cls(id=sample_id, features=features, tumor_class=tumor_class)

        except (ValueError, KeyError, TypeError):
            return None  # Invalid sample


class TumorDataset:
    """
    Collection of tumor samples with utility methods
    
    This class manages a collection of TumorSample objects and provides
    methods for filtering, validation, and conversion to arrays.
    
    Attributes:
    -----------
    samples : List[TumorSample]
        List of tumor samples in the dataset
    """

    def __init__(self, samples: Optional[List[TumorSample]] = None):
        """
        Initialize dataset with optional samples
        
        Parameters:
        -----------
        samples : Optional[List[TumorSample]]
            Initial list of samples (default: empty list)
        """
        self._samples: List[TumorSample] = []
        self._id_set: set = set()

        if samples:
            for sample in samples:
                self.add_sample(sample)

    def add_sample(self, sample: TumorSample) -> bool:
        """
        Add a sample to the dataset
        
        Duplicate IDs are automatically ignored.
        
        Parameters:
        -----------
        sample : TumorSample
            Sample to add
            
        Returns:
        --------
        bool
            True if the sample was added, False if duplicate ID
        """
        if sample.id in self._id_set:
            return False  # Duplicate ID

        self._samples.append(sample)
        self._id_set.add(sample.id)
        return True

    def __len__(self) -> int:
        """Return number of samples in dataset"""
        return len(self._samples)

    def __getitem__(self, index: int) -> TumorSample:
        """Get sample by index"""
        return self._samples[index]

    def __iter__(self):
        """Iterate over samples"""
        return iter(self._samples)

    @property
    def samples(self) -> List[TumorSample]:
        """Get a list of all samples"""
        return self._samples.copy()

    def to_arrays(self) -> tuple[np.ndarray[tuple[int, int], np.dtype[int]], np.ndarray[tuple[int], np.dtype[int]]]:
        """
        Convert dataset to numpy arrays for ML algorithms
        
        Returns:
        --------
        tuple[np.ndarray[tuple[int, int], np.dtype[int]], np.ndarray[tuple[int], np.dtype[int]]]
            Features matrix (n_samples, 9) and labels array (n_samples)
        """
        if len(self._samples) == 0:
            return np.array([]).reshape(0, 9), np.array([])

        features = np.array([sample.features.to_array() for sample in self._samples])
        labels = np.array([sample.tumor_class for sample in self._samples])

        return features, labels

    def get_ids(self) -> List[int]:
        """
        Get a list of all sample IDs
        
        Returns:
        --------
        List[int]
            List of sample IDs
        """
        return [sample.id for sample in self._samples]

    def get_by_id(self, sample_id: int) -> Optional[TumorSample]:
        """
        Get the sample by ID
        
        Parameters:
        -----------
        sample_id : int
            ID of the sample to retrieve
            
        Returns:
        --------
        Optional[TumorSample]
            The sample if found, None otherwise
        """
        for sample in self._samples:
            if sample.id == sample_id:
                return sample
        return None

    def filter_by_class(self, tumor_class: int) -> 'TumorDataset':
        """
        Create a new dataset with only samples of a specific class
        
        Parameters:
        -----------
        tumor_class : int
            Class to filter by (2 or 4)
            
        Returns:
        --------
        TumorDataset
            New dataset containing only samples of the specified class
        """
        filtered_samples = [s for s in self._samples if s.tumor_class == tumor_class]
        return TumorDataset(filtered_samples)

    def split(self, indices: np.ndarray) -> 'TumorDataset':
        """
        Create a new dataset from a subset of samples by indices
        
        Parameters:
        -----------
        indices : np.ndarray
            Array of indices to include in the new dataset
            
        Returns:
        --------
        TumorDataset
            New dataset containing the selected samples
        """
        subset_samples = [self._samples[i] for i in indices]
        return TumorDataset(subset_samples)

    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get distribution of classes in the dataset
        
        Returns:
        --------
        Dict[int, int]
            Dictionary mapping class labels to counts
        """
        distribution = {2: 0, 4: 0}
        for sample in self._samples:
            distribution[sample.tumor_class] += 1
        return distribution

    def __repr__(self) -> str:
        """String representation of the dataset"""
        dist = self.get_class_distribution()
        return f"TumorDataset(samples={len(self)}, benign={dist[2]}, malignant={dist[4]})"
