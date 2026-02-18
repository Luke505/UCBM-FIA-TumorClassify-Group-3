"""
Validation strategy implementations

This module implements the holdout, random subsampling, and K-fold
cross-validation strategies using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Generator, Union

import numpy as np

from src.model import TumorDataset


class BaseValidationStrategy(ABC):
    """
    Abstract base class for validation strategies
    
    All validation strategies must implement the split method that yields
    train and test indices for each fold/iteration.
    
    Strategies work with both TumorDataset and numpy arrays.
    """

    @abstractmethod
    def split(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for cross-validation
        
        Parameters:
        -----------
        features : Union[np.ndarray, TumorDataset]
            Feature matrix or TumorDataset
        labels : np.ndarray, optional
            Label vector (not needed if X is TumorDataset)

        Yields:
        -------
        Tuple[np.ndarray, np.ndarray]
            Train indices and test indices for each split
        """
        pass

    @abstractmethod
    def get_n_splits(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> int:
        """
        Return the number of splits/folds
        
        Parameters:
        -----------
        features : Union[np.ndarray, TumorDataset]
            Feature matrix or TumorDataset
        labels : np.ndarray, optional
            Label vector (not needed if X is TumorDataset)
            
        Returns:
        --------
        int
            Number of splits
        """
        pass

    def _convert_to_arrays(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert input to numpy arrays
        
        Parameters:
        -----------
        features : Union[np.ndarray, TumorDataset]
            Features or dataset
        labels : np.ndarray, optional
            Labels
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Features and labels as arrays
        """
        if isinstance(features, TumorDataset):
            return features.to_arrays()
        else:
            return features, labels


class HoldoutValidation(BaseValidationStrategy):
    """
    Simple holdout validation strategy
    
    Splits the dataset once into training and test sets
    
    Parameters:
    -----------
    test_size : float
        Proportion of dataset to include in the test split (default: 0.3)
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(self, test_size: float = 0.3, random_state: int = None):
        """
        Initialize holdout validation
        
        Parameters:
        -----------
        test_size : float
            Proportion of dataset for testing (between 0 and 1)
        random_state : int, optional
            Random seed for reproducibility
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        self.test_size = test_size
        self.random_state = random_state

    def split(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate a single train/test split
        
        Parameters:
        -----------
        features : Union[np.ndarray, TumorDataset]
            Feature matrix or dataset
        labels : np.ndarray, optional
            Label vector

        Yields:
        -------
        Tuple[np.ndarray, np.ndarray]
            Train indices and test indices
        """
        features, labels = self._convert_to_arrays(features, labels)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = len(features)
        n_test = int(n_samples * self.test_size)

        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        yield train_indices, test_indices

    def get_n_splits(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> int:
        """Return number of splits (always 1 for holdout)"""
        return 1


class RandomSubsampling(BaseValidationStrategy):
    """
    Random subsampling validation (repeated holdout)
    
    Performs multiple random train/test splits
    
    Parameters:
    -----------
    n_splits : int
        Number of re-shuffling and splitting iterations
    test_size : float
        Proportion of dataset for testing
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(self, n_splits: int = 10, test_size: float = 0.3, random_state: int = None):
        """
        Initialize random subsampling
        
        Parameters:
        -----------
        n_splits : int
            Number of random splits
        test_size : float
            Proportion of dataset for testing
        random_state : int, optional
            Random seed
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate multiple random train/test splits
        
        Parameters:
        -----------
        features : Union[np.ndarray, TumorDataset]

            Feature matrix or dataset

        labels : np.ndarray, optional

            Label vector
            
        Yields:
        -------
        Tuple[np.ndarray[tuple[int, int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]]
            Train indices and test indices for each split
        """
        features, _ = self._convert_to_arrays(features, labels)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = len(features)
        n_test = int(n_samples * self.test_size)

        for _ in range(self.n_splits):
            indices = np.random.permutation(n_samples)
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
            yield train_indices, test_indices

    def get_n_splits(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> int:
        """Return number of splits"""
        return self.n_splits


class KFoldCrossValidation(BaseValidationStrategy):
    """
    K-Fold Cross-Validation
    
    Splits the dataset into K folds and uses each fold once as the test set
    
    Parameters:
    -----------
    n_splits : int
        Number of folds (K)
    shuffle : bool
        Whether to shuffle data before splitting
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = None):
        """
        Initialize K-Fold cross-validation
        
        Parameters:
        -----------
        n_splits : int
            Number of folds
        shuffle : bool
            Whether to shuffle before splitting
        random_state : int, optional
            Random seed
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate K-Fold train/test splits
        
        Parameters:
        -----------
        features : Union[np.ndarray, TumorDataset]

            Feature matrix or dataset

        labels : np.ndarray, optional

            Label vector

        Yields:
        -------
        Tuple[np.ndarray[tuple[int, int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]]
            Train indices and test indices for each fold
        """
        features, _ = self._convert_to_arrays(features, labels)

        n_samples = len(features)
        indices = np.arange(n_samples)

        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)

        # Split indices into K folds
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop

    def get_n_splits(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> int:
        """Return number of splits"""
        return self.n_splits


class ValidationStrategyFactory:
    """
    Factory class for creating validation strategies
    
    This implements the Factory design pattern to provide a unified interface
    for creating different validation strategies
    """

    # noinspection SpellCheckingInspection
    _strategies = {
        'holdout': HoldoutValidation,
        'random_subsampling': RandomSubsampling,
        'kfold': KFoldCrossValidation,
        'k-fold': KFoldCrossValidation,
    }

    @classmethod
    def create_strategy(cls, strategy_name: str, **kwargs) -> BaseValidationStrategy:
        """
        Create a validation strategy by name
        
        Parameters:
        -----------
        strategy_name : str
            Name of the validation strategy
        **kwargs
            Additional arguments for the strategy
            
        Returns:
        --------
        BaseValidationStrategy
            An instance of the requested validation strategy
            
        Raises:
        -------
        ValueError
            If the strategy name is not recognized
        """
        strategy_name = strategy_name.lower().replace(' ', '_')

        strategy_class = cls._strategies.get(strategy_name)
        if strategy_class is None:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(f"Unknown validation strategy: {strategy_name}. Available: {available}")

        return strategy_class(**kwargs)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        List all available validation strategies
        
        Returns:
        --------
        List[str]
            List of available strategy names
        """
        return list(cls._strategies.keys())
