"""
Validation strategy implementations

This module implements various validation strategies using the Strategy pattern,
including Holdout, K-fold CV, Leave-one-out, Stratified CV, Bootstrap, etc.
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


class LeaveOneOutCrossValidation(BaseValidationStrategy):
    """
    Leave-One-Out Cross-Validation (LOOCV)
    
    Each sample is used once as the test set while the remaining samples form the training set.
    This is equivalent to K-Fold where K = n_samples
    
    Note: This can be very slow for large datasets
    """

    def __init__(self):
        """Initialize Leave-One-Out cross-validation"""
        pass

    def split(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate LOOCV train/test splits
        
        Parameters:
        -----------
        features : Union[np.ndarray, TumorDataset]

            Feature matrix or dataset

        labels : np.ndarray, optional

            Label vector

        Yields:
        -------
        Tuple[np.ndarray[tuple[int, int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]]
            Train indices and test indices for each iteration
        """
        features, _ = self._convert_to_arrays(features, labels)

        n_samples = len(features)

        for i in range(n_samples):
            test_indices = np.array([i])
            train_indices = np.concatenate([np.arange(0, i), np.arange(i + 1, n_samples)])
            yield train_indices, test_indices

    def get_n_splits(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> int:
        """Return the number of splits (equal to the number of samples)"""
        return len(features)


class LeavePOutCrossValidation(BaseValidationStrategy):
    """
    Leave-P-Out Cross-Validation
    
    Uses P samples as the test set in each iteration
    
    Note: This generates C(n, p) splits which can be extremely large
    For practical use, we limit the number of splits
    
    Parameters:
    -----------
    p : int
        Number of samples to leave out
    max_splits : int
        Maximum number of splits to generate (to avoid combinatorial explosion)
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(self, p: int = 2, max_splits: int = 100, random_state: int = None):
        """
        Initialize Leave-P-Out cross-validation
        
        Parameters:
        -----------
        p : int
            Number of samples to leave out
        max_splits : int
            Maximum number of splits
        random_state : int, optional
            Random seed
        """
        if p < 1:
            raise ValueError("p must be at least 1")

        self.p = p
        self.max_splits = max_splits
        self.random_state = random_state

    def split(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate Leave-P-Out train/test splits
        
        Parameters:
        -----------
        features : Union[np.ndarray, TumorDataset]

            Feature matrix or dataset

        labels : np.ndarray, optional

            Label vector
            
        Yields:
        -------
        Tuple[np.ndarray[tuple[int, int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]]
            Train indices and test indices for each iteration
        """
        features, _ = self._convert_to_arrays(features, labels)

        n_samples = len(features)

        if self.p >= n_samples:
            raise ValueError(f"p ({self.p}) must be less than n_samples ({n_samples})")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Generate random combinations instead of all combinations
        indices = np.arange(n_samples)

        for _ in range(min(self.max_splits, self._n_combinations(n_samples, self.p))):
            # Randomly select p indices for the test set
            test_indices = np.sort(np.random.choice(indices, size=self.p, replace=False))
            train_indices = np.array([i for i in indices if i not in test_indices])
            yield train_indices, test_indices

    def get_n_splits(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> int:
        """Return number of splits"""
        n_samples = len(features)
        return min(self.max_splits, self._n_combinations(n_samples, self.p))

    @staticmethod
    def _n_combinations(n: int, k: int) -> int:
        """Calculate the number of combinations C(n, k)"""
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1

        # Use the smaller of k and n-k for efficiency
        k = min(k, n - k)

        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)

        return result


class StratifiedCrossValidation(BaseValidationStrategy):
    """
    Stratified K-Fold Cross-Validation
    
    Ensures that each fold has approximately the same proportion of samples
    from each class as the complete dataset
    
    Parameters:
    -----------
    n_splits : int
        Number of folds
    shuffle : bool
        Whether to shuffle before splitting
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = None):
        """
        Initialize Stratified K-Fold cross-validation
        
        Parameters:
        -----------
        n_splits : int
            Number of folds
        shuffle : bool
            Whether to shuffle
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
        Generate stratified K-Fold train/test splits
        
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
        _, labels_array = self._convert_to_arrays(features, labels)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Get unique classes and their indices
        unique_classes = np.unique(labels_array)
        class_indices = {cls: np.where(labels_array == cls)[0] for cls in unique_classes}

        # Shuffle indices within each class
        if self.shuffle:
            for cls in unique_classes:
                np.random.shuffle(class_indices[cls])

        # Split each class into folds
        class_folds = {}
        for cls in unique_classes:
            indices = class_indices[cls]
            n_samples = len(indices)
            fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
            fold_sizes[:n_samples % self.n_splits] += 1

            current = 0
            folds = []
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                folds.append(indices[start:stop])
                current = stop

            class_folds[cls] = folds

        # Combine folds from all classes
        for i in range(self.n_splits):
            test_indices = np.concatenate([class_folds[cls][i] for cls in unique_classes])
            train_indices = np.concatenate([
                np.concatenate([class_folds[cls][j] for j in range(self.n_splits) if j != i])
                for cls in unique_classes
            ])

            yield train_indices, test_indices

    def get_n_splits(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> int:
        """Return number of splits"""
        return self.n_splits


class StratifiedShuffleSplit(BaseValidationStrategy):
    """
    Stratified Shuffle Split
    
    Similar to Random Subsampling but maintains class proportions in each split
    
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
        Initialize Stratified Shuffle Split
        
        Parameters:
        -----------
        n_splits : int
            Number of splits
        test_size : float
            Proportion for testing
        random_state : int, optional
            Random seed
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate stratified shuffle splits
        
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
        _, labels_array = self._convert_to_arrays(features, labels)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        unique_classes = np.unique(labels_array)
        class_indices = {cls: np.where(labels_array == cls)[0] for cls in unique_classes}

        for _ in range(self.n_splits):
            train_indices_list = []
            test_indices_list = []

            # For each class, split proportionally
            for cls in unique_classes:
                indices = class_indices[cls].copy()
                np.random.shuffle(indices)

                n_test = int(len(indices) * self.test_size)
                test_indices_list.append(indices[:n_test])
                train_indices_list.append(indices[n_test:])

            train_indices = np.concatenate(train_indices_list)
            test_indices = np.concatenate(test_indices_list)

            yield train_indices, test_indices

    def get_n_splits(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> int:
        """Return number of splits"""
        return self.n_splits


class Bootstrap(BaseValidationStrategy):
    """
    Bootstrap resampling validation
    
    Generates training sets by sampling with replacement from the original dataset.
    Samples not selected (out-of-bag) are used as the test set
    
    Parameters:
    -----------
    n_splits : int
        Number of bootstrap iterations
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(self, n_splits: int = 10, random_state: int = None):
        """
        Initialize Bootstrap validation
        
        Parameters:
        -----------
        n_splits : int
            Number of bootstrap iterations
        random_state : int, optional
            Random seed
        """
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, features: Union[np.ndarray, TumorDataset], labels: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate bootstrap train/test splits
        
        Parameters:
        -----------
        features : Union[np.ndarray, TumorDataset]

            Feature matrix or dataset

        labels : np.ndarray, optional

            Label vector

        Yields:
        -------
        Tuple[np.ndarray[tuple[int, int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]]
            Train indices (with possible duplicates) and test indices (out-of-bag)
        """
        features_array, _ = self._convert_to_arrays(features, labels)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = len(features_array)
        indices = np.arange(n_samples)

        for _ in range(self.n_splits):
            # Sample with replacement for the training set
            train_indices = np.random.choice(indices, size=n_samples, replace=True)

            # Out-of-bag samples for the test set
            test_indices = np.array([i for i in indices if i not in train_indices])

            # If all samples were selected (very rare), use a small subset as the test set
            if len(test_indices) == 0:
                test_indices = np.random.choice(indices, size=max(1, n_samples // 10), replace=False)

            yield train_indices, test_indices

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
        'loocv': LeaveOneOutCrossValidation,
        'leave-one-out': LeaveOneOutCrossValidation,
        'lpocv': LeavePOutCrossValidation,
        'leave-p-out': LeavePOutCrossValidation,
        'stratified': StratifiedCrossValidation,
        'stratified_kfold': StratifiedCrossValidation,
        'stratified_shuffle': StratifiedShuffleSplit,
        'bootstrap': Bootstrap,
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
