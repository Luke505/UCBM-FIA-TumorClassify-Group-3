"""
k-Nearest Neighbors (k-NN) classifier implementation

This module implements a k-NN classifier from scratch without using
external machine learning libraries like scikit-learn
"""

import random
from collections import Counter
from typing import Union

import numpy as np

from src.model import TumorDataset


class KNNClassifier:
    """
    k-Nearest Neighbors classifier
    
    This classifier predicts the class of a sample based on the k nearest
    neighbors in the training data, using Euclidean distance as the metric
    
    Attributes:
    -----------
    k : int
        Number of neighbors to consider
    features_train : np.ndarray[tuple[int, int], np.dtype[np.float64]]
        Training feature matrix
    labels_train : np.ndarray[tuple[int], np.dtype[np.float64]]
        Training labels
    """

    def __init__(self, k: int = 3):
        """
        Initialize the k-NN classifier
        
        Parameters:
        -----------
        k : int, optional
            Number of neighbors to consider (default: 3)
            Must be a positive odd number for binary classification
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")

        self.k = k
        self.features_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None
        self.labels_train: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None

    def fit(self, features_train: Union[np.ndarray, TumorDataset], labels_train: np.ndarray = None) -> 'KNNClassifier':
        # noinspection GrazieInspection
        """
        Fit the classifier with training data

        For k-NN, this simply stores the training data

        Parameters:
        -----------
        features_train : Union[np.ndarray, TumorDataset]
            Training feature matrix of shape (n_samples, n_features) or TumorDataset
        labels_train : np.ndarray, optional
            Training labels of shape (n_samples,). Not needed if features_train is TumorDataset.

        Returns:
        --------
        KNNClassifier
            The fitted classifier (self)
        """
        if isinstance(features_train, TumorDataset):
            self.features_train, self.labels_train = features_train.to_arrays()
        else:
            self.features_train = np.array(features_train)
            self.labels_train = np.array(labels_train) if labels_train is not None else None
        return self

    def predict(self, features_test: Union[np.ndarray, TumorDataset]) -> np.ndarray:
        # noinspection GrazieInspection
        """
        Predict class labels for test samples
        
        Parameters:
        -----------
        features_test : Union[np.ndarray, TumorDataset]
            Test feature matrix of shape (n_samples, n_features) or TumorDataset
            
        Returns:
        --------
        np.ndarray
            Predicted class labels of shape (n_samples,)
        """
        if self.features_train is None or self.labels_train is None:
            raise ValueError("Classifier must be fitted before making predictions")

        # Convert TumorDataset to array if needed
        if isinstance(features_test, TumorDataset):
            features_test, _ = features_test.to_arrays()

        predictions = []
        for test_sample in features_test:
            prediction = self._predict_single(test_sample)
            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, features_test: Union[np.ndarray, TumorDataset]) -> np.ndarray:
        """
        Predict class probabilities for test samples
        
        The probability is calculated as the proportion of neighbors
        belonging to each class
        
        Parameters:
        -----------
        features_test : Union[np.ndarray, TumorDataset]
            Test feature matrix of shape (n_samples, n_features) or TumorDataset
            
        Returns:
        --------
        np.ndarray
            Class probabilities of shape (n_samples, n_classes)
            For binary classification: [prob_class_2, prob_class_4]
        """
        if self.features_train is None or self.labels_train is None:
            raise ValueError("Classifier must be fitted before making predictions")

        # Convert TumorDataset to array if needed
        if isinstance(features_test, TumorDataset):
            features_test, _ = features_test.to_arrays()

        probabilities = []
        for test_sample in features_test:
            proba = self._predict_proba_single(test_sample)
            probabilities.append(proba)

        return np.array(probabilities)

    def _predict_single(self, test_sample: np.ndarray[tuple[int, int], np.dtype[np.float64]]) -> float:
        # noinspection GrazieInspection
        """
        Predict the class label for a single test sample
        
        Parameters:
        -----------
        test_sample : np.ndarray[tuple[int, int], np.dtype[np.float64]]
            A single test sample of shape (n_features,)
            
        Returns:
        --------
        float
            The predicted class label (2 or 4)
        """
        # Calculate distances to all training samples
        distances = self._calculate_distances(test_sample)

        # Get indices of k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]

        # Get labels of k nearest neighbors
        k_nearest_labels: np.ndarray[tuple[int], np.dtype[np.float64]] = self.labels_train[k_nearest_indices]

        # Count occurrences of each label
        label_counts: Counter[float] = Counter(k_nearest_labels)

        # Handle ties: if there's a tie, randomly select among the tied labels
        max_count = max(label_counts.values())
        most_common_labels = [label for label, count in label_counts.items() if count == max_count]

        if len(most_common_labels) > 1:
            # Random tie-breaking
            return random.choice(most_common_labels)
        else:
            return most_common_labels[0]

    def _predict_proba_single(self, test_sample: np.ndarray[tuple[int, int], np.dtype[np.float64]]) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        # noinspection GrazieInspection
        """
        Predict class probabilities for a single test sample
        
        Parameters:
        -----------
        test_sample : np.ndarray[tuple[int, int], np.dtype[np.float64]]
            A single test sample of shape (n_features,)
            
        Returns:
        --------
        np.ndarray[tuple[int, int], np.dtype[np.float64]]
            Class probabilities [prob_class_2, prob_class_4]
        """
        # Calculate distances to all training samples
        distances = self._calculate_distances(test_sample)

        # Get indices of k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]

        # Get labels of k nearest neighbors
        k_nearest_labels: np.ndarray[tuple[int], np.dtype[np.float64]] = self.labels_train[k_nearest_indices]

        # Calculate probabilities
        unique_labels = np.unique(self.labels_train)
        probabilities = []

        for label in sorted(unique_labels):
            prob = np.sum(k_nearest_labels == label) / self.k
            probabilities.append(prob)

        return np.array(probabilities)

    def _calculate_distances(self, test_sample: np.ndarray[tuple[int, int], np.dtype[np.float64]]) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        # noinspection GrazieInspection
        """
        Calculate Euclidean distances between the test sample and all training samples
        
        Euclidean distance: sqrt(sum((x1_i - x2_i)^2))
        
        Parameters:
        -----------
        test_sample : np.ndarray[tuple[int, int], np.dtype[np.float64]]
            A single test sample of shape (n_features,)
            
        Returns:
        --------
        np.ndarray[tuple[int, int], np.dtype[np.float64]]
            Distances to all training samples of shape (n_train_samples,)
        """
        # Calculate squared differences
        squared_diffs = (self.features_train - test_sample) ** 2

        # Sum across features and take square root
        distances = np.sqrt(np.sum(squared_diffs, axis=1))

        return distances

    def __str__(self) -> str:
        """String representation of the classifier"""
        return f"KNNClassifier(k={self.k})"

    def __repr__(self) -> str:
        """Official string representation of the classifier"""
        return self.__str__()
