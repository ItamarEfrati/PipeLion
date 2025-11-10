"""
Example DataLoader implementation for SimpleFeatureArranger output.

This dataloader is designed to work with the feature arrangement output
that separates features, labels, and metadata into different files.
"""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.constants import DATA_FOLDER, FOR_MODELING_FOLDER, FOR_INFERENCE_FOLDER


class ExampleDataLoader:
    """
    Simplified DataLoader for SimpleFeatureArranger output format.
    
    Loads features, labels, and metadata from separate CSV files
    and provides train/test/predict methods with groups support.
    """
    
    def __init__(self, 
                 version: str = "features_1",
                 is_inference: bool = False,
                 features_file: str = "features.csv",
                 labels_file: str = "labels.csv",
                 metadata_file: str = "metadata.csv",
                 train_split: float = 0.8,
                 test_split: float = 0.2,
                 shuffle: bool = True,
                 seed: int = 42):
        """
        Initialize the ExampleDataLoader.
        
        Args:
            version: Version folder name in data directory (e.g., "features_1")
            is_inference: True for inference mode, False for training mode
            features_file: Name of features file
            labels_file: Name of labels file (only used in training mode)
            metadata_file: Name of metadata file (hierarchy columns)
            train_split: Fraction of data for training (only used in training mode)
            test_split: Fraction of data for testing (only used in training mode)
            shuffle: Whether to shuffle data before splitting (only used in training mode)
            seed: Random seed for reproducibility
        """
        self.version = version
        self.is_inference = is_inference
        
        # Set data directory based on mode
        if self.is_inference:
            self.data_dir = os.path.join(DATA_FOLDER, FOR_INFERENCE_FOLDER)
        else:
            self.data_dir = os.path.join(DATA_FOLDER, FOR_MODELING_FOLDER)
            
        self.features_file = features_file
        self.labels_file = labels_file
        self.metadata_file = metadata_file
        self.train_split = train_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.seed = seed
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.features = None
        self.labels = None
        self.metadata = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.groups_train = None
        self.groups_test = None
        
        # Load data automatically
        self._load_data()
        
        # Create splits only in training mode
        if not self.is_inference:
            self._create_splits()
    
    def _load_data(self) -> None:
        """Load features, labels, and metadata from files."""
        # Construct full paths
        version_dir = os.path.join(self.data_dir, self.version)
        features_path = os.path.join(version_dir, self.features_file)
        metadata_path = os.path.join(version_dir, self.metadata_file)
        
        # Check if required files exist
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load features and metadata (always required)
        self.features = pd.read_csv(features_path)
        self.metadata = pd.read_csv(metadata_path)
        
        # Load labels only in training mode
        if not self.is_inference:
            labels_path = os.path.join(version_dir, self.labels_file)
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Labels file not found: {labels_path}")
            self.labels = pd.read_csv(labels_path)
            
            # Validate sample counts in training mode
            n_features = len(self.features)
            n_labels = len(self.labels)
            n_metadata = len(self.metadata)
            
            if not (n_features == n_labels == n_metadata):
                raise ValueError(f"Inconsistent sample counts: features({n_features}), labels({n_labels}), metadata({n_metadata})")
            
            self.logger.info(f"Loaded {n_features} samples with {self.features.shape[1]} features for training")
        else:
            # Inference mode - no labels
            self.labels = None
            
            # Validate features and metadata alignment
            n_features = len(self.features)
            n_metadata = len(self.metadata)
            
            if n_features != n_metadata:
                raise ValueError(f"Inconsistent sample counts: features({n_features}), metadata({n_metadata})")
            
            self.logger.info(f"Loaded {n_features} samples with {self.features.shape[1]} features for inference")
    
    def _create_splits(self) -> None:
        """Create train/test splits while maintaining sample alignment."""
        n_samples = len(self.features)
        
        # Create indices
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Calculate split points
        train_size = int(n_samples * self.train_split)
        
        # Split indices
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # Create splits for features and labels
        self.X_train = self.features.iloc[train_indices].values
        self.X_test = self.features.iloc[test_indices].values
        self.y_train = self.labels.iloc[train_indices].iloc[:, 0].values  # First column as target
        self.y_test = self.labels.iloc[test_indices].iloc[:, 0].values
        
        # Create groups from metadata (combine source and hierarchy_1 for grouping)
        groups = self.metadata['source'] + '_' + self.metadata['hierarchy_1'].astype(str)
        self.groups_train = groups.iloc[train_indices].values
        self.groups_test = groups.iloc[test_indices].values
        
        self.logger.info(f"Created splits: {len(self.X_train)} train, {len(self.X_test)} test samples")
    
    def get_train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training data.
        
        Returns:
            Tuple of (X_train, y_train, groups_train)
            
        Raises:
            ValueError: If called in inference mode or data not available
        """
        if self.is_inference:
            raise ValueError("get_train() is only available in training mode")
            
        if self.X_train is None or self.y_train is None or self.groups_train is None:
            raise ValueError("Training data not available. Ensure data loading was successful.")
        
        return self.X_train, self.y_train, self.groups_train
    
    def get_test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get test data.
        
        Returns:
            Tuple of (X_test, y_test, groups_test)
            
        Raises:
            ValueError: If called in inference mode or data not available
        """
        if self.is_inference:
            raise ValueError("get_test() is only available in training mode")
            
        if self.X_test is None or self.y_test is None or self.groups_test is None:
            raise ValueError("Test data not available. Ensure data loading was successful.")
        
        return self.X_test, self.y_test, self.groups_test
    
    def get_predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all data for prediction (no labels).
        Available in both training and inference modes.
        
        Returns:
            Tuple of (X, groups) for all data
        """
        if self.features is None or self.metadata is None:
            raise ValueError("Data not available. Ensure data loading was successful.")
        
        X = self.features.values
        groups = self.metadata.astype(str).apply(lambda row: '_'.join(row), axis=1).values        
        return X, groups
    
    def get_feature_names(self) -> list:
        """
        Get feature column names.
        
        Returns:
            List of feature names
        """
        if self.features is None:
            raise ValueError("Features not available. Ensure data loading was successful.")
        
        return list(self.features.columns)