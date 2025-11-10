"""
Abstract base class for feature arrangement with different implementations.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

import pandas as pd


class AbstractFeatureArranger(ABC):
    """
    Abstract base class for feature arrangement implementations.
    Concrete implementations must define how to parse filenames and handle join columns.
    """
    
    def __init__(self, 
                 feature_types: List[str],
                 label_file_path: Optional[str] = None,
                 description: Optional[str] = None,
                 n_workers: int = 4):
        """
        Initialize abstract feature arranger.
        
        Args:
            feature_types: List of feature types to concatenate
            label_file_path: Path to CSV file with labels
            description: Optional user description for the dataset
            n_workers: Number of workers for parallel processing
        """
        self.feature_types = feature_types
        self.label_file_path = label_file_path
        self.description = description or "Feature arrangement output"
        self.n_workers = n_workers
        
    # region Abstract Methods
    
    @abstractmethod
    def _create_output_directory(self, is_inference: bool = False) -> str:
        """Create output directory with appropriate naming strategy."""
        pass
    
    @abstractmethod
    def _load_labels(self) -> pd.DataFrame:
        """Load labels from CSV file."""
        pass
    
    @abstractmethod
    def _join_features_labels(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform inner join between features and labels."""
        pass
    
    @abstractmethod
    def _get_features_without_labels(self, features_df: pd.DataFrame, features_with_labels: pd.DataFrame) -> pd.DataFrame:
        """Get features that don't have corresponding labels."""
        pass
    
    @abstractmethod
    def _save_features_and_metadata(self, features_df: pd.DataFrame, output_dir: str, suffix: str = "", is_inference: bool = False):
        """Save features and metadata files with optional suffix."""
        pass
    
    @abstractmethod
    def _save_labels(self, labels_df: pd.DataFrame, output_dir: str):
        """Save labels separately."""
        pass
    
    @abstractmethod
    def combine_features_from_sources(self) -> pd.DataFrame:
        """
        Combine feature files from sources and feature types.
        
        Returns:
            DataFrame with combined features and standardized join columns
        """
        pass
    
    # endregion
    
    # region Concrete Public Methods
    
    def run(self) -> str:
        """
        Main orchestration method that combines features from sources.
        
        Returns:
            Output directory path with the arranged features
        """
        # Determine if this is for inference (no labels provided)
        is_inference = self.label_file_path is None
        
        # Combine features from all sources and feature types
        features_df = self.combine_features_from_sources()
        
        # Create output directory based on scenario
        output_dir = self._create_output_directory(is_inference)
        
        if is_inference:
            # Inference scenario: save only features and metadata
            self._save_features_and_metadata(features_df, output_dir, is_inference=True)
            print(f"Saved inference data")
        else:
            # Training scenario: load labels and create splits
            labels_df = self._load_labels()
            
            # Create features with labels (inner join)
            features_with_labels, labels_matched = self._join_features_labels(features_df, labels_df)
            
            # Create features without labels
            features_without_labels = self._get_features_without_labels(features_df, features_with_labels)
            
            # Save files
            self._save_features_and_metadata(features_with_labels, output_dir)
            self._save_labels(labels_matched, output_dir)
            
            if len(features_without_labels) > 0:
                self._save_features_and_metadata(features_without_labels, output_dir, "_without_labels")
            
            print(f"Saved training data with {len(features_with_labels)} labeled samples")
        
        return output_dir
    
    # endregion