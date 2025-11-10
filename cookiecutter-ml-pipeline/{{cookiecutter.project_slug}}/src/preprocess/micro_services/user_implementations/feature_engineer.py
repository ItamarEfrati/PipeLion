# region Imports
"""
Feature engineering microservice for preprocessing pipeline.

This microservice reads statistics artifacts and generates random features.
"""

import os
import logging
import json
from typing import Tuple, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np

from src.preprocess.micro_services.micro_service import FeatureCalculator
from src.utils.constants import ARTIFACTS, DATA_FOLDER, PREPROCESS, OUTCOMES
# endregion


class FeatureEngineer(FeatureCalculator):
    """
    Performs feature engineering based on statistics artifacts.
    
    This microservice reads statistics artifacts and generates 
    random features based on the calculated statistics.
    """
    
    # region Initialization
    def __init__(self, 
                 hierarchy_level: int = 2,
                 input_folder: str = "raw_data",
                 feed_output_queue_automatically: bool = True,
                 n_random_features: int = 10):
        """
        Initialize the FeatureEngineer.
        
        Args:
            hierarchy_level: Level in processing hierarchy
            input_folder: Folder to read data from ("raw_data" or "preprocessed")
            feed_output_queue_automatically: Whether to automatically feed output to next service
            n_random_features: Number of random features to generate
        """
        super().__init__(hierarchy_level, input_folder, feed_output_queue_automatically)
        self.n_random_features = n_random_features
        self.logger = logging.getLogger(__name__)
    # endregion
    
    # region Properties
    @property
    def name(self) -> str:
        """Return the name of this microservice."""
        return 'feature_engineer'
    
    @property
    def file_name(self) -> str:
        """Return the filename pattern for saving results."""
        return '{}_{}_engineered_features.csv'
    # endregion
    
    # region Private Methods
    def _engineer_features_from_statistics(self, source: str, hierarchy_1: str, hierarchy_2: str, statistics: dict) -> pd.DataFrame:
        """
        Generate random features based on statistics artifacts.
        
        Args:
            source: Source identifier
            hierarchy_1: First hierarchy level identifier  
            hierarchy_2: Second hierarchy level identifier
            statistics: Statistics dictionary
            
        Returns:
            DataFrame with generated features
        """
        features_dict = {}
        
        # Generate random features
        np.random.seed(hash(f"{source}_{hierarchy_1}_{hierarchy_2}") % 2**32)
        
        for i in range(self.n_random_features):
            feature_name = f"random_feature_{i+1}"
            features_dict[feature_name] = np.random.normal(0, 1)
        
        return pd.DataFrame([features_dict])
    
    def _load_statistics_artifact(self, source: str, hierarchy_1: str, hierarchy_2: str) -> dict:
        """
        Load statistics artifact for the given hierarchy.
        
        Args:
            source: Source identifier
            hierarchy_1: First hierarchy level identifier
            hierarchy_2: Second hierarchy level identifier
            
        Returns:
            Dictionary containing statistics or None if not found
        """
        artifacts_dir = Path(DATA_FOLDER, PREPROCESS, f"{OUTCOMES}_{source}", hierarchy_1, hierarchy_2, ARTIFACTS)
        stats_filename = f"statistics.json"
        stats_path = os.path.join(artifacts_dir, stats_filename)    
        
        with open(stats_path, 'r') as f:
            statistics = json.load(f)
        
        return statistics
    # endregion

    # region Public Methods
    def handle_queue_values(self, queue_values: Tuple) -> Tuple[Tuple, Any, str]:
        """
        Handle input queue values and perform feature engineering.
        
        Args:
            queue_values: Tuple containing hierarchy information + save_dir
            
        Returns:
            Tuple of (features, source) where source can be string or tuple
        """
        (source, hierarchy_1, hierarchy_2) = queue_values
        
        # Load statistics artifact
        statistics = self._load_statistics_artifact(source, hierarchy_1, hierarchy_2)
        
        # Generate features
        engineered_features = self._engineer_features_from_statistics(source, hierarchy_1, hierarchy_2, statistics)

        results = (engineered_features, source, hierarchy_1, hierarchy_2)
        return results
    
    def save_features(self, features: pd.DataFrame, save_dir: str, source: Union[str, Tuple]) -> None:
        """
        Save engineered features to the features directory.
        
        Args:
            features: DataFrame with engineered features
            save_dir: Features directory path
            source: Source identifier (can be string or tuple)
        """
        (source, hierarchy_1, hierarchy_2) = source

        source_dir = os.path.join(save_dir, source, self.name)
        os.makedirs(source_dir, exist_ok=True)

        output_path = os.path.join(source_dir, self.file_name.format(hierarchy_1, hierarchy_2))
        features.to_csv(output_path, index=False)
    # endregion
    
 