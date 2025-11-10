# region Imports
"""
Statistics Calculator microservice for preprocessing pipeline.

This microservice reads individual data files, calculates statistics,
and saves artifacts to outcomes_{source} folder.
"""

import os
import logging
import json
from typing import Tuple, Any
from pathlib import Path

import pandas as pd
import numpy as np

from src.preprocess.micro_services.micro_service import MicroService
from src.utils.constants import ARTIFACTS, DATA_FOLDER, PREPROCESS, OUTCOMES, RAW_DATA
# endregion


class StatisticsCalculator(MicroService):
    """
    Calculates statistics from individual data files.
    
    This microservice:
    - Reads data.csv files from raw_data hierarchy
    - Calculates descriptive statistics
    - Saves artifacts to outcomes_{source} folder
    """
    
    # region Initialization
    def __init__(self, 
                 hierarchy_level: int = 2,
                 input_folder: str = "raw_data",
                 feed_output_queue_automatically: bool = True):
        """
        Initialize the StatisticsCalculator.
        
        Args:
            hierarchy_level: Level in processing hierarchy
            input_folder: Folder to read data from ("raw_data" or "preprocessed")
            feed_output_queue_automatically: Whether to automatically feed output to next service
        """
        super().__init__(feed_output_queue_automatically)
        self.hierarchy_level = hierarchy_level
        self.input_folder = input_folder
        self.logger = logging.getLogger(__name__)
    # endregion
    
    # region Properties
    @property
    def name(self) -> str:
        """Return the name of this microservice."""
        return 'statistics_calculator'
    
    @property
    def file_name(self) -> str:
        """Return the filename pattern for saving results."""
        return 'statistics.json'
    # endregion


    # region Private Methods
    def _calculate_statistics(self, source: str, hierarchy_1: str, hierarchy_2: str) -> dict:
        """
        Calculate statistics from the data file.
        
        Args:
            source: Source identifier
            hierarchy_1: First hierarchy level identifier
            hierarchy_2: Second hierarchy level identifier
            
        Returns:
            Dictionary containing calculated statistics
        """
        data_path = Path(DATA_FOLDER, RAW_DATA, source, hierarchy_1, hierarchy_2, "data.csv")
        
        df = pd.read_csv(data_path)
        
        statistics = {
            'source': source,
            'hierarchy_1': hierarchy_1,
            'hierarchy_2': hierarchy_2,
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            summary_features = {
                'mean_of_means': float(df[numeric_cols].mean().mean()),
                'std_of_means': float(df[numeric_cols].mean().std()),
                'mean_of_stds': float(df[numeric_cols].std().mean()),
                'overall_range': float(df[numeric_cols].max().max() - df[numeric_cols].min().min()),
                'correlation_matrix_trace': float(np.trace(df[numeric_cols].corr().fillna(0))),
                'total_variance': float(df[numeric_cols].var().sum())
            }
            statistics['summary_features'] = summary_features
        
        return statistics
    # endregion

    # region Public Methods
    def handle_queue_values(self, queue_values: Tuple) -> Tuple[Tuple, Any, str]:
        """
        Handle input queue values and calculate statistics.
        
        Args:
            queue_values: Tuple containing hierarchy information + save_dir
            
        Returns:
            Tuple of (output_queue_values, statistics_data, save_dir)
        """
        (source, hierarchy_1, hierarchy_2) = queue_values
        
        statistics_data = self._calculate_statistics(source, hierarchy_1, hierarchy_2)
        save_dir = os.path.join(DATA_FOLDER, PREPROCESS, f"{OUTCOMES}_{source}", hierarchy_1, hierarchy_2, ARTIFACTS)
        
        return queue_values, statistics_data, save_dir
    
    def save_results(self, results: Any, save_dir: str) -> None:
        """
        Save statistics to the outcomes_{source} directory.
        
        Args:
            results: Statistics data dictionary
            save_dir: Directory where to save results (will be modified to outcomes folder)
        """
        
        os.makedirs(save_dir, exist_ok=True)

        stats_filename = f"statistics.json"
        stats_path = os.path.join(save_dir, stats_filename)

        with open(stats_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    # endregion
    
    