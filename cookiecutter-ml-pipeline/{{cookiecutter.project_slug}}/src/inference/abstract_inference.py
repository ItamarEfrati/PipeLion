"""
Abstract base class for inference implementations.

Author: i <ii>
"""

#region Imports
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from src.utils.constants import ASSETS_FOLDER, INFERENCE, RESULTS, RESULTS
#endregion


class InferenceHandler(ABC):
    """
    Abstract base class for inference implementations.
    
    Manages versioned folder structure for model loading while allowing
    concrete implementations to define how to load and use specific model types.
    
    Directory structure expected:
    assets/results/{study_name}/
    ├── {study_name}.db
    ├── ver_1/
    │   ├── best_model.pkl
    │   └── run_results/
    └── ver_2/
        ├── best_model.pkl
        └── run_results/

    will save the inference results under the input version in inference folder.
    """
    
    #region Initialization
    def __init__(self, study_name: str, version: str, inference_source_name: str) -> None:
        """
        Initialize the inference handler with study name and version.
        
        Args:
            study_name: Name of the study (folder name in results)
            version: Version of the study (e.g., "ver_1", "ver_2")
            inference_source_name: Name of the source initiating the inference
        """
        self.model = None
        self.study_name = study_name
        self.version = version
        self.actual_version = None  # Will be set after loading model
        self.logger = logging.getLogger(__name__)
        self.inference_source_name = inference_source_name
        
        # Set up base paths
        self.study_base_dir = Path(ASSETS_FOLDER, RESULTS, study_name)
        self.version_dir = None  # Will be set after determining actual version
        
        # Inference results paths will be set after version is determined
        self.inference_results_dir = None
    #endregion

    #region Abstract Methods
    @abstractmethod
    def _load_model_from_dir(self, model_dir: Path) -> None:
        """
        Load a trained model from the specified directory.
        
        This method must be implemented by all concrete inference classes.
        The implementation decides the specific model filename and loading mechanism.
        
        Args:
            model_dir: Directory containing the model file(s)
        """
        pass
    
    @abstractmethod
    def _perform_inference(self, dataloader: Any) -> None:
        """
        Perform the actual inference logic using the loaded model and dataloader.
        
        This method must be implemented by all concrete inference classes.
        It should focus only on the inference logic, not result saving.
        
        Args:
            dataloader: Dataloader instance with methods to load inference data
        """
        pass

    @abstractmethod
    def save_results(self, results: Dict[str, Union[np.ndarray, pd.DataFrame]]) -> None:
        """
        Save inference results to the standardized folder structure.
        
        This method must be implemented by all concrete inference classes.
        
        Args:
            results: Dictionary containing prediction results
        """
        pass

    #endregion

    #region Public Methods
    def load_model(self) -> None:
        """
        Load a trained model for inference using the versioned folder structure.
        
        Automatically determines the version to use and delegates actual loading to the
        concrete implementation via _load_model_from_dir.
        """
        # Determine which version to use
        self.version_dir = self.study_base_dir.joinpath(self.version)
        self._load_model_from_dir(self.version_dir)

        self.logger.info(f"Loaded model from study: {self.study_name}, version: {self.version}")

    def infer(self, dataloader: Any) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Perform inference and save results using standardized folder structure.
        
        This method enforces the folder structure and delegates actual inference
        to the concrete implementation via _perform_inference.
        
        Args:
            dataloader: Dataloader instance with methods to load inference data
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.logger.info(f"Starting inference for study: {self.study_name}, version: {self.version}")
        
        # Perform the actual inference
        results = self._perform_inference(dataloader)
        self.inference_results_dir = Path(ASSETS_FOLDER, RESULTS, self.study_name, self.version, INFERENCE, self.inference_source_name)

        self.inference_results_dir.mkdir(parents=True, exist_ok=True)
        self.save_results(results, self.inference_results_dir)
        
        self.logger.info(f"Inference completed successfully. Results saved to: {self.inference_results_dir}")
        
    #endregion