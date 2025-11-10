"""
Example InferenceHandler implementation.

This is a concrete implementation of the abstract InferenceHandler class.
Replace this with your specific inference logic.
"""

#region Imports

import json
import logging
from os import pread
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from src.inference.abstract_inference import InferenceHandler

#endregion


class ExampleInferenceHandler(InferenceHandler):
    """
    Example implementation of InferenceHandler for generic ML models.
    
    This is a template implementation - replace with your specific inference logic.
    Uses pickle to load models from the versioned study structure.
    """
    
    #region Initialization
    def __init__(self, 
                 study_name: str,
                 version: str,
                 inference_source_name: str,
                 model_filename: str = "best_model.pkl") -> None:
        """
        Initialize the ExampleInferenceHandler.
        
        Args:
            study_name: Name of the study
            version: Version of the study (e.g., "ver_1", "ver_2")
            model_filename: Name of the model file to load (default: "best_model.pkl")
        """
        super().__init__(study_name, version, inference_source_name)
        self.model_filename = model_filename
        self.logger = logging.getLogger(__name__)
    #endregion


    #region Abstract Method Implementations
    def _load_model_from_dir(self, model_dir: Path) -> None:
        """
        Load a trained model from the specified directory using pickle.
        
        Args:
            model_dir: Directory containing the model file
            
        Raises:
            FileNotFoundError: If model file cannot be found
            Exception: If model loading fails
        """
        model_path = model_dir.joinpath(self.model_filename)
        
        try:
            self.logger.info(f"Loading model from: {model_path}")
            self.logger.info(f"Study: {self.study_name}, Version: {self.version}")
            
            if not model_path.exists():
                available_files = [f.name for f in model_dir.iterdir() if f.is_file()]
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    f"Available files in {model_dir}: {available_files}"
                )
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.logger.info("Model loaded successfully")
            
        except FileNotFoundError:
            self.logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _perform_inference(self, dataloader: Any) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Perform inference using the loaded sklearn model and dataloader.
        
        Args:
            dataloader: Dataloader instance with methods to load inference data
            
        Returns:
            Dictionary containing prediction results
        """
        self.logger.info("Starting inference logic...")
        
        # Load inference data from dataloader
        # ExampleDataLoader has get_predict() method that returns (X, groups)
        try:
            X_inference, groups = dataloader.get_predict()
            self.logger.info(f"Loaded {len(X_inference)} samples for inference")
                
        except Exception as e:
            self.logger.error(f"Failed to load inference data: {e}")
            raise
        
        # Perform inference
        self.logger.info(f"Running inference on {len(X_inference)} samples...")
        
        # Get predictions from sklearn model
        predictions = self.model.predict(X_inference)
        self.logger.info("Generated predictions")
        
        # Get probabilities from sklearn model
        probabilities = self.model.predict_proba(X_inference)
        self.logger.info("Calculated prediction probabilities")
        
        self.logger.info(f"Inference logic completed successfully. Generated {len(predictions)} predictions.")

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'groups': groups
        }

    def save_results(self, results: Dict, version_dir: Path) -> None:
                
        """
        Save inference results to the standardized folder structure.    

        """
        predictions = results['predictions']
        probabilities = results['probabilities']
        groups = results['groups']

        results_df = {}
        results_df['sample_id'] = range(len(predictions))
        results_df['group'] = groups
        
        # Add predictions
        results_df['prediction'] = predictions
        
        # Add probabilities if available
        if probabilities is not None:
            for i, class_prob in enumerate(probabilities.T):
                results_df[f'probability_class_{i}'] = class_prob
        
        # Create DataFrame for saving
        results_df = pd.DataFrame(results_df)
        
        # Create output directory structure
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results as CSV
        predictions_path = version_dir.joinpath("predictions.csv")
        results_df.to_csv(predictions_path, index=False)

    # endregion
