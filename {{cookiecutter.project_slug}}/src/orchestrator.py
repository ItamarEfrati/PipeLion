"""
Pipeline orchestrator for data processing workflows.
Handles the complete workflow from raw data to model inference.
"""

import logging
import pathlib

from hydra.utils import instantiate
from omegaconf import DictConfig

from src.training.hyperparameters_tuning.abstract_study import Study
from src.inference.abstract_inference import InferenceHandler
from src.utils.constants import DATA_FOLDER, PREPROCESS, OUTCOMES

class Orchestrator:
    """
    Robust pipeline orchestrator for data processing.
    Handles the complete workflow from raw data to model inference.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        

    # region Validation
    
    def validate_raw_data(self) -> bool:
        """Validate that raw data exists and is properly structured."""
        validation_success = False
        
        try:
            raw_data_path = pathlib.Path(DATA_FOLDER, 'raw_data')
            if not raw_data_path.exists():
                self.logger.error(f"Raw data directory not found: {raw_data_path}")
            elif not any(raw_data_path.rglob('*')):
                self.logger.error("No files found in raw data directory")
            else:
                self.logger.info(f"Raw data validation passed. Found data in: {raw_data_path}")
                validation_success = True
        except Exception as e:
            self.logger.error(f"Error validating raw data: {e}")
            
        return validation_success

    def validate_preprocessed_data(self) -> bool:
        """Validate that processed data exists and is ready for training."""
        validation_success = False
        
        try:
            dataset_type = getattr(self.cfg.dataloader, 'pipeline_type', 'unknown')
            processed_path = pathlib.Path(DATA_FOLDER, PREPROCESS, f"{OUTCOMES}_{dataset_type}")
            
            if not processed_path.exists():
                self.logger.error(f"Processed data directory not found: {processed_path}")
            else:
                # Check for features and labels
                features_path = processed_path / 'features'
                if not features_path.exists():
                    self.logger.error(f"Features directory not found: {features_path}")
                else:
                    features_file = features_path / 'features.csv'
                    labels_file = features_path / 'labels.csv'
                    
                    if not features_file.exists():
                        self.logger.error(f"Features file not found: {features_file}")
                    elif not labels_file.exists():
                        self.logger.error(f"Labels file not found: {labels_file}")
                    else:
                        self.logger.info("Processed data validation passed")
                        validation_success = True
        except Exception as e:
            self.logger.error(f"Error validating processed data: {e}")
            
        return validation_success

    # endregion

    def run_preprocessing(self) -> None:
        """Execute the data processing pipeline."""
        try:
            self.logger.info("Starting data processing pipeline...")
            
            # Create DataProcessor with the pipeline configuration
            data_processor = instantiate(self.cfg.preprocess.data_processor)
            data_processor = data_processor(pipeline_config=self.cfg.preprocess)
            data_processor.run()
            
            self.logger.info("Data processing pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            raise

    def run_hyperparameter_tuning(self) -> None:
        """Execute hyperparameter tuning and model training."""
        try:
            self.logger.info("Starting hyperparameter tuning...")
            
            dataloader = instantiate(self.cfg.dataloader)
            study: Study = instantiate(self.cfg.study)
            study.run_study(dataloader)
            
            self.logger.info("Hyperparameter tuning completed successfully")
        except Exception as e:
            self.logger.error(f"Hyperparameter tuning failed: {e}")
            raise

    def run_inference(self) -> None:
        """
        Run inference pipeline with dataloader and inference handler.
        """
        try:
            self.logger.info("Starting inference...")
            
            # Use dataloader from config
            dataloader = instantiate(self.cfg.dataloader)
            
            # Instantiate inference handler from config
            inference_handler: InferenceHandler = instantiate(self.cfg.inference)
            
            # Load model through the inference handler
            inference_handler.load_model()
            
            # Process inference with dataloader (model is stored in handler)
            inference_handler.infer(dataloader=dataloader)
            
            self.logger.info("Inference completed successfully")
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise