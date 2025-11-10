"""
Example Study implementation for hyperparameter tuning.

This is a concrete implementation of the abstract Study class.
Replace this with your specific machine learning model and hyperparameter tuning logic.
"""

#region Imports
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import BaseSampler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

from src.training.hyperparameters_tuning.abstract_study import Study
#endregion


class ExampleStudy(Study):
    """
    Example implementation of Study class using Random Forest.
    
    This is a template implementation - replace with your specific model and logic.
    """
    
    #region Initialization
    def __init__(self, 
                 study_name: str,
                 sampler: BaseSampler,
                 direction: str,
                 optimize_metric: str,
                 n_trials: int,
                 model_params: Optional[Dict[str, Any]] = None,
                 n_jobs: int = 1,
                 cv: int = 5,
                 seed: int = 42) -> None:
        """
        Initialize the ExampleStudy.
        
        Args:
            study_name: Name of the study
            sampler: Optuna sampler for hyperparameter optimization
            direction: Direction of optimization ("maximize" or "minimize")
            optimize_metric: Metric to optimize (e.g., "accuracy", "f1", "roc_auc")
            n_trials: Number of optimization trials
            model_params: Dictionary containing hyperparameter ranges for the model
            n_jobs: Number of parallel jobs
            cv: Number of cross-validation folds
            seed: Random seed for reproducibility
        """
        super().__init__(study_name, sampler, direction, optimize_metric, n_trials, n_jobs, cv, seed)
        self.model_params = model_params or {}
        self.logger = logging.getLogger(__name__)
    #endregion
    
    #region Private Methods
    def _compile_all_metrics(self, study: optuna.Study, best_model_metrics: Dict[str, float], 
                           test_metrics: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """
        Compile all metrics into a single dictionary structure.
        
        Args:
            study: Completed Optuna study
            best_model_metrics: Metrics from best model training
            test_metrics: Test metrics (optional)
            
        Returns:
            Dictionary containing all compiled metrics
        """
        all_metrics = {
            'study_info': {
                'study_name': self.study_name,
                'current_version': self.current_version,
                'n_trials': len(study.trials),
                'best_value': study.best_value,
                'best_params': study.best_params,
                'direction': self.direction,
                'optimize_metric': self.optimize_metric
            },
            'training_metrics': best_model_metrics,
            'test_metrics': test_metrics or {},
            'hyperparameter_importance': {}
        }
        
        # Add hyperparameter importance if available
        try:
            importance = optuna.importance.get_param_importances(study)
            all_metrics['hyperparameter_importance'] = importance
        except Exception as e:
            self.logger.warning(f"Could not calculate parameter importance: {e}")
        
        return all_metrics
    #endregion

    #region Abstract Method Implementations
    def objective(self, trial: optuna.Trial, dataloader: Any) -> float:
        """
        Define the objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object for hyperparameter suggestions
            dataloader: Dataloader instance to get training data
            
        Returns:
            Objective value to optimize (accuracy in this example)
        """
        # Load training data from dataloader
        # DataLoader returns (X, y, groups) for get_train()
        X_train, y_train, groups_train = dataloader.get_train()
        
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': self.seed
        }
        
        # Create and evaluate model
        model = RandomForestClassifier(**params)
        
        # Use cross-validation for robust evaluation
        # Use GroupKFold with groups for group-aware CV
        from sklearn.model_selection import GroupKFold
        cv_splitter = GroupKFold(n_splits=self.cv)
        scores = cross_val_score(
            model, X_train, y_train, 
            groups=groups_train,
            cv=cv_splitter, 
            scoring=self.optimize_metric.replace('_', '_'),  # e.g., 'roc_auc'
            n_jobs=1  # Use 1 to avoid nested parallelism
        )
        
        return scores.mean()
    
    def train_best_model(self, best_params: Dict[str, Any], dataloader) -> Tuple[Any, Dict[str, float]]:
        """
        Train the final model with best hyperparameters.
        
        Args:
            best_params: Best hyperparameters from optimization
            
        Returns:
            Tuple of (trained_model, training_metrics)
        """
        self.logger.info(f"Training best model with params: {best_params}")
        
        # Load training data
        X_train, y_train, groups_train = dataloader.get_train()
        
        # Add fixed parameters
        model_params = {**best_params, 'random_state': self.seed}
        
        # Train final model
        best_model = RandomForestClassifier(**model_params)
        best_model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_predictions = best_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        training_metrics = {
            'train_accuracy': train_accuracy,
            'train_samples': len(X_train),
            'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
        }
        
        self.logger.info(f"Training completed. Accuracy: {train_accuracy:.4f}")
        
        return best_model, training_metrics
    
    def run_test(self, model: Any, dataloader) -> Optional[Dict[str, float]]:
        """
        Run the best model on test data from dataloader.
        
        Args:
            model: Trained model
            dataloader: Dataloader instance to get test data
            
        Returns:
            Test metrics dictionary or None if no test data available
        """
        # Get test data from dataloader - returns (X, y, groups)
        X_test, y_test, groups_test = dataloader.get_test()
        
        # Make predictions
        test_predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Generate detailed classification report
        class_report = classification_report(y_test, test_predictions, output_dict=True)
        
        test_metrics = {
            'test_accuracy': test_accuracy,
            'test_samples': len(X_test),
            'precision_macro': class_report['macro avg']['precision'],
            'recall_macro': class_report['macro avg']['recall'],
            'f1_macro': class_report['macro avg']['f1-score']
        }
        
        self.logger.info(f"Test evaluation completed. Accuracy: {test_accuracy:.4f}")
        
        return test_metrics
    
    def save_metrics(self, study: optuna.Study, best_model_metrics: Dict[str, float], 
                     test_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Save all relevant metrics (hypertuning + test results) to run_results folder.
        
        Args:
            study: Completed Optuna study
            best_model_metrics: Metrics from best model training
            test_metrics: Test metrics (optional)
        """
        all_metrics = self._compile_all_metrics(study, best_model_metrics, test_metrics)
        
        # Save metrics to JSON in run_results directory
        metrics_path = self.run_results_dir.joinpath("study_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        # Save study trials dataframe
        trials_df = study.trials_dataframe()
        trials_path = self.run_results_dir.joinpath("study_trials.csv")
        trials_df.to_csv(trials_path, index=False)
        
        self.logger.info(f"Metrics saved to {metrics_path}")
        self.logger.info(f"Trials data saved to {trials_path}")

    def save_best_model(self, model: Any, model_save_dir: Path) -> None:
        """
        Save the best model to the specified path using pickle.
        
        Args:
            model: Trained model to save
            model_save_dir: Path where to save the model
        """
        model_save_path = model_save_dir.joinpath("best_model.pkl")
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)

        self.logger.info(f"Best model saved to {model_save_path}")
    #endregion

   