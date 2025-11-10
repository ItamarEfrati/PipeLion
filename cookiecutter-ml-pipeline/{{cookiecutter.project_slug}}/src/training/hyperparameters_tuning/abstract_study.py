"""
aa

Author: i <ii>
"""

#region Imports

from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import optuna
from optuna.samplers import BaseSampler

from src.utils.constants import *
#endregion


class Study(ABC):
    """
    Abstract base class for hyperparameter tuning studies.
    
    Manages folder structure and versioning for study runs while allowing
    concrete implementations to define study-specific content and logic.
    
    Directory structure created:
    assets/results/{study_name}/
    ├── {study_name}.db                    # Shared Optuna database
    ├── ver_1/
    │   ├── best_model.pkl                 # Model saved directly here
    │   └── run_results/                   # Metrics folder
    └── ver_2/
        ├── best_model.pkl
        └── run_results/
    """

    #region Initialization
    def __init__(self,
                 study_name: str,
                 sampler: BaseSampler,
                 direction: str,
                 optimize_metric: str,
                 n_trials: int,
                 n_jobs: int = 1,
                 cv: int = 5,
                 seed: int = 42) -> None:
        """
        Initialize the Study with versioned folder structure.
        
        Args:
            study_name: Name of the study (used for folder and database naming)
            sampler: Optuna sampler for hyperparameter optimization
            direction: Direction of optimization ("maximize" or "minimize")
            optimize_metric: Metric to optimize (e.g., "accuracy", "f1", "roc_auc")
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            cv: Number of cross-validation folds
            seed: Random seed for reproducibility
        """
        self.study_name = study_name
        self.sampler = sampler
        self.direction = direction
        self.optimize_metric = optimize_metric
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.cv = cv
        self.seed = seed
        
        # Create base study directory structure
        self.study_base_dir = Path(ASSETS_FOLDER, "results", study_name)
        self.study_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Set current version and create version directory
        self.current_version = self._get_next_version()
        self.current_version_dir = self.study_base_dir.joinpath(self.current_version)
        self._setup_version_structure()
    #endregion

    #region Private Methods
    def _get_next_version(self) -> str:
        """
        Determine the next version number by checking existing version folders.
        
        Returns:
            Version string like 'ver_1', 'ver_2', etc.
        """
        existing_versions = [d.name for d in self.study_base_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('ver_')]
        
        if not existing_versions:
            return 'ver_1'
        
        # Extract version numbers and find the maximum
        version_numbers = []
        for version in existing_versions:
            try:
                num = int(version.split('_')[1])
                version_numbers.append(num)
            except (IndexError, ValueError):
                continue
        
        next_version = max(version_numbers) + 1 if version_numbers else 1
        return f'ver_{next_version}'
    
    def _setup_version_structure(self) -> None:
        """
        Create the folder structure for the current version:
        - ver_X/
        - ver_X/run_results/
        """
        self.current_version_dir.mkdir(parents=True, exist_ok=True)
        self.run_results_dir = self.current_version_dir.joinpath("run_results")
        self.run_results_dir.mkdir(parents=True, exist_ok=True)

    def _run_hyperparameter_tuning(self, dataloader: Any) -> optuna.Study:
        """
        Run the Optuna hyperparameter optimization.
        
        Args:
            dataloader: Dataloader instance for training data
            
        Returns:
            Completed Optuna study object
        """
        storage_path = self.study_base_dir.joinpath(f"{self.study_name}.db")
        storage_name = f"sqlite:///{storage_path}"
        
        study = optuna.create_study(
            study_name=self.current_version,
            sampler=self.sampler,
            storage=storage_name,
            pruner=optuna.pruners.MedianPruner(),
            direction=self.direction,
            load_if_exists=True
        )
        
        # Pass dataloader to objective function
        objective_with_dataloader = partial(self.objective, dataloader=dataloader)
        study.optimize(objective_with_dataloader, n_trials=self.n_trials, n_jobs=self.n_jobs)
        
        return study
    #endregion

    #region Abstract Methods
    @abstractmethod
    def objective(self, trial: optuna.Trial, dataloader: Any) -> float:
        """
        Define the objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object for hyperparameter suggestions
            dataloader: Dataloader instance to get training data
            
        Returns:
            Objective value to optimize
        """
        pass

    @abstractmethod
    def train_best_model(self, best_params: Dict[str, Any], dataloader: Any) -> Tuple[Any, Dict[str, float]]:
        """
        Train the final model with best hyperparameters.
        
        Args:
            best_params: Best hyperparameters from optimization
            dataloader: Dataloader instance to get training data
            
        Returns:
            Tuple of (trained_model, training_metrics)
        """
        pass

    @abstractmethod
    def run_test(self, model: Any, dataloader: Any) -> Optional[Dict[str, float]]:
        """
        Run the best model on test data from dataloader.
        
        Args:
            model: Trained model
            dataloader: Dataloader instance to get test data
            
        Returns:
            Test metrics dictionary or None if no test data available
        """
        pass

    @abstractmethod
    def save_metrics(self, study: optuna.Study, best_model_metrics: Dict[str, float], 
                    test_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Save all relevant metrics (hypertuning + test results).
        
        Args:
            study: Completed Optuna study
            best_model_metrics: Metrics from best model training
            test_metrics: Test metrics (optional)
        """
        pass
    
    @abstractmethod
    def save_best_model(self, model: Any, model_save_dir: Path) -> None:
        """
        Save the best model to the specified path.
        
        Args:
            model: Trained model to save
            model_save_dir: Path where to save the model
        """
        pass
    #endregion

    #region Public Methods

    def run_study(self, dataloader: Any) -> Tuple[optuna.Study, Any]:
        """
        Execute the complete study workflow.
        
        1. Load data (via dataloader passed to objective)
        2. Hyperparameter tuning
        3. Choose best model to save
        4. Run test data over best model
        5. Save relevant metrics
        
        Args:
            dataloader: Dataloader instance for training and test data
            
        Returns:
            Tuple of (completed_study, best_model)
        """
        # 1. & 2. Hyperparameter tuning (dataloader used within objective)
        study = self._run_hyperparameter_tuning(dataloader)
        
        # 3. Choose best model and train it
        best_model, best_model_metrics = self.train_best_model(study.best_params, dataloader)
        
        self.save_best_model(best_model, self.current_version_dir)
        
        # 4. Run test data over best model (let implementation decide if test data exists)
        test_metrics = self.run_test(best_model, dataloader)
        
        # 5. Save relevant metrics (implementation decides what to save)
        self.save_metrics(study, best_model_metrics, test_metrics)
        
        return study, best_model
    #endregion