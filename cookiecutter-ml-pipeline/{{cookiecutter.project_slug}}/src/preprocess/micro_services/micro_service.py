# region Imports

import logging
import queue
import threading
import pathlib

from abc import ABC, abstractmethod
from tkinter import N
from typing import Tuple, Any, Union

from src.utils.constants import DATA_FOLDER, PREPROCESS, PREPROCESSED_DATA
# endregion



class MicroService(ABC, threading.Thread):
    def __init__(self, feed_output_queue_automatically=True):
        super().__init__()
        self._input_queue = None
        self._output_queue = None
        self.is_running = True
        self.log = logging.getLogger()
        self.feed_output_queue_automatically = feed_output_queue_automatically

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def file_name(self):
        pass

    @property
    def input_queue(self):
        return self._input_queue

    @property
    def output_queue(self):
        return self._output_queue

    @output_queue.setter
    def output_queue(self, queue):
        self._output_queue = queue

    @input_queue.setter
    def input_queue(self, queue):
        self._input_queue = queue

    def run(self):
        """The main loop for processing input and generating output."""
        while self.is_running or not self.input_queue.empty():
            try:
                queue_values = self.input_queue.get(timeout=10)
            except queue.Empty:
                continue

            output_queue_values = None
            try:
                output = self.handle_queue_values(queue_values)
                if self.__class__.__name__ == "FeatureEngineer":
                    results = output
                    save_dir = None
                else:
                    output_queue_values, results, save_dir = output
                self.save_results(results, save_dir)
            except Exception as e:
                self.log.exception(f'Failed to handle queue values {queue_values}: {e}')

            self.input_queue.task_done()
            if (output_queue_values is not None) and self.feed_output_queue_automatically:
                for q in self.output_queue:
                    q.put(output_queue_values)

    @abstractmethod
    def handle_queue_values(self, queue_values):
        pass

    @abstractmethod
    def save_results(self, output_queue_values, save_dir=None):
        pass


class FeatureCalculator(MicroService, ABC):
    """
    Abstract base class for feature calculators.
    
    Implements save_results to enforce saving under data/preprocess/features/{source}/{calculator_name}/
    """
    
    # region Initialization
    def __init__(self, 
                 hierarchy_level: int = 2,
                 input_folder: str = "raw_data",
                 feed_output_queue_automatically: bool = True) -> None:
        """
        Initialize the FeatureCalculator.
        
        Args:
            hierarchy_level: Level in processing hierarchy
            input_folder: Folder to read data from ("raw_data" or "preprocessed")
            feed_output_queue_automatically: Whether to feed output to next service
        """
        super().__init__(feed_output_queue_automatically)
        self.hierarchy_level = hierarchy_level
        self.input_folder = input_folder
        self.logger = logging.getLogger(__name__)
    # endregion
    
    # region Public Methods
    @abstractmethod
    def save_features(self, features: Any, save_dir: str, source: Union[str, Tuple]) -> None:
        """
        Save features with source information.
        
        Args:
            features: Features to save (any type)
            save_dir: Directory path where to save features
            source: Source identifier (can be string or tuple)
        """
        pass

    def save_results(self, results: Tuple[Any, Union[str, Tuple]], save_dir: str = None) -> None:
        """
        Save results to the features directory.
        
        Args:
            results: Tuple of (features, source) where source can be string or tuple
            save_dir: Directory path where to save results
        """
        features, source = results[0], results[1:]
        
        features_save_dir = pathlib.Path(DATA_FOLDER, PREPROCESS, PREPROCESSED_DATA)
        features_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_features(features, str(features_save_dir), source)
    # endregion