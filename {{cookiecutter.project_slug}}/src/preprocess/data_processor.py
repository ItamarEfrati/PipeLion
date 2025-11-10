import logging
import os
import pathlib
import queue
import time
from collections import defaultdict
from typing import List

import omegaconf
from hydra.utils import instantiate

from src.preprocess.feature_arrangement.abstract_feature_arranger import AbstractFeatureArranger
from src.utils.constants import *


class DataProcessor:

    def __init__(self,
                 pipeline_config,
                 source_list: List[str],
                 feature_arranger: AbstractFeatureArranger = None,
                 source_sample: str = None):
        self.feature_arranger = feature_arranger
        self.source_list = source_list
        self.source_sample = source_sample  # Specific sample within a source for debugging
        
        # Validation: source_sample requires exactly one source
        if source_sample and len(source_list) != 1:
            raise ValueError(f"source_sample '{source_sample}' requires exactly one source in source_list, got {len(source_list)}: {source_list}")
        
        # Input directory is determined dynamically based on source and processing needs
        self.input_directory = None  # Will be set dynamically when needed
        self.output_queues = defaultdict(lambda: [queue.Queue()])
        self.input_queues = defaultdict(lambda: [queue.Queue()])
        self.services = self.initiate_mini_services(pipeline_config)
        self.run_preprocess = self.services is None or len(self.services) != 0

        if self.run_preprocess:
            # Step 2: Get hierarchy from first service
            self.hierarchy = self._get_first_service_hierarchy()
            
            # Step 3: Set first service input folder (raw or outcomes)
            self.first_service_input_folder = self._get_first_service_input_folder()

            # Input directory validation will be done dynamically when needed

            self.init_process_folder()
        self.log = logging.getLogger()

#   region Arrange Features

    def run_feature_arrangement(self):
        """Run feature arrangement as the final step of preprocessing using the provided FeatureArranger."""        
        try:
            self.log.info(f"Starting feature arrangement for sources: {self.source_list}")
            
            # Run the feature arrangement using the provided arranger
            output_dir = self.feature_arranger.run()
            
            self.log.info(f"Feature arrangement completed successfully. Output saved to: {output_dir}")
            
        except Exception as e:
            self.log.error(f"Feature arrangement failed: {e}")
            raise

    # endregion
    
    # region Init

    def initiate_mini_services(self, service_config):
        """Initialize microservices for preprocessing pipeline."""
        services_dict = defaultdict(list)
        for micro_service, service_kwargs in service_config.items():
            if not isinstance(service_kwargs, omegaconf.dictconfig.DictConfig):
                continue
            if not any(item.endswith('threads') for item in service_kwargs.keys()):
                continue
            for key in service_kwargs.keys():
                if not key.endswith('_n_threads'):
                    break
            for service in list(filter(lambda x: x.endswith('n_threads'), service_kwargs.keys())):
                service_name = service[:-len('_n_threads')]
                n_threads = service_kwargs[f'{service_name}_n_threads']
                services = []
                for i in range(n_threads):
                   services.append(instantiate(service_kwargs[service_name]))
                services_dict[micro_service].append(services)
            services_dict[micro_service] += [service_kwargs['share_input']]
        return services_dict

    def _get_first_service_hierarchy(self) -> int:
        """Step 2: Extract hierarchy level from first service"""
        if not self.services:
            raise ValueError("No services configured")
        
        first_service_name = list(self.services.keys())[0]
        
        # Get the first microservice instance to check its hierarchy requirement
        first_service_threads = self.services[first_service_name][0]  # Get first thread group
        if first_service_threads:
            first_service_instance = first_service_threads[0]  # Get first instance
            
            # Check if service has hierarchy attribute
            if hasattr(first_service_instance, 'hierarchy_level'):
                return first_service_instance.hierarchy_level
        
        # If no hierarchy attribute found, raise error
        raise ValueError(f"First service '{first_service_name}' does not have 'hierarchy_level' attribute. "
                        f"Service class: {first_service_instance.__class__.__name__ if first_service_threads else 'Unknown'}")
    
    def _get_first_service_input_folder(self) -> str:
        """Step 3: Determine if first service reads from raw or outcomes"""
        if not self.services:
            raise ValueError("No services configured")
        
        first_service_name = list(self.services.keys())[0]
        
        # Get the first microservice instance to check its input folder requirement
        first_service_threads = self.services[first_service_name][0]  # Get first thread group
        if first_service_threads:
            first_service_instance = first_service_threads[0]  # Get first instance
            
            # Check if service has input_folder attribute
            if hasattr(first_service_instance, 'input_folder'):
                return first_service_instance.input_folder
        
        # If no input_folder attribute found, raise error
        raise ValueError(f"First service '{first_service_name}' does not have 'input_folder' attribute. "
                        f"Service class: {first_service_instance.__class__.__name__ if first_service_threads else 'Unknown'}")

    def init_process_folder(self):
        # Create folders for each source
        for source in self.source_list:
            source_folder = OUTCOMES + '_' + source
            pathlib.Path(DATA_FOLDER, PREPROCESS, source_folder).mkdir(parents=True, exist_ok=True)

    def _set_queue_structure(self):
        for pipeline_level, (service_name, service_threads) in enumerate(self.services.items()):
            share_input = service_threads[-1]
            input_quque = [queue.Queue() for _ in range(
                len(service_threads[:-1]))] if not share_input else [queue.Queue()]
            self.input_queues[pipeline_level] = input_quque
            if pipeline_level == 0:
                continue
            self.output_queues[pipeline_level - 1] = input_quque

    def init_threads(self):
        self._set_queue_structure()
        for pipeline_level, (service_name, service_threads) in enumerate(self.services.items()):
            input_queues = self.input_queues[pipeline_level]
            for j, threads in enumerate(service_threads[:-1]):
                for single_thread in threads:
                    single_thread.input_queue = input_queues[j]
                    single_thread.output_queue = self.output_queues[pipeline_level]
                    single_thread.start()

        # Determine input directory dynamically based on sources
        self._feed_inputs_from_sources(self.input_queues[0])

    def _feed_inputs_from_sources(self, input_queue):
        """Feed inputs from sources - iterate directly over sources."""
        if self.source_sample:
            self._feed_single_sample_debug(input_queue)
        else:
            self._feed_all_sources(input_queue)

    def _feed_single_sample_debug(self, input_queue):
        """Debug mode: process only a specific sample by constructing hierarchy directly."""
        self.log.info(f"Debug mode: processing sample '{self.source_sample}' at hierarchy level {self.hierarchy}")
        
        if len(self.source_list) != 1:
            raise ValueError(f"Debug mode requires exactly one source, got {len(self.source_list)}")
        
        source = self.source_list[0]
        
        # Parse source_sample to extract hierarchy components
        # Assuming source_sample format like "chip_id" or "chip_id_pen_id" etc.
        sample_components = self.source_sample.split('_')
        
        # Construct full hierarchy tuple: (source, component1, component2, ...)
        # Take only the number of components needed for the hierarchy level
        hierarchy_components = sample_components[:self.hierarchy]
        full_hierarchy = [source] + hierarchy_components
        
        # Validate we have enough components for the hierarchy level
        if len(full_hierarchy) != self.hierarchy + 1:  # +1 for source
            raise ValueError(f"source_sample '{self.source_sample}' doesn't provide enough components for hierarchy level {self.hierarchy}. "
                           f"Expected {self.hierarchy} components, got {len(hierarchy_components)}: {hierarchy_components}")
        
        # Queue the hierarchy tuple directly
        q_input = tuple(full_hierarchy)
        list(map(lambda q: q.put(q_input), input_queue))
        self.log.info(f"Queued debug sample with hierarchy {full_hierarchy}")

    def _feed_all_sources(self, input_queue):
        """Step 4: Feed all sources using hierarchy and first service input folder"""
        self.log.info(f"Processing all sources at hierarchy level {self.hierarchy} from {self.first_service_input_folder}")
        
        for source in self.source_list:
            self.log.info(f"Processing source: {source}")
            
            # Step 4: Construct input path using first service input folder
            if self.first_service_input_folder == RAW_DATA:
                base_path = pathlib.Path(DATA_FOLDER, RAW_DATA, source)
            else:
                # For preprocessed data
                base_path = pathlib.Path(DATA_FOLDER, PREPROCESS, f"{OUTCOMES}_{source}")
            
            if not base_path.exists():
                self.log.warning(f"Base path not found for source '{source}': {base_path}")
                continue
            
            # Step 6 & 7: Handle hierarchy levels
            if self.hierarchy == 0:
                # Hierarchy 0: source level only - queue the source as full hierarchy
                q_input = (source,)
                list(map(lambda q: q.put(q_input), input_queue))
                self.log.info(f"Queued source '{source}' at hierarchy level 0")
                
            else:
                # Hierarchy > 0: need to iterate through folder structure
                self._iterate_hierarchy_levels(base_path, source, input_queue, [], 1)
                
        self.log.info(f"Completed queuing all sources at hierarchy level {self.hierarchy}")

    def _iterate_hierarchy_levels(self, current_path: pathlib.Path, source: str, input_queue, path_hierarchy: List[str], current_level: int):
        """Step 7: Recursively iterate through hierarchy levels"""
        if current_level > self.hierarchy:
            # Reached target hierarchy level - queue the whole hierarchy from source to last item
            full_hierarchy = [source] + path_hierarchy
            q_input = tuple(full_hierarchy)
            
            list(map(lambda q: q.put(q_input), input_queue))
            self.log.info(f"Queued item with full hierarchy {full_hierarchy} at level {current_level-1}")
        else:
            # Step 6: Use scandir instead of iterdir for better performance
            try:
                with os.scandir(current_path) as entries:
                    for entry in entries:
                        if entry.is_dir():
                            new_hierarchy = path_hierarchy + [entry.name]
                            new_path = pathlib.Path(entry.path)
                            self._iterate_hierarchy_levels(new_path, source, input_queue, new_hierarchy, current_level + 1)
            except (OSError, FileNotFoundError) as e:
                self.log.warning(f"Cannot scan directory {current_path}: {e}")

    # endregion

    # region Threads

    def wait_for_finish(self):
        for pipeline_level, (service_name, service_threads) in enumerate(self.services.items()):
            share_input = service_threads[-1]
            if not share_input:
                while any(list(map(lambda q: q.unfinished_tasks > len(service_threads), self.input_queues[pipeline_level]))):
                    time.sleep(5)
            else:
                while self.input_queues[pipeline_level][0].unfinished_tasks > len(service_threads):
                    time.sleep(5)
            for threads in service_threads[:-1]:
                for single_thread in threads:
                    single_thread.is_running = False
            for threads in service_threads[:-1]:
                for single_thread in threads:
                    single_thread.join()
            self.log.info(f"{service_name} threads finished")
        self.log.info("All threads finish, shutting down program")

    # endregion


    def run(self):
        if self.run_preprocess:
            self.init_threads()
            self.wait_for_finish()
        
        # Add feature arrangement as final step if feature_arranger is provided
        if self.feature_arranger is not None:
            self.run_feature_arrangement()