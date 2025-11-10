"""
Simple concrete implementation of feature arranger for the current project.
Assumes fixed hierarchy: source -> hierarchy_1 -> hierarchy_2
Filename format: {hierarchy_1}_{hierarchy_2}_{feature_type}.csv
"""

import json
import logging
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import pandas as pd
from tqdm import tqdm

from src.preprocess.feature_arrangement.abstract_feature_arranger import AbstractFeatureArranger
from src.utils.constants import (
    DATA_FOLDER, PREPROCESS, PREPROCESSED_DATA, FOR_MODELING_FOLDER, FOR_INFERENCE_FOLDER,
    FEATURES_FOLDER_PREFIX, FEATURES_FILENAME, METADATA_FILENAME, LABELS_FILENAME,
    CSV_EXTENSION, JSON_EXTENSION
)


class ExampleFeatureArranger(AbstractFeatureArranger):
    """
    Simple feature arranger with fixed hierarchy structure.
    Knows exactly how files are structured and named.
    """
    
    def __init__(self, 
                 sources: List[str],
                 feature_types: List[str],
                 label_file_path: Optional[str] = None,
                 description: Optional[str] = None,
                 n_workers: int = 4):
        """
        Initialize simple feature arranger.
        
        Args:
            sources: List of source names to process (e.g., ['Source_A', 'Source_B'])
            feature_types: List of feature types to concatenate (e.g., ['feature_engineer', 'statistics_calculator'])
            label_file_path: Path to CSV file with labels
            description: Optional user description for the dataset
            n_workers: Number of workers for parallel processing
        """
        super().__init__(feature_types, label_file_path, description, n_workers)
        self.sources = sources
        self.logger = logging.getLogger(__name__)

    # region Private Methods
    
    def _csv_file_generator(self, folder_path: pathlib.Path):
        """Generator for CSV files in a folder."""
        try:
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(CSV_EXTENSION):
                        yield pathlib.Path(entry.path)
        except (OSError, FileNotFoundError) as e:
            self.logger.error(f"Cannot scan directory {folder_path}: {e}")
    
    def _process_single_feature_file(self, csv_file, source: str, feature_type: str):
        """Process a single feature CSV file using simple filename parsing."""
        try:
            # Simple filename parsing: hierarchy_1_hierarchy_2_feature_type.csv
            filename_parts = csv_file.stem.split('_')
            
            if len(filename_parts) < 2:
                self.logger.warning(f"Filename {csv_file.name} doesn't follow expected pattern: hierarchy1_hierarchy2_featuretype.csv")
                return None
                
            hierarchy_1 = filename_parts[0]
            hierarchy_2 = filename_parts[1]
            
            # Load CSV file
            df = pd.read_csv(csv_file)
            
            # Handle the case where first column is an index
            if len(df.columns) > 0 and (df.columns[0] in ['', 'Unnamed: 0'] or str(df.columns[0]).isdigit()):
                df = df.iloc[:, 1:]  # Skip first column
            
            # Add prefix to feature columns to avoid conflicts
            feature_columns = [col for col in df.columns]
            df.columns = [f"{feature_type}_{col}" for col in feature_columns]
            
            # Add hierarchy columns - we know exactly what they are
            df['source'] = source
            df['hierarchy_1'] = hierarchy_1
            df['hierarchy_2'] = hierarchy_2
            
            # If multiple rows in CSV, take the first one (assuming single sample per file)
            if len(df) > 0:
                return df.iloc[0]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing file {csv_file}: {e}")
            return None
    
    def _load_feature_type(self, feature_type_path: pathlib.Path, source: str, feature_type: str) -> pd.DataFrame:
        """Load all CSV files for a specific feature type using ThreadPoolExecutor."""
        feature_rows = []
        
        # Get all CSV files in the feature type directory
        csv_files = list(self._csv_file_generator(feature_type_path))
        
        if not csv_files:
            return pd.DataFrame()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for df_row in tqdm(executor.map(lambda csv_file: self._process_single_feature_file(csv_file, source, feature_type), csv_files),
                              total=len(csv_files),
                              desc=f"Loading {feature_type}"):
                if df_row is not None:
                    feature_rows.append(df_row)
        
        if feature_rows:
            feature_df = pd.DataFrame(feature_rows)
            feature_df = feature_df.reset_index(drop=True)
            return feature_df
        else:
            return pd.DataFrame()
    
    def _merge_feature_types(self, feature_dfs: List[pd.DataFrame], source: str) -> pd.DataFrame:
        """Merge multiple feature types for a single source on (source, hierarchy_1, hierarchy_2)."""
        if len(feature_dfs) == 1:
            return feature_dfs[0]
        
        # Start with the first dataframe
        merged_df = feature_dfs[0]
        
        # Merge with each subsequent feature type on known join columns
        join_columns = ['source', 'hierarchy_1', 'hierarchy_2']
        for i, feature_df in enumerate(feature_dfs[1:], 1):
            merged_df = pd.merge(
                merged_df, 
                feature_df, 
                on=join_columns, 
                how='outer'
            )
        
        self.logger.info(f"Merged {len(feature_dfs)} feature types for source {source}: {len(merged_df)} samples")
        return merged_df
    
    def _create_metadata(self, n_samples: int, has_labels: Optional[bool], feature_columns: List[str], is_inference: bool = False) -> Dict:
        """Create metadata dictionary."""
        join_columns = ['source', 'hierarchy_1', 'hierarchy_2']
        
        metadata = {
            "creation_date": datetime.now().isoformat(),
            "description": self.description,
            "sources": list(self.sources),
            "feature_types": list(self.feature_types),
            "n_samples": n_samples,
            "n_features": len([col for col in feature_columns if col not in join_columns]),
            "feature_columns": feature_columns,
            "index_columns": join_columns,
            "has_labels": has_labels,
            "is_inference": is_inference,
            "label_file_path": self.label_file_path if has_labels else None,
            "hierarchy_structure": "source -> hierarchy_1 -> hierarchy_2",
            "filename_convention": "hierarchy_1_hierarchy_2_feature_type.csv"
        }
        
        return metadata
    
    # endregion

            
    # region Abstract Methods
    
    def _create_output_directory(self, is_inference: bool = False) -> str:
        """Create output directory with incremental index for modeling or source-based for inference."""
        if is_inference:
            # For inference: use source names in for_inference folder
            base_path = pathlib.Path(DATA_FOLDER, FOR_INFERENCE_FOLDER)
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Use source names as folder name
            source_names = "_".join(sorted(self.sources))
            output_dir = os.path.join(base_path, source_names)
            os.makedirs(output_dir, exist_ok=True)

            return str(output_dir)
        else:
            # For modeling: use incremental index in for_modeling folder
            base_path = pathlib.Path(DATA_FOLDER, FOR_MODELING_FOLDER)
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Find next available index
            existing_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith(FEATURES_FOLDER_PREFIX)]
            indices = []
            for d in existing_dirs:
                try:
                    idx = int(d.name.split('_')[1])
                    indices.append(idx)
                except (IndexError, ValueError):
                    continue
            
            next_index = max(indices, default=0) + 1
            output_dir = os.path.join(base_path, f'{FEATURES_FOLDER_PREFIX}{next_index}')
            os.makedirs(output_dir, exist_ok=True)

            return str(output_dir)
    
    def _load_labels(self) -> pd.DataFrame:
        """Load labels from CSV file."""
        if not os.path.exists(self.label_file_path):
            raise FileNotFoundError(f"Label file not found: {self.label_file_path}")
        
        # Read CSV with all columns as string except label column
        labels_df = pd.read_csv(self.label_file_path, dtype={'source': str, 'hierarchy_1': str, 'hierarchy_2': str})

        # Convert hierarchy columns to string for consistent joining
        for col in ['source', 'hierarchy_1', 'hierarchy_2']:
            if col in labels_df.columns:
                labels_df[col] = labels_df[col].astype(str)
        
        # Validate required columns
        required_cols = ['source', 'hierarchy_1', 'hierarchy_2']
        missing_cols = [col for col in required_cols if col not in labels_df.columns]
        if missing_cols:
            raise ValueError(f"Label file missing required columns: {missing_cols}")
        
        return labels_df
    
    def _join_features_labels(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform inner join between features and labels on known columns."""
        join_columns = ['source', 'hierarchy_1', 'hierarchy_2']
        
        # Perform inner join to ensure row alignment
        merged_df = pd.merge(features_df, labels_df, on=join_columns, how='inner')
        
        self.logger.info(f"Inner join result: {len(merged_df)} samples with both features and labels")
        
        # Split back into features and labels
        feature_cols = [col for col in features_df.columns if col not in join_columns]
        label_cols = [col for col in labels_df.columns if col not in join_columns]
        
        features_with_labels = merged_df[join_columns + feature_cols].copy()
        labels_matched = merged_df[join_columns + label_cols].copy()
        
        return features_with_labels, labels_matched
    
    def _get_features_without_labels(self, features_df: pd.DataFrame, features_with_labels: pd.DataFrame) -> pd.DataFrame:
        """Get features that don't have corresponding labels using anti-join."""
        join_columns = ['source', 'hierarchy_1', 'hierarchy_2']
        
        # Use anti-join to get features without labels
        features_without_labels = features_df.merge(
            features_with_labels[join_columns], 
            on=join_columns, 
            how='left', 
            indicator=True
        ).query('_merge == "left_only"').drop('_merge', axis=1)
        
        self.logger.info(f"Features without labels: {len(features_without_labels)} samples")
        
        return features_without_labels
    
    def _save_features_and_metadata(self, features_df: pd.DataFrame, output_dir: str, suffix: str = "", is_inference: bool = False):
        """Save features and metadata files with optional suffix."""
        join_columns = ['source', 'hierarchy_1', 'hierarchy_2']
        
        # Extract only feature columns (excluding index columns)
        feature_cols = [col for col in features_df.columns if col not in join_columns]
        
        # Determine file names with suffix
        features_filename = f'{FEATURES_FILENAME}{suffix}{CSV_EXTENSION}' if suffix else f'{FEATURES_FILENAME}{CSV_EXTENSION}'
        metadata_filename = f'{METADATA_FILENAME}{suffix}{CSV_EXTENSION}' if suffix else f'{METADATA_FILENAME}{CSV_EXTENSION}'
        json_metadata_filename = f'{METADATA_FILENAME}{suffix}{JSON_EXTENSION}' if suffix else f'{METADATA_FILENAME}{JSON_EXTENSION}'
        
        # Save features only (no index columns)
        features_only = features_df[feature_cols].copy()
        features_file = os.path.join(output_dir, features_filename)
        features_only.to_csv(features_file, index=False)
        
        # Save metadata (index columns only)
        metadata_df = features_df[join_columns].copy()
        metadata_file = os.path.join(output_dir, metadata_filename)
        metadata_df.to_csv(metadata_file, index=False)
        
        # Create JSON metadata
        has_labels = suffix == "" and not is_inference  # Only main features have labels and not inference
        json_metadata = self._create_metadata(len(features_df), has_labels, features_df.columns.tolist(), is_inference)
        json_metadata_file = os.path.join(output_dir, json_metadata_filename)
        with open(json_metadata_file, 'w') as f:
            json.dump(json_metadata, f, indent=2)
        
        self.logger.info(f"Saved features: {features_file}")
        self.logger.info(f"Saved metadata indices: {metadata_file}")
        self.logger.info(f"Saved JSON metadata: {json_metadata_file}")
    
    def _save_labels(self, labels_df: pd.DataFrame, output_dir: str):
        """Save labels separately."""
        join_columns = ['source', 'hierarchy_1', 'hierarchy_2']
        
        # Extract only label columns (excluding index columns)
        label_cols = [col for col in labels_df.columns if col not in join_columns]
        
        # Save labels only (no index columns)
        labels_only = labels_df[label_cols].copy()
        labels_file = os.path.join(output_dir, f'{LABELS_FILENAME}{CSV_EXTENSION}')
        labels_only.to_csv(labels_file, index=False)
        
        self.logger.info(f"Saved labels: {labels_file}")
    
    def combine_features_from_sources(self) -> pd.DataFrame:
        """
        Combine feature files from multiple sources and feature types.
        
        Returns:
            DataFrame with combined features and standardized join columns (source, hierarchy_1, hierarchy_2)
        """
        all_features = []
        
        for source in self.sources:
            self.logger.info(f"Processing source: {source}")
            
            # Source features path: data/preprocessed/{source}/
            source_features_path = pathlib.Path(DATA_FOLDER, PREPROCESS, PREPROCESSED_DATA, source)
            
            if not source_features_path.exists():
                self.logger.warning(f"Features path not found for source {source}: {source_features_path}")
                continue
            
            # Load features from each specified feature type
            source_feature_dfs = []
            for feature_type in self.feature_types:
                feature_type_path = os.path.join(source_features_path, feature_type)

                if not os.path.exists(feature_type_path):
                    self.logger.warning(f"Feature type {feature_type} not found for source {source}")
                    continue
                
                self.logger.info(f"Loading {feature_type} features...")
                feature_df = self._load_feature_type(feature_type_path, source, feature_type)
                
                if not feature_df.empty:
                    source_feature_dfs.append(feature_df)
            
            # Combine all feature types for this source
            if source_feature_dfs:
                source_combined = self._merge_feature_types(source_feature_dfs, source)
                if not source_combined.empty:
                    all_features.append(source_combined)
        
        # Combine all sources
        if all_features:
            final_features = pd.concat(all_features, ignore_index=True)
            self.logger.info(f"Combined features from {len(self.sources)} sources: {len(final_features)} total samples")
        else:
            self.logger.exception("No features found for any source")
            raise

        return final_features
    
    # endregion
    