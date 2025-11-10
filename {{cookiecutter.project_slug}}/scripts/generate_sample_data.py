#!/usr/bin/env python3
"""
Generate sample raw data for the preprocessing pipeline.

This script creates sample data following the expected hierarchy structure:
raw_data/source/hierarchy_1/hierarchy_2/data.csv

The hierarchy levels are generic and can represent any domain-specific structure.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_sample_data():
    """Generate sample data with hierarchy: source/hierarchy_1/hierarchy_2/"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configuration
    sources = ['Source_A', 'Source_B', 'Source_C']
    hierarchy_1_per_source = 3
    hierarchy_2_per_level1 = 5
    
    base_path = Path('data/raw_data')
    
    print(f"Generating sample data structure:")
    print(f"- Sources: {sources}")
    print(f"- Hierarchy level 1 per source: {hierarchy_1_per_source}")
    print(f"- Hierarchy level 2 per level 1: {hierarchy_2_per_level1}")
    print(f"- Base path: {base_path}")
    
    for source in sources:
        print(f"\nProcessing source: {source}")
        
        for level1_idx in range(1, hierarchy_1_per_source + 1):
            hierarchy_1_id = f'{level1_idx}'
            
            for level2_idx in range(1, hierarchy_2_per_level1 + 1):
                hierarchy_2_id = f'{level2_idx}'
                
                # Create directory structure
                data_path = os.path.join(base_path, source, hierarchy_1_id, hierarchy_2_id)
                os.makedirs(data_path , exist_ok=True)
                
                # Generate sample time-series data
                n_rows = np.random.randint(100, 300)  # Variable number of measurements
                
                # Create realistic sensor/measurement data
                data = {
                    'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='30min'),
                    'measurement_1': np.random.normal(50.0, 10.0, n_rows),      # Primary measurement
                    'measurement_2': np.random.exponential(5.0, n_rows),        # Secondary measurement
                    'measurement_3': np.random.beta(2, 5, n_rows) * 100,       # Percentage-type measurement
                    'sensor_reading_a': np.random.gamma(3, 2, n_rows),          # Sensor A reading
                    'sensor_reading_b': np.random.lognormal(2, 0.5, n_rows),   # Sensor B reading
                    'quality_metric': np.random.beta(8, 2, n_rows) * 100,      # Quality score
                    'status_flag': np.random.choice(['normal', 'warning', 'alert'], 
                                                   n_rows, p=[0.8, 0.15, 0.05]) # Status indicators
                }
                
                df = pd.DataFrame(data)
                
                # Add realistic data imperfections
                # Missing values in some measurements (2-5% missing)
                for col in ['measurement_1', 'measurement_2', 'sensor_reading_a']:
                    missing_rate = np.random.uniform(0.02, 0.05)
                    mask = np.random.random(n_rows) < missing_rate
                    df.loc[mask, col] = np.nan
                
                # Ensure realistic value ranges
                df['measurement_3'] = df['measurement_3'].clip(0, 100)
                df['quality_metric'] = df['quality_metric'].clip(0, 100)
                df['sensor_reading_a'] = df['sensor_reading_a'].clip(lower=0)
                df['sensor_reading_b'] = df['sensor_reading_b'].clip(lower=0)
                
                # Save main data file
                csv_path = os.path.join(data_path, 'data.csv')
                df.to_csv(csv_path, index=False)
                
                print(f"  Generated: {source}/{hierarchy_1_id}/{hierarchy_2_id} ({n_rows} records)")
    
    print(f"\nSample data generation complete!")
    print(f"Total structure created:")
    print(f"- {len(sources)} sources")
    print(f"- {len(sources) * hierarchy_1_per_source} hierarchy level 1 items")  
    print(f"- {len(sources) * hierarchy_1_per_source * hierarchy_2_per_level1} hierarchy level 2 items")
    print(f"\nEach hierarchy level 2 item contains:")
    print(f"- data.csv (main time-series data)")

if __name__ == "__main__":
    generate_sample_data()