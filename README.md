# 🚀 Cookiecutter ML Pipeline Template

A machine learning pipeline template for starting a machine learning projects including preprocessing, model selection and training, and model inference.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Template Options](#template-options)
4. [Project Structure](#project-structure)
5. [Framework Architecture](#framework-architecture)
6. [Best Practices](#best-practices)

## Overview

This framework follows three main steps in the pipeline: **preprocess**, **training**, and **inference**. The design is flexible, allowing users to choose and implement their own logic for each step. The template is suitable for both classical machine learning and deep learning workflows, and is intended for research and experimentation rather than production.

**Key Features:**
- Modular architecture for preprocess, training, and inference
- Abstract base classes for extensibility
- Versioned results and experiment tracking
- Hydra-based configuration management
- Optuna integration for hyperparameter optimization
- Example option for quick exploration

## Quick Start

### 1. Install Cookiecutter
```bash
pip install cookiecutter
```

### 2. Generate Project
```bash
cookiecutter https://github.com/ItamarEfrati/PipeLion.git --directory ml-project
```

## Template Options

The template includes a very simple example pipeline. To use the example, first run the sample data generation script provided in the template. This will create example data for you to explore the pipeline. The example implementation demonstrates the full pipeline (preprocess, training, inference) and is intended for research and learning—not for production use. When generating the template, you can choose to include or exclude these examples.

**With examples:**
- Example study, dataloader, inference handler
- Sample data and configs
- Data generation script to create example data

**Without examples:**
- Only the abstract base classes and core structure
- No sample data or example implementations

## Project Structure

```
ml_project/
├── 📄 requirements.txt                 # ✅ Core dependencies
├── 📁 src/                            # Main source code
│   ├── 📄 __init__.py                 # ✅ Package initialization  
│   ├── 📄 main.py                     # ✅ Entry point orchestrator
│   ├── 📄 orchestrator.py             # ✅ Pipeline coordination
│   │
│   ├── 📁 training/                   # Training pipeline components
│   │   ├── 📄 __init__.py             # ✅ Always included
│   │   ├── 📁 dataloaders/            # Data loading abstractions
│   │   │   ├── 📄 __init__.py         # ✅ Always included
│   │   │   └── 📄 example_dataloader.py # 🎯 CSV-based example
│   │   └── 📁 hyperparameters_tuning/ # Study management
│   │       ├── 📄 __init__.py         # ✅ Always included
│   │       ├── 📄 abstract_study.py   # ✅ Base Study interface
│   │       └── 📁 user_studies/       # Your custom studies
│   │           ├── 📄 __init__.py     # ✅ Always included
│   │           └── 📄 example_study.py # 🎯 RandomForest study
│   │
│   ├── 📁 inference/                  # Inference pipeline
│   │   ├── 📄 __init__.py             # ✅ Always included
│   │   ├── 📄 abstract_inference.py   # ✅ Base inference interface
│   │   └── 📄 example_inference.py    # 🎯 Working inference example
│   │
│   ├── 📁 preprocess/                 # Data preprocessing
│   │   ├── 📄 __init__.py             # ✅ Always included
│   │   ├── 📄 data_processor.py       # ✅ Core preprocessing logic
│   │   ├── 📁 feature_arrangement/    # Feature engineering
│   │   │   ├── 📄 __init__.py         # ✅ Always included
│   │   │   ├── 📄 abstract_feature_arranger.py # ✅ Base interface
│   │   │   └── 📄 example_feature_arranger.py  # 🎯 Example implementation
│   │   └── 📁 micro_services/         # Modular processing services
│   │       ├── 📄 __init__.py         # ✅ Always included
│   │       ├── 📄 micro_service.py    # ✅ Base service class
│   │       └── 📁 user_implementations/ # Your custom services
│   │           ├── 📄 __init__.py     # ✅ Always included
│   │           ├── 📄 feature_engineer.py    # 🎯 Example feature eng.
│   │           └── 📄 statistics_calculator.py # 🎯 Example stats
│   │
│   └── 📁 utils/                      # Shared utilities
│       ├── 📄 __init__.py             # ✅ Always included
│       └── 📄 constants.py            # ✅ Global constants
│
├── 📁 config/                         # Configuration management
│   ├── 📄 config.yaml                 # ✅ Main configuration
│   ├── 📄 hydra/default.yaml          # ✅ Hydra settings
│   ├── 📄 inference_config.yaml       # 🎯 Inference configuration
│   ├── 📄 train_config.yaml           # 🎯 Training configuration
│   ├── 📁 dataloader/                 # DataLoader configs
│   │   ├── 📄 example_dataloader_train.yaml      # 🎯 Examples
│   │   └── 📄 example_dataloader_inference.yaml  # 🎯 Examples
│   ├── 📁 study/                      # Study configurations
│   │   └── 📄 example_study.yaml      # 🎯 Example study config
│   ├── 📁 inference/                  # Inference configurations
│   │   └── 📄 example_inference.yaml  # 🎯 Example inference config
│   └── 📁 preprocess/                 # Preprocessing configs
│       ├── 📄 example_preprocess_train.yaml     # 🎯 Examples
│       └── 📄 example_preprocess_inference.yaml # 🎯 Examples
│
├── 📁 data/                           # Data organization
│   ├── 📁 raw_data/                   # Original datasets
│   │   ├── 📄 labels.csv              # 🎯 Example labels
│   │   ├── 📁 Source_A/               # 🎯 Example data source
│   │   ├── 📁 Source_B/               # 🎯 Example data source  
│   │   └── 📁 Source_C/               # 🎯 Example data source
│   ├── 📁 preprocessed/               # Processed datasets
│   │   ├── 📁 features/               # 🎯 Example processed features
│   │   ├── 📁 outcomes_Source_A/      # 🎯 Example outcomes
│   │   ├── 📁 outcomes_Source_B/      # 🎯 Example outcomes
│   │   └── 📁 outcomes_Source_C/      # 🎯 Example outcomes
│   ├── 📁 for_modeling/               # Ready-for-training data
│   │   └── 📁 features_1/             # 🎯 Example modeling data
│   └── 📁 for_inference/              # Inference-ready data
│       └── 📁 Source_C/               # 🎯 Example inference data
│
├── 📁 assets/                         # Generated assets
│   └── 📁 results/                    # Study results and models
│       └── 📁 example_study/          # 🎯 Example study outputs
│           ├── 📄 example_study.db    # Optuna database
│           └── 📁 ver_1/              # Versioned results
│               ├── 📄 best_model.pkl  # Trained model
│               ├── 📁 inference/      # Inference results
│               └── 📁 run_results/    # Training metrics
│
├── 📁 scripts/                        # Utility scripts
│   └── 📄 generate_sample_data.py     # 🎯 Sample data generator
│
└── 📁 notebooks/                      # Jupyter notebooks (empty)
```

**Legend:**
- ✅ **Always included** - Core framework files
- 🎯 **Examples only** - Included when `include_examples: "yes"`

## Framework Architecture

The framework is organized around abstract modules for each pipeline step: preprocess, training, and inference. Each module provides a base interface, and you are free to implement your own logic for any step. You can use any ML or deep learning library (e.g., scikit-learn, PyTorch, TensorFlow) and design your own workflow. The framework does not enforce any specific algorithm or data format—it's up to you to decide how to structure your pipeline.

**Core abstractions:**
- Preprocess: Define how raw data is transformed and features are engineered
- Training: Implement your own study logic for model training and hyperparameter optimization
- Inference: Create custom handlers for model loading and prediction

You choose what to implement and how to connect the steps. The framework provides the structure and flexibility for research workflows.

## Best Practices

### 1. **Configuration Management**
- Keep configs small and focused
- Use Hydra composition for complex scenarios
- Override parameters via command line when needed
- Version control your configuration files

### 2. **Code Organization**
- One class per file for major components
- Use descriptive names for studies and handlers
- Keep business logic separate from framework code
- Add docstrings to all public methods

### 3. **Data Management**
- Organize data by source and processing stage
- Use consistent naming conventions
- Keep raw data immutable
- Document data transformations

### 4. **Experiment Tracking**
- Use meaningful study names
- Let the framework handle versioning
- Save important artifacts (plots, reports)
- Document significant findings

### 5. **Testing Strategy**
- Test with small datasets first
- Validate preprocessing pipelines separately  
- Use example components as templates
- Monitor training metrics closely