#!/usr/bin/env python3
"""
Post-generation hook for cookiecutter ML pipeline template.

This script handles conditional file creation/removal based on user choices.
"""

import os
import shutil
from pathlib import Path


def remove_examples():
    """Remove example files when include_examples is 'no'."""
    
    # Get the project directory
    project_dir = Path.cwd()
    
    # Example files with 'example' prefix
    example_files = [
        "config/dataloader/example_dataloader_train.yaml",
        "config/dataloader/example_dataloader_inference.yaml",
        "config/study/example_study.yaml",
        "config/preprocess/example_preprocess_train.yaml",
        "config/preprocess/example_preprocess_inference.yaml",
        "config/inference/example_inference.yaml",
        "src/inference/example_inference.py",
        "src/preprocess/feature_arrangement/example_feature_arranger.py",
        "src/training/dataloaders/example_dataloader.py",
        "src/training/hyperparameters_tuning/user_studies/example_study.py",
    ]
    
    # Configuration files (part of examples)
    config_files = [
        "config/inference_config.yaml",
        "config/train_config.yaml",
    ]
    
    # User implementation content (example implementations)
    user_implementation_files = [
        "src/preprocess/micro_services/user_implementations/feature_engineer.py",
        "src/preprocess/micro_services/user_implementations/statistics_calculator.py",
    ]
    
    # Example scripts
    example_scripts = [
        "scripts/generate_sample_data.py",
    ]
    
    # Example data directories
    example_data_dirs = [
        "data/raw_data/Source_A/",
        "data/raw_data/Source_B/",
        "data/raw_data/Source_C/",
        "data/preprocessed/features/",
        "data/preprocessed/outcomes_Source_A/",
        "data/preprocessed/outcomes_Source_B/",
        "data/preprocessed/outcomes_Source_C/",
        "data/for_inference/Source_C/",
        "data/for_modeling/features_1/",
        "assets/results/example_study/",
    ]
    
    # Example data files
    example_data_files = [
        "data/raw_data/labels.csv",
    ]
    
    # Combine all items to remove
    items_to_remove = (example_files + config_files + user_implementation_files + 
                      example_scripts + example_data_dirs + example_data_files)
    
    removed_count = 0
    for item_path in items_to_remove:
        full_path = project_dir / item_path
        if full_path.exists():
            try:
                if full_path.is_dir():
                    shutil.rmtree(full_path)
                    print(f"  📁 Removed directory: {item_path}")
                else:
                    full_path.unlink()
                    print(f"  📄 Removed file: {item_path}")
                removed_count += 1
            except Exception as e:
                print(f"  ❌ Error removing {item_path}: {e}")
    
    print(f"\n  ✅ Removed {removed_count} example items")


def main():
    """Main post-generation logic."""
    
    include_examples = "{{ cookiecutter.include_examples }}"
    is_vscode_ide = "{{ cookiecutter.is_vscode_ide }}"
    project_slug = "{{ cookiecutter.project_slug }}"
    
    print(f"🚀 Setting up project...")
    
    # Remove .vscode folder if not using VS Code
    if is_vscode_ide == "no":
        vscode_path = Path.cwd().joinpath(".vscode")
        if vscode_path.exists() and vscode_path.is_dir():
            shutil.rmtree(vscode_path)
            print("Removed .vscode folder (not using VS Code)")
    
        if include_examples == "no":
            print("\n📦 Setting up clean template without examples...")
            remove_examples()
            print("✅ Clean template setup complete!")
        else:
            print("\n📦 Setting up template with examples...")
            print("✅ Full template with examples setup complete!")
    
    print(f"\n🎉 Project has been created successfully!")
    print(f"📁 Project directory: {project_slug}")
    
    if include_examples == "yes":
        print("\n📚 Examples included - you can run the example right away!")
        print(f"   cd {project_slug}")
        print("   pip install -r requirements.txt")
        print("   python src/main.py")
    else:
        print("\n🏗️  Clean template created - implement your custom classes:")
        print("   - Extend abstract classes in src/inference/ and src/training/")
        print("   - Add your implementations in user_implementations/ directories")
        print("   - Create configuration files in config/ subdirectories")
        print("   - Add your data processing logic")
        
        print("\n📋 Next steps:")
        print(f"   cd {project_slug}")
        print("   pip install -r requirements.txt")
        print("   # Implement your custom classes and configurations")


if __name__ == "__main__":
    main()