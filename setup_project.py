#!/usr/bin/env python3
"""
Setup script to ensure all project directories and __init__.py files exist
"""

import os
import sys

def create_init_files():
    """Create __init__.py files in all necessary directories"""
    
    # List of directories that need __init__.py
    directories = [
        'src',
        'src/models',
        'src/data',
        'src/evaluation',
        'src/utils',
        'configs'
    ]
    
    # Create directories and __init__.py files
    for directory in directories:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Create __init__.py file
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# This file makes the directory a Python package\n')
            print(f"Created {init_file}")
        else:
            print(f"{init_file} already exists")

def create_additional_directories():
    """Create additional project directories"""
    
    additional_dirs = [
        'checkpoints',
        'logs',
        'scripts'
    ]
    
    for directory in additional_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def check_required_files():
    """Check if all required Python files exist"""
    
    required_files = {
        'main.py': 'Main entry point',
        'src/train.py': 'Training script',
        'src/models/bcan_model.py': 'BCAN model implementation',
        'src/models/losses.py': 'Loss functions',
        'src/data/flickr30k_dataset.py': 'Dataset loader',
        'src/evaluation/evaluator.py': 'Evaluation metrics',
        'src/utils/logger.py': 'Logging utilities',
        'configs/config.py': 'Configuration file'
    }
    
    print("\nChecking required files:")
    missing_files = []
    
    for filepath, description in required_files.items():
        if os.path.exists(filepath):
            print(f"✓ {filepath} - {description}")
        else:
            print(f"✗ {filepath} - {description} [MISSING]")
            missing_files.append(filepath)
    
    if missing_files:
        print(f"\nWARNING: {len(missing_files)} required files are missing!")
        print("Please create these files with the appropriate content.")
        print("\nMissing files:")
        for file in missing_files:
            print(f"  - {file}")
    else:
        print("\nAll required files are present!")

def check_python_version():
    """Check if Python version is compatible"""
    
    version = sys.version_info
    print(f"\nPython version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("WARNING: Python 3.8+ is recommended for this project")
    else:
        print("Python version is compatible ✓")

def display_project_structure():
    """Display the expected project structure"""
    
    print("\nExpected project structure:")
    print("""
clip-bcan-project/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bcan_model.py
│   │   └── losses.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── flickr30k_dataset.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py
│   └── train.py
├── configs/
│   ├── __init__.py
│   └── config.py
├── scripts/
│   └── prepare_flickr30k.py
├── checkpoints/
├── logs/
├── main.py
├── setup_project.py (this file)
└── requirements.txt
    """)

def main():
    """Main setup function"""
    
    print("=" * 60)
    print("CLIP-BCAN Project Setup")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Create directories and __init__.py files
    print("\nCreating project structure...")
    create_init_files()
    create_additional_directories()
    
    # Check for required files
    check_required_files()
    
    # Display project structure
    display_project_structure()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Add any missing Python files listed above")
    print("3. Download Flickr8k or Flickr30k dataset")
    print("4. Run: python main.py --data_path /path/to/dataset --dataset_name flickr8k")
    print("=" * 60)

if __name__ == '__main__':
    main()