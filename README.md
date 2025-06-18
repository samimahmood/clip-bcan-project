# Complete Guide: CLIP-Enhanced BCAN for Cross-Modal Retrieval

This comprehensive guide covers everything you need to set up, run, and switch between Flickr8k and Flickr30k datasets on both Mac and Windows systems.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation Guide](#installation-guide)
3. [Dataset Setup](#dataset-setup)
4. [Running the Project](#running-the-project)
5. [Switching Between Datasets](#switching-between-datasets)
6. [Troubleshooting](#troubleshooting)
7. [Performance Expectations](#performance-expectations)

## Project Overview

This project implements a Bidirectional Correct Attention Network (BCAN) enhanced with CLIP features for cross-modal image-text retrieval. Key features:

- **CLIP Integration**: Uses frozen CLIP ViT-B/16 for feature extraction
- **BCAN Architecture**: Global and Local Correct Units for semantic alignment
- **Dataset Flexibility**: Seamlessly works with both Flickr8k and Flickr30k
- **Cross-Platform**: Supports Mac (CPU/MPS) and Windows (CPU/CUDA)

### Project Structure

```
clip-bcan-project/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bcan_model.py      # BCAN model implementation
│   │   └── losses.py           # Loss functions
│   ├── data/
│   │   ├── __init__.py
│   │   └── flickr30k_dataset.py # Dataset loader (works for both 8k/30k)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py        # Evaluation metrics
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py           # Logging utilities
│   └── train.py                # Training script
├── configs/
│   ├── __init__.py
│   └── config.py               # Configuration
├── scripts/
│   └── prepare_flickr30k.py    # Data preparation
├── checkpoints/                # Model checkpoints
├── logs/                       # Training logs
├── main.py                     # Main entry point
├── requirements.txt            # Dependencies
├── setup_project.py            # Project setup helper
└── quick_prepare_flickr8k.py   # Quick Flickr8k setup
```

## Installation Guide

### Prerequisites

- Python 3.8 or higher
- Git
- 8GB+ RAM (16GB recommended for Flickr30k)
- GPU optional but recommended for faster training

### Mac Installation

#### Step 1: Create Project Directory

```bash
mkdir clip-bcan-project
cd clip-bcan-project

# Create directory structure
mkdir -p src/{models,data,evaluation,utils}
mkdir -p configs scripts checkpoints logs
```

#### Step 2: Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Step 3: Install PyTorch (Mac-specific)

```bash
# For Mac (both Intel and Apple Silicon)
pip3 install torch torchvision torchaudio

# If that fails, try:
pip3 install torch==2.0.1 torchvision==0.15.2
```

#### Step 4: Install Other Dependencies

```bash
# Install remaining packages
pip install numpy pillow tqdm matplotlib scikit-learn pandas ftfy regex

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

### Windows Installation

#### Step 1: Create Project Directory

```cmd
mkdir clip-bcan-project
cd clip-bcan-project

# Create directory structure
mkdir src\models src\data src\evaluation src\utils
mkdir configs scripts checkpoints logs
```

#### Step 2: Setup Python Environment

```cmd
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### Step 3: Install PyTorch (Windows-specific)

```cmd
# For Windows with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Windows CPU only
pip install torch torchvision torchaudio
```

#### Step 4: Install Other Dependencies

```cmd
# Install remaining packages
pip install numpy pillow tqdm matplotlib scikit-learn pandas ftfy regex

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

### Create **init**.py Files (Both Platforms)

Run the setup script:

```bash
# Mac/Linux
python3 setup_project.py

# Windows
python setup_project.py
```

Or manually create them:

```bash
# Mac/Linux
touch src/__init__.py src/models/__init__.py src/data/__init__.py src/evaluation/__init__.py src/utils/__init__.py configs/__init__.py

# Windows
echo. > src\__init__.py && echo. > src\models\__init__.py && echo. > src\data\__init__.py && echo. > src\evaluation\__init__.py && echo. > src\utils\__init__.py && echo. > configs\__init__.py
```

## Dataset Setup

### Flickr8k Setup (Development - Recommended to Start)

#### Dataset Structure

```
Flicker8k/                      # Note the typo in "Flicker"!
├── Flicker8k_Dataset/          # 8,091 images
├── Flickr8k.token.txt          # 40,455 captions
├── Flickr_8k.trainImages.txt   # 6,000 training images (optional)
├── Flickr_8k.devImages.txt     # 1,000 validation images (optional)
└── Flickr_8k.testImages.txt    # 1,000 test images (optional)
```

#### Download and Prepare

1. Download Flickr8k from official sources or Kaggle
2. Extract maintaining the structure above
3. Prepare splits:

```bash
# Mac/Linux
python3 scripts/prepare_flickr30k.py --data_path /path/to/Flicker8k --dataset_name flickr8k

# Windows
python scripts\prepare_flickr30k.py --data_path C:\path\to\Flicker8k --dataset_name flickr8k
```

### Flickr30k Setup (Production)

#### Dataset Structure

```
flickr30k/
├── flickr30k-images/           # 31,783 images
└── results_20130124.token      # 158,915 captions
```

#### Download and Prepare

1. Download Flickr30k from official sources
2. Extract maintaining the structure above
3. Prepare splits:

```bash
# Mac/Linux
python3 scripts/prepare_flickr30k.py --data_path /path/to/flickr30k --dataset_name flickr30k

# Windows
python scripts\prepare_flickr30k.py --data_path C:\path\to\flickr30k --dataset_name flickr30k
```

## Running the Project

### Mac Commands

#### Development with Flickr8k

```bash
# CPU (all Macs)
python3 main.py \
    --data_path /path/to/Flicker8k \
    --dataset_name flickr8k \
    --batch_size 4 \
    --num_epochs 2 \
    --device cpu

# Apple Silicon with MPS (M1/M2/M3)
python3 main.py \
    --data_path /path/to/Flicker8k \
    --dataset_name flickr8k \
    --batch_size 8 \
    --num_epochs 20 \
    --device mps
```

#### Production with Flickr30k

```bash
# Just change dataset path and name!
python3 main.py \
    --data_path /path/to/flickr30k \
    --dataset_name flickr30k \
    --batch_size 32 \
    --num_epochs 20 \
    --device mps  # or cpu
```

### Windows Commands

#### Development with Flickr8k

```cmd
# CPU
python main.py ^
    --data_path C:\path\to\Flicker8k ^
    --dataset_name flickr8k ^
    --batch_size 4 ^
    --num_epochs 2 ^
    --device cpu

# GPU (NVIDIA)
python main.py ^
    --data_path C:\path\to\Flicker8k ^
    --dataset_name flickr8k ^
    --batch_size 16 ^
    --num_epochs 20 ^
    --device cuda
```

#### Production with Flickr30k

```cmd
# GPU recommended for Flickr30k
python main.py ^
    --data_path C:\path\to\flickr30k ^
    --dataset_name flickr30k ^
    --batch_size 32 ^
    --num_epochs 20 ^
    --device cuda
```

### Command Line Arguments

| Argument          | Description                             | Default       |
| ----------------- | --------------------------------------- | ------------- |
| `--data_path`     | Path to dataset directory               | Required      |
| `--dataset_name`  | Dataset type: 'flickr8k' or 'flickr30k' | 'flickr8k'    |
| `--batch_size`    | Training batch size                     | 32            |
| `--num_epochs`    | Number of training epochs               | 20            |
| `--learning_rate` | Learning rate                           | 1e-4          |
| `--device`        | Device: 'cpu', 'cuda', or 'mps'         | Auto-detected |
| `--clip_model`    | CLIP model variant                      | 'ViT-B/16'    |
| `--embed_size`    | BCAN embedding dimension                | 1024          |
| `--margin`        | Contrastive loss margin                 | 0.2           |
| `--resume`        | Path to checkpoint to resume            | None          |
| `--test_only`     | Only run evaluation                     | False         |

## Switching Between Datasets

The beauty of this implementation is that **no code changes are needed** to switch between Flickr8k and Flickr30k!

### Quick Switch Guide

1. **During Development**: Use Flickr8k

   ```bash
   --data_path /path/to/Flicker8k --dataset_name flickr8k
   ```

2. **For Production**: Switch to Flickr30k
   ```bash
   --data_path /path/to/flickr30k --dataset_name flickr30k
   ```

### What Changes Automatically

- Image directory paths (`Flicker8k_Dataset` vs `flickr30k-images`)
- Caption file names (`Flickr8k.token.txt` vs `results_20130124.token`)
- Split file detection and creation
- All other code remains identical!

## Troubleshooting

### Common Issues and Solutions

#### 1. "No module named 'torch'"

```bash
# Mac
pip3 install torch torchvision

# Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. "Torch not compiled with CUDA enabled" (Mac)

```bash
# Use CPU or MPS instead
--device cpu  # or --device mps for Apple Silicon
```

#### 3. "0 images loaded"

```bash
# Check split files exist
ls -la /path/to/dataset/*.txt  # Mac/Linux
dir C:\path\to\dataset\*.txt   # Windows

# Run prepare script
python3 scripts/prepare_flickr30k.py --data_path /path/to/dataset --dataset_name flickr8k
```

#### 4. "ModuleNotFoundError: No module named 'src.evaluation'"

```bash
# Create __init__.py files
python3 setup_project.py

# Or run with PYTHONPATH
PYTHONPATH=. python3 main.py ...  # Mac/Linux
set PYTHONPATH=. && python main.py ...  # Windows
```

#### 5. Memory Issues

- Reduce batch_size (try 4 or 8)
- Use CPU if GPU runs out of memory
- For Flickr30k, use cloud GPU if local resources insufficient

### Platform-Specific Tips

#### Mac Tips

- Use `mps` device on Apple Silicon for 2-3x speedup over CPU
- Install PyTorch with: `pip3 install torch torchvision`
- Use `python3` explicitly to avoid Python 2.x

#### Windows Tips

- Use CUDA if you have NVIDIA GPU
- Use forward slashes in paths or escape backslashes
- Run commands in Anaconda Prompt if using conda

## Performance Expectations

### Hardware Requirements

| Dataset   | Min RAM | Recommended RAM | Min GPU       | Training Time/Epoch |
| --------- | ------- | --------------- | ------------- | ------------------- |
| Flickr8k  | 8GB     | 16GB            | None (CPU OK) | 15-20 min (CPU)     |
| Flickr30k | 16GB    | 32GB            | 8GB VRAM      | 60-90 min (CPU)     |

### Expected Results

| Dataset   | Epochs | Expected R@sum | Time to Train |
| --------- | ------ | -------------- | ------------- |
| Flickr8k  | 20     | ~65-70%        | 5-6 hours     |
| Flickr30k | 20     | >76%           | 20-30 hours   |

### Optimization Tips

1. **Start Small**: Test with 2 epochs on Flickr8k
2. **Batch Size**: Larger = faster but more memory
3. **Device Selection**:
   - Mac: Use `mps` on Apple Silicon, `cpu` on Intel
   - Windows: Use `cuda` with NVIDIA GPU, otherwise `cpu`
4. **Monitoring**: Check `logs/` directory for training progress

## Advanced Usage

### Resume Training

```bash
python3 main.py --data_path /path/to/dataset --dataset_name flickr8k --resume checkpoints/checkpoint_epoch_10.pth
```

### Evaluation Only

```bash
python3 main.py --data_path /path/to/dataset --dataset_name flickr8k --test_only
```

### Custom Hyperparameters

```bash
python3 main.py \
    --data_path /path/to/dataset \
    --dataset_name flickr8k \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --margin 0.3 \
    --embed_size 2048
```

## Summary

This implementation provides a complete, cross-platform solution for cross-modal retrieval that:

- Works identically on Mac and Windows
- Seamlessly switches between Flickr8k and Flickr30k
- Automatically detects best available device (CPU/CUDA/MPS)
- Provides comprehensive logging and checkpointing

Start with Flickr8k for quick development cycles, then scale to Flickr30k for production results!
