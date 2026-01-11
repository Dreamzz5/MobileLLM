# MobileLLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Spatio-Temporal Large Language Model for Mobile Traffic Prediction.

A PyTorch implementation of spatio-temporal prediction models using large language models with cross-modal attention for mobile network traffic forecasting tasks.

## Features

- 🚀 **Large Language Model Integration**: Leverages GPT-2 with LoRA fine-tuning for efficient spatio-temporal modeling
- 🔄 **Cross-Modal Attention**: Advanced attention mechanisms for fusing spatial and temporal features
- 📊 **Multiple Datasets**: Support for TRENTO, MILAN, and SHANGHAI mobile traffic datasets
- ⚡ **Efficient Training**: Optimized with PyTorch Lightning and modern deep learning practices

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install from Source

```bash
git clone https://github.com/tmc-research/st-llm-plus.git
cd st-llm-plus
pip install -e .
```

## Quick Start

### Training

Train the model on TRENTO dataset:

```bash
python train.py -d TRENTO -g 0
```

### Available Datasets

- `TRENTO`: Trento traffic dataset
- `MILAN`: Milan traffic dataset
- `SHANGHAI`: Shanghai traffic dataset

### Dataset Download

Download the datasets from the following Google Drive link:
[Mobile Traffic Datasets](https://drive.google.com/drive/folders/1TvMI6cjiXDvWwUXm_Z8r3j8nMNA-RDWi)

### Command Line Arguments

- `-d, --dataset`: Dataset name (TRENTO, MILAN, SHANGHAI)
- `-g, --gpu`: GPU device ID (default: 0)

## Project Structure

```
MobileLLM/
├── mobilellm/               # Main package
│   ├── __init__.py          # Package initialization
│   ├── MobileLLM.py         # Main model implementation
│   ├── data_prepare.py      # Data loading utilities
│   ├── metrics.py           # Evaluation metrics
│   ├── utils.py             # Utility functions
│   └── ranger21.py          # Ranger21 optimizer
├── __init__.py              # Root package initialization
├── train.py                 # Training script
├── MobileLLM.yaml           # Model configuration
├── pyproject.toml           # Build configuration
├── requirements.txt         # Dependencies
├── README.md                # Documentation
├── LICENSE                  # License file
├── data/                    # Dataset storage (not included)
├── logs/                    # Training logs (generated)
└── saved_models/            # Model checkpoints (generated)
```
