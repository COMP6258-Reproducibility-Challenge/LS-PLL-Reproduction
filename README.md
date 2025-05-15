# LS-PLL-Reproduction

> **This repository contains the code and resources to reproduce the experiments from the paper:**
> **[Does Label Smoothing Help Deep Partial Label Learning?](https://openreview.net/pdf?id=drjjxmi2Ha)** by _Gong et al_.

## Overview

Partial Label Learning (PLL) addresses the problem where each training example is associated with a set of candidate labels, only one of which is correct. This project investigates the effect of label smoothing on deep PLL methods by:

* Training base models (LeNet5, ResNet18, ResNet56) on clean datasets.
* Generating partial labels with controlled ambiguity.
* Evaluating PLL performance with and without label smoothing.
* Comparing accuracy across different smoothing rates and noise levels.

## Features

* **Model Architectures** : Implements LeNet5, ResNet18, and ResNet56.
* **Datasets** : Support for FashionMNIST, KuzushijiMNIST, CIFAR-10, and CIFAR-100.
* **Partial Label Generation** : Configurable average candidate label size (Avg.#CL) and Top-K predictions.
* **Label Smoothing** : Custom `LS_PLL_CrossEntropy` loss for smoothing within candidate sets.
* **Experiments Pipeline** : End-to-end script to train, generate partial labels, and evaluate.

## Experimental Configuration

The hyperparameters and experiments are defined in `codes/main.py`:

* **BATCH_SIZE** : 128
* **LEARNING_RATE** : 0.01
* **EPOCHS** : 200
* **WEIGHTING_PARAM** : 0.9
* **MOMENTUM** : 0.9
* **SMOOTHING_RATE** : [0.1, 0.3, 0.5, 0.7, 0.9]
* **EXPERIMENTS** :
  
  | Dataset | Model | k | Avg.#CL |
  | ---- | ---- | ---- | ---- |
  | Fashion-MNIST | LeNet-5 | 6 | 3, 4, 5 |
  | Kuzushiji-MNIST | LeNet-5 | 6 | 3, 4, 5 |
  | CIFAR-10 | ResNet-18 | 6 | 3, 4, 5 |
  | CIFAR-100 | ResNet-56 | 20 | 7, 9, 11 |

## Example FashionMNIST TSNE Grid

We provide sample TSNE plots illustrating feature representations on test sets under different candidate label sizes and smoothing configurations. In each grid, columns represent smoothing rates (no smoothing, r=0.1, 0.3, …, 0.9) and rows correspond to Avg.#CL levels.

![TSNE Grid for FashionMNIST](doc/figures/tsne_grid_FashionMNIST.png)

## Repository Structure

```bash
LS-PLL-Reproduction/
├── codes/                   # Source code
│   ├── LeNet5.py            # LeNet5 model implementation
│   ├── ResNet18.py          # ResNet18 model implementation
│   ├── ResNet56.py          # ResNet56 model implementation
│   ├── prepare_data.py      # Dataset loading & partial label generation
│   ├── train.py             # Training routines & PLL loss
│   └── utils.py             # Utility functions
├── datasets/                # Downloaded datasets & generated partial labels
├── models/                  # Saved pretrained and fine-tuned model weights
├── logs/                    # Training and evaluation logs
├── run.sh                   # Quick start script (setup & run all experiments)
├── requirements.txt         # Python package dependencies
└── README.md                # This file
```

## Requirements

* Python 3.13+
* numpy==2.2.5
* torch==2.7.0
* torchvision==0.22.0
* scikit-learn==1.6.1
* scipy==1.15.3
* pandas==2.2.3
* seaborn==0.13.2
* matplotlib==3.10.1

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/COMP6258-Reproducibility-Challenge/LS-PLL-Reproduction.git
   cd LS-PLL-Reproduction
   ```

2. **Create a virtual environment**

   The `run.sh` script will create a `.venv`, install dependencies, and run the experiments.

   Alternatively, manually install dependencies

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Running All Experiments

```bash
bash run.sh
```

This will:

1. Download datasets to `datasets/`.
2. Train base models on each dataset and save to `models/`.
3. Generate partial labels with different AvgCL values and save to `datasets/`.
4. Train PLL models without smoothing and with smoothing rates: 0.1, 0.3, 0.5, 0.7, 0.9.
5. Log outputs to `logs/YYYYMMDD_HHMMSS.log`.
6. Figures outputs to `./doc/figures.`

### Custom Run

```bash
./.venv/bin/python -u ./codes/main.py   \
  --model_path ./models                 \
  --dataset_path ./datasets             \
  --figure_path ./doc/figures
```

* `--model_path`: Path to save or load model weights (default: `./models`).
* `--dataset_path`: Path for input datasets and generated partial labels (default: `./datasets`).
* `--figure_path`: Path to save TSNE plots generated (default: `./doc/figures`).
