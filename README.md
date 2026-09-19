# Bach Chorale Note Prediction (GRU)

A PyTorch project that trains a GRU-based recurrent neural network on the [JSB Chorales dataset](https://github.com/czhuang/JSB-Chorales-dataset) to predict the next set of notes in a Bach chorale, and uses the trained model to autoregressively generate new, Bach-like note sequences.

## Overview

Each chorale in the dataset is a time series of 4 values per timestep. The model is trained to look at a window of past timesteps and predict the notes at the next timestep. Once trained, it can generate new sequences by feeding its own predictions back in as input.

## How it works

1. **Data loading** — Chorales are loaded from per-piece CSV files split into `train`, `valid`, and `test` folders. Each file contains one chorale as a `(timesteps, 4)` array of note values.
2. **Normalization** — Mean and standard deviation are computed per voice from the training set only, then used to standardize all splits.
3. **Windowing** — A sliding-window `Dataset` (`ChoraleDataset`) turns each chorale into `(window, next_step)` training pairs, with a window length of 98 timesteps.
4. **Model** — A single-layer GRU (`SimpleRNNModel`) with a hidden size of 32, followed by a linear layer, maps the last hidden state to a 4-value prediction for the next timestep.
5. **Training** — The model is trained with Huber loss and SGD (with momentum), tracked using Mean Absolute Error, and uses a `ReduceLROnPlateau` learning rate scheduler based on validation performance.
6. **Evaluation** — A random test window is used to sanity-check a single next-step prediction against the ground truth.
7. **Generation** — Starting from a seed window taken from a test chorale, the model repeatedly predicts the next timestep and appends it to the input window (dropping the oldest step), generating a new sequence of notes step by step.

## Requirements

- Python 3.12
- `torch`
- `torchmetrics`
- `pandas`
- `numpy`
- A CUDA-capable GPU (the notebook moves the model and data to `"cuda"`; adapt to `"cpu"` if unavailable)

Install dependencies:

```bash
pip install torch torchmetrics pandas numpy
```

## Data

Download the [JSB Chorales dataset](https://github.com/czhuang/JSB-Chorales-dataset) and place it so the following paths exist relative to the notebook:

```
jsb_chorales/jsb_chorales/train/*.csv
jsb_chorales/jsb_chorales/valid/*.csv
jsb_chorales/jsb_chorales/test/*.csv
```

Each CSV should contain one chorale, with 4 columns (one per voice) and one row per timestep.

## Usage

Open and run `chorale_predict.ipynb` top to bottom in Jupyter:

```bash
jupyter notebook chorale_predict.ipynb
```

The notebook will:
- Load and inspect the data
- Compute normalization statistics
- Build windowed datasets and data loaders
- Define and train the GRU model for 50 epochs
- Print a sample next-step prediction vs. ground truth
- Generate a new sequence of predicted notes from a seed chorale

## Key hyperparameters

| Parameter | Value |
|---|---|
| Window length | 98 |
| Hidden size | 32 |
| Batch size | 32 |
| Epochs | 50 |
| Loss function | Huber Loss |
| Optimizer | SGD (lr=0.003, momentum=0.9) |
| LR scheduler | ReduceLROnPlateau (patience=10, factor=0.1) |
| Metric | Mean Absolute Error |

