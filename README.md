Hull Tactical Market Prediction with Machine Learning

This project explores machine learning methods for predicting daily excess returns of the U.S. equity market using the Hull Tactical Market Prediction dataset. We compare linear models, ensemble learners, and sequential neural networks under a unified, leakage-free time-series evaluation pipeline.

Project Structure
hull-tactical-ml/
├── configs/                 # Experiment configurations (YAML)
│   ├── baseline_ridge.yaml
│   ├── xgboost.yaml
│   └── lstm.yaml
├── data/
│   └── raw/
│       └── train.csv        # Kaggle Hull Tactical dataset
├── scripts/
│   └── train.py             # Main training entrypoint
├── src/
│   ├── data/
│   │   ├── load.py          # CSV loading & time sorting
│   │   ├── split.py         # Walk-forward time splits
│   │   └── sequence.py      # Sequence construction for LSTM
│   ├── features/
│   │   └── build.py         # Feature engineering (lags, rolling)
│   └── models/
│       ├── baseline/
│       │   └── linear.py    # Ridge regression
│       ├── ensemble/
│       │   └── xgb.py       # XGBoost
│       └── sequential/
│           └── lstm.py      # LSTM model (PyTorch)
└── README.md

Setup
1. Environment

Python 3.9+ is recommended.

Install required dependencies:

pip install numpy pandas scikit-learn pyyaml torch xgboost


GPU is optional. LSTM will automatically run on CPU if CUDA is unavailable.

2. Dataset

Download the Hull Tactical Market Prediction dataset from Kaggle and place it at:

data/raw/train.csv


The dataset contains extensive missing values in early periods; preprocessing and feature selection are handled automatically by the pipeline.

Running Experiments

All experiments are configured via YAML files under configs/.
Training is executed through a single unified entrypoint.

1. Ridge Regression (Linear Baseline)
python -m scripts.train --config configs/baseline_ridge.yaml


Linear baseline with optional lagged features

Evaluated using expanding-window walk-forward validation

2. XGBoost (Ensemble Learner)
python -m scripts.train --config configs/xgboost.yaml


Gradient-boosted decision trees

Explicit exclusion of forward-looking variables to prevent data leakage

Strong non-linear baseline

3. LSTM (Sequential Model)
python -m scripts.train --config configs/lstm.yaml


Sliding-window sequence modeling (PyTorch)

Input and target variables are standardized using training-only statistics

RMSE is reported on the original target scale

Evaluation Protocol

Time-aware walk-forward validation (no shuffling)

Expanding training window with fixed-size test windows

RMSE reported on the original scale of market excess returns

All preprocessing (scaling, feature selection) is fit on training data only

Notes on Data Leakage

During development, anomalously strong performance from high-capacity models revealed the presence of forward-looking variables in the dataset. These variables were explicitly removed from the feature set, and all results reported are based on a corrected, leakage-free pipeline.

Reproducibility

Random seeds are fixed for NumPy and PyTorch

Model behavior is fully specified via configuration files

All experiments can be reproduced by re-running the commands above