# Hull Tactical Market Prediction (ML)

This project studies machine learning approaches for predicting daily U.S. equity market excess returns using the Hull Tactical Market Prediction dataset. Linear, ensemble, and sequential models are evaluated under a unified time-series pipeline.

---

## Setup

Python 3.9+ is required.

Install dependencies:

    pip install numpy pandas scikit-learn pyyaml torch xgboost

Download the Kaggle dataset and place it at:

    data/raw/train.csv

---

## Run Experiments

All experiments are driven by YAML configuration files.

Ridge Regression (Linear Baseline):

    python -m scripts.train --config configs/baseline_ridge.yaml

XGBoost (Ensemble Learner):

    python -m scripts.train --config configs/xgboost.yaml

LSTM (Sequential Model):

    python -m scripts.train --config configs/lstm.yaml

---

## Evaluation

- Time-aware walk-forward validation (no shuffling)
- Expanding training window with fixed test size
- RMSE reported on the original return scale
- All preprocessing is fit on training data only

---

## Notes

Forward-looking variables were explicitly removed to prevent information leakage.
All results are produced using a corrected, leakage-free pipeline.
