# Hull Tactical Market Prediction (CSCI-4364/6364 F25)

This repo contains a config-driven, leakage-aware ML pipeline for the Kaggle Hull Tactical Market Prediction dataset.

## Data layout (local only, not committed)
- data/raw/train.csv
- data/raw/test.csv (optional)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train.py --config configs/baseline_ridge.yaml
python scripts/eval.py --run_dir runs/<RUN_ID>
```
