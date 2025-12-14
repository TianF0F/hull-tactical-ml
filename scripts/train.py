# training entrypoint
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.data.load import load_csv_time_sorted
from src.models.baseline.linear import make_model


def main():
    # ===== 1. read configs =====
    config_path = Path("configs/baseline_ridge.yaml")
    cfg = yaml.safe_load(config_path.read_text())

    time_col = cfg["run"]["time_col"]
    target_col = cfg["run"]["target_col"]

    # ===== 2. read data =====
    df = load_csv_time_sorted(
        cfg["data"]["raw_path"],
        time_col=time_col,
    )

    # drop cols
    for c in cfg["data"].get("drop_cols", []):
        if c in df.columns:
            df = df.drop(columns=c)

    # ===== 3. construct X / y =====
    y = df[target_col].values
    X = df.drop(columns=[target_col, time_col])

    # only numeric features
    X = X.select_dtypes(include="number")

    # ===== 3.5 drop cols with high missing ratio (DO THIS BEFORE dropping rows) =====
    max_missing = cfg["features"].get("max_missing_ratio", None)
    if max_missing is not None:
        miss_ratio = X.isna().mean()
        keep_cols = miss_ratio[miss_ratio <= max_missing].index
        X = X[keep_cols]

    # ===== 3.6 discard rows that still have NaNs (optional) =====
    if cfg["features"].get("drop_na_rows", False):
        mask = ~X.isna().any(axis=1)
        X = X.loc[mask].reset_index(drop=True)
        y = y[mask.to_numpy()]

    # ===== 4. time split =====
    n = len(X)
    train_ratio = cfg["split"]["train_ratio"]
    split = int(n * train_ratio)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    # ===== 5. preprocess =====
    if cfg["preprocess"]["scale"] == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values

    # ===== 6. Train Model =====
    model = make_model(**cfg["model"]["params"])
    model.fit(X_train, y_train)

    # ===== 7. Evaluation =====
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print("=" * 50)
    print("Baseline Ridge Result")
    print(f"Samples used      : {n}")
    print(f"Train / Test size : {len(X_train)} / {len(X_test)}")
    print(f"RMSE              : {rmse:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
