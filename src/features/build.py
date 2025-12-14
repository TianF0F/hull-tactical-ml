from src.features.base import add_lag_features, add_rolling_features


def build_features(df, cfg):
    feature_cols = df.select_dtypes("number").columns.tolist()

    # grouping columns by prefix
    market_cols = [c for c in feature_cols if c.startswith("M")]
    vol_cols = [c for c in feature_cols if c.startswith("V")]
    sent_cols = [c for c in feature_cols if c.startswith("S")]

    df = add_lag_features(
        df,
        cols=market_cols + vol_cols + sent_cols,
        lags=cfg["features"]["lags"],
    )

    df = add_rolling_features(
        df,
        cols=market_cols + vol_cols,
        windows=cfg["features"]["rolling_windows"],
    )

    return df
