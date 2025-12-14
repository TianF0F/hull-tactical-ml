import pandas as pd


def add_lag_features(df, cols, lags):
    lagged_feats = {}

    for lag in lags:
        for c in cols:
            lagged_feats[f"{c}_lag{lag}"] = df[c].shift(lag)

    if lagged_feats:
        df = pd.concat([df, pd.DataFrame(lagged_feats, index=df.index)], axis=1)

    return df


def add_rolling_features(df, cols, windows, stats=("mean",)):
    feats = {}
    for w in windows:
        for c in cols:
            if "mean" in stats:
                feats[f"{c}_roll{w}_mean"] = df[c].shift(1).rolling(w).mean()
    if feats:
        df = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
    return df

