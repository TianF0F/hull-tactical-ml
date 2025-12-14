import pandas as pd


def build_features(df, target_col, time_col):
    y = df[target_col].values
    X = df.drop(columns=[target_col, time_col])
    X = X.select_dtypes(include="number")
    return X, y
