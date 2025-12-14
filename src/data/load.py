import pandas as pd


def load_csv_time_sorted(path: str, time_col: str):
    df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)
    return df
