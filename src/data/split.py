import numpy as np


def simple_time_split(df, train_ratio=0.7):
    n = len(df)
    split = int(n * train_ratio)
    idx = np.arange(n)
    return idx[:split], idx[split:]
