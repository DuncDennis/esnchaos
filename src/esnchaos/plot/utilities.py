import pandas as pd
import numpy as np

### Utility:
def get_param_cols(df: pd.DataFrame) -> list:
    """Get all columns starting with <P >."""
    return [x for x in df.columns if x.startswith("P ")]


def get_metric_cols(df: pd.DataFrame) -> list:
    """Get all columns starting with <M >."""
    return [x for x in df.columns if x.startswith("M ")]


### statistical functions:
def mean(x):
    return np.mean(x)


def std_low(x):
    return np.std(x) * 0.5


def std_high(x):
    return np.std(x) * 0.5


def median(x):
    return np.median(x)


def quartile_low(x):
    return np.median(x) - np.quantile(x, q=0.25)


def quartile_high(x):
    return np.quantile(x, q=0.75) - np.median(x)
