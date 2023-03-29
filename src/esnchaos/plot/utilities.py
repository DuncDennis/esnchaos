from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np


### Utility:
def get_param_cols(df: pd.DataFrame) -> list:
    """Get all columns starting with <P >."""
    return [x for x in df.columns if x.startswith("P ")]


def get_metric_cols(df: pd.DataFrame) -> list:
    """Get all columns starting with <M >."""
    return [x for x in df.columns if x.startswith("M ")]


def get_x_param(df: pd.DataFrame,
                x_param: str | None) -> str:
    """Utility to get the x_param."""
    if x_param is None:
        x_params = get_param_cols(df)
        if len(x_params) == 0:
            raise ValueError("No parameter available in the df.")
        elif len(x_params) == 1:
            x_param = x_params[0]
        else:
            raise ValueError(f"Choose one of the values in {x_params} as x_param.")
    return x_param


def get_y_metric(df: pd.DataFrame,
                 y_metric: str | None) -> str:
    """Utility to get the x_param."""
    if y_metric is None:
        y_metrics = get_metric_cols(df)
        if len(y_metrics) == 0:
            raise ValueError("No metric available in the df.")
        elif len(y_metrics) == 1:
            y_metric = y_metrics[0]
        else:
            raise ValueError(f"Choose one of the values in {y_metrics} as y_metric.")
    return y_metric


def overwrite_plot_params(new_params: dict[str, Any] | None,
                          default_params: dict[str, Any]) -> dict[str, Any]:
    """Overwrite the default parameter dict with some values."""
    out_params = default_params.copy()
    if new_params is not None:
        for k, v in new_params.items():
            if k in out_params.keys():
                out_params[k] = v
            else:
                raise ValueError(f"The given key {k} is not available in plot_params."
                                 f"PLOT_ONE_DIM_PARAMS.")
    return out_params


def get_auto_axis_title(x_param_or_y_metric: str,
                        transform_dict: dict[str, str] = None,
                        latex_text_size: str = None
                        ) -> str:
    """Get the automatic axis title."""
    if transform_dict is None:
        title = str(x_param_or_y_metric)
    else:
        if x_param_or_y_metric in transform_dict.keys():
            title = transform_dict[x_param_or_y_metric]
            if latex_text_size is not None:
                title = rf"$\{latex_text_size}" + title
        else:
            title = str(x_param_or_y_metric)
    return title

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
