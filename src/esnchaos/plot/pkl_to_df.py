"""Functionality to plot the output of sweep_experiments."""

from __future__ import annotations

import pathlib

import pandas as pd

import esnchaos.plot.utilities as plot_utils


def read_pkl(path: str | pathlib.Path) -> pd.DataFrame:
    """Read the pkl file and return a Pandas Dataframe."""
    if type(path) is str:
        path_obj = pathlib.Path(path)
    else:
        path_obj = path
    df = pd.read_pickle(path_obj)
    return df


def pre_filter_df(
        df: pd.DataFrame,
        excluded_params: None | dict[str, list] = None,
        rmv_const_cols: bool = True) -> pd.DataFrame:
    """Exclude some rows and delete constant columns."""

    df_out = df.copy()

    # Drop rows corresponding to specs in excluded_params.
    if excluded_params:
        for key, val in excluded_params.items():
            to_remove = ~df_out[key].isin(val)
            df_out = df_out[to_remove]

    # Remove all parameter columns which are constant throughout the DF.
    if rmv_const_cols:
        parameter_cols = plot_utils.get_param_cols(df_out)
        for col in parameter_cols:
            if len(df_out[col].unique()) == 1:
                df_out.drop(col, inplace=True, axis=1)

    return df_out


def aggregate_df(df: pd.DataFrame,
                 metrics_subset: None | list[str] = None,
                 avg_mode: str = "median_and_quartile"):
    """Aggregate data over ensemble. """

    # average mode:
    if avg_mode == "mean_and_std":
        avg_str = "mean"
        error_high_str = "std_high"
        error_low_str = "std_low"
        avg = plot_utils.mean
        error_high = plot_utils.std_high
        error_low = plot_utils.std_low

    elif avg_mode == "median_and_quartile":
        avg_str = "median"
        error_high_str = "quartile_high"
        error_low_str = "quartile_low"
        avg = plot_utils.median
        error_high = plot_utils.quartile_high
        error_low = plot_utils.quartile_low

    else:
        raise ValueError("avg_mode must be either mean_and_std or median_and_quartile.")

    stat_funcs = [avg, error_high, error_low]

    # aggregate:
    parameter_cols = plot_utils.get_param_cols(df)
    metric_cols = plot_utils.get_metric_cols(df)
    if metrics_subset is not None:
        for metric in metrics_subset:
            if metric not in metric_cols:
                raise ValueError(
                    f"Chosen metric {metric} is not in metric_cols: {metric_cols}")
        metric_cols = metrics_subset
    group_obj = df.groupby(parameter_cols, as_index=False)
    df_agg = group_obj[metric_cols].agg(stat_funcs).reset_index(inplace=False)

    # rename columns:
    avg_mode_rename = {
        avg_str: "avg",
        error_high_str: "error_high",
        error_low_str: "error_low",
    }
    df_agg.columns = df_agg.columns.map('|'.join).str.strip('|')
    prev_cols = df_agg.columns
    new_cols = []
    for x in prev_cols:
        if str(x).endswith(avg_str):
            new_cols.append(str(x).replace(avg_str, avg_mode_rename[avg_str]))
        elif str(x).endswith(error_high_str):
            new_cols.append(
                str(x).replace(error_high_str, avg_mode_rename[error_high_str]))
        elif str(x).endswith(error_low_str):
            new_cols.append(
                str(x).replace(error_low_str, avg_mode_rename[error_low_str]))
        else:
            new_cols.append(str(x))

    df_agg.rename(columns=dict(zip(prev_cols, new_cols)), inplace=True)

    return df_agg


