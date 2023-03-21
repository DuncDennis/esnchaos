"""Functionality to plot the output of sweep_experiments."""

from __future__ import annotations

import pathlib

import pandas as pd

def read_pkl(path: str | pathlib.Path) -> pd.DataFrame:
    """Read the pkl file and return a Pandas Dataframe."""
    if type(path) is str:
        path_obj = pathlib.Path(path)
    else:
        path_obj = path
    df = pd.read_pickle(path_obj)
    return df
