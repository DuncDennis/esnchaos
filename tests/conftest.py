import pytest

import pandas as pd

import esnchaos.plot.pkl_to_df as pkl_to_df

import tests.example_sweep_experiment as ese
import tests.example_sweep_experiment_pccutoff as ese_pcc


@pytest.fixture
def results_df(tmp_path) -> pd.DataFrame:
    """Simple fixture to run the experiment and to return the dataframe."""
    file_path = ese.run(tmp_path)

    df = pkl_to_df.read_pkl(file_path)
    return df

@pytest.fixture
def filtered_df(results_df) -> pd.DataFrame:
    """Simple fixture to return a df with const cols removed and only one P column."""
    return pkl_to_df.pre_filter_df(results_df,
                                   excluded_params={"P r_dim": [30]},
                                   rmv_const_cols=True)

@pytest.fixture
def agg_df(results_df) -> pd.DataFrame:
    """Simple fixture to return the aggregated df."""
    return pkl_to_df.aggregate_df(results_df)

@pytest.fixture
def fil_agg_df(filtered_df) -> pd.DataFrame:
    """Simple fixture to return the aggregated df."""

    return pkl_to_df.aggregate_df(filtered_df)

@pytest.fixture
def results_df_pcc_metric(tmp_path) -> pd.DataFrame:
    """Simple fixture to run the experiment with pcmetric and to return the dataframe."""
    file_path = ese_pcc.run(tmp_path)

    df = pkl_to_df.read_pkl(file_path)
    return df

@pytest.fixture
def filtered_df_pcc_metric(results_df_pcc_metric) -> pd.DataFrame:
    """Simple fixture to return a df with const cols removed and only one P column using the pcc_metric."""
    return pkl_to_df.pre_filter_df(results_df_pcc_metric,
                                   excluded_params={"P r_dim": [30]},
                                   rmv_const_cols=True)

@pytest.fixture
def fil_agg_df_pcc_metric(filtered_df_pcc_metric) -> pd.DataFrame:
    """Simple fixture to return the aggregated df."""

    return pkl_to_df.aggregate_df(filtered_df_pcc_metric)
