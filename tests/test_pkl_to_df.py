import pytest

import pandas as pd

import esnchaos.plot.pkl_to_df as pkl_to_df
import esnchaos.plot.utilities as plot_utils

import tests.example_sweep_experiment as ese



def test_run_and_load_sweep_exp(tmp_path):
    """Run the example sweep exp and load the resulting pkl as a Pandas df."""
    file_path = ese.run(tmp_path)

    df = pkl_to_df.read_pkl(file_path)
    assert type(df) == pd.DataFrame


def test_results_df_shape(results_df):
    """Test the shape of the results_df."""
    df = results_df
    assert df.shape == (48, 37)


def test_prefilter_df(results_df):
    """Test if the pre-filtering of the dataframe works."""

    # Check if nothing is done if rmv_const_cols=False is parsed.
    not_filtered_df = pkl_to_df.pre_filter_df(results_df,
                                              excluded_params=None,
                                              rmv_const_cols=False)
    assert results_df.equals(not_filtered_df)

    # Check if const cols are removed:
    const_col_rmv_df = pkl_to_df.pre_filter_df(results_df,
                                               excluded_params=None,
                                               rmv_const_cols=True)
    parameter_cols = [x for x in const_col_rmv_df.columns if x.startswith("P ")]
    assert all(const_col_rmv_df[parameter_cols].nunique() != 1)

    # Check if parameters are excluded with excluded_params:
    excl_params_df = pkl_to_df.pre_filter_df(results_df,
                                             excluded_params={"P r_dim": [30]},
                                             rmv_const_cols=False)
    assert excl_params_df.shape[0] * 2 == results_df.shape[0]
    assert excl_params_df.shape[1] == results_df.shape[1]


def test_data_aggregation(results_df):
    """Test the average and error calculation."""
    # get df_agg_mean and df_agg_median
    df_agg_mean = pkl_to_df.aggregate_df(results_df,
                                         avg_mode="mean_and_std")
    df_agg_median = pkl_to_df.aggregate_df(results_df,
                                           avg_mode="median_and_quartile")

    # check columns:
    orig_metric_cols = plot_utils.get_metric_cols(results_df)
    new_metric_cols = plot_utils.get_metric_cols(df_agg_mean)
    for orig_metric_col in orig_metric_cols:
        assert orig_metric_col + "|avg" in new_metric_cols
        assert orig_metric_col + "|error_low" in new_metric_cols
        assert orig_metric_col + "|error_high" in new_metric_cols

    # check output size:
    parameter_cols = plot_utils.get_param_cols(results_df)
    assert df_agg_mean.shape[0] == results_df.value_counts(subset=parameter_cols).size

    # check that df_agg_mean and df_agg_median have different values:
    assert not df_agg_mean.equals(df_agg_median)

    # check metrics subset:
    df_agg_metrics_sub = pkl_to_df.aggregate_df(results_df,
                                                metrics_subset=["M VALIDATE MSE"])

    metric_cols_subset = plot_utils.get_metric_cols(df_agg_metrics_sub)
    assert "M VALIDATE MSE|avg" in metric_cols_subset
    assert "M VALIDATE MSE|error_low" in metric_cols_subset
    assert "M VALIDATE MSE|error_high" in metric_cols_subset
    assert len(metric_cols_subset) == 3

    # Check error raise:
    with pytest.raises(ValueError):
        _ = pkl_to_df.aggregate_df(results_df,
                                   metrics_subset=["M DOES NOT EXIST"])



