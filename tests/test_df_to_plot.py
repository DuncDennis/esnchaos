import plotly.graph_objs as go

import esnchaos.plot.df_to_plot as df_to_plot


def test_plot_one_dim_sweep(fil_agg_df):
    fig = df_to_plot.plot_one_dim_sweep(
        fil_agg_df,
        x_param="P node_bias_scale",
        y_metric="M VALIDATE VT",
    )
    assert type(fig) == go.Figure
