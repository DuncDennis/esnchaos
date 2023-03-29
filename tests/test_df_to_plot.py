import plotly.graph_objs as go

import esnchaos.plot.df_to_plot as df_to_plot


def test_plot_one_dim_sweep(fil_agg_df):
    params = dict(
        # y_axis_dict=dict(
        #     range=[-0.5, 9.0],
        #     tick0=0,
        #     dtick=2),
        # x_axis_dict=dict(
        #     range=[0, 0.5]
        # ),
        vertical_line_val=0.1,
        log_x=True,
        # default_param_vals_dict={
        #     "P node_bias_scale": 0.1
        # },
        # log_x_params=["P node_bias_scale"]
    )

    fig = df_to_plot.plot_one_dim_sweep(
        fil_agg_df,
        x_param="P node_bias_scale",
        y_metric="M VALIDATE VT",
        params=params
    )
    assert type(fig) == go.Figure
