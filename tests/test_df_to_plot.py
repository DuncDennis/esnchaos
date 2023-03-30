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


def test_plot_one_dim_violin_sweep(filtered_df_pcc_metric):
    params = dict(
        # y_axis_dict=dict(
        #     range=[-0.5, 9.0],
        #     tick0=0,
        #     dtick=2),
        # x_axis_dict=dict(
        #     range=[0, 0.5]
        # ),
        log_x=False,
        # color_green_at_val=0.0,
        color_green_at_val=None,
        no_xaxis_title=False,
        x_param_val_transform_dict={
            0: "a",
            0.1: "b",
            1/3: "c",
        },
    )

    fig = df_to_plot.plot_one_dim_violin_sweep(
        filtered_df_pcc_metric,
        x_param="P node_bias_scale",
        # y_metric="M VALIDATE VT",
        y_metric="M TRAIN PCMAX",
        # y_metric="M TRAIN MSE",
        params=params
    )
    assert type(fig) == go.Figure


def test_plot_two_dim_sweep(agg_df):
    params = dict(
        hex_color_list=["#000000"],
        line_style_list=["dash", "dot"],
        color_alpha=0.5,
        col_param_val_rename_func=lambda x: {
            30: "a",
            40: "b",
        }[x],
        col_param_val_order_dict={
            "a": 2,
            "b": 1,
        },
        no_xaxis_title=True
    )

    fig = df_to_plot.plot_two_dim_sweep(
        agg_df,
        x_param="P node_bias_scale",
        y_metric="M VALIDATE VT",
        col_param="P r_dim",
        params=params
    )
    assert type(fig) == go.Figure
