"""Create plots from dataframes created from sweep experiments."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objs as go

import esnchaos.plot.plot_params as pp
import esnchaos.plot.utilities as pu

def plot_one_dim_sweep(agg_df: pd.DataFrame,
                       x_param: None | str = None,
                       y_metric: None | str = None,
                       params: None | dict = None,
                       ) -> go.Figure:
    """Plot one dimensional sweep x_param vs y_metric with error bars."""

    x_param = pu.get_x_param(agg_df, x_param)
    y_metric = pu.get_y_metric(agg_df, y_metric)


    # Modify parameters if applicable.
    p = pu.overwrite_plot_params(params,
                                 default_params=pp.DEFAULT_PLOT_ONE_DIM_PARAMS)

    fig = go.Figure()

    # Add trace:
    fig.add_trace(
        go.Scatter(
            x=agg_df[x_param],
            y=agg_df[y_metric + "|avg"],
            error_y={"array": agg_df[y_metric + "|error_high"],
                     "arrayminus": agg_df[y_metric + "|error_low"],
                     "width": p["error_barwidth"],
                     "thickness": p["error_thickness"]
                     },
            line=dict(
                width=p["line_width"],
                color=p["line_color"]
            )
        )
    )

    # vertical line to indicate some sweep-value:
    if p["vertical_line_val"] is not None:
        fig.add_vline(
            x=p["vertical_line_val"],
            **p["vertical_line_dict"]
        )

    # logarithmic x-axis
    if p["log_x"]:
        fig.update_xaxes(type="log",
                         exponentformat=p["exponentformat"])

    # x axis title:
    xaxis_title = pu.get_auto_axis_title(x_param,
                                         p["param_transform_ltx"],
                                         p["latex_text_size"])
    fig.update_layout(
        xaxis_title=xaxis_title
    )

    # y axis title:
    yaxis_title = pu.get_auto_axis_title(y_metric,
                                         p["metric_transform_ltx"],
                                         p["latex_text_size"])
    fig.update_layout(
        yaxis_title=yaxis_title
    )

    # Axis ticks and range:
    fig.update_yaxes(**p["y_axis_dict"])
    fig.update_xaxes(**p["x_axis_dict"])

    # Grid:
    fig.update_yaxes(**p["grid_settings_dict"])

    # layout:
    fig.update_layout(
        width=p["width"],
        height=p["height"],
        template=p["template"],
        font=dict(
            size=p["font_size"],
            family=p["font_family"]
        ),
        margin=p["margin_dict"]
    )

    # show:
    fig.show()

    return fig


def plot_one_dim_violin_sweep(df: pd.DataFrame,
                              x_param: None | str = None,
                              y_metric: None | str = None,
                              params: None | dict = None,
                              ) -> go.Figure:
    """Plot violin plots of y_metric over ensemble of calculations."""

    x_param = pu.get_x_param(df, x_param)
    y_metric = pu.get_y_metric(df, y_metric)

    # Modify parameters if applicable.
    p = pu.overwrite_plot_params(params,
                                 default_params=pp.DEFAULT_PLOT_ONE_DIM_VIOLIN_PARAMS)

    fig = go.Figure()

    # Add traces - Violin plots:
    for i, val in enumerate(df[x_param].unique()):
        sub_df = df[df[x_param] == val]

        if p["color_green_at_val"] is not None:
            if val == p["color_green_at_val"]:
                color = "green"
            else:
                color = "black"
        else:
            color = None

        fig.add_trace(
            go.Violin(x=sub_df[x_param],
                      y=sub_df[y_metric],
                      box_visible=True,
                      line_color=color,
                      points="all",
                      marker_size=3,
                      # points=False
                      ))

    # parameter values transformation:
    if p["x_param_val_transform_dict"] is not None:
        trans_dict = p["x_param_val_transform_dict"]
        original_ticks = df[x_param].unique()
        new_ticks = [trans_dict[x] for x in original_ticks]
        fig.update_xaxes(ticktext=new_ticks,
                         tickvals=original_ticks)

    # logarithmic x-axis
    if p["log_x"]:
        fig.update_xaxes(type="log",
                         exponentformat=p["exponentformat"])

    # y axis title:
    yaxis_title = pu.get_auto_axis_title(y_metric,
                                         p["metric_transform_ltx"],
                                         p["latex_text_size"])
    fig.update_layout(
        yaxis_title=yaxis_title
    )

    # x axis title:
    if p["no_xaxis_title"]:
        xaxis_title = None
    else:
        xaxis_title = pu.get_auto_axis_title(x_param,
                                             p["param_transform_ltx"],
                                             p["latex_text_size"])
    fig.update_layout(
        xaxis_title=xaxis_title
    )

    # Axis ticks and range:
    fig.update_yaxes(**p["y_axis_dict"])
    fig.update_xaxes(**p["x_axis_dict"])

    # Grid:
    fig.update_yaxes(**p["grid_settings_dict"])

    # layout:
    fig.update_layout(
        width=p["width"],
        height=p["height"],
        template=p["template"],
        font=dict(
            size=p["font_size"],
            family=p["font_family"]
        ),
        margin=p["margin_dict"],
        showlegend=False
    )

    fig.show()
    return fig
