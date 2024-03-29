"""Create plots from dataframes created from sweep experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import itertools

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

    return fig


def plot_two_dim_sweep(agg_df: pd.DataFrame,
                       x_param: None | str = None,
                       y_metric: None | str = None,
                       col_param: None | str = None,
                       params: None | dict = None,
                       ) -> go.Figure:
    """Plot two-dimensional sweep x_param vs y_metric with error bars."""

    x_param = pu.get_x_param(agg_df, x_param)
    y_metric = pu.get_y_metric(agg_df, y_metric)

    if col_param is None: # TODO: also make it smart when None is selected.
        raise NotImplementedError("col_param = None is not implemented yet. Please "
                                  "specify the col_param.")

    # Modify parameters if applicable.
    p = pu.overwrite_plot_params(params,
                                 default_params=pp.DEFAULT_PLOT_TWO_DIM_PARAMS)

    # rename and order col_param values:
    if p["col_param_val_rename_func"] is not None:
        agg_df[col_param] = agg_df[col_param].apply(p["col_param_val_rename_func"])

    col_param_vals = pd.Series(agg_df[col_param].value_counts().index)
    if p["col_param_val_order_dict"] is not None:
        col_param_vals.sort_values(
            key=lambda series: series.apply(lambda x: p["col_param_val_order_dict"][x]),
            inplace=True)

    fig = go.Figure()

    # color list:
    if p["hex_color_list"] is None:
        hex_color_list = pio.templates[p["template"]].layout.colorway
    else:
        hex_color_list = p["hex_color_list"]
    col_pal_iterator = itertools.cycle(hex_color_list)

    # line style list:
    if p["line_style_list"] is None:
        line_style_list = ["solid"]
    else:
        line_style_list = p["line_style_list"]
    line_style_iterator = itertools.cycle(line_style_list)


    for col_param_val in col_param_vals:
        sub_df = agg_df[agg_df[col_param] == col_param_val]

        # name:
        trace_name = str(col_param_val)

        # Line and color:
        hex_color = next(col_pal_iterator)
        rgba_line = pu.hex_to_rgba(hex_color, p["color_alpha"])
        line_style = next(line_style_iterator)

        # Add trace:
        fig.add_trace(
            go.Scatter(
                x=sub_df[x_param],
                y=sub_df[y_metric + "|avg"],
                error_y={"array": sub_df[y_metric + "|error_high"],
                         "arrayminus": sub_df[y_metric + "|error_low"],
                         "width": p["error_barwidth"],
                         "thickness": p["error_thickness"]},
                line=dict(
                    width=p["line_width"],
                    color=rgba_line,
                    dash=line_style
                ),
                name=trace_name
            )
        )
    # logarithmic x-axis
    if p["log_x"]:
        fig.update_xaxes(type="log",
                         exponentformat=p["exponentformat"])

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
    fig.update_xaxes(**p["x_grid_settings_dict"])
    fig.update_yaxes(**p["y_grid_settings_dict"])

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

    fig.update_layout(
        legend=p["legend_dict"],
        showlegend=p["show_legend"]
    )

    return fig


def plot_m_vs_m_scatter(agg_df: pd.DataFrame,
                        x_metric: str,
                        y_metric: str,
                        col_param: str,
                        params: dict[str, Any]) -> go.Figure:
    """Scatter plot with error bars metric vs metric with colored points."""


    # Modify parameters if applicable.
    p = pu.overwrite_plot_params(params,
                                 default_params=pp.DEFAULT_PLOT_M_VS_M_PARAMS)

    col_title = pu.get_auto_axis_title(col_param,
                                       p["param_transform_html"])

    # log color:
    if p["log_col"]:
        agg_df[col_param] = np.log10(agg_df[col_param])
        tick_prefix = "10<sup>"
        tick_suffix = "</sup>"
    else:
        tick_prefix = ""
        tick_suffix = ""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=agg_df[x_metric + "|" + "avg"],
                   y=agg_df[y_metric + "|" + "avg"],
                   error_y=dict(
                       array=agg_df[y_metric + "|" + "error_high"],
                       arrayminus=agg_df[y_metric + "|" + "error_low"],
                       thickness=p["error_thickness"]
                   ),
                   error_x=dict(
                       array=agg_df[x_metric + "|" + "error_high"],
                       arrayminus=agg_df[x_metric + "|" + "error_low"],
                       thickness=p["error_thickness"]
                   ),
                   mode="markers",
                   marker=dict(color=agg_df[col_param],

                               colorbar=dict(
                                   len=0.95,
                                   tickprefix=tick_prefix,
                                   ticksuffix=tick_suffix,
                                   dtick=p["color_dtick"],
                                   tick0=p["color_tick0"],
                                   ticks="outside",
                                   title=col_title,
                                   orientation="v",
                               ),
                               colorscale="portland",
                               size=p["marker_size"],
                               line=dict(width=p["marker_line_width"],
                                         color=p["marker_line_color"]))
                   ),
    )


    # y axis title:
    yaxis_title = pu.get_auto_axis_title(y_metric,
                                         p["metric_transform_ltx"],
                                         p["latex_text_size"])
    fig.update_layout(
        yaxis_title=yaxis_title
    )

    # x axis title:
    xaxis_title = pu.get_auto_axis_title(x_metric,
                                         p["metric_transform_ltx"],
                                         p["latex_text_size"])
    fig.update_layout(
        xaxis_title=xaxis_title
    )

    # Axis ticks and range:
    fig.update_yaxes(**p["y_axis_dict"])
    fig.update_xaxes(**p["x_axis_dict"])

    # Grid:
    fig.update_xaxes(**p["x_grid_settings_dict"])
    fig.update_yaxes(**p["y_grid_settings_dict"])

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
    return fig
