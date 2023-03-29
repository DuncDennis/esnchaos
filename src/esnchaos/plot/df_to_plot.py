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
    """Plot one dimensional sweep x_param vs y_metric."""

    # get x_param and y_metric
    if x_param is None:
        x_params = pu.get_param_cols(agg_df)
        if len(x_params) == 0:
            raise ValueError("No parameter available in the df.")
        elif len(x_params) == 1:
            x_param = x_params[0]
        else:
            raise ValueError(f"Choose one of the values in {x_params} as x_param.")
    if y_metric is None:
        y_metrics = pu.get_metric_cols(agg_df)
        if len(y_metrics) == 0:
            raise ValueError("No metric available in the df.")
        elif len(y_metrics) == 1:
            y_metric = y_metrics[0]
        else:
            raise ValueError(f"Choose one of the values in {y_metrics} as y_metric.")

    # Modify parameters if applicable.
    p = pp.DEFAULT_PLOT_ONE_DIM_PARAMS.copy()
    if params is not None:
        for k, v in params.items():
            if k in p.keys():
                p[k] = v
            else:
                raise ValueError(f"The given key {k} is not available in plot_params."
                                 f"PLOT_ONE_DIM_PARAMS.")

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
    if x_param in p["param_transform_ltx"].keys():
        xaxis_title = p["param_transform_ltx"][x_param]
        latex_text_size = p["latex_text_size"]
        xaxis_title = rf"$\{latex_text_size}" + xaxis_title
    else:
        xaxis_title = str(x_param)

    # y axis title:
    if y_metric in p["metric_transform_ltx"].keys():
        yaxis_title = p["metric_transform_ltx"][y_metric]
        latex_text_size = p["latex_text_size"]
        yaxis_title = rf"$\{latex_text_size}" + yaxis_title
    else:
        yaxis_title = str(y_metric)

    # set Axis title:
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
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
