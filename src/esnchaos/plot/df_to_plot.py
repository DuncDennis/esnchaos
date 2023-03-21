"""Create plots from dataframes created from sweep experiments."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objs as go

import esnchaos.plot.utilities as plot_utils

def plot_one_dim_sweep(agg_df: pd.DataFrame,
                       x_param: None | str = None,
                       y_metric: None | str = None,
                       ) -> go.Figure:
    fig = go.Figure()

    # Add trace:
    fig.add_trace(
        go.Scatter(
            x=agg_df[x_param],
            y=agg_df[y_metric + "|avg"],
            error_y={"array": agg_df[y_metric + "|error_high"],
                     "arrayminus": agg_df[y_metric + "|error_low"],
            #          "width": ERRORWIDTH,
            #          "thickness": ERRORLINEWIDTH
                     },
            # line=dict(
            #     width=LINEWIDTH,
            #     color=LINECOLOR
            # )
        )
    )

    # show:
    fig.show()

    return fig
