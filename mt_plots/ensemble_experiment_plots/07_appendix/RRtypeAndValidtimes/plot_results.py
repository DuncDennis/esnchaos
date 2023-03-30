"""Read all pkl files from a designated folder and save the plots.
"""

from pathlib import Path
import plotly.io as pio

import esnchaos.plot.pkl_to_df as pkl_to_df
import esnchaos.plot.df_to_plot as df_to_plot
import esnchaos.plot.utilities as plot_utils

col_pal_simple_white = pio.templates["simple_white"].layout.colorway

RESULTS_PKL_DIR = "original_data"
OUTPUT_DIR = "created_plots"
Y_METRIC = "M VALIDATE VT_centered"
X_PARAM = "P data_offset"
COL_PARAM = "P rr_type"
AVG_MODE = "median_and_quartile"

Y_AXIS_DICT = dict(
    dtick=2
)

X_AXIS_DICT = dict(
)

GRID_SETTINGS = dict(
    showgrid=True,
    gridwidth=1,
    gridcolor="gray"
)

if __name__ == "__main__":
    path = Path(RESULTS_PKL_DIR)

    pkl_files = list(path.glob("*.pkl"))

    for i, pkl_file_path in enumerate(pkl_files):
        out_plot_name = pkl_file_path.name.split(".")[0]
        df = pkl_to_df.read_pkl(pkl_file_path)

        df_pre_fil = pkl_to_df.pre_filter_df(df, rmv_const_cols=True)

        df_agg = pkl_to_df.aggregate_df(df_pre_fil,
                                        avg_mode=AVG_MODE)

        # global params:
        params = dict(
            y_axis_dict=Y_AXIS_DICT,
            x_axis_dict=X_AXIS_DICT,
            width=700,
            height=500,
            line_width=3,
            error_thickness=2,
            color_alpha=1.0,
            y_grid_settings_dict=GRID_SETTINGS,
            metric_transform_ltx={
                "M VALIDATE VT_centered": r" t_\text{v} \lambda_\text{max}$"}
        )

        fig = df_to_plot.plot_two_dim_sweep(df_agg,
                                            x_param=X_PARAM,
                                            y_metric=Y_METRIC,
                                            col_param=COL_PARAM,
                                            params=params)

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        fig.write_image(f"{OUTPUT_DIR}/{out_plot_name}.png", scale=3)
