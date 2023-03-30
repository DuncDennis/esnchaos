"""Read all pkl files from a designated folder and save the plots.
TODO: For now does not handle the saving of the plots -> has to be implemented.
"""

from pathlib import Path
import plotly.io as pio

import esnchaos.plot.pkl_to_df as pkl_to_df
import esnchaos.plot.df_to_plot as df_to_plot
import esnchaos.plot.utilities as plot_utils

col_pal_simple_white = pio.templates["simple_white"].layout.colorway

RESULTS_PKL_DIR = "original_data"
Y_METRIC = "M VALIDATE VT"
COL_PARAM = "P perform_pca_bool"
AVG_MODE = "median_and_quartile"

Y_AXIS_DICT = dict(
    dtick=5
)

X_AXIS_DICT = dict(
    dtick=2
)

GRID_SETTINGS = dict(
    showgrid=True,
    gridwidth=1,
    gridcolor="gray"
)

# Log x-axis for some parameters:
LOG_X_PARAMS = [
    "P reg_param",
    "P n_avg_deg"
]

if __name__ == "__main__":
    path = Path(RESULTS_PKL_DIR)

    pkl_files = list(path.glob("*.pkl"))

    for i, pkl_file_path in enumerate(pkl_files):

        df = pkl_to_df.read_pkl(pkl_file_path)

        df_pre_fil = pkl_to_df.pre_filter_df(df, rmv_const_cols=True)

        df_agg = pkl_to_df.aggregate_df(df_pre_fil,
                                        avg_mode=AVG_MODE)

        # global params:
        params = dict(
            y_axis_dict=Y_AXIS_DICT,
            x_axis_dict=X_AXIS_DICT,
            hex_color_list=[col_pal_simple_white[1], # orange
                            col_pal_simple_white[2]], # green
            width=550,
            height=400,
            line_width=4,
            error_thickness=1.7,
            color_alpha=0.75,
            x_grid_settings_dict=GRID_SETTINGS,
            y_grid_settings_dict=GRID_SETTINGS,
            col_param_val_rename_func=lambda x: {True: "PC-transform", False: "no PC-transform"}[x]
        )

        # get x_param
        x_param = plot_utils.get_param_cols(df_agg)[0]

        # log x:
        if x_param in LOG_X_PARAMS:
            params["log_x"] = True

        fig = df_to_plot.plot_two_dim_sweep(df_agg,
                                            x_param=x_param,
                                            y_metric=Y_METRIC,
                                            col_param=COL_PARAM,
                                            params=params)

        fig.write_image(f"test.png", scale=3)
