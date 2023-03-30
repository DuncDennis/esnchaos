"""Read all pkl files from a designated folder and save the plots.
"""

from pathlib import Path

import esnchaos.plot.pkl_to_df as pkl_to_df
import esnchaos.plot.df_to_plot as df_to_plot
import esnchaos.plot.utilities as plot_utils


RESULTS_PKL_DIR = "original_data"
OUTPUT_DIR = "created_plots"
X_METRIC = "M TRAIN PCMAX"
Y_METRIC = "M VALIDATE VT"
AVG_MODE = "median_and_quartile"

Y_AXIS_DICT = dict(
    range=[0, 9],
)

X_AXIS_DICT = dict(
    range=[0, 520],
)

# Log x-axis for some parameters:
LOG_COL_PARAMS = [
    "P reg_param",
    "P n_avg_deg",
    "P n_rad",
]

if __name__ == "__main__":
    path = Path(RESULTS_PKL_DIR)

    pkl_files = list(path.glob("*.pkl"))

    for i, pkl_file_path in enumerate(pkl_files):
        out_plot_name = pkl_file_path.name.split(".")[0]
        df = pkl_to_df.read_pkl(pkl_file_path)

        excluded_params = None
        color_dtick = None
        color_tick0 = None
        if out_plot_name.startswith("pc_max_effect_of_node_deg"):
            excluded_params = {"P n_avg_deg": [10**-3, 10**(-2.5), 10**(2.5)]}
        elif out_plot_name.startswith("pc_max_effect_of_nrad"):
            excluded_params = {"P n_rad": [10 ** -5, 10 ** (-4.5)]}
        elif out_plot_name.startswith("pc_max_effect_of_reg_param"):
            color_dtick = 3
            color_tick0 = -13
        df_pre_fil = pkl_to_df.pre_filter_df(df,
                                             rmv_const_cols=True,
                                             excluded_params=excluded_params)

        df_agg = pkl_to_df.aggregate_df(df_pre_fil,
                                        avg_mode=AVG_MODE)

        param_cols = plot_utils.get_param_cols(df_agg)
        col_param = param_cols[0]

        if col_param in LOG_COL_PARAMS:
            log_col = True
        else:
            log_col = False

        # global params:
        # TODO: add log x dependency
        params = dict(
            y_axis_dict=Y_AXIS_DICT,
            x_axis_dict=X_AXIS_DICT,
            log_col = log_col,
            color_dtick=color_dtick,
            color_tick0=color_tick0,
        )

        fig = df_to_plot.plot_m_vs_m_scatter(df_agg,
                                            x_metric=X_METRIC,
                                            y_metric=Y_METRIC,
                                            col_param=col_param,
                                            params=params)

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        fig.write_image(f"{OUTPUT_DIR}/{out_plot_name}.png", scale=3)
