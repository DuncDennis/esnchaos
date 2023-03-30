"""Read all pkl files from a designated folder and save the plots.
"""

from pathlib import Path

import esnchaos.plot.pkl_to_df as pkl_to_df
import esnchaos.plot.df_to_plot as df_to_plot
import esnchaos.plot.utilities as plot_utils

RESULTS_PKL_DIR = "original_data"
OUTPUT_DIR = "created_plots"
Y_METRIC = "M VALIDATE VT"
AVG_MODE = "median_and_quartile"

Y_AXIS_DICT = dict(
    range=[-0.5, 9.0],
    tick0=0,
    dtick=2)

X_AXIS_DICT_PARAM_DEPENDENT = {
    "P node_bias_scale": dict(
        tick0=0,
        dtick=0.2),
    "P reg_param": dict(
        dtick=2),
    "P n_avg_deg": {"dtick": 1},
}

# DEFAULT PARAMETERS:
VERTICAL_LINE_VAL_PARAM_DEPENDENT = {
    "P node_bias_scale": 0.4,
    "P t_train": 2000,
    "P r_dim": 500,
    "P reg_param": 1e-7,
    "P w_in_scale": 1.0,
    "P n_avg_deg": 5.0,
    "P n_rad": 0.4,
    "P dt": 0.1,
    "P x_train_noise_scale": 0.0,
}

# Log x-axis for some parameters:
LOG_X_PARAMS = [
    "P reg_param",
    "P n_avg_deg"
]

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
        )

        # get x_param
        x_param = plot_utils.get_param_cols(df_agg)[0]

        # set x_axis_dict per parameter:
        if x_param in X_AXIS_DICT_PARAM_DEPENDENT.keys():
            params["x_axis_dict"] = X_AXIS_DICT_PARAM_DEPENDENT[x_param]

        # vertical line to mark one parameter:
        if x_param in VERTICAL_LINE_VAL_PARAM_DEPENDENT.keys():
            params["vertical_line_val"] = VERTICAL_LINE_VAL_PARAM_DEPENDENT[x_param]

        # log x:
        if x_param in LOG_X_PARAMS:
            params["log_x"] = True

        fig = df_to_plot.plot_one_dim_sweep(df_agg,
                                            x_param=x_param,
                                            y_metric=Y_METRIC,
                                            params=params)
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        fig.write_image(f"{OUTPUT_DIR}/{out_plot_name}.png", scale=3)
