"""Read all pkl files from a designated folder and save the plots.
"""

from pathlib import Path
import plotly.io as pio

import esnchaos.plot.pkl_to_df as pkl_to_df
import esnchaos.plot.df_to_plot as df_to_plot
import esnchaos.plot.utilities as plot_utils

col_pal_simple_white = pio.templates["simple_white"].layout.colorway

RESULTS_PKL_DIR = "sweep_results_fitted"
OUTPUT_DIR = "created_plots_fitted"
Y_METRIC = "M VALIDATE VT"
COL_PARAM = "P predictor_type"
AVG_MODE = "median_and_quartile"

Y_AXIS_DICT = dict(
)

X_AXIS_DICT = dict(
    dtick=0.2,
    tick0=0,
)

GRID_SETTINGS = dict(
    showgrid=True,
    gridwidth=1,
    gridcolor="gray"
)

# Log x-axis for some parameters:
LOG_X_PARAMS = [
    "P model_error_eps",
    "P x_train_noise_scale"
]

HYBRIDORDER = {"only reservoir": 1,
               "input hybrid": 2,
               "output hybrid": 3,
               "full hybrid": 4,
               "only model": 5,
               "model fitted": 6}

HYBRIDNAMES = {"no_hybrid": "only reservoir",
               "input_hybrid": "input hybrid",
               "output_hybrid": "output hybrid",
               "full_hybrid": "full hybrid",
               "model_predictor": "only model",
               "model_predictor_fitted": "model fitted"}

# NAMES FOR SYSTEMS:
def rename_windmi(x: str):
    if x == "WindmiAttractor":
        return "Windmi"
    else: return x

if __name__ == "__main__":
    path = Path(RESULTS_PKL_DIR)

    pkl_files = list(path.glob("*.pkl"))

    for i, pkl_file_path in enumerate(pkl_files):
        out_plot_name = pkl_file_path.name.split(".")[0]
        df = pkl_to_df.read_pkl(pkl_file_path)

        df_pre_fil = pkl_to_df.pre_filter_df(df, rmv_const_cols=True)

        df_agg = pkl_to_df.aggregate_df(df_pre_fil,
                                        avg_mode=AVG_MODE)

        # get x_param:
        param_cols = plot_utils.get_param_cols(df_agg)
        no_xaxis_title=False
        if "P system" in param_cols:
            x_param = "P system"
            no_xaxis_title=True
            df_agg["P system"] = df_agg["P system"].apply(rename_windmi)
        elif "P r_dim" in param_cols:
            x_param = "P r_dim"
        elif "P model_error_eps" in param_cols:
            x_param = "P model_error_eps"
        elif "P t_train" in param_cols:
            x_param = "P t_train"
        elif "P x_train_noise_scale" in param_cols:
            x_param = "P x_train_noise_scale"
        elif "P n_rad" in param_cols:
            x_param = "P n_rad"
        else:
            x_param = None
            print("Not recognized")

        # adjust dtick for some files:
        if out_plot_name.startswith("lorenz_eps_model_effect_of_r_dim"):
            y_axis_dict = {"dtick": 5}
        elif out_plot_name.startswith("all_systems_dimselect_large_res"):
            y_axis_dict = {"dtick": 5}
        else:
            y_axis_dict = Y_AXIS_DICT

        # global params:
        params = dict(
            y_axis_dict=y_axis_dict,
            x_axis_dict=X_AXIS_DICT,
            hex_color_list=[
                "#000000",  # black
                col_pal_simple_white[1],  # blue
                col_pal_simple_white[3],  # red
                col_pal_simple_white[2],  # green
                "#797979",  # gray,
                "#2596be"  # blue,

            ],
            line_style_list=[
                "dash",
                "solid",
                "solid",
                "solid",
                "dot",
                "dot"
            ],
            width=650,
            height=450,
            line_width=4,
            error_thickness=1.7,
            color_alpha=0.75,
            y_grid_settings_dict=GRID_SETTINGS,
            col_param_val_rename_func=lambda x: HYBRIDNAMES[x],
            col_param_val_order_dict=HYBRIDORDER,
            no_xaxis_title=no_xaxis_title
        )

        # log x:
        if x_param in LOG_X_PARAMS:
            params["log_x"] = True

        fig = df_to_plot.plot_two_dim_sweep(df_agg,
                                            x_param=x_param,
                                            y_metric=Y_METRIC,
                                            col_param=COL_PARAM,
                                            params=params)

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        fig.write_image(f"{OUTPUT_DIR}/{out_plot_name}.png", scale=3)
