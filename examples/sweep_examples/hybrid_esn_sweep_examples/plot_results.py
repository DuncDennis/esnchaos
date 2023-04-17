"""Read all pkl files from a designated folder and save the plots."""

from pathlib import Path
import plotly.io as pio

import esnchaos.plot.pkl_to_df as pkl_to_df
import esnchaos.plot.df_to_plot as df_to_plot

col_pal_simple_white = pio.templates["simple_white"].layout.colorway

RESULTS_PKL_DIR = "sweep_results"
OUTPUT_DIR = "created_plots"
Y_METRIC = "M VALIDATE VT"
X_PARAM = "P model_error_eps"
COL_PARAM = "P predictor_type"
AVG_MODE = "median_and_quartile"

Y_AXIS_DICT = dict(
)

X_AXIS_DICT = dict(
)

GRID_SETTINGS = dict(
    showgrid=True,
    gridwidth=1,
    gridcolor="gray"
)

HYBRIDORDER = {"only reservoir": 1,
               "input hybrid": 2,
               "output hybrid": 3,
               "full hybrid": 4,
               "only model": 5}

HYBRIDNAMES = {"no_hybrid": "only reservoir",
               "input_hybrid": "input hybrid",
               "output_hybrid": "output hybrid",
               "full_hybrid": "full hybrid",
               "model_predictor": "only model"}

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

        # global params:
        params = dict(
            y_axis_dict=Y_AXIS_DICT,
            x_axis_dict=X_AXIS_DICT,
            hex_color_list=[
                "#000000",  # black
                col_pal_simple_white[1],  # blue
                col_pal_simple_white[3],  # red
                col_pal_simple_white[2],  # green
                "#797979"  # gray
            ],
            line_style_list=[
                "dash",
                "solid",
                "solid",
                "solid",
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
            log_x=True,
        )

        fig = df_to_plot.plot_two_dim_sweep(df_agg,
                                            x_param=X_PARAM,
                                            y_metric=Y_METRIC,
                                            col_param=COL_PARAM,
                                            params=params)

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        fig.write_image(f"{OUTPUT_DIR}/{out_plot_name}.png", scale=3)
