"""Read all pkl files from a designated folder and save the plots.
TODO: For now does not handle the saving of the plots -> has to be implemented.
"""

from pathlib import Path

import esnchaos.plot.pkl_to_df as pkl_to_df
import esnchaos.plot.df_to_plot as df_to_plot
import esnchaos.plot.utilities as plot_utils

RESULTS_PKL_DIR = "original_data"
Y_METRIC = "M VALIDATE VT"

# Y-axis range and ticks:
Y_AXIS_DICT = dict(
    range=[0, 20],
    tick0 = 0,
    dtick = 5,
)

# DEFAULT PARAMETERS:
DEFAULT_VAL_PARAM_DEPENDENT = {
    "P rr_type": "b",
    "P r_to_rgen_opt": "linear_r",
}

# Value transform:
PARAM_VAL_TRANSFORM_PARAM_DEPENDENT = {
    "P r_to_rgen_opt": {
        "linear_r": r"Linear",
        "linear_and_square_r_alt": r"Lu",
        "linear_and_square_r": r"ext. Lu",
    },
    "P dim_subset": {
        0: "1",
        1: "2",
        2: "3"},
}

if __name__ == "__main__":
    path = Path(RESULTS_PKL_DIR)

    pkl_files = list(path.glob("*.pkl"))

    for i, pkl_file_path in enumerate(pkl_files):

        df = pkl_to_df.read_pkl(pkl_file_path)

        df_pre_fil = pkl_to_df.pre_filter_df(df, rmv_const_cols=True)

        # global params:
        params = dict(
            y_axis_dict=Y_AXIS_DICT,
        )

        # get x_param
        x_param = plot_utils.get_param_cols(df_pre_fil)[0]

        # x tick transform:
        if x_param in PARAM_VAL_TRANSFORM_PARAM_DEPENDENT.keys():
            params["x_param_val_transform_dict"] = PARAM_VAL_TRANSFORM_PARAM_DEPENDENT[x_param]

        # default value green:
        if x_param in DEFAULT_VAL_PARAM_DEPENDENT.keys():
            params["color_green_at_val"] = DEFAULT_VAL_PARAM_DEPENDENT[x_param]

        fig = df_to_plot.plot_one_dim_violin_sweep(df_pre_fil,
                                                   x_param=x_param,
                                                   y_metric=Y_METRIC,
                                                   params=params)

        name = x_param.split()[-1]
        fig.write_image(f"{name}.png", scale=3)
