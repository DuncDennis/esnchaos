"""Read all pkl files from a designated folder and save the plots.
TODO: For now does not handle the saving of the plots -> has to be implemented.
"""

from pathlib import Path

import esnchaos.plot.pkl_to_df as pkl_to_df
import esnchaos.plot.df_to_plot as df_to_plot
import esnchaos.plot.utilities as plot_utils

RESULTS_PKL_DIR = "original_data"
Y_METRIC_LIST = ["M VALIDATE VT", "M TRAIN PCMAX"]

# Y-axis range and ticks:
Y_AXIS_DICT_MASTER = {
    "M VALIDATE VT": dict(
        range=[-0.5, 16],
        tick0=0,
        dtick=5,
    ),
    "M TRAIN PCMAX": dict(
        range=[-10, 305],
        tick0=0,
        dtick=100,
    ),
}

FONT_SIZE = 22

if __name__ == "__main__":
    path = Path(RESULTS_PKL_DIR)

    pkl_files = list(path.glob("*.pkl"))

    for i, pkl_file_path in enumerate(pkl_files):

        for y_metric in Y_METRIC_LIST:

            df = pkl_to_df.read_pkl(pkl_file_path)

            df_pre_fil = pkl_to_df.pre_filter_df(df, rmv_const_cols=True)

            # global params:
            params = dict(
                y_axis_dict=Y_AXIS_DICT_MASTER[y_metric],
                no_xaxis_title=True,
                font_size=FONT_SIZE
            )

            # get x_param
            x_param = plot_utils.get_param_cols(df_pre_fil)[0]


            fig = df_to_plot.plot_one_dim_violin_sweep(df_pre_fil,
                                                       x_param=x_param,
                                                       y_metric=y_metric,
                                                       params=params)
            fig.write_image(f"{y_metric}.png", scale=3)
