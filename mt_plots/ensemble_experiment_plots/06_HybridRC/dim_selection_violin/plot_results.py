"""Read all pkl files from a designated folder and save the plots.
TODO: For now does not handle the saving of the plots -> has to be implemented.
"""

from pathlib import Path

import esnchaos.plot.pkl_to_df as pkl_to_df
import esnchaos.plot.df_to_plot as df_to_plot
import esnchaos.plot.utilities as plot_utils

RESULTS_PKL_DIR = "original_data"
Y_METRIC = "M VALIDATE VT"

Y_AXIS_DICT_SYS_DEP = {
    "ChuaCircuit": dict(
        range=[0, 15],
        tick0=0,
        dtick=5,
    ),
    "WindmiAttractor": dict()
}

if __name__ == "__main__":
    path = Path(RESULTS_PKL_DIR)

    pkl_files = list(path.glob("*.pkl"))

    full_system_list = ["Lorenz63", "ChuaCircuit", "WindmiAttractor"]

    for i, pkl_file_path in enumerate(pkl_files):

            for system in ["ChuaCircuit", "WindmiAttractor"]:
                to_exclude = full_system_list.copy()
                to_exclude.remove(system)

                print(to_exclude)

                df = pkl_to_df.read_pkl(pkl_file_path)

                df_pre_fil = pkl_to_df.pre_filter_df(df,
                                                     excluded_params={
                                                         "P system": to_exclude,
                                                         "P predictor_type": ["no_hybrid"]
                                                     },
                                                     rmv_const_cols=True)

                # global params:
                params = dict(
                    no_xaxis_title=False,
                    y_axis_dict=Y_AXIS_DICT_SYS_DEP[system],
                    color_green_at_val=-1,
                    x_param_val_transform_dict={0: "1",
                                                1: "2",
                                                2: "3"}
                )

                # get x_param
                x_param = plot_utils.get_param_cols(df_pre_fil)[0]


                fig = df_to_plot.plot_one_dim_violin_sweep(df_pre_fil,
                                                           x_param=x_param,
                                                           y_metric=Y_METRIC,
                                                           params=params)
                fig.write_image(f"test_{system}.png", scale=3)
