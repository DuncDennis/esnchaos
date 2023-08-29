import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import esnchaos.simulations as sims
import esnchaos.utilities as utilities
import esnchaos.sweep_experiments as sweep

from simulation_dict import dict_of_sys_params

# TRACKED PARAMETERS:
parameters = {

    "system": [
        "Lorenz63",
        "Roessler",
        "Chen",
        "ChuaCircuit",
        "Thomas",
        "WindmiAttractor",
        "Rucklidge",
        "Halvorsen",
        "DoubleScroll",
    ],

    # Preprocess:
    "normalize_and_center": False,

    # Data steps (used in sweep.time_series_creator(ARGS):
    "t_train_disc": 1000,
    "t_train_sync": 100,
    "t_train": 2000,
    "t_validate_disc": 1000,
    "t_validate_sync": 100,
    "t_validate": 2000,
    "n_train_sects": 1,
    "n_validate_sects": 1,
}

p = parameters.copy()
fig = make_subplots(rows=9,
                    cols=1,
                    # shared_xaxes=True,
                    shared_xaxes=False,
                    vertical_spacing=None,
                    print_grid=True,
                    x_title="Lyap times",
                    subplot_titles=p["system"]
                    )

for i_sys, system in enumerate(p["system"]):

    sys_class = sims.SYSTEM_DICT[system]

    params_for_sys = dict_of_sys_params[system]
    sys_args = utilities.remove_invalid_args(sys_class.__init__, params_for_sys)
    sys_obj = sys_class(**sys_args)
    lle = params_for_sys["lle"]
    dt = params_for_sys["dt"]

    # Create Data:
    ts_creator_args = utilities.remove_invalid_args(sweep.time_series_creator, p)
    train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                             **ts_creator_args)

    train_plot = train_data_list[0]

    steps, x_dim = train_plot.shape
    lyap_times = np.arange(0, steps) * dt * lle

    for i_x in range(x_dim):
        fig.add_trace(
            go.Scatter(
                x=lyap_times,
                y=train_plot[:, i_x]
            ),
            row=i_sys + 1, col=1
        )
# fig.update_layout(title=system)
fig.update_layout(
    template="simple_white",
    font=dict(
        size=18,
        family="Times New Roman"
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.25,
        xanchor="right",
        x=1,
        font=dict(size=20)
    ),
    showlegend=False,
)

fig.update_layout(
    width=1000,
    height=1000,
)
# fig.update_layout(
#     margin=dict(l=15, r=40, t=10, b=50),
# )

fig.show()

file_name = f"visualize_training_times"
fig.write_image(file_name + ".png", scale=5)
print(train_plot.shape)
