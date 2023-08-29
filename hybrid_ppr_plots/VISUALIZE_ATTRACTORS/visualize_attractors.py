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
        "Chen",
        "ChuaCircuit",
        "DoubleScroll",
        "Halvorsen",
        "Roessler",
        "Rucklidge",
        "Thomas",
        "WindmiAttractor",
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
fig = make_subplots(rows=3,
                    cols=3,
                    shared_xaxes=False,
                    vertical_spacing=0.1,
                    print_grid=True,
                    subplot_titles=[dict_of_sys_params[x]["alias"] for x in
                                    p["system"]],
                    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"},
                            {"type": "scatter3d"}],
                           [{"type": "scatter3d"}, {"type": "scatter3d"},
                            {"type": "scatter3d"}],
                           [{"type": "scatter3d"}, {"type": "scatter3d"},
                            {"type": "scatter3d"}]]
                    )

i_row = 1
i_col = 0

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

    i_col += 1
    if i_col == 4:
        i_col = 1
        i_row += 1

    fig.add_trace(
        go.Scatter3d(
            x=train_plot[:, 0],
            y=train_plot[:, 1],
            z=train_plot[:, 2],
            line=dict(
                color="black",
                width=2
            ),
            mode="lines",
        ),
        row=i_row, col=i_col
    )

    # for i_x in range(x_dim):
    #     fig.add_trace(
    #         go.Scatter(
    #             x=lyap_times,
    #             y=train_plot[:, i_x]
    #         ),
    #         row=i_sys + 1, col=1
    #     )
# fig.update_layout(title=system)
fig.update_layout(
    template="simple_white",
    font=dict(
        size=18,
        family="Times New Roman"
    ),
    showlegend=False,
)

fig.update_scenes(
    xaxis_title="",
    yaxis_title="",
    zaxis_title="",

    xaxis_showticklabels=False,
    yaxis_showticklabels=False,
    zaxis_showticklabels=False,

    xaxis_showgrid=True,
    xaxis_showline=False,
    xaxis_showspikes=False,
    xaxis_ticklen=0,
    xaxis_gridcolor="rgba(50, 50, 50, 0.65)",

    yaxis_showgrid=True,
    yaxis_showline=False,
    yaxis_showspikes=False,
    yaxis_ticklen=0,
    yaxis_gridcolor="rgba(50, 50, 50, 0.65)",

    zaxis_showgrid=True,
    zaxis_showline=False,
    zaxis_showspikes=False,
    zaxis_ticklen=0,
    zaxis_gridcolor="rgba(50, 50, 50, 0.65)",

    # camera=dict(
    #     eye=dict(x=0.7, y=1.5, z=-1.3),
    #     up=dict(x=0, y=1, z=0)
    # )
)

fig.update_layout(
    width=500,
    height=500,
)
fig.update_layout(
    margin=dict(l=0, r=0, t=20, b=0),
)

fig.show()
file_name = f"visualize_attractors"
fig.write_image(file_name + ".png", scale=5)

# fig.write_image(file_name + ".eps", scale=3)
print(train_plot.shape)
