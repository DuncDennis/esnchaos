"""Create the PCA intro plot. """
from sklearn.decomposition import PCA
import plotly.graph_objects as go

import esnchaos.esn as esn
import esnchaos.simulations as sims
import esnchaos.sweep_experiments as sweep
import esnchaos.utilities as utilities

# Create data:
sys_obj = sims.Lorenz63(dt=0.1)
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 2000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 400,
                    "n_train_sects": 1,
                    "n_validate_sects": 1,
                    "normalize_and_center": False,
                    }

n_train = ts_creation_args["n_train_sects"]
train_sync_steps = ts_creation_args["t_train_sync"]
train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                         **ts_creation_args)

# Build RC args:
build_args = {
    "x_dim": 3,
    "r_dim": 500,
    "n_rad": 0.4,
    "n_avg_deg": 5.0,
    "n_type_opt": "erdos_renyi",
    "r_to_rgen_opt": "linear_r",
    "act_fct_opt": "tanh",
    "node_bias_opt": "random_bias",
    "node_bias_scale": 0.4,
    "w_in_opt": "random_sparse",
    "w_in_scale": 1.0,
    "x_train_noise_scale": 0.0,
    "reg_param": 1e-7,
    "ridge_regression_opt": "bias",
    "scale_input_bool": True,
}

# seeds:
seed = 300

# Do experiment:

# Build rc:
esn_obj = esn.ESN()
with utilities.temp_seed(seed):
    esn_obj.build(**build_args)

# Train (i.e. Drive):
# Train RC:
train_data = train_data_list[0]

fit, true, more_out = esn_obj.train(train_data, sync_steps=train_sync_steps, more_out_bool=True)
res_states = more_out["r"]
pca = PCA()
res_pca_states = pca.fit_transform(res_states)

r_input = train_data[train_sync_steps:-1, :]

# pca_input = PCA()
# true_pca = pca_input.fit_transform(r_input)


color = "black"
linewidth = 0.5
height = 500
# width = int(1.4 * height)
width = 500

for i in range(3):

    # Input:
    if i == 0:
        name = "input"
        x = r_input[:, 0].tolist()
        y = r_input[:, 1].tolist()
        z = r_input[:, 2].tolist()
        camera = dict(eye=dict(x=1.25, y=-1.25, z=1.25))

    # PCA reservoir 1-2-3:
    if i == 1:
        name = "pca_res_first"
        x = res_pca_states[:, 0].tolist()
        y = res_pca_states[:, 1].tolist()
        z = res_pca_states[:, 2].tolist()
        camera = dict(
            # eye=dict(x=0.8, y=0.9, z=1.25),
            eye=dict(x=1.2, y=1.0, z=1.3),
            up=dict(x=0, y=1, z=0)
        )

    # PCA reservoir 3-4-5:
    if i == 2:
        name = "pca_res_later"
        x = res_pca_states[:, 3].tolist()
        y = res_pca_states[:, 4].tolist()
        z = res_pca_states[:, 5].tolist()
        camera = dict(
            eye=dict(x=0.7, y=1.5, z=-1.3),
            up=dict(x=0, y=1, z=0)
            )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(x=x, y=y, z=z,
                     line=dict(
                         color=color,
                         width=linewidth
                     ),
                     mode="lines",
                     # mode="markers",
                     # marker=dict(size=2),
                     meta=dict())
    )
    fig.update_layout(
        # template="plotly_white",
        template="simple_white",
        showlegend=False,
    )

    fig.update_scenes(
        xaxis_title="",
        yaxis_title="",
        zaxis_title="",

        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
        zaxis_showticklabels=False
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                gridcolor="rgba(50, 50, 50, 0.65)",
                showgrid=True,
                showline=False,
                showspikes=False,
                ticklen=0,
                # zerolinecolor="white",
            ),
            yaxis=dict(
                gridcolor="rgba(50, 50, 50, 0.65)",
                showgrid=True,
                showline=False,
                showspikes=False,
                ticklen=0,
                # zerolinecolor="white"
            ),
            zaxis=dict(
                gridcolor="rgba(50, 50, 50, 0.65)",
                showgrid=True,
                showline=False,
                showspikes=False,
                ticklen=0,
                # zerolinecolor="white",
            )),
        scene_camera=camera,
                      width=width,
                      height=height,
                      )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
    )

    # SAVE
    file_name = f"traj_input_and_pcs__{name}.png"
    fig.write_image(file_name, scale=2)

    print(fig.layout)
