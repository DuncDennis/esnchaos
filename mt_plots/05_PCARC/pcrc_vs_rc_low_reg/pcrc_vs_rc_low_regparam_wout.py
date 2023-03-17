import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import itertools
import plotly.express as px
col_pal = px.colors.qualitative.Plotly
col_pal_iterator = itertools.cycle(col_pal)


import esnchaos.esn as esn
import esnchaos.simulations as sims
import esnchaos.sweep_experiments as sweep
import esnchaos.utilities as utilities


def get_color_value(val: float,
                    full_val_range: list[float, float],
                    log10_bool: bool = False):

    if log10_bool:
        val = np.log10(val)
        full_val_range = [np.log10(full_val_range[0]),
                          np.log10(full_val_range[1])]

    print(val, full_val_range)

    ratio = (val - full_val_range[0])/(full_val_range[1] - full_val_range[0])
    if ratio < 0 or ratio > 1:
        return "rgb(15, 15, 15)"
    else:
        return sample_colorscale('Portland', [ratio])[0]


def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return "rgba" + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))

# Create data:

ts_creation_args = {"t_train_disc": 5000,
                    "t_train_sync": 100,
                    "t_train": 2000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 2000,
                    "n_train_sects": 1,
                    "n_validate_sects": 1,
                    "normalize_and_center": False,
                    }
n_train = ts_creation_args["n_train_sects"]
train_sync_steps = ts_creation_args["t_train_sync"]

# Build RC args:
build_args = {
    "x_dim": 3,
    "r_dim": 500,
    "n_rad": 0.4,
    "n_avg_deg": 5,
    "n_type_opt": "erdos_renyi",
    "r_to_rgen_opt": "linear_r",
    "act_fct_opt": "tanh",
    "node_bias_opt": "random_bias",
    "node_bias_scale": 0.4,
    "w_in_opt": "random_sparse",
    "w_in_scale": 1.0,
    "x_train_noise_scale": 0.0,
    "reg_param": 1e-12,
    "ridge_regression_opt": "bias",
    "scale_input_bool": True,
}

# Ensemble size:
n_ens = 1

# seeds:
seed = 300
rng = np.random.default_rng(seed)
seeds = rng.integers(0, 10000000, size=n_ens)

sys_obj = sims.Lorenz63(dt=0.05)
train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                         **ts_creation_args)

train_data = train_data_list[0]

# no pc:
esn_obj = esn.ESN()
with utilities.temp_seed(seeds[0]):
    esn_obj.build(**build_args)
_, _, more_out = esn_obj.train(train_data, sync_steps=train_sync_steps, more_out_bool=True)
res_states = more_out["rfit"]
pca = PCA()
res_pca_states = pca.fit_transform(res_states)
w_out = esn_obj.get_w_out()
if build_args["ridge_regression_opt"] == "bias":
    w_out = w_out[:, :-1]
w_out_pca = w_out @ pca.components_.T

# with pc:
esn_obj = esn.ESN()
build_args_pc = build_args.copy()
build_args_pc["perform_pca_bool"] = True
with utilities.temp_seed(seeds[0]):
    esn_obj.build(**build_args_pc)
train_data = train_data_list[0]
_, _, more_out = esn_obj.train(train_data,
                               sync_steps=train_sync_steps,
                               more_out_bool=True)
w_out_pc = esn_obj.get_w_out()
if build_args["ridge_regression_opt"] == "bias":
    w_out_pc = w_out_pc[:, :-1]

# PLOT BOTH:
x = np.arange(1, 501)
fig = go.Figure()

import plotly.io as pio
col_pal_simple_white = pio.templates["simple_white"].layout.colorway

# no pc transform:
y = np.sum(np.abs(w_out_pca), axis=0)
fig.add_trace(
    go.Scatter(x=x,
               y=y,
               name="no PC-transform",
               line=dict(color=hex_to_rgba(col_pal_simple_white[1], 1))
               )
)

# pc transform:
y = np.sum(np.abs(w_out_pc), axis=0)
fig.add_trace(
    go.Scatter(x=x,
               y=y,
               name="PC-transform",
               line=dict(color=hex_to_rgba(col_pal_simple_white[2], 1))
               )
)

fig.update_layout(
    template="simple_white",
    width=560,
    height=380,
    font=dict(
        size=25,
        family="Times New Roman"
    ),
    xaxis=dict(
        title=r'$\Large \text{principal component } i$',
        range=[0, 500],
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.2)',
    ),
    yaxis=dict(
        title=r"$\Large\sum_j|\text{W}_{\text{pc}, ji}|$",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.2)',
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        # y=0.85,  # 0.99
        y=1,  # 0.99
        xanchor="left",
        x=0.05,
        font=dict(size=20),
        bordercolor="grey",
        borderwidth=2,
    ),
    margin=dict(l=5, r=15, t=5, b=5),

)



fig.write_image(f"pcrc_vs_rc_low_reg.png", scale=2)
