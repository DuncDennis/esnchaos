"""RC plot: train vs predict trajectory lorenz """
import numpy as np
import plotly.graph_objects as go

import esnchaos.esn as esn
import esnchaos.simulations as sims
import esnchaos.sweep_experiments as sweep
import esnchaos.utilities as utilities
import esnchaos.measures as meas


# model error:
eps = 1.0

# Create data:
dict_of_sys_params = {
    "KuramotoSivashinsky": {"sys_dim": 64,
                            "sys_length": 35,
                            "lle": 0.07489,  # calculated.
                            "dt": 0.25,
                            "hybridparam": ["eps"]},
}

ts_creation_args = {"t_train_disc": 7000,
                    "t_train_sync": 200,
                    "t_train": 10000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 200,
                    "t_validate": 1500,
                    "n_train_sects": 1,
                    "n_validate_sects": 1,
                    "normalize_and_center": False,
                    }

sys_class = sims.SYSTEM_DICT["KuramotoSivashinsky"]
params_for_sys = dict_of_sys_params["KuramotoSivashinsky"]
sys_args = utilities.remove_invalid_args(sys_class.__init__, params_for_sys)
sys_obj = sys_class(**sys_args)
lle = params_for_sys["lle"]
dt = params_for_sys["dt"]

# Create Data:
n_train = ts_creation_args["n_train_sects"]
train_sync_steps = ts_creation_args["t_train_sync"]
pred_sync_steps = ts_creation_args["t_validate_sync"]
train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                         **ts_creation_args)

# Hybrid model:
sys_args_hybrid = params_for_sys.copy()
# modify the parameter in sys_args_hybrid:
for x in params_for_sys["hybridparam"]:
    sys_args_hybrid[x] = eps  # MODIFIED FOR KS!!
# remove non-sys parameters:
sys_args_hybrid = utilities.remove_invalid_args(sys_class.__init__, sys_args_hybrid)
wrong_sim_obj = sys_class(**sys_args_hybrid)
wrong_model = wrong_sim_obj.iterate

# Build RC args:
build_args = {
    "x_dim": train_data_list[0].shape[1],
    "r_dim": 4000,
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
seed = 10

# Do experiment:

predictor_type = [
    "input_hybrid",
    "output_hybrid",
    "full_hybrid",
    "no_hybrid",
    "model_predictor_fitted",
    "model_predictor",
]

results = {}

for i_pred_type, pred_type in enumerate(predictor_type):

    if pred_type == "model_predictor":
        class model_predictor:
            """Class of the same "shape" as the esn classes, but only simulates data wrong,
            i.e. there is no training or building.
            """

            def __init__(self):
                """Fake init."""
                pass

            def build(self):
                """Fake build."""
                pass

            def train(self, use_for_train: np.ndarray, sync_steps: int = 0):
                """Fake train."""
                to_return = use_for_train[sync_steps:, :]
                return to_return, to_return

            def predict(self, use_for_pred: np.ndarray, sync_steps: int = 0):
                """Predict using the "wrong" iterator function."""
                y_true = use_for_pred[sync_steps:, :]
                steps_to_predict = y_true.shape[0]
                starting_point = y_true[0, :]
                y_pred = wrong_sim_obj.simulate(steps_to_predict, starting_point=starting_point)
                return y_pred, y_true

        model_class = model_predictor
        esn_obj = model_class()
    else:
        p = build_args.copy()

        model_class = esn.ESNHybrid
        if pred_type == "no_hybrid":
            output_model = None
            input_model = None
        elif pred_type == "full_hybrid":
            output_model = wrong_model
            input_model = wrong_model
        elif pred_type == "input_hybrid":
            output_model = None
            input_model = wrong_model
        elif pred_type == "output_hybrid":
            output_model = wrong_model
            input_model = None
        elif pred_type == "model_predictor_fitted":
            output_model = wrong_model
            input_model = None
            p["r_dim"] = 0
            p["scale_input_bool"] = False

        p["output_model"] = output_model
        p["input_model"] = input_model
        p["scale_input_model_bool"] = p["scale_input_bool"]

        # Build ESN:
        p = utilities.remove_invalid_args(model_class.build, p)

        esn_obj = model_class()

        with utilities.temp_seed(seed):
            esn_obj.build(**p)

    # Train RC:
    train_data = train_data_list[0]
    fit, true = esn_obj.train(train_data, sync_steps=train_sync_steps)
    # Predict RC
    pred_data = validate_data_list_of_lists[0][0]
    pred, true_pred = esn_obj.predict(pred_data,
                                      sync_steps=pred_sync_steps)

    if i_pred_type == 0:
        results["True"] = true_pred

    results[pred_type] = pred


# PLOT:

import plotly.graph_objs as go
from plotly.subplots import make_subplots

# titles = ["true", "input hybrid", "output hybrid", "full hybrid", "only reservoir",
#           "only model", "model fitted"]
titles = ["true", "input hybrid", "output hybrid", "full hybrid", "reservoir-only",
          "KBM-fitted", "KBM-only"]


fig = make_subplots(rows=len(results.keys()),
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    print_grid=True,
                    x_title=r"$\large " + r't \cdot \lambda_\text{max}$',
                    y_title=r"$\large " + r'\boldsymbol{u}(t)$',
                    subplot_titles=titles
                    )

# for i_plot, (key, result_for_key) in enumerate(results.items()):
#     fig.add_trace(
#         go.Heatmap(z=result_for_key.T),
#         row=i_plot + 1, col=1,
#     )

lyap_times = np.arange(pred.shape[0]) * dt * lle
sys_sizes = np.arange(64) / 64 * 35
for i_plot, (key, result_for_key) in enumerate(results.items()):
    if key == "True":
        fig.add_trace(
            go.Heatmap(z=result_for_key.T,
                       x=lyap_times,
                       y=sys_sizes),
            row=i_plot + 1, col=1,
        )
    else:
        true_pred = results["True"]
        pred = result_for_key
        fig.add_trace(
            # go.Heatmap(z=pred.T - true_pred.T,
            #            x=lyap_times),
            go.Heatmap(z=pred.T,
                       x=lyap_times,
                       y=sys_sizes),
            row=i_plot + 1, col=1,
        )

        error_series_ts = meas.error_over_time(y_pred=pred,
                                               y_true=true_pred,
                                               normalization="root_of_avg_of_spacedist_squared")
        vt = meas.valid_time_index(error_series_ts, error_threshold=0.4)
        vt = lle * dt * vt
        print(f"vt = {vt}")
        fig.add_vline(x=vt,
                      line_width=3,
                      line_dash="solid",
                      line_color="red",
                      row=i_plot + 1,
                      col=1,
                      opacity=1
                      )

fig.update_xaxes(range=[0, 10])
fig.update_yaxes(tickvals=[10, 30])

fig.update_layout(
    template="simple_white",
    font=dict(
        size=20,
        family="Times New Roman"
    ),
)

fig.update_layout(
    width=550,
    height=550,
)
fig.update_layout(
    # margin=dict(l=15, r=40, t=10, b=50),
    # margin=dict(l=30, r=20, t=10, b=50),
    margin=dict(l=60, r=20, t=10, b=50),
)
fig.update_traces(showscale=False)

fig.show()

file_name = f"visualize_ks_prediction.png"
fig.write_image(file_name, scale=5)
