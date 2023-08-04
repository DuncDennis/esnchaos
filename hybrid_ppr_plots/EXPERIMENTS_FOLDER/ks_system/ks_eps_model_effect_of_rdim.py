
from __future__ import annotations
from datetime import datetime
import pathlib

import numpy as np

import esnchaos.simulations as sims
import esnchaos.esn as esn
import esnchaos.utilities as utilities
import esnchaos.sweep_experiments as sweep

# get ensemble parameters:
from ensemble_parameters import n_ens_file, n_train_sects_file, n_validate_sects_file

name_of_file = "ks_eps_model_effect_of_rdim"

# Parameters for each system:
dict_of_sys_params = {
    "KuramotoSivashinsky": {"sys_dim": 64,
                            "sys_length": 35,
                            "lle": 0.07489,  # calculated.
                            "dt": 0.25,
                            "hybridparam": ["eps"]},
}

# TRACKED PARAMETERS:
parameters = {

    # Hybrid args:
    "model_error_eps": [1],

    # Predictor Meta:
    "predictor_type": [
        "model_predictor",
        "model_predictor_fitted",
        "full_hybrid",
        "input_hybrid",
        "output_hybrid",
        "no_hybrid",
    ],

    "system": [
        "KuramotoSivashinsky",
    ],

    # Preprocess:
    "normalize_and_center": False,

    # Data steps (used in sweep.time_series_creator(ARGS):
    "t_train_disc": [7000],
    "t_train_sync": [200],
    "t_train": [10000],
    "t_validate_disc": [1000],
    "t_validate_sync": [200],
    "t_validate": [1500],
    "n_train_sects": [n_train_sects_file],
    "n_validate_sects": [n_validate_sects_file],

    # ESN Build (data that is fed into esn.build(ARGS):
    "r_dim": [50, 100, 200, 500, 1000, 2000, 4000, 6000],
    "n_rad": [0.4],
    "n_avg_deg": [5.0],
    "n_type_opt": ["erdos_renyi"],
    "r_to_rgen_opt": ["linear_r"],
    "act_fct_opt": ["tanh"],
    "node_bias_opt": ["random_bias"],
    "node_bias_scale": [0.4],
    "w_in_opt": ["random_sparse"],
    "w_in_scale": [1.0],
    "x_train_noise_scale": [0.0],
    "reg_param": [1e-7],
    "ridge_regression_opt": ["bias"],
    "scale_input_bool": [True],

    # Experiment parameters:
    "seed": [300],
    "n_ens": n_ens_file
}


# PARAMETER TO ARGUMENT TRANSFOMER FUNCTION:
def parameter_transformer(parameters: dict[str, float | int | str]):
    """Transform the parameters to be usable by PredModelEnsembler.

    Args:
        parameters: The parameter dict defining the sweep experiment.
            Each key value pair must be like: key is a string, value is either a string,
            int or float.

    Returns:
        All the data needed for PredModelEnsembler.
    """
    p = parameters.copy()

    # System:
    sys_class = sims.SYSTEM_DICT[p["system"]]
    params_for_sys = dict_of_sys_params[p["system"]]

    sys_args = utilities.remove_invalid_args(sys_class.__init__, params_for_sys)
    sys_obj = sys_class(**sys_args)
    lle = params_for_sys["lle"]
    dt = params_for_sys["dt"]

    # Create Data:
    ts_creator_args = utilities.remove_invalid_args(sweep.time_series_creator, p)
    train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                             **ts_creator_args)

    # Hybrid model:
    sys_args_hybrid = params_for_sys.copy()
    # modify the parameter in sys_args_hybrid:
    for x in params_for_sys["hybridparam"]:
        # sys_args_hybrid[x] = sys_args_hybrid[x] * (1 + p["model_error_eps"])
        sys_args_hybrid[x] = p["model_error_eps"]  # MODIFIED FOR KS!!
    # remove non-sys parameters:
    sys_args_hybrid = utilities.remove_invalid_args(sys_class.__init__, sys_args_hybrid)
    wrong_sim_obj = sys_class(**sys_args_hybrid)
    wrong_model = wrong_sim_obj.iterate

    if p["predictor_type"] == "model_predictor":
        # Create model predictor class (that also needs a train and predict method).
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

        n_ens = 1  # There is no randomness here, so save cpu time.
        build_args = {}
        model_class = model_predictor

    else:  # if model is not only the model predictor:

        model_class = esn.ESNHybrid
        if p["predictor_type"] == "no_hybrid":
            output_model = None
            input_model = None
        elif p["predictor_type"] == "full_hybrid":
            output_model = wrong_model
            input_model = wrong_model
        elif p["predictor_type"] == "input_hybrid":
            output_model = None
            input_model = wrong_model
        elif p["predictor_type"] == "output_hybrid":
            output_model = wrong_model
            input_model = None
        elif p["predictor_type"] == "model_predictor_fitted":
            output_model = wrong_model
            input_model = None
            p["r_dim"] = 0
            p["scale_input_bool"] = False
            p["n_ens"] = 1  # no randomness in model anyway.

        p["output_model"] = output_model
        p["input_model"] = input_model
        p["scale_input_model_bool"] = p["scale_input_bool"]

        # Build ESN:
        build_args = utilities.remove_invalid_args(model_class.build, p)
        build_args["x_dim"] = train_data_list[0].shape[1]

        # Experiment args:
        n_ens = p["n_ens"]

    # seed:     
    seed = p["seed"]

    # Build model args:
    build_models_args = {"model_class": model_class,
                         "build_args": build_args,
                         "n_ens": n_ens,
                         "seed": seed}

    # Train validate_test_args:
    train_validate_test_args = {
        "train_data_list": train_data_list,
        "validate_data_list_of_lists": validate_data_list_of_lists,
        "train_sync_steps": p["t_train_sync"],
        "validate_sync_steps": p["t_validate_sync"],
        "opt_validate_metrics_args": {"VT": {"dt": dt,
                                             "lle": lle}}
    }

    return build_models_args, train_validate_test_args


# Set up Sweeper.
sweeper = sweep.PredModelSweeper(parameter_transformer)

# Print start time:
start = datetime.now()
start_str = start.strftime("%Y-%m-%d %H:%M:%S")
print(f"Start: {start_str}")

# Sweep:
results_df = sweeper.sweep(parameters)

# End time:
end = datetime.now()
end_str = end.strftime("%Y-%m-%d %H:%M:%S")
print(f"End: {end_str}")

# Ellapsed time:
diff = end - start
diff_str = diff
print(f"Time difference: {diff_str}")

# Save results:
print("Saving...")
directory = pathlib.Path.joinpath(pathlib.Path().absolute(), "sweep_results")
file_path = sweep.save_pandas_to_pickles(df=results_df,
                                         directory=directory,
                                         name=name_of_file)
print(f"Saved to: {file_path}")
print("FINISHED! ")
