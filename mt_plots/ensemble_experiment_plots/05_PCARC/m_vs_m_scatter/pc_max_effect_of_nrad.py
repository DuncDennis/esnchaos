"""A test experiment template to show how to start an ensemble experiment.
# SYSTEM: LORENZ
# Some notes about the experiment:
10/01/2023: 
"""

from __future__ import annotations
from datetime import datetime
import pathlib

import esnchaos.simulations as sims
import esnchaos.esn as esn
import esnchaos.utilities as utilities
import esnchaos.sweep_experiments as sweep

# import src.ensemble_src.sweep_experiments as sweep

name_of_file = "pc_max_effect_of_nrad"

# TRACKED PARAMETERS:
parameters = {

    # Lorenz:
    "system": ["Lorenz63"],
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 8 / 3,
    "dt": 0.1,
    "lle": 0.9059,

    # Henon:
    # "system": ["Henon"],
    # "a": 1.4,
    # "b": 0.3,
    # "dt": 1.0,
    # "lle": 0.4189,

    # Preprocess:
    "normalize_and_center": False,

    # Data steps (used in sweep.time_series_creator(ARGS):
    "t_train_disc": [1000],
    "t_train_sync": [100],
    "t_train": [2000],
    "t_validate_disc": [1000],
    "t_validate_sync": [100],
    "t_validate": [2000],
    "n_train_sects": [15],
    "n_validate_sects": [10],

    # TODO: MIGRATING TO ESNCHAOS NOTE: The MT plot ignores the data points for
    # TODO: 10^-5 and 10^-4.5.
    "r_dim": [500],
    "n_rad": [0,
              10**(-5),
              10**(-4.5),
              10**(-4.0),
              10**(-3.5),
              10**(-3.0),
              10**(-2.5),
              10**(-2.0),
              10**(-1.5),
              10**(-1.0),
              10**(-0.5),
              1],
    "n_avg_deg": [5.0],
    "n_type_opt": ["erdos_renyi"],
    "r_to_rgen_opt": ["linear_r"],
    "act_fct_opt": ["tanh"],
    "node_bias_opt": ["random_bias"],
    "node_bias_scale": [0.4],
    "w_in_opt": ["random_sparse"],
    "w_in_scale": [1.0],
    "input_noise_scale": [0.0],
    "reg_param": [1e-7],
    "ridge_regression_opt": ["bias"],
    "scale_input_bool": [True],

    # Experiment parameters:
    "seed": [300],
    "n_ens": 15
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
    sys_args = utilities.remove_invalid_args(sys_class.__init__, p)
    sys_obj = sys_class(**sys_args)

    # Create Data:
    ts_creator_args = utilities.remove_invalid_args(sweep.time_series_creator, p)
    train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                             **ts_creator_args)
    # Build ESN:
    model_class = esn.ESN
    build_args = utilities.remove_invalid_args(model_class.build, p)
    build_args["x_dim"] = train_data_list[0].shape[1]

    # Experiment args:
    n_ens = p["n_ens"]
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
        "train_metrics": {},
        "opt_train_args": {"more_out_bool": True},
        "train_more_metrics": {"PCMAX": sweep.pc_max_index},
        "opt_train_more_metrics_args": {"PCMAX": {"reg_param": p["reg_param"],
                                                  "train_steps": p["t_train"]}},
        "validate_metrics": {"VT": sweep.valid_time},
        "opt_validate_metrics_args": {"VT": {"dt": p["dt"],
                                             "lle": p["lle"]}}
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
