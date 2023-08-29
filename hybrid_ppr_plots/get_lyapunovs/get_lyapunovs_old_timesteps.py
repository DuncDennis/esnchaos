"""Calculate the Lyapunov exponents of all systems tested."""
import numpy as np

from esnchaos.measures import largest_lyapunov_exponent as lle
import esnchaos.simulations as sims
import esnchaos.utilities as utilities


# The sprott values are given in the "lle" keys.
dict_of_sys_params = {
    # ERROR Sprot is 0.9056
    "Lorenz63": {"sigma": 10.0,
                 "rho": 28.0,
                 "beta": 8 / 3,
                 "dt": 0.1,  # Paper value
                 # "dt": 0.05,  # New for better lyapunov
                 "lle": 0.9056},

    # Sprott check
    "Roessler": {"a": 0.2,
                 "b": 0.2,
                 "c": 5.7,
                 "dt": 0.2,  # Paper value
                 # "dt": 0.1,  # New for better lyapunov
                 "lle": 0.0714},

    # Sprott check
    "Chen": {"a": 35.0,
             "b": 3.0,
             "c": 28.0,
             "dt": 0.03,  # Paper value
             # "dt": 0.02,  # New for better lyapunov
             "lle": 2.0272},

    # Sprott check
    "ChuaCircuit": {"alpha": 9.0,
                    "beta": 100 / 7,
                    "a": 8 / 7,
                    "b": 5 / 7,
                    "dt": 0.2, # Paper value
                    # "dt": 0.1, # New for better lyapunov
                    "lle": 0.3271},

    # Sprott error: actually 0.0349
    "Thomas": {"b": 0.18,
               "dt": 0.4, # Paper value
               # "dt": 0.3, # New for better lyapunov
               "lle": 0.0349,
               },

    # # Sprott check
    # "WindmiAttractor": {"a": 0.7,
    #                     "b": 2.5,
    #                     "dt": 0.2,  # Paper value (is good)
    #                     "lle": 0.0755},

    # # Sprott check
    # "Rucklidge": {"kappa": 2.0,
    #               "lam": 6.7,
    #               "dt": 0.1,  # Paper value (is good)
    #               "lle": 0.0643},

    # # Sprott check
    # "Halvorsen": {"a": 1.27,
    #               "dt": 0.05,  # Paper value (is good)
    #               "lle": 0.7899},
    #
    # # Sprott check
    # "DoubleScroll": {"a": 0.8,
    #                  "dt": 0.3,  # Paper value (is good)
    #                  "lle": 0.0497},

    # # Kuramoto Sivashinsky:
    # "KuramotoSivashinsky": {"sys_dim": 64,
    #                         "sys_length": 35,
    #                         "dt": 0.25,
    #                         "lle": 0.07}  # Value from pathak
}

lles_perc_diff = []

for sysname, sysparams in dict_of_sys_params.items():
    print(sysname)
    # get iterate function and default_starting_point and dt:
    sys_class = sims.SYSTEM_DICT[sysname]

    # modify dt:
    # sysparams["dt"] = 0.05

    sys_args = utilities.remove_invalid_args(sys_class.__init__, sysparams)
    sys_obj = sys_class(**sys_args)
    iterate_func = sys_obj.iterate
    default_starting_point = sys_obj.default_starting_point
    dt = sysparams["dt"]

    # calculate lle:
    lle_value = lle(iterate_func,
                    starting_point=default_starting_point,
                    deviation_scale=1e-10,
                    steps_skip=500,
                    part_time_steps=15,
                    steps=3000,
                    dt=dt)

    sprott_lle = sysparams["lle"]

    perc_diff = np.abs(sprott_lle - lle_value) / sprott_lle * 100

    print(f"Sprott: {sprott_lle}, calc: {lle_value}, perc_diff: {perc_diff}")

    lles_perc_diff.append(perc_diff)

    # save lle:
    dict_of_sys_params[sysname]["lle_calc"] = lle_value

    print("--------------------------------------------------------")

import numpy as np
print(f"average perc difference: {np.mean(lles_perc_diff)}")
