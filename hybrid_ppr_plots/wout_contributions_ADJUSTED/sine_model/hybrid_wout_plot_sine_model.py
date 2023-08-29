"""
File that plots the w_out matrix of a hybrid-rc setup, in order to see the effect of
output-hybrid.

- See how the wout distribution changes if output hybrid is added:
"""
import numpy as np
import plotly.graph_objects as go

import esnchaos.esn as esn
import esnchaos.simulations as sims
import esnchaos.sweep_experiments as sweep
import esnchaos.utilities as utilities

from simulation_dict import dict_of_sys_params


# flow model:
for system, filename_extra in (
        ("Lorenz63", "Lorenz63"),
        ("Thomas", "Thomas"),
        ("Roessler", "Roessler"),
        ("Chen", "Chen"),
        ("ChuaCircuit", "ChuaCircuit"),
        ("WindmiAttractor", "WindmiAttractor"),
        ("Rucklidge", "Rucklidge"),
        ("Halvorsen", "Halvorsen"),
        ("DoubleScroll", "DoubleScroll"),
):

    sys_class = sims.SYSTEM_DICT[system]
    params_for_sys = dict_of_sys_params[system]
    sys_args = utilities.remove_invalid_args(sys_class.__init__, params_for_sys)
    sys_obj = sys_class(**sys_args)

    # Hybrid model:
    # sinus model:
    def wrong_model(x):
        return np.sin(x)

    kbm_dim = wrong_model(np.array([0, 0, 0])).size

    # Time steps:
    ts_creation_args = {"t_train_disc": 1000,
                        "t_train_sync": 100,
                        "t_train": 2000,
                        "t_validate_disc": 1000,
                        "t_validate_sync": 100,
                        "t_validate": 1000,
                        "n_train_sects": 15,
                        "n_validate_sects": 1, # keep it at 1 here.
                        "normalize_and_center": False,
                        }

    n_train = ts_creation_args["n_train_sects"]
    train_sync_steps = ts_creation_args["t_train_sync"]
    pred_sync_steps = ts_creation_args["t_validate_sync"]
    train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                             **ts_creation_args)
    # x_dim:
    x_dim = sys_obj.sys_dim

    # No hybrid build RC args:
    build_args = {
        "x_dim": x_dim,
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
        "x_train_noise_scale": 0.0, # 1e-6,
        "reg_param": 1e-7,
        "ridge_regression_opt": "bias",
        "scale_input_bool": True,
        "perform_pca_bool": False,
        "scale_input_model_bool": True,
    }

    # hybrid build args:
    hybrid_build_args = build_args.copy()
    # Input hybrid:
    # hybrid_build_args["input_model"] = wrong_model
    # Output hybrid:
    hybrid_build_args["output_model"] = wrong_model


    # Ensemble size:
    n_ens = 15

    # seeds:
    seed = 300
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10000000, size=n_ens)

    # Do experiment:
    for i_ens in range(n_ens):
        print(f"{i_ens=}")

        # # Build normal rc:
        # esn_obj = esn.ESNHybrid()
        # with utilities.temp_seed(seeds[i_ens]):
        #     esn_obj.build(**build_args)

        # Build hybrid rc:
        esn_hyb_obj = esn.ESNHybrid()
        with utilities.temp_seed(seeds[i_ens]):
            esn_hyb_obj.build(**hybrid_build_args)

        for i_train in range(n_train):
            train_data = train_data_list[i_train]

            # Train hybrid RC:
            x_train_fit, x_train, more_out = esn_hyb_obj.train(train_data,
                                                               sync_steps=train_sync_steps,
                                                               more_out_bool=True)
            r_dim = build_args["r_dim"]

            # Get wout matrices:
            # Get wout matrices:
            w_out_hybrid = esn_hyb_obj.get_w_out()  # shape: y_dim, rfit_dim
            w_out_res = w_out_hybrid[:, :r_dim]
            w_out_kbm = w_out_hybrid[:, r_dim: -1]
            w_out_bias = w_out_hybrid[:, -1][:, np.newaxis]

            # res-rfit states:
            res_rfit_states = more_out["rfit"][:, :build_args["r_dim"]]
            kbm_rfit_states = more_out["rfit"][:, build_args["r_dim"]:]

            # outputs:
            res_rfit_output = (w_out_res @ res_rfit_states.T).T
            kbm_rfit_output = (w_out_kbm @ kbm_rfit_states.T).T
            bias_rfit_output = (w_out_kbm @ np.ones(x_train.shape).T).T

            if i_train == 0 and i_ens == 0:
                # new
                results_w_out_res_std = np.zeros((n_ens,
                                          n_train,
                                          x_dim))
                results_w_out_kbm_std = np.zeros((n_ens,
                                          n_train,
                                          x_dim))


                # old:
                res_w_out_hybrid = np.zeros((n_ens,
                                             n_train,
                                             w_out_hybrid.shape[0],
                                             w_out_hybrid.shape[1]))

            # new:
            results_w_out_res_std[i_ens, i_train, :] = np.std(res_rfit_output, axis=0)
            results_w_out_kbm_std[i_ens, i_train, :] = np.std(kbm_rfit_output, axis=0)

            # old
            res_w_out_hybrid[i_ens, i_train, :, :] = w_out_hybrid

    # OLD Get absolute value of w_out and calculate median:
    # Hybrid:
    abs_res_w_out_hybrid = np.abs(res_w_out_hybrid)

    # split to reservoir contribution:
    abs_w_out_hyb_res = abs_res_w_out_hybrid[:, :, :, 0: -1 - kbm_dim]

    # split hybrid contribution
    abs_w_out_hyb_kbm = abs_res_w_out_hybrid[:, :, :, build_args["r_dim"]: -1]

    abs_w_out_hyb_res_avg = np.mean(abs_w_out_hyb_res, axis=3)
    abs_w_out_hyb_kbm_avg = np.mean(abs_w_out_hyb_kbm, axis=3)

    median_abs_w_out_hyb_res_avg = np.median(abs_w_out_hyb_res_avg, axis=(0, 1))
    median_abs_w_out_hyb_kbm_avg = np.median(abs_w_out_hyb_kbm_avg, axis=(0, 1))

    low_abs_w_out_hyb_res_avg = np.quantile(abs_w_out_hyb_res_avg, q=0.25, axis=(0, 1))
    low_abs_w_out_hyb_kbm_avg = np.quantile(abs_w_out_hyb_kbm_avg, q=0.25, axis=(0, 1))

    high_abs_w_out_hyb_res_avg = np.quantile(abs_w_out_hyb_res_avg, q=0.75, axis=(0, 1))
    high_abs_w_out_hyb_kbm_avg = np.quantile(abs_w_out_hyb_kbm_avg, q=0.75, axis=(0, 1))

    x = np.arange(x_dim) + 1
    # Hybrid:
    fig = go.Figure()
    fig.add_traces([
        go.Bar(x=x,
               y=median_abs_w_out_hyb_res_avg,
               error_y=dict(symmetric=False,
                            thickness=2,
                            width=6,
                            array=high_abs_w_out_hyb_res_avg - median_abs_w_out_hyb_res_avg,
                            arrayminus=median_abs_w_out_hyb_res_avg - low_abs_w_out_hyb_res_avg),
               marker_color="#019355",
               name="Reservoir"
               ),
        go.Bar(x=x,
               y=median_abs_w_out_hyb_kbm_avg,
               error_y=dict(symmetric=False,
                            thickness=2,
                            width=6,
                            array=high_abs_w_out_hyb_kbm_avg - median_abs_w_out_hyb_kbm_avg,
                            arrayminus=median_abs_w_out_hyb_kbm_avg - low_abs_w_out_hyb_kbm_avg),
               marker_color="#F79503",
               name="KBM"
               )
    ])

    # add hor lines:
    for x_val in x[:-1]:
        fig.add_vline(x=x_val + 0.5,
                      line_width=1,
                      line_dash="dash",
                      line_color="black",
                      opacity=0.5,
                      )

    fig.update_layout(
        # title="hybrid",
        barmode="group",
        template="simple_white",
        showlegend=True,
        # width=600,
        width=400,
        height=300,
        # yaxis_title=r"$\large \text{partial } |\text{W}_\text{out}|$",
        yaxis_title=r"$\large \Omega_{\text{res} / \text{kbm},\, j}$",
        xaxis_title=r"$\large \text{output dimension } j$",

        font=dict(
            size=23,
            family="Times New Roman"
        ),
        # yaxis=dict(dtick=0.4),  # use for thomas sinus.
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.3,
            xanchor="right",
            x=0.99,
            font=dict(size=23),
            entrywidth=0,
            bordercolor="grey",
            borderwidth=2,
        ),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    # SAVE
    fig.write_image(f"hybrid_wout_plot__sine_{filename_extra}.png", scale=4)


    # NEW Get absolute value of w_out x states and calculate median:
    # Hybrid:

    median_states_w_out_res = np.median(results_w_out_res_std, axis=(0, 1))
    median_states_w_out_kbm = np.median(results_w_out_kbm_std, axis=(0, 1))

    low_states_w_out_res = np.quantile(results_w_out_res_std, q=0.25, axis=(0, 1))
    low_states_w_out_kbm = np.quantile(results_w_out_kbm_std, q=0.25, axis=(0, 1))

    high_states_w_out_res = np.quantile(results_w_out_res_std, q=0.75, axis=(0, 1))
    high_states_w_out_kbm = np.quantile(results_w_out_kbm_std, q=0.75, axis=(0, 1))

    x = np.arange(x_dim) + 1
    # Hybrid:
    fig = go.Figure()
    fig.add_traces([
        go.Bar(x=x,
               y=median_states_w_out_res,
               error_y=dict(symmetric=False,
                            thickness=2,
                            width=6,
                            array=high_states_w_out_res - median_states_w_out_res,
                            arrayminus=median_states_w_out_res - low_states_w_out_res),
               marker_color="#019355",
               name="Reservoir"
               ),
        go.Bar(x=x,
               y=median_states_w_out_kbm,
               error_y=dict(symmetric=False,
                            thickness=2,
                            width=6,
                            array=high_states_w_out_kbm - median_states_w_out_kbm,
                            arrayminus=median_states_w_out_kbm - low_states_w_out_kbm),
               marker_color="#F79503",
               name="KBM"
               )
    ])

    # add hor lines:
    for x_val in x[:-1]:
        fig.add_vline(x=x_val + 0.5,
                      line_width=1,
                      line_dash="dash",
                      line_color="black",
                      opacity=0.5,
                      )

    fig.update_layout(
        # title="hybrid",
        barmode="group",
        template="simple_white",
        showlegend=True,
        # width=600,
        width=400,
        height=300,
        yaxis_title=r"$\large \text{std}(y_{\text{res} / \text{kbm},\, j})$",
        xaxis_title=r"$\large \text{output dimension } j$",

        font=dict(
            size=23,
            family="Times New Roman"
        ),
        # yaxis=dict(dtick=0.4),  # use for thomas sinus.
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.3,
            xanchor="right",
            x=0.99,
            font=dict(size=23),
            entrywidth=0,
            bordercolor="grey",
            borderwidth=2,
        ),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    # SAVE
    fig.write_image(f"states_times_wout_plot__sine_{filename_extra}.png", scale=4)
