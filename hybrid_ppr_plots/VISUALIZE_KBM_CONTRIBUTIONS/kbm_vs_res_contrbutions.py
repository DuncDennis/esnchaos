import numpy as np
import plotly.graph_objects as go

import esnchaos.esn as esn
import esnchaos.simulations as sims
import esnchaos.sweep_experiments as sweep
import esnchaos.utilities as utilities

from simulation_dict import dict_of_sys_params


# system = "Lorenz63"
# system = "ChuaCircuit"
# system = "WindmiAttractor"
system = "Lorenz63"
# system = "Thomas"
# system = "Roessler"
# system = "Chen"
# system = "Rucklidge"

# epsilon model:
for eps, filename_extra in (
        # (1e-4, "1em4"),
        (0.1, "0p1"),
        (1, "1p0"),
        (10, "10"),
        # (100, "100"),
        # (1000, "1000"),
):

    sys_class = sims.SYSTEM_DICT[system]
    params_for_sys = dict_of_sys_params[system]
    sys_args = utilities.remove_invalid_args(sys_class.__init__, params_for_sys)
    sys_obj = sys_class(**sys_args)

    # Hybrid model:

    # eps model:
    sys_args_hybrid = params_for_sys.copy()
    # modify the parameter in sys_args_hybrid:
    for x in params_for_sys["hybridparam"]:
        sys_args_hybrid[x] = sys_args_hybrid[x] * (1 + eps)

    sys_args_hybrid = utilities.remove_invalid_args(sys_class.__init__, sys_args_hybrid)
    wrong_sim_obj = sys_class(**sys_args_hybrid)
    wrong_model = wrong_sim_obj.iterate


    # Flow model:
    # wrong_model = sys_obj.flow

    # dim-selection model:
    # dim_select = 2
    # wrong_model = lambda x: sys_obj.iterate(x)[[0, 2]]

    # sinus model:
    # def wrong_model(x):
    #     return np.sin(x)

    # def wrong_model(x):
    #     return np.exp(x)
    #
    # def wrong_model(x):
    #     return x**3

    # def wrong_model(x):
    #     return np.cos(x)

    # wrong_model = sims.Thomas().iterate
    # wrong_model = sims.WindmiAttractor().iterate

    kbm_dim = wrong_model(np.array([0, 0, 0])).size

    # Time steps:
    ts_creation_args = {"t_train_disc": 1000,
                        "t_train_sync": 100,
                        "t_train": 2000,
                        "t_validate_disc": 1000,
                        "t_validate_sync": 100,
                        "t_validate": 1000,
                        "n_train_sects": 1,
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
        # "r_dim": 50,
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
        "perform_pca_bool": False,
        "scale_input_model_bool": True,
        # "scale_output_model_bool": True,  # ADDED FOR TEST
    }

    # hybrid build args:
    hybrid_build_args = build_args.copy()
    # Input hybrid:
    # hybrid_build_args["input_model"] = wrong_model
    # Output hybrid:
    hybrid_build_args["output_model"] = wrong_model


    # Ensemble size:
    n_ens = 1

    # seeds:
    seed = 300
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10000000, size=n_ens)

    # Do experiment:
    for i_ens in range(n_ens):
        print(i_ens)

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

            # Train normal RC:
            # _, _ = esn_obj.train(train_data,
            #                         sync_steps=train_sync_steps,
            #                         more_out_bool=False)

            # Train hybrid RC:
            x_train_fit, x_train, more_out = esn_hyb_obj.train(train_data,
                                                               sync_steps=train_sync_steps,
                                                               more_out_bool=True)

            r_dim = build_args["r_dim"]

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
            bias_rfit_output = (w_out_bias @ np.ones((x_train.shape[0], 1)).T).T


    # # MAYBE REMOVE:
    # x_train_fit = x_train_fit - bias_rfit_output

    # Plot:

    def hex_to_rgba(h, alpha):
        '''
        converts color value in hex format to rgba format with alpha transparency
        '''
        return "rgba" + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))


    true_out_color = '#000000'  # red
    res_out_color = '#019355'  # blue
    kbm_out_color = '#F79503'  # green

    true_out_color = hex_to_rgba(true_out_color, 1.0)
    res_out_color = hex_to_rgba(res_out_color, 1.0)
    kbm_out_color = hex_to_rgba(kbm_out_color, 1.0)

    linewidth = 2
    height = 500
    # width = int(1.4 * height)
    width = 500



    # LORENZ:
    # cx, cy, cz = 1.25, -1.25, 1.25
    # cx, cy, cz = 1.25, -1.25, 1.0
    cx, cy, cz = 1.25, -1.25, 0.5
    # f = 1.2
    f = 1.5
    camera = dict(eye=dict(x=f * cx,
                           y=f * cy,
                           z=f * cz))

    fig = go.Figure()

    # name = "Reservoir (+ KBM)"
    name = "Reservoir"
    x = res_rfit_output[:, 0].tolist()
    y = res_rfit_output[:, 1].tolist()
    z = res_rfit_output[:, 2].tolist()
    fig.add_trace(
        go.Scatter3d(x=x, y=y, z=z,
                     line=dict(
                         width=linewidth,
                         color=res_out_color,
                     ),
                     name=name,
                     mode="lines",
                     meta=dict())
    )
    name = "KBM"
    x = kbm_rfit_output[:, 0].tolist()
    y = kbm_rfit_output[:, 1].tolist()
    z = kbm_rfit_output[:, 2].tolist()
    fig.add_trace(
        go.Scatter3d(x=x, y=y, z=z,
                     line=dict(
                         width=linewidth,
                         color=kbm_out_color,
                     ),
                     name=name,
                     mode="lines",
                     meta=dict())
    )
    # TRUE:
    name = "Both"
    x = x_train_fit[:, 0].tolist()
    y = x_train_fit[:, 1].tolist()
    z = x_train_fit[:, 2].tolist()
    fig.add_trace(
        go.Scatter3d(x=x, y=y, z=z,
                     line=dict(
                         color=true_out_color,
                         width=linewidth
                     ),
                     name=name,
                     mode="lines",
                     meta=dict())
    )

    fig.update_layout(template="simple_white",
                      # showlegend=False,
                      font=dict(
                          size=18,
                          family="Times New Roman"
                      ),
                      legend=dict(
                          orientation="h",
                          yanchor="top",
                          y=0.8,
                          xanchor="right",
                          x=0.8,
                          font=dict(size=20),
                          entrywidth=0,
                          bordercolor="grey",
                          borderwidth=2,
                      ),

                      )



    fig.update_scenes(
        # xaxis_title=r"$x(t)$",
        # yaxis_title=r"$y(t)$",
        # zaxis_title=r"$z(t)$",

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
    )

    fig.update_layout(scene_camera=camera,
                      width=width,
                      height=height,
                      )

    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(range=[-200, 200]),
    #         yaxis=dict(range=[-200, 200]),
    #         zaxis=dict(range=[-200, 50]),
    #     )
    # )

    fig.update_layout(
        # margin=dict(l=5, r=5, t=5, b=5),
        margin=dict(l=0, r=0, t=0, b=0),
    )


    print(fig.layout.scene)

    # fig.show()

    # SAVE
    file_name = f"kbm_vs_res_out_3d_traj_{filename_extra}.png"
    fig.write_image(file_name, scale=6)


# bar plot:
# fig = go.Figure()
# x = np.arange(x_dim) + 1
# fig.add_traces([
#     go.Bar(x=x,
#            y=np.std(res_rfit_output, axis=0),
#            # error_y=dict(symmetric=False,
#            #              thickness=2,
#            #              width=6,
#            #              array=high_abs_w_out_hyb_res_avg - median_abs_w_out_hyb_res_avg,
#            #              arrayminus=median_abs_w_out_hyb_res_avg - low_abs_w_out_hyb_res_avg),
#            marker_color="#019355",
#            name="Reservoir"
#            ),
#     go.Bar(x=x,
#            y=np.std(kbm_rfit_output, axis=0),
#            # error_y=dict(symmetric=False,
#            #              thickness=2,
#            #              width=6,
#            #              array=high_abs_w_out_hyb_kbm_avg - median_abs_w_out_hyb_kbm_avg,
#            #              arrayminus=median_abs_w_out_hyb_kbm_avg - low_abs_w_out_hyb_kbm_avg),
#            marker_color="#F79503",
#            name="KBM"
#            )
# ])
#
# fig.show()
