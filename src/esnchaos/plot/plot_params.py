
# PARAMETER TRANSFORMATIONS latex:
PARAM_TRANSFORM_LTX = {
    "P node_bias_scale": r"\text{Node bias scale } \sigma_\text{b}$",
    "P t_train": r"\text{Training size } N_\text{T}$",
    "P r_dim": r"\text{Reservoir dimension } r_\text{dim}$",
    "P reg_param": r"\text{Regularization parameter } \beta$",
    "P w_in_scale": r"\text{Input strength } \sigma$",
    "P n_avg_deg": r"\text{Avg. node degree } d$",
    "P n_rad": r"\text{Spectral radius } \rho_0$",
    "P dt": r"\text{Time step of system } \Delta t$",
    "P x_train_noise_scale": r"\text{Train noise scale } \sigma_\text{T}$",
    "P model_error_eps": r"\text{model error }\epsilon$",
    "P predictor_type": r"\text{hybrid type}$",
    "P system": r"\text{system}$",
    "P rr_type": r"\text{ridge regression type}$",
    "P data_offset": r"\text{data offset } \delta$",
    "P dim_subset": r"\text{selected dimension } l$",
    "P r_to_rgen_opt": r"\text{Readout function } \Psi$",
}

# PARAMETER TRANSFORMATIONS html:
PARAM_TRANSFORM_HTML = {
    "P node_bias_scale": "<i>σ</i><sub>b</sub>",
    "P r_dim": "<i>r</i><sub>dim</sub>",
    "P reg_param": "<i>β</i>",
    "P w_in_scale": "<i>σ</i>",
    "P n_avg_deg": "d",
    "P n_rad": "<i>ρ</i><sub>0</sub>",
}

# METRIC TRANSFORMATIONS latex:
METRIC_TRANSFORM_LTX = {
    "M TRAIN PCMAX": r" i_\text{co}$",
    "M VALIDATE VT": r" t_\text{v} \lambda_\text{max}$",
    "M TRAIN MSE": r" \text{MSE train}$",
    "M VALIDATE MSE": r" \text{MSE validate}$",
}

# global default plot params
DEFAULT_PLOT_PARAMS = dict(
    exponentformat="power",
    template="simple_white",
    font_family="Times New Roman",
    font_size=25,
    latex_text_size="large",  # normalsize, large, Large, huge,
    param_transform_ltx=PARAM_TRANSFORM_LTX,
    param_transform_html=PARAM_TRANSFORM_HTML,
    metric_transform_ltx=METRIC_TRANSFORM_LTX,
)

# One dim params:
DEFAULT_PLOT_ONE_DIM_PARAMS = dict(
    # Error bar:
    error_barwidth=8,
    error_thickness=2,

    # line:
    line_width=3,
    line_color="Black",

    # Layout:
    width=600,
    height=int(0.50 * 600),
    margin_dict=dict(l=5, r=5, t=5, b=5),

    # Grid:
    grid_settings_dict=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor="gray"
    ),

    # x axis dict:
    x_axis_dict=dict(),

    # y axis dict:
    y_axis_dict=dict(),

    # standard param vertical line settings:
    vertical_line_dict=dict(
        line_width=5,
        line_dash="dash",
        line_color="green",
        opacity=0.6
    ),

    # plot a vertical line at value:
    vertical_line_val=None,

    # logarthmic axes:
    log_x=False,
)

DEFAULT_PLOT_ONE_DIM_PARAMS = DEFAULT_PLOT_ONE_DIM_PARAMS | DEFAULT_PLOT_PARAMS

DEFAULT_PLOT_ONE_DIM_VIOLIN_PARAMS = dict(
    # Layout:
    width=550,
    height=int(0.65*550),
    margin_dict=dict(l=5, r=5, t=5, b=5),

    # logarthmic axes:
    log_x = False,

    # Highlight one x value:
    color_green_at_val = None,

    # Grid:
    grid_settings_dict=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor="gray"
    ),

    # no xaxis title:
    no_xaxis_title=False,

    # x axis dict:
    x_axis_dict=dict(),

    # y axis dict:
    y_axis_dict=dict(),

    # x_param value transformation:
    x_param_val_transform_dict=None,
)

DEFAULT_PLOT_ONE_DIM_VIOLIN_PARAMS = DEFAULT_PLOT_ONE_DIM_VIOLIN_PARAMS | DEFAULT_PLOT_PARAMS


DEFAULT_PLOT_TWO_DIM_PARAMS = dict(
    # Layout:
    width=550,
    height=int(0.65 * 550),
    margin_dict=dict(l=5, r=5, t=5, b=5),

    # Error bar:
    error_barwidth=8,
    error_thickness=2,

    # line:
    line_width=3,

    # x Grid:
    x_grid_settings_dict=dict(
    ),

    # y Grid:
    y_grid_settings_dict=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor="gray"
    ),

    # logarthmic axes:
    log_x=False,

    # no xaxis title:
    no_xaxis_title=False,

    # x axis dict:
    x_axis_dict=dict(),

    # y axis dict:
    y_axis_dict=dict(),

    # show legend:
    show_legend=True,

    # legend dict:
    legend_dict=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,  # 0.99
        xanchor="left",
        # x=0.01,
        font=dict(size=20),
        bordercolor="grey",
        borderwidth=2,
    ),

    # line color and style:
    hex_color_list = None,
    line_style_list = None,
    color_alpha = 1.0,

    # order of param values:
    col_param_val_rename_func=None,
    col_param_val_order_dict=None,
)
DEFAULT_PLOT_TWO_DIM_PARAMS = DEFAULT_PLOT_TWO_DIM_PARAMS | DEFAULT_PLOT_PARAMS


DEFAULT_PLOT_M_VS_M_PARAMS = dict(
    # Layout:
    width=600,
    height=int(0.7 * 600),
    margin_dict=dict(l=20, r=20, t=20, b=20),

    # x axis dict:
    x_axis_dict=dict(),

    # y axis dict:
    y_axis_dict=dict(),

    # Error bar:
    # error_barwidth=8,
    error_thickness=1,

    # x Grid:
    x_grid_settings_dict=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.2)'
    ),

    # y Grid:
    y_grid_settings_dict=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.2)'
    ),

    # Color logscale:
    log_col = False,

    # marker styling:
    marker_line_width=1,
    marker_line_color="DarkSlateGrey",
    marker_size=12,

    # colorbar ticks:
    color_dtick=None,
    color_tick0=None,

)
DEFAULT_PLOT_M_VS_M_PARAMS = DEFAULT_PLOT_M_VS_M_PARAMS | DEFAULT_PLOT_PARAMS
