
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
    "M VALIDATE VT": r" t_\text{v} \lambda_\text{max}$"
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

