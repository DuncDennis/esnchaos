"""RC plot: train vs predict trajectory lorenz """
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import esnchaos.simulations as sims
import esnchaos.measures as meas

predicted_color = '#EF553B'  # red
true_color = '#636EFA'  # blue


# Create data:
# dt = 0.1
mle = 0.9059


# Lorenz:
sigma = 10.0
rho = 28.0
beta = 8 / 3
dt = 0.1
sys_obj = sims.Lorenz63(dt=dt,
                        sigma=sigma,
                        rho=rho,
                        beta=beta)

# initial condition:
skip_steps = 1000
initial_condition_new = sys_obj.simulate(time_steps=skip_steps)[-1, :]

# perturbed initial condition:
pert_scale = 1e-4
# pert_init_cond = initial_condition_new + pert_scale
pert_init_cond = initial_condition_new + np.array([pert_scale, 0, 0])

steps = 2000

# Baseline traj:
base_traj = sys_obj.simulate(steps, starting_point=initial_condition_new)

# Perturbed traj:
pert_traj = sys_obj.simulate(steps, starting_point=pert_init_cond)


true_pred = base_traj

# eps kbm prediction:
pred = pert_traj



error_series_ts = meas.error_over_time(y_pred=pred,
                                       y_true=true_pred,
                                       normalization="root_of_avg_of_spacedist_squared")
vt = meas.valid_time_index(error_series_ts, error_threshold=0.4)
vt = mle * dt * vt
vt = np.round(vt, 1)

latex_text_size = "large"
linewidth = 3
height = 350
width = int(2 * height)

dim = 0
t_max = 250
x = np.arange(pred.shape[0])[:t_max] * dt * mle
xaxis_title =  rf"$\{latex_text_size} " + r't \cdot \lambda_\text{max}$'
fig = make_subplots(rows=3,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=None,
                    print_grid=True,
                    x_title=xaxis_title,
                    )



index_to_dimstr = {0: rf"$\{latex_text_size} " + r"x(t)$",
                   1: rf"$\{latex_text_size} " + r"y(t)$",
                   2: rf"$\{latex_text_size} " + r"z(t)$",}

for i_x in range(3):
    dimstr = index_to_dimstr[i_x]
    if i_x == 0:
        showlegend=True
    else:
        showlegend=False
    # TRUE:
    name = "basis"
    # true_color = "black"
    y = true_pred[:t_max, i_x]
    fig.add_trace(
        go.Scatter(x=x, y=y,
                   line=dict(
                       color=true_color,
                       width=linewidth,
                   ),
                   showlegend=showlegend,
                   name=name,
                   mode="lines"),
        row=i_x + 1, col=1
    )

    # TRUE:
    name = r"perturbed"
    y = pred[:t_max, i_x]
    fig.add_trace(
        go.Scatter(x=x, y=y,
                   line=dict(
                       color=predicted_color,
                       width=linewidth,
                   ),
                   showlegend=showlegend,
                   name=name,
                   mode="lines"),
        row=i_x + 1, col=1
    )

    fig.update_yaxes(
        title_text=dimstr,
        row=i_x + 1,
        col=1,
        title_standoff=5,
    )

    fig.update_yaxes(
        # showticklabels=False,
        showticklabels=True,
        row=i_x + 1,
        col=1,
    )

    fig.update_xaxes(
        range=[0, np.max(x)],
        row=i_x + 1,
        col=1,
    )

fig.update_yaxes(
    tick0 = -15,
    dtick = 15,
    row=1,
    col=1,
)

fig.update_yaxes(
    tick0 = -20,
    dtick = 20,
    row=2,
    col=1,
)

fig.update_yaxes(
    tick0 = 10,
    dtick = 30,
    row=3,
    col=1,
)

fig.update_layout(
    template="simple_white",
    font=dict(
        size=18,
        family="Times New Roman"
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.25,
        xanchor="right",
        x=1,
        font=dict(size=20)
    )
)


fig.update_layout(
    width=width,
    height=height,
)
fig.update_layout(
    margin=dict(l=15, r=40, t=10, b=50),
)

# SAVE
file_name = f"perturbed_vs_true_lorenz_1dim_all.png"
fig.write_image(file_name, scale=3)

