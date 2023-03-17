"""RC plot: train vs predict trajectory lorenz """
import plotly.graph_objects as go

import esnchaos.simulations as sims


def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return "rgba" + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))


predicted_color = '#EF553B'  # red
true_color = '#636EFA'  # blue
predicted_color = hex_to_rgba(predicted_color, 0.9)
true_color = hex_to_rgba(true_color, 0.9)



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
pert_scale = 1e-5
pert_init_cond = initial_condition_new + pert_scale

steps = 4000

# Baseline traj:
base_traj = sys_obj.simulate(steps, starting_point=initial_condition_new)

# Perturbed traj:
pert_traj = sys_obj.simulate(steps, starting_point=pert_init_cond)


true_pred = base_traj

# eps kbm prediction:
pred = pert_traj



linewidth = 2
height = 500
# width = int(1.4 * height)
width = 500



# LORENZ:
cx, cy, cz = 1.25, -1.25, 0.5
f = 1.5
camera = dict(eye=dict(x=f * cx,
                       y=f * cy,
                       z=f * cz))

fig = go.Figure()

# TRUE:
name = "basis"
x = true_pred[:, 0].tolist()
y = true_pred[:, 1].tolist()
z = true_pred[:, 2].tolist()
fig.add_trace(
    go.Scatter3d(x=x, y=y, z=z,
                 line=dict(
                     color=true_color,
                     width=linewidth
                 ),
                 name=name,
                 mode="lines",
                 meta=dict())
)

fig.update_layout(template="simple_white",
                  showlegend=False,
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
                      font=dict(size=20)
                  ),
                  # xaxis_title=r"$test$"

                  )



fig.update_scenes(

    xaxis_showticklabels=False,
    yaxis_showticklabels=False,
    zaxis_showticklabels=False
)



fig.update_layout(scene_camera=camera,
                  width=width,
                  height=height,
                  )


fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
)


print(fig.layout.scene)

# SAVE
file_name = f"perturbed_vs_true_lorenz_traj.png"
fig.write_image(file_name, scale=3)

