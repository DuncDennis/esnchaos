from typing import Callable

import numpy as np
from sklearn.decomposition import PCA

import esnchaos.simulations as sims


def get_rom(
        time_series: np.ndarray,
        sys_obj,
        n_pod: int,
):
    # get pca object and fit to data:
    pca = PCA(n_components=n_pod)
    pca.fit(time_series)

    # get pca component matrix and mean:
    pc_matrix = pca.components_
    # pc_mean = pca.mean_

    # get transform func:
    def transform_func(x: np.ndarray) -> np.ndarray:
        try:
            return pca.transform(x[np.newaxis, :])[0, :]
        except:  # really bad programming.
            temp = np.empty(n_pod)
            temp[:] = np.nan
            return temp
    # transform_func = lambda x: pca.transform(x[np.newaxis, :])[0, :]

    # get original flow:
    original_flow = sys_obj.flow

    # pod flow:
    def pod_flow(pod_state: np.ndarray) -> np.ndarray:
        arg = pca.inverse_transform(pod_state[np.newaxis, :])[0, :]
        flow_out = original_flow(arg)
        return flow_out @ pc_matrix.T
        # return pc_matrix @ original_flow(arg)  # doesnt matter which one to use.

    # pod iterator:
    pod_iterator = lambda x: sims._runge_kutta(pod_flow, dt=sys_obj.dt, x=x)

    def kbm(x: np.ndarray) -> np.ndarray:
        x_projected = transform_func(x)
        return pod_iterator(x_projected)

    return pod_iterator, transform_func, kbm


if __name__ == "__main__":
    # sys_obj = sims.Lorenz63()
    # sys_obj = sims.Roessler()
    # sys_obj = sims.ChuaCircuit()
    # sys_obj = sims.WindmiAttractor()
    # sys_obj = sims.Rucklidge()
    sys_obj = sims.Lorenz96()
    print(sys_obj.sys_dim)

    # original data:
    steps = 1000
    time_series = sys_obj.simulate(steps)

    n_pod = 25

    pod_iterator, transform_func, _ = get_rom(time_series,
                                           sys_obj,
                                           n_pod=n_pod)


    # experiment:
    initial_point = transform_func(time_series[0, :])

    pod_states = np.zeros((steps, n_pod))
    pod_states[0, :] = initial_point

    time_series_projected = np.zeros((steps, n_pod))
    time_series_projected[0, :] = initial_point

    for i in range(1, steps):
        pod_states[i, :] = pod_iterator(pod_states[i-1, :])
        time_series_projected[i, :] = transform_func(time_series[i, :])
    # print(pod_states)

    import plotly.graph_objs as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=pod_states[:, 0],
                   y=pod_states[:, 1])
    )

    # original transformed:
    fig.add_trace(
        go.Scatter(x=time_series_projected[:, 0],
                   y=time_series_projected[:, 1])
    )
    fig.show()
