import pandas as pd

import esnchaos.sweep_plots as sp

import tests.example_sweep_experiment as ese

def test_sweep_experiment(tmp_path):
    file_path = ese.run(tmp_path)

    df = sp.read_pkl(file_path)
    assert type(df) == pd.DataFrame
