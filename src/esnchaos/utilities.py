"""Utility functions."""
from __future__ import annotations

from typing import Callable

import contextlib #  for temp_seed
import inspect

import numpy as np

@contextlib.contextmanager
def temp_seed(seed: int | None = None):
    """
    from https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    Use like:
    with temp_seed(5):
        <do_smth_that_uses_np.random>
    """
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)


def remove_invalid_args(func, args_dict):
    """Return dictionary of valid args and kwargs with invalid ones removed

    CREDIT: Taken from https://github.com/GLSRC/rescomp.

    Adjusted from:
    https://stackoverflow.com/questions/196960/can-you-list-the-keyword-arguments-a-function-receives

    Args:
        func (fct): function to check if the arguments are valid or not
        args_dict (dict): dictionary of arguments

    Returns:
        dict: dictionary of valid arguments

    """
    valid_args = inspect.signature(func).parameters
    # valid_args = func.func_code.co_varnames[:func.func_code.co_argcount]
    return dict((key, value) for key, value in args_dict.items() if key in valid_args)


def vectorize(func: Callable[[np.ndarray], np.ndarray],
              args_two_dim: tuple[np.ndarray, ...]) -> np.ndarray:
    """Vectorize a function func that takes a 1d-array as input to 2d arrays.

    Args:
        func: The function that works like: func(args[0][i, :], args[1][i, :], ...) -> out
              where args[j] has the shape (time steps, inp_dim), and out has the shape (out_dim).
        args: a tuple of 2d arrays.
    Returns:
        The vectorized result of shape (time steps, outdim).
    """

    steps = args_two_dim[0].shape[0]
    results = None
    for i in range(steps):
        args_one_dim = [args[i, :] for args in args_two_dim]
        out = func(*args_one_dim)
        if results is None:
            results = np.zeros((steps, out.size))
        results[i, :] = out

    return results


def sigmoid(x):
    """The sigmoid activation function. """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """The relu activation function."""
    return x * (x > 0)


def add_one(x):
    """Add a new dimension with value one to vector.

    Args:
        x: input vector of size (x_dim, ).

    Returns:
        output vector of size (x_dim+1, ).
    """
    return np.hstack((x, 1))
