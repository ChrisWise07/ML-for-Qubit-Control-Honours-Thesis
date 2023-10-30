import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from time_series_to_noise.constants import EPSILON


def expected_improvement_for_minimising_obj_func(
    x: np.ndarray, gp_model: GaussianProcessRegressor, y_min: float
) -> float:
    """
    Objective function for the Bayesian optimisation. This is the
    expected improvement function.

    Args:
        x (np.ndarray):
            The input to the objective function.
        gp_model (GaussianProcessRegressor):
            The Gaussian process model.
        y_min (float):
            The minimum value of the objective function.

    Returns:
        ei (float):
            The expected improvement.
    """
    x = x.reshape(1, -1)
    mu, sigma = gp_model.predict(x, return_std=True)
    sigma += EPSILON
    improvement = y_min - mu
    Z = improvement / sigma
    return improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
