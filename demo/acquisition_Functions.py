import pickle
import numpy as np
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import random
import warnings
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
import math

import numpy as np
from scipy.stats import norm


def gama_expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1, a=1, b=1):
    """
    This method is used to balance exploration and exploitation in optimization by considering not just the expected improvement
    but also incorporating the variance and probability of improvement with scaling factors `a` and `b`.

    :param x: Input parameter(s) to be evaluated, assumed to be a single point or set of points.
    :param gaussian_process: A trained Gaussian Process model for predicting mean and variance at the given point.
    :param evaluated_loss: The losses observed so far (the objective function values already evaluated).
    :param greater_is_better: If True, the goal is to maximize the objective function, else minimize it.
    :param n_params: Number of parameters (dimensions) of the input `x`.
    :param a: Scaling factor for the expected improvement.
    :param b: Scaling factor for the variance term (exploration).

    :return: A combined measure of expected improvement, probability of improvement, and variance.
    """

    # Ensure that `x` is reshaped as a column vector or set of column vectors to match the input format of the Gaussian Process
    x_to_predict = x.reshape(-1, n_params)

    # Predict the mean (`mu`) and standard deviation (`sigma`) of the Gaussian process at the input `x`
    try:
        mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)
    except Exception as e:
        raise ValueError(f"Error in Gaussian Process prediction: {e}")

    # Determine the optimal value of the evaluated loss (either max or min based on `greater_is_better`)
    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)  # Maximize the objective function
    else:
        loss_optimum = np.min(evaluated_loss)  # Minimize the objective function

    # Determine the scaling factor for improvement calculations (inversion for minimization problems)
    scaling_factor = (-1) ** (not greater_is_better)

    # Safeguard against cases where sigma equals zero (avoid division by zero)
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma  # Standardized improvement

        # Calculate the classical Expected Improvement (EI)
        expected = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Handle cases where sigma is zero (no uncertainty about the prediction)
        expected[sigma == 0.0] = 0.0

    # Recompute the scaling factor for further calculations (probability of improvement)
    scaling_factor = (-1) ** (not greater_is_better)

    # Recalculate Z for probability estimation
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma

    # Calculate the probability of improvement based on the current model predictions
    probability = norm.cdf(Z)

    # Variance (sigma^2) is used to enhance exploration (exploit variance for uncertain regions)
    variance = sigma ** 2

    # Compute the modified Expected Improvement formula with scaling terms `a` and `b` for exploration-exploitation trade-off
    # a * Expected Improvement + Probability of Improvement + b * Variance
    modified_ei = a * expected + probability + b * variance

    return modified_ei

