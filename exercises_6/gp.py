import numpy as np
from scipy.stats import multivariate_normal

def gaussian_process(x, cov_func, hyperparameters):
    """
    calculate gaussian process with mean zero and squared exponential covaraiance plot
    x: a finite set of points on the unit inteval
    hyper parameters: a tuple b, tau squared 1, tau squared 2
    """
    multivariate_normal(x, 0, get_covariance_matrix(x, cov_func, hyperparameters))


def squared_exponential_cov(x1, x2, hyperparameters):
    """squared exponential covariance function"""
    d = np.linalg.norm(x1-x2)
    return hyperparameters["tau_sq_1"] * np.exp(-.5 * (d/hyperparameters["b"])**2) + hyperparameters["tau_sq_2"]*np.kron(x1, x2)


def matern_5_2(x1, x2, hyperparameters):
    """squared exponential covariance function"""
    d = np.linalg.norm(x1-x2)
    return hyperparameters["tau_sq_1"] * (1 + np.sqrt(5)*d/hyperparameters["b"] + (5*d**2)/(3*hyperparameters["b"]**2))*np.exp(-np.sqrt(5)*d/hyperparameters["b"]) + hyperparameters["tau_sq_2"]*np.kron(x1, x2)

def get_covariance_matrix(x, cov_func, hyperparameters):
    """returns a covariance matrix for the given covariance function"""
    return [[cov_func(x[i], x[j], hyperparameters) for i in range(len(x))] for j in range(len(x))]