import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.spatial import distance_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

def gaussian_process(x, y, cov_func, hyperparameters):
    """
    calculate gaussian process with mean zero and squared exponential covaraiance
    x: a finite set of points on the unit inteval
    hyperparameters: a tuple b, tau squared 1, tau squared 2
    """
    return multivariate_normal(np.zeros(len(x)), get_covariance_matrix(x, y, cov_func, hyperparameters), allow_singular=True)


def squared_exponential_cov(x1, x2, hyperparameters):
    """squared exponential covariance function"""
    d = np.linalg.norm(x1-x2)
    return hyperparameters["tau_sq_1"] * np.exp(-.5 * (d/hyperparameters["b"])**2) + hyperparameters["tau_sq_2"]*int(np.all(x1==x2))

def matern_5_2(x1, x2, hyperparameters):
    """squared exponential covariance function"""
    d = np.linalg.norm(x1-x2)
    return hyperparameters["tau_sq_1"] * (1 + np.sqrt(5)*d/hyperparameters["b"] + (5*d**2)/(3*hyperparameters["b"]**2))*np.exp(-np.sqrt(5)*d/hyperparameters["b"]) + hyperparameters["tau_sq_2"]*int(np.all(x1==x2))

def get_covariance_matrix(x, y, cov_func, hyperparameters):
    """returns a covariance matrix for the given covariance function"""
    return np.array([[cov_func(x[i], y[j], hyperparameters) for i in range(len(x))] for j in range(len(y))])

def get_gp_params(x, y, x_pred, cov_func, hyperparameters, sigma_squared=1):
    """get mean, covariance, smoothing for gp"""
    sigma_11 = get_covariance_matrix(x, x, cov_func, hyperparameters)
    sigma_22 = get_covariance_matrix(x_pred, x_pred, cov_func, hyperparameters)
    sigma_21 = get_covariance_matrix(x, x_pred, cov_func, hyperparameters)
    pinv_mat = np.linalg.pinv(sigma_11 + np.eye(len(x))*sigma_squared)
    H = sigma_21.T @ pinv_mat
    mean = sigma_21.T @ pinv_mat @ y
    cov = sigma_22 - (sigma_21 @ pinv_mat @ sigma_21.T)
    return mean, cov, H

def get_faster_covariance_matrix(x1, x2, hyperparameters):
    """use distance matrix function and matrix operations to speed up calculation"""
    d = distance_matrix(x1, x2)
    return hyperparameters["tau_sq_1"] * np.exp(-.5 * (d/hyperparameters["b"])**2) + hyperparameters["tau_sq_2"]*np.eye(x1.shape[0], x2.shape[0])

    
def get_gp_params_faster(x, y, x_pred, hyperparameters, sigma_squared=1):
    """speed up with matrix math"""
    sigma_11 = get_faster_covariance_matrix(x, x, hyperparameters)
    sigma_22 = get_faster_covariance_matrix(x_pred, x_pred, hyperparameters)
    sigma_21 = get_faster_covariance_matrix(x, x_pred, hyperparameters)
    pinv_mat = np.linalg.pinv(sigma_11 + np.eye(len(x))*sigma_squared)
    H = sigma_21.T @ pinv_mat
    mean = H @ y
    cov = sigma_22 - (sigma_21.T @ pinv_mat @ sigma_21)
    return mean, cov, H

def predict(x, y, x_pred, cov_func, hyperparameters, sigma_squared=1):
    """get predictions and CI"""
    y_pred, _, H = get_gp_params(x, y, x_pred, cov_func, hyperparameters, sigma_squared)
    ci = calculate_ci(y_pred, H)
    return y_pred, ci

def calculate_ci(y_pred, H, sigma_squared=1, sig_level=.05):
    """calculate CIs for predictions"""
    var = sigma_squared * np.sum(H ** 2, axis=1)
    z = norm(0, 1).ppf(1 - sig_level / 2)
    lower = y_pred.flatten() - z * np.sqrt(var)
    upper = y_pred.flatten() + z * np.sqrt(var)
    return lower, upper

def plot_ci(x, y, y_pred, ci):
    """plotting helper for confidence intervals"""
    plt.scatter(x, y, alpha =.5, color='black')
    plt.fill_between(x, ci[0], ci[1], alpha=.5)
    plt.plot(x, y_pred, color='red')

def calculate_marginal(x, y, cov_func, hyperparameters, sigma_squared = 1):
    """calculate marginal p(y|b,t1^2) given hyperparameters"""
    cov_mat = get_covariance_matrix(x,x,cov_func, hyperparameters)
    cov = sigma_squared * np.eye(x.shape[0]) + cov_mat
    dist = multivariate_normal(cov=cov, allow_singular=True)
    return dist.logpdf(y)

def grid_optimize(x, y, cov_func, hyperparameter_grid):
    """do a grid search for marginals"""
    all_marginals = []
    hyperparameter_pairs = hyperparameter_grid.reshape(hyperparameter_grid.shape[0]*hyperparameter_grid.shape[1],2)
    for n in tqdm(range(len(hyperparameter_pairs)), "searching hyperparameters:"):
        hyperparameters = {"b": hyperparameter_pairs[n][0], "tau_sq_1": hyperparameter_pairs[n][1], "tau_sq_2":0}
        marginal = calculate_marginal(x, y, cov_func, hyperparameters)
        all_marginals.append(marginal)
    return np.array(all_marginals).reshape(hyperparameter_grid.shape[0], hyperparameter_grid.shape[1]), hyperparameter_pairs[np.argmax(all_marginals)]

def get_hyperparameter_grid(x_min, x_max, y_min, y_max, n_interval=10):
    """get a grid of hyperparameters for searching"""
    x_range = np.linspace(x_min, x_max, n_interval)
    y_range = np.linspace(y_min, y_max, n_interval)
    return np.array([[[x_range[i], y_range[j]] for i in range(len(x_range))] for j in range(len(y_range))])