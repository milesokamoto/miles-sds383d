"""implement gradient descent to fit a logistic regression model"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def standardize(X):
    """standardize/scale X"""
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    return (X - means) / stds


def binomial_link_function(nu):
    """link function from binomial (sigmoid)"""
    return 1 / (1 + np.exp(-nu))


def b_theta_binomial(theta):
    """b_theta from binomial"""
    return np.log(1 + np.exp(theta))


def log_likelihood(X, y, beta, b_theta):
    """log likelihood with generic parameters"""
    return np.sum((np.array(y) * (X @ beta) - b_theta(X @ beta)))


def score_function(X, y, beta, link_function):
    """for the canonical link function, score function simplifies"""
    nu = X @ beta
    mu = link_function(nu)
    return X.T @ (y - mu)


def hessian_function(X, beta, link_function):
    """for the canonical link function, hessian function simplifies"""
    nu = X @ beta
    mu = link_function(nu)
    return X.T @ (np.diag(mu * (1 - mu)) @ X)


def fit_glm(
    X,
    y,
    link_function,
    b_theta,
    # weight,
    # dispersion,
    lr=0.001,
    thresh=0.000001,
    add_intercept=True,
    max_iter=100000,
):
    """generalized function to fit a glm given a link function, b(theta) and data"""
    if add_intercept:
        X.insert(0, 1, np.ones(len(X)))
    iter = 0
    log_liks = []  # maybe preallocate this to NAs for the max_iter value
    ll_old = None
    beta = np.random.randn(X.shape[1])
    while True and iter < max_iter:
        ll = -log_likelihood(X, y, beta, b_theta)
        log_liks.append(ll)
        if ll_old:
            if abs(ll - ll_old) / abs(ll_old) <= thresh:
                break
        grad = score_function(X, y, beta, link_function)
        beta += lr * grad
        ll_old = ll
        iter += 1

    return beta, log_liks


def fit_glm_newton(
    X,
    y,
    link_function,
    b_theta,
    # weight,
    # dispersion,
    lr=0.001,
    thresh=0.000001,
    add_intercept=True,
    max_iter=100000,
):
    """generalized function to fit a glm given a link function, b(theta) and data"""
    if add_intercept:
        X.insert(0, 1, np.ones(len(X)))
    iter = 0
    log_liks = []  # maybe preallocate this to NAs for the max_iter value
    ll_old = None
    beta = np.zeros(X.shape[1])
    while True and iter < max_iter:
        ll = -log_likelihood(X, y, beta, b_theta)
        log_liks.append(ll)
        if ll_old:
            if abs(ll - ll_old) / abs(ll_old) <= thresh:
                break
        hessian = hessian_function(X, beta, link_function)
        grad = score_function(X, y, beta, link_function)
        beta += np.linalg.inv(hessian) @ grad
        ll_old = ll
        iter += 1

    return beta, log_liks


def main():
    """main method"""
    np.random.seed(1)
    data = pd.read_csv("data/wdbc.csv", header=None).iloc[:, 1:12]
    X = data.iloc[:, 1:]
    X = standardize(X)
    y = [1 if y == "M" else 0 for y in data.iloc[:, 0]]

    beta, log_liks = fit_glm_newton(X, y, binomial_link_function, b_theta_binomial)

    simple_plot(log_liks, "log_lik_plot.png")

    # preds = link_function(X@beta)
    # sum((preds-y)**2)/len(y)


def simple_plot(series, filename):
    """simple plotting utility"""
    plt.plot(np.arange(0, len(series)), series)
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":
    main()


# from sklearn.linear_model import LogisticRegression

# clf = LogisticRegression(solver="newton-cg", C=1e6, fit_intercept=True).fit(
#     X, np.ravel(y)
# )
# np.set_printoptions(suppress=True)
# print(clf.coef_, clf.intercept_)
