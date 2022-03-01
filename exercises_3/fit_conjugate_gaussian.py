"""Code to fit a conjugate gaussian linear model"""
import pandas as pd
import numpy as np


def update_K(K, X, lam):
    """calculating K star"""
    return X.T @ lam @ X + K


def update_m(K, X, lam, y, m):
    return np.linalg.solve(update_K(K, X, lam), X.T @ lam @ y + K @ m)


def update_d(d, n):
    return d + n


def update_eta(m, K, eta, y, la
m):
    return eta + y.T @ lam @ y + m.T @ K @ m - m.T @ K @ m


def fit_cglm(X, y, lam, K, m, d, eta, add_intercept=True):
    if add_intercept:
        np.insert(X, 0, np.ones(len(X)), axis=1)
    return update_m(K, X, lam, y, m)


def main():
    """main method"""
    data = pd.read_csv("./data/greenbuildings.csv")
    X = np.array(
        data[["City_Market_Rent", "green_rating", "age", "class_a", "class_b"]]
    )
    y = np.array(data["Rent"] * data["leasing_rate"]) / 100  # new revenue variable
    lam = np.identity(X.shape[0])  # lambda = I
    K = np.identity(X.shape[1]) * 0.001
    m = np.ones(X.shape[1])
    d = 1
    eta = 1
    betas = fit_cglm(X, y, lam, K, m, d, eta, add_intercept=True)
    plt.plot(
        np.array(data["City_Market_Rent"]), y - X @ betas, "bo", markersize=2, alpha=0.1
    )
    plt.savefig("res_vs_market.png")


if __name__ == "__main__":
    main()
