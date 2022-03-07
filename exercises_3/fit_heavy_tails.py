"""Code to fit a conjugate gaussian linear model"""
import pandas as pd
import numpy as np
from scipy.stats import gamma, multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt


class HeavyTailedModel:
    def __init__(
        self,
        X: np.array,
        y: np.array,
        lam,
        K,
        m,
        d,
        eta,
        h,
        n_iter: int = 1000,
        burn_in: int = 500,
        add_intercept: bool = True,
    ):
        self.add_intercept = add_intercept
        self.X = self._process_X(X)
        self.y = y
        self.lam = lam
        self.K = K
        self.m = m
        self.d = d
        self.eta = eta
        self.h = h
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.n = self.X.shape[0]
        self.beta = np.zeros(self.X.shape[1])
        self.lambdas = np.array([np.zeros_like(np.diagonal(lam))] * (self.n_iter))
        self.betas = np.array([np.zeros_like(self.beta)] * (self.n_iter))

    def _process_X(self, X):
        if self.add_intercept:
            return np.insert(X, 0, np.ones(len(X)), axis=1)
        return X

    def _update_lambda(self):
        a = (self.h + 1) / 2
        b = 1 / (self.h + self.omega * (self.y - self.X @ self.beta)) / 2
        sample = gamma.rvs(a, b)
        self.lam = np.diag(sample)

    def _update_omega(self, d_star, eta_star):
        self.omega = gamma.rvs(d_star / 2, 2 / eta_star)

    def _update_beta(self, m_star, K_star):
        """update betas"""
        self.beta = multivariate_normal.rvs(
            mean=m_star, cov=np.linalg.inv(self.omega * K_star)
        )

    def _update_K(self):
        """calculating K star"""
        return self.X.T * self.lam.diagonal() @ self.X + self.K

    def _update_m(self):
        return np.linalg.solve(
            self.X.T * self.lam.diagonal() @ self.X + self.K,
            self.X.T * self.lam.diagonal() @ self.y + self.K @ self.m,
        )

    def _update_d(self):
        return self.d + self.n

    def _update_eta(self):
        return (
            self.eta
            + self.y.T * self.lam.diagonal() @ self.y
            + self.m.T @ self.K @ self.m
            - self.m.T @ self.K @ self.m
        )

    def _store_iter(self, iter):
        self.lambdas[iter] = np.diagonal(self.lam)
        self.betas[iter] = self.beta

    def fit(self):
        iter = 0
        for _ in tqdm(range(self.n_iter + self.burn_in)):

            # update
            d_star = self._update_d()
            eta_star = self._update_eta()
            K_star = self._update_K()
            m_star = self._update_m()

            # gibbs sampling
            self._update_omega(d_star, eta_star)
            self._update_beta(m_star, K_star)
            self._update_lambda()

            # store each step
            if iter >= self.burn_in:
                self._store_iter(iter - self.burn_in)

            # store updates
            self.d = d_star
            self.eta = eta_star
            self.K = K_star
            self.m = m_star
            iter += 1

    def predict(self):
        return self.X @ self.beta


def main():
    """main method"""
    data = pd.read_csv("./data/greenbuildings.csv")
    X = np.array(
        data[["City_Market_Rent", "green_rating", "age", "class_a", "class_b"]]
    )
    y = np.array(data["Rent"] * data["leasing_rate"]) / 100  # new revenue variable

    lam = np.identity(X.shape[0])  # lambda = I
    K = np.identity(X.shape[1] + 1) * 0.001
    m = np.ones(X.shape[1] + 1)
    d = 1
    eta = 1
    h = 1

    htm = HeavyTailedModel(X, y, lam, K, m, d, eta, h)
    htm.fit()

    plt.plot(
        np.array(data["City_Market_Rent"]),
        y - htm.predict(),
        "bo",
        markersize=2,
        alpha=0.1,
    )
    plt.savefig("res_vs_market_htm.png")

    plt.clf()
    plt.hist(np.array([np.mean(x) for x in htm.lambdas]), bins=20)
    plt.savefig("lambdas.png")


# X = np.insert(X, 0, np.ones(len(X)), axis=1)
# d_star = d + n
# eta_star = eta + y.T @ lam @ y + m.T @ K @ m - m.T @ K @ m
# K_star = X.T @ lam @ X + K
# m_star = np.linalg.solve(K_star, X.T @ lam @ y + K @ m)
# omega = gamma.rvs(d_star / 2, 2 / eta_star)
# beta = multivariate_normal.rvs(mean=m_star, cov=np.linalg.inv(omega * K_star))
# a = (h + 1) / 2
# b = 1 / (h + omega * (y - X @ beta)) / 2
# sample = gamma.rvs(a, b)
# lam_star = np.diag(sample)

# d = d_star
# eta = eta_star
# K = K_star
# m = m_star
# lam = lam_star
