"""hierarchical model for cheese"""
import numpy as np
from scipy.stats import multivariate_normal, gamma
from tqdm import tqdm

class SDModel:
    """defines class for a hierarchical linear regression model"""
    def __init__(self, X_grouped, y_grouped, groups, n_iter:int = 1000, burn:int=500):
        self.n_iter = n_iter
        self.burn = burn
        self.groups = groups
        self.X_grouped = X_grouped
        self.y_grouped = y_grouped
        self.n_groups = len(self.groups)
        self.n_betas = X_grouped[0].shape[1]
        self.N = sum([len(x) for x in y_grouped])
        self.n_iter = n_iter
        self.traces = {'betas': np.zeros([self.n_iter, self.n_betas, len(X_grouped)]),
                       'gamma': np.zeros((self.n_iter, self.n_betas)),
                       'sigma_squared': np.zeros(self.n_iter),
                       'lam': np.zeros((self.n_iter, self.n_betas))}

        self.lam = np.diag([1] * self.n_groups)
        self.sigma_squared = 1
        self.gamma = np.ones(self.n_betas)
        self.betas = np.ones((self.n_betas, len(X_grouped)))

    def _update_betas(self):
        """update for betas"""
        for group in range(self.n_groups):
            K_inv = np.linalg.inv(self.X_grouped[group].T @ self.X_grouped[group] / self.sigma_squared +
                                  self.inv_lam / self.sigma_squared)
            m = K_inv @ (self.X_grouped[group].T @ self.y_grouped[group] / self.sigma_squared +
                         self.inv_lam @ self.gamma / self.sigma_squared)
            self.betas[:, group] = multivariate_normal.rvs(m, K_inv)

    def _update_sigma_squared(self):
        """update for sigma squared"""
        alpha = (self.N + self.n_groups) / 2
        beta = 1 / 2 * sum(
            [(self.y_grouped[group_number] - self.X_grouped[group_number] @ self.betas[:, group_number]).T @ (
                    self.y_grouped[group_number] - self.X_grouped[group_number] @ self.betas[:, group_number]) + (
                     self.betas[:, group_number] - self.gamma).T @ self.inv_lam @ (
                     self.betas[:, group_number] - self.gamma) for group_number in range(self.n_groups)])
        self.sigma_squared = 1 / gamma.rvs(a=alpha, scale=1 / beta)

    def _update_gamma(self):
        """update for gamma"""
        k_inv = np.linalg.inv(self.n_groups * self.inv_lam / self.sigma_squared)
        m = k_inv @ self.inv_lam @ np.sum(self.betas, axis=1) / self.sigma_squared
        self.gamma = multivariate_normal(m, k_inv).rvs()

    def _update_lam(self):
        """update for lambda"""
        alpha = (self.n_groups + 1) / 2
        beta = 0.5 * (((self.betas - self.gamma[:, None]) ** 2).sum(axis=1) / self.sigma_squared + 1)
        lam_ii = 1 / gamma.rvs(a=alpha, scale=1 / beta)
        self.lam = np.diag(lam_ii)
        self.inv_lam = np.linalg.inv(self.lam)


    def _update_traces(self, it):
        """recording gibbs sampling traces"""
        self.traces['betas'][it, :, :] = self.betas
        self.traces['gamma'][it, :] = self.gamma
        self.traces['sigma_squared'][it] = self.sigma_squared
        self.traces['lam'][it, :] = np.diag(self.lam)

    def fit(self):
        """fit model using gibbs sampling"""
        for it in tqdm(range(self.n_iter + self.burn)):
            self._update_lam()
            self._update_gamma()
            self._update_sigma_squared()
            self._update_betas()
            if it >= self.burn:
                self._update_traces(it-self.burn)
