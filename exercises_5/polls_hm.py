"""hierarchical model using data augmentation based on Albert + Chib (1993)"""
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, truncnorm, gamma
from tqdm import tqdm
import matplotlib.pyplot as plt

ENCODE_EDU = {
    "NoHS": 0,
    "HS": 1,
    "SomeColl": 2,
    "Bacc": 3
}

ENCODE_AGE = {
    "18to29": 0,
    "30to44": 1,
    "45to64": 2,
    "65plus": 3
}

class ProbitRegression:
    def __init__(self, X, y, groups, n_iter:int = 1000, burn:int=500):
        self.n_iter = n_iter
        self.burn = burn

        self.n_betas = X.shape[1]
        self.groups_idx = pd.get_dummies(groups)
        self.n_groups = self.groups_idx.shape[1]
        self.X_sparse = self.get_X_sparse(X)
        self.y = y

        self.B_0 = 1
        self.betas = np.ones((self.n_betas * self.n_groups))
        self.b_0 = np.ones(self.n_betas * self.n_groups)
        self.z = np.ones(len(y)).T

        self.traces = {'betas': np.zeros((self.n_iter, self.n_betas, self.n_groups)),
            'B_0': np.zeros(self.n_iter),
            'b_0': np.zeros((self.n_iter, self.n_betas))}

    def get_X_sparse(self, X):
        return np.array([
        np.concatenate(
            [
                np.zeros(self.n_betas*np.argmax(self.groups_idx.iloc[row])), X[row], np.zeros(self.n_betas* (self.groups_idx.shape[1]-1-np.argmax(self.groups_idx.iloc[row])))
            ]
        ) for row in range(len(X))])

    def update_betas(self):
        B = np.linalg.inv(1/self.b_0*np.eye(self.n_betas*self.n_groups) + self.X_sparse.T @ self.X_sparse)
        b = B @ (self.b_0 / self.b_0 + self.X_sparse.T @ self.z)
        self.betas = multivariate_normal.rvs(b, B)

    # update B_0
    def update_B_0(self):
        alpha = (self.n_groups + 1) / 2
        beta = 0.5 * (((self.betas - self.b_0) ** 2).sum() + 1)
        self.B_0 = 1 / gamma.rvs(a=alpha, scale=1 / beta)

    # update b_0
    def update_b_0(self):
        cov = self.b_0 / self.n_groups * np.eye(self.n_betas*self.n_groups)
        mean = cov / self.b_0 @ self.betas
        self.b_0 = multivariate_normal.rvs(mean, cov)

    # update z
    def update_z(self):
        self.z = truncnorm.rvs([-np.inf if x == 0 else 0 for x in self.y], [0 if x == 0 else np.inf for x in self.y], loc=0, scale=1)

    def update_traces(self, it):
        self.traces['betas'][it, :, :] = self.betas.reshape(self.n_betas, int(len(self.betas)/self.n_betas))
        self.traces['b_0'][it, :] = self.b_0
        self.traces['B_0'][it] = self.B_0

    def fit(self):
        for it in tqdm(range(self.n_iter + self.burn)):
            self.update_B_0()
            self.update_betas()
            self.update_b_0()
            self.update_z()
            if it >= self.burn:
                self.update_traces(it-self.burn)
