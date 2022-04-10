"""hierarchical model for cheese"""
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm, gamma
import scipy as sp
from tqdm import tqdm
import matplotlib.pyplot as plt

class ElasticityModel:
    def __init__(self, X, y, groups, n_iter:int = 1000, burn:int=500):
        self.n_iter = n_iter
        self.burn = burn
        self.n_betas = X.shape[1] + 1
        self.groups_idx = pd.get_dummies(groups)
        self.n_groups = self.groups_idx.shape[1]
        self.X_sparse = self.get_X_sparse(X)
        self.y = y
        self.n_iter = n_iter
        self.traces = {'betas': np.zeros([self.n_iter, self.n_betas, self.n_groups]),
                        'omega': np.zeros((self.n_iter, self.n_groups)), 'lam': np.zeros(self.n_iter),
                        'mu': np.zeros((self.n_iter, self.n_betas))}

        self.omega = np.diag(np.ones(self.n_betas*self.n_groups))
        self.lam = np.diag(np.ones(self.n_betas*self.n_groups))
        self.betas = np.ones((self.n_betas, self.n_groups))
        self.mu = np.ones(self.n_betas * self.n_groups)


    def update_betas(self):
        lam_star_inv = np.linalg.inv(self.omega @ (self.X_sparse.T @ self.X_sparse + self.lam))
        mu_star = lam_star_inv @ (self.omega @ (self.X_sparse.T @ self.y + self.lam @ self.mu))
        self.betas = multivariate_normal.rvs(mu_star, lam_star_inv)

    def update_omega(self):
        N_i = self.groups_idx.sum(axis=0)
        alpha = (N_i + self.n_groups)/2
        beta = .5 * ((self.y-self.X_sparse @ self.betas).T@(self.y-self.X_sparse @ self.betas) + ((self.betas-self.mu).T@self.lam@(self.betas-self.mu)))
        self.omega = np.diag(np.concatenate([[x]*self.n_betas for x in gamma.rvs(list(alpha), beta)]))

    def update_lam(self):
        alpha = (1+self.n_groups)/2
        beta = .5 * ((self.betas-self.mu).T@self.omega@(self.betas-self.mu) + 1)
        self.lam = np.eye(self.n_betas*self.n_groups) * gamma.rvs(alpha,beta)

    def update_mu(self):
        sigma_star = np.linalg.inv(np.eye(self.n_betas)*(self.n_groups * sum(np.diagonal(self.omega)[::self.n_betas]) * self.lam[0,0]))
        mu_star = np.sum(self.betas.reshape(self.n_betas,self.n_groups),axis=1)/self.n_groups
        self.mu = np.repeat(multivariate_normal.rvs(mu_star, sigma_star), self.n_groups)

    def fit(self):
        for it in tqdm(range(self.n_iter + self.burn)):
            self.update_betas()
            self.update_omega()
            self.update_lam()
            self.update_mu()
            if it >= self.burn:
                self.update_traces(it-self.burn)

    def update_traces(self, it):
        self.traces['betas'][it, :, :] = self.betas.reshape(self.n_betas, self.n_groups)
        self.traces['omega'][it, :] = np.diagonal(self.omega)[::self.n_betas]
        self.traces['lam'][it] = self.lam[0,0]
        self.traces['mu'][it, :] = self.mu[:self.n_betas]

    def get_X_sparse(self, X):
        return np.array([
        np.concatenate(
            [
                np.zeros(self.n_betas*np.argmax(self.groups_idx.iloc[row])), [1], X[row], np.zeros(self.n_betas* (self.groups_idx.shape[1]-1-np.argmax(self.groups_idx.iloc[row])))
            ]
        ) for row in range(len(X))])


def main():
    """main method"""
    cheese = pd.read_csv('./experimental/mokamoto/cheese.csv')
    # cheese = cheese.iloc[100:120].reset_index(drop=True)
    # cheese.store = pd.Categorical(cheese.store)
    # cheese["store_code"] = cheese.store.cat.codes
    stores = cheese["store"]
    cheese = cheese.sort_values("store")
    X = np.array([[1, np.log(cheese.loc[i, "price"]), cheese.loc[i, "disp"], np.log(cheese.loc[i, "price"]) * cheese.loc[i, "disp"]] for i in range(len(cheese))])
    y = np.log(cheese["vol"])

    em = ElasticityModel(X, y, stores, burn=500)
    em.fit()

    plt.close('all')
    plt.clf()
    for j in range(em.n_betas):
        for i in range(em.n_groups):
            plt.hist(em.traces['betas'][:, j, i], density=True, alpha=.5, bins=50)
        plt.title(f'beta {j + 1}', fontsize=16)
        plt.xlim([min(em.traces['betas'][:, j, :].flatten()), max(em.traces['betas'][:, j, :].flatten())])
        plt.ylim([0, 100])
        plt.savefig(f'beta{j + 1}.png', dpi=600)
        plt.figure()


    for i in range(em.n_betas):
        for j in range(em.n_groups):
            print(np.mean(em.traces['betas'][:,i,j]))


if __name__ == "__main__":
    main()
