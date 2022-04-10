"""hierarchical model using data augmentation based on Albert + Chib (1993)"""
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, truncnorm, invgamma, norm
from tqdm import tqdm
import matplotlib.pyplot as plt

class ProbitRegression:
    def __init__(self, X, y, groups, n_iter:int = 1000, burn:int=500):
        self.n_iter = n_iter
        self.burn = burn

        self.n_betas = X.shape[1] + 1
        self.groups_idx = pd.get_dummies(groups)
        self.n_groups = self.groups_idx.shape[1]
        self.X_sparse = self.get_X_sparse(X)
        self.y = y

        self.B_0 = 1
        self.betas = np.ones((self.n_betas * self.n_groups))
        self.b_0 = np.ones(self.n_betas * self.n_groups)
        self.z = np.concatenate([np.random.choice([0, 1], size=len(x)) for x in self.y])

        self.traces = {'betas': np.zeros((self.n_iter, self.n_betas, self.n_groups)),
            'B_0': np.zeros(self.n_iter),
            'b_0': np.zeros((self.n_iter, self.n_betas, self.n_groups))
        }

    def get_X_sparse(self, X):
        return np.array([
        np.concatenate(
            [
                np.zeros(self.n_betas*np.argmax(self.groups_idx.iloc[row])), [1], X[row], np.zeros(self.n_betas* (self.groups_idx.shape[1]-1-np.argmax(self.groups_idx.iloc[row])))
            ]
        ) for row in range(len(X))])

    def update_betas(self):
        B = np.linalg.inv(1/self.B_0*np.eye(self.n_betas*self.n_groups) + self.X_sparse.T @ self.X_sparse)
        b = B @ (self.b_0 / self.B_0 + self.X_sparse.T @ self.z)
        self.betas = multivariate_normal.rvs(b, B)

    # update B_0
    def update_B_0(self):
        alpha = (self.n_groups + 1) / 2
        beta = 0.5 * (((self.betas - self.b_0) ** 2).sum() + 1)
        self.B_0 = invgamma.rvs(a=alpha, scale=1 / beta)

    # update b_0
    def update_b_0(self):
        cov = self.B_0 / self.n_groups * np.eye(self.n_betas*self.n_groups)
        mean = cov / self.B_0 @ self.betas
        self.b_0 = multivariate_normal.rvs(mean, cov)

    # update z
    def update_z(self):
        self.z = truncnorm.rvs([-np.inf if x == 0 else 0 for x in self.y], [0 if x == 0 else np.inf for x in self.y], loc=self.X_sparse@self.betas, scale=1) #location should be xTb?


    def update_traces(self, it):
        x = self.betas.reshape(self.n_betas, int(len(self.betas)/self.n_betas))
        self.traces['b_0'][it, :, :] = self.b_0.reshape(self.n_betas, int(len(self.betas)/self.n_betas))
        self.traces['B_0'][it] = self.B_0

    def fit(self):
        for it in tqdm(range(self.n_iter + self.burn)):
            self.update_B_0()
            self.update_betas()
            self.update_b_0()
            self.update_z()
            if it >= self.burn:
                self.update_traces(it-self.burn)


def main():
    """main method"""
    data = pd.read_csv("experimental/mokamoto/polls.csv")
    data_cleaned = data[data["bush"].notna()].sort_values("state")
    data_encoded = data_cleaned.join(pd.get_dummies(data_cleaned["edu"]))
    data_encoded = data_encoded.join(pd.get_dummies(data_cleaned["age"]))
    data_cols = ["female", "black", "Bacc", "HS", "NoHS", "SomeColl", "18to29", "30to44", "45to64", "65plus"]
    groups = data_encoded[["state"]]
    X = data_encoded[data_cols].to_numpy()
    y = data_encoded[["bush"]].to_numpy()
    pr = ProbitRegression(X, y, groups)
    pr.fit()

    labels = ["intercept"]+data_cols

    plt.close('all')
    plt.clf()
    for j in range(pr.n_betas):
        for i in range(pr.n_groups):
            plt.hist(pr.traces['betas'][:, j, i], density=True, alpha=.5, bins=50)
        plt.title(f'beta {j + 1}', fontsize=16)
        plt.xlim([min(pr.traces['betas'][:, j, :].flatten()), max(pr.traces['betas'][:, j, :].flatten())])
        plt.ylim([0, 100])
        plt.savefig(f'beta{j + 1}.png', dpi=600)
        plt.figure()


    for i in range(pr.n_betas):
        for j in range(pr.n_groups):
            print(np.mean(pr.traces['betas'][:,i,j]))

    plt.clf()
    plt.rcParams.update({'font.size': 3})
    beta_means = np.mean(pr.traces["betas"], axis=0)
    fig, axs = plt.subplots(len(labels),1)
    for i in range(pr.n_betas):
        axs[i].axhline(y=0, color='r', linestyle='-', linewidth=.5, alpha=.5)
        axs[i].scatter(groups["state"].unique(), beta_means[i], s=2)
        axs[i].set_ylim([-2,2])
        axs[i].set_ylabel(f"beta {labels[i]}")
    plt.savefig("betas.png",  dpi=1000)

    plt.clf()
    plt.scatter(labels, np.mean)

if __name__ == "__main__":
    main()

