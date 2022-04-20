"""Code to fit a hierarchical model"""
import pandas as pd
import numpy as np
from scipy.stats import gamma, norm, ttest_ind
from tqdm import tqdm
import matplotlib.pyplot as plt


class HierarchicalModel:
    def __init__(
        self,
        y: np.array,
        group: np.array,  # person
        xi: np.array,
        theta=None,
        mu=None,
        beta=1.0,
        omega=1.0,
        lam=1.0,
        n_iter: int = 1000,
        burn_in: int = 500,
    ):
        self.y = y
        self.xi = xi
        self.group = group
        self.n_i = self._get_n_i()
        self.theta = self._init_theta(theta)
        self.beta = beta

        self.mu = self._init_mu(mu)
        self.omega = omega
        self.lam = lam
        self.P = len(self.theta)
        self.N = len(self.y)

        self.n_iter = n_iter
        self.burn_in = burn_in
        self.omegas = np.zeros(n_iter)
        self.lambdas = np.zeros(n_iter)
        self.betas = np.zeros(n_iter)
        self.thetas = np.array([np.zeros_like(self.theta)] * (self.n_iter))
        self.mus = np.zeros(n_iter)

    def _get_n_i(self):
        return np.array(
            [len(self.group[self.group == i]) for i in np.sort(np.unique(self.group))]
        )

    def _init_theta(self, theta):
        """initialize theta with given value or class averages if None"""
        if theta is None:
            return np.array(
                [
                    np.mean(self.y[np.where(self.group == i)])
                    for i in np.sort(np.unique(self.group))
                ]
            )
        return theta

    def _init_mu(self, mu):
        """initialize mu with given value or grand mean if None"""
        if mu is None:
            return np.mean(self.theta)
        return mu

    def _update_lam(self):
        self.lam = gamma.rvs(
            (self.P + 1) / 2,
            self.lam
            / 2
            * (
                np.sum((self.theta - (self.mu + self.betas * self.xi)) ** 2)
                + 1 / self.lam
            ),
        )

    def _update_omega(self):
        print(self.mu + self.betas * self.xi)
        print(np.sum((self.theta - (self.mu + self.betas * self.xi)) ** 2))
        print(self.lam * np.sum((self.theta - (self.mu + self.betas * self.xi)) ** 2))
        print(
            np.sum(
                np.sum(
                    [
                        (self.y[j] - self.theta[self.group[j] - 1]) ** 2
                        for j in range(len(self.y))
                    ]
                )
            )
            + self.lam * np.sum((self.theta - (self.mu + self.betas * self.xi)) ** 2)
        )
        beta = np.sum(
            np.sum(
                [
                    (self.y[j] - self.theta[self.group[j] - 1]) ** 2
                    for j in range(len(self.y))
                ]
            )
        ) + self.lam * np.sum((self.theta - (self.mu + self.betas * self.xi)) ** 2)
        self.omega = gamma.rvs((self.N + self.P) / 2, beta)

    def _update_mu(self):
        self.mu = norm.rvs(
            np.mean(self.theta - self.beta * self.xi),
            1 / (self.P * self.omega * self.lam),
        )

    def _update_theta(self):
        y_bar = np.array(
            [
                np.mean(self.y[np.where(self.group == i)])
                for i in np.sort(np.unique(self.group))
            ]
        )
        mean = (y_bar * self.n_i + self.lam * self.mu) / (self.n_i / self.lam)
        var = 1 / (self.omega * (self.n_i + self.lam))
        norm.rvs(mean, var)

    def _update_beta(self):
        self.beta = norm.rvs(
            (np.mean(self.theta * self.xi) - self.mu * np.mean(self.xi))
            / np.mean(self.xi ** 2),
            1 / (self.omega * self.lam * self.P * np.mean(self.xi ** 2)),
        )

    def _store_iter(self, iter):
        self.omegas[iter] = self.omega
        self.lambdas[iter] = self.lam
        self.mus[iter] = self.mu
        self.thetas[iter] = self.theta
        self.betas[iter] = self.beta

    def fit(self):
        iter = 0
        for _ in tqdm(range(self.n_iter + self.burn_in)):
            # gibbs sampling
            self._update_mu()
            self._update_omega()
            self._update_lam()
            self._update_theta()
            self._update_beta()

            # store each step
            if iter >= self.burn_in:
                self._store_iter(iter - self.burn_in)

            iter += 1


def main():
    """main method"""
    data = pd.read_csv("./data/bloodpressure.csv")
    treatment = np.array(data.groupby("subject").agg({"treatment": "mean"})) - 1
    subject = np.array(data[["subject"]])
    y = np.array(data[["systolic"]])

    ttest_ind(
        data[data["treatment"] == 1]["systolic"],
        data[data["treatment"] == 2]["systolic"],
    )
    pooled = data.groupby(["subject", "treatment"], as_index=False).agg(
        {"systolic": "mean"}
    )
    ttest_ind(
        pooled[pooled["treatment"] == 1]["systolic"],
        pooled[pooled["treatment"] == 2]["systolic"],
    )

    # PLOTS FOR PART B
    averages = data.groupby(["subject", "treatment"], as_index=False).agg(
        {"systolic": ["size", "mean"]}
    )
    plt.plot(
        np.array(averages["treatment"].astype(str)),
        np.array(averages["systolic"]["mean"]),
        "bo",
    )
    plt.title("Systolic Average vs Treatment")
    plt.xlabel("treatment")
    plt.ylabel("mean_score")
    plt.savefig("systolic_averages.png")
    plt.clf()

    # Fit Model

    hm = HierarchicalModel(y, subject, treatment, n_iter=4000, burn_in=1000)
    hm.fit()

    plt.clf()
    plt.scatter(
        np.sort(np.unique(hm.group)),
        hm.thetas.mean(axis=0),
        marker="o",
        c=treatment,
        cmap=plt.get_cmap("Dark2"),
    )
    plt.title("theta_posterior_means")
    plt.xlabel("subject")
    plt.ylabel("theta_est")
    plt.savefig("posterior_thetas_bp.png")

    kappas = hm.lam / (hm.n_i + hm.lam)


# df.groupby("group")["values"].mean().to_numpy().flatten()
# [np.mean(self.y[np.where(self.group == i)]) for i in np.sort(np.unique(self.group))]


# import time

# iter = 1000
# start = time.perf_counter()
# for _ in np.range(iter):
#     df.groupby("group")["values"].mean().to_numpy().flatten()
# end = time.perf_counter()

# print((end - start) / iter)

# df.groupby("group")["values"].mean().to_numpy().flatten()
