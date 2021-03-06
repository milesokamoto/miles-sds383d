"""Code to fit a hierarchical model"""
import pandas as pd
import numpy as np
from scipy.stats import gamma, norm
from tqdm import tqdm
import matplotlib.pyplot as plt


class HierarchicalModel:
    def __init__(
        self,
        y: np.array,
        cl: np.array,  # class
        theta=None,
        mu=None,  # grand mean
        omega=1.0,  # precision of data
        lam=1.0,  # precision signal:noise
        n_iter: int = 1000,
        burn_in: int = 500,
    ):
        self.y = y
        self.cl = cl
        self.n_i = self._get_n_i()
        self.theta = self._init_theta(theta)

        self.mu = self._init_mu(mu)
        self.omega = omega
        self.lam = lam
        self.P = len(self.theta)
        self.N = len(self.y)

        self.n_iter = n_iter
        self.burn_in = burn_in
        self.omegas = np.zeros(n_iter)
        self.lambdas = np.zeros(n_iter)
        self.thetas = np.array([np.zeros_like(self.theta)] * (self.n_iter))
        self.mus = np.zeros(n_iter)

    def _get_n_i(self):
        return np.array(
            [len(self.cl[self.cl == i]) for i in np.sort(np.unique(self.cl))]
        )

    def _group_means(self):
        return np.array(
            [
                np.mean(self.y[np.where(self.cl == i)])
                for i in np.sort(np.unique(self.cl))
            ]
        )

    def _init_theta(self, theta):
        """initialize theta with given value or class averages if None"""
        if theta is None:
            return self._group_means()
        return theta

    def _init_mu(self, mu):
        """initialize mu with given value or grand mean if None"""
        if mu is None:
            return np.mean(self.theta)
        return mu

    def _update_lam(self):
        self.lam = gamma.rvs(
            (self.P + 1) / 2,
            1 / 2 * (self.lam * np.sum((self.theta - self.mu) ** 2) + 1),
        )

    def _update_omega(self):
        beta = np.sum(
            np.sum(
                [
                    (self.y[j] - self.theta[self.cl[j] - 1]) ** 2
                    for j in range(len(self.y))
                ]
            )
        ) + self.lam * np.sum((self.theta - self.mu) ** 2)
        self.omega = gamma.rvs((self.N + self.P) / 2, beta)

    def _update_mu(self):
        print(np.mean(self.theta))
        print(1 / (self.P * self.omega * self.lam))
        self.mu = norm.rvs(np.mean(self.theta), 1 / (self.P * self.omega * self.lam))

    def _update_theta(self):
        y_bar = np.array(
            [
                np.mean(self.y[np.where(self.cl == i)])
                for i in np.sort(np.unique(self.cl))
            ]
        )
        # print(np.array([(i * self.lam) + self.lam * self.omega for i in self.n_i]))
        # print(
        #     (self.n_i * self.lam * y_bar + self.lam * self.omega * self.mu)
        #     / (self.n_i * self.lam + self.lam * self.omega)
        # )

        # self.theta = norm.rvs(
        #     (self.n_i * self.lam * y_bar + self.lam * self.omega * self.mu)
        #     / (self.n_i * self.lam + self.lam * self.omega),
        #     (self.n_i * self.lam + self.lam * self.omega),
        # )
        mean = (y_bar * self.n_i + self.lam * self.mu) / (self.n_i / self.lam)
        var = 1 / (self.omega * (self.n_i + self.lam))
        norm.rvs(mean, var)

    def _store_iter(self, iter):
        self.omegas[iter] = self.omega
        self.lambdas[iter] = self.lam
        self.mus[iter] = self.mu
        self.thetas[iter] = self.theta

    def fit(self):
        iter = 0
        for _ in tqdm(range(self.n_iter + self.burn_in)):

            # gibbs sampling
            self._update_mu()
            print(f"mu: {self.mu}")
            self._update_omega()
            print(f"omega: {self.omega}")
            self._update_lam()
            print(f"lambda: {self.lam}")
            self._update_theta()
            print(f"theta: {self.theta}")

            # store each step
            if iter >= self.burn_in:
                self._store_iter(iter - self.burn_in)

            iter += 1


def main():
    """main method"""
    data = pd.read_csv("./data/mathtest.csv")
    cl = np.array(data[["school"]])
    y = np.array(data[["mathscore"]])

    # PLOTS FOR PART B
    averages = data.groupby("school").agg({"mathscore": ["size", "mean"]})
    plt.plot(
        np.array(averages["mathscore"]["size"]),
        np.array(averages["mathscore"]["mean"]),
        "bo",
    )
    plt.title("School Average vs Number of Samples")
    plt.xlabel("n")
    plt.ylabel("mean_score")
    plt.savefig("samples_average.png")
    plt.clf()
    score_std = np.std(np.array(averages["mathscore"]["mean"]))
    plt.plot(
        np.array(averages["mathscore"]["size"]),
        abs(
            np.array(averages["mathscore"]["mean"])
            - np.mean(np.array(averages["mathscore"]["mean"]))
        )
        / score_std,
        "bo",
    )
    plt.title("School average std from mean vs Number of Samples")
    plt.xlabel("n")
    plt.ylabel("mean_score_std_from_mean")
    plt.savefig("samples_std.png")
    plt.clf()

    # Fit Model

    hm = HierarchicalModel(y, cl, n_iter=0, burn_in=1)
    hm.fit()

    plt.clf()
    plt.plot(np.sort(np.unique(hm.cl)), hm.thetas.mean(axis=0), "bo")
    plt.title("theta_posterior_means")
    plt.xlabel("school")
    plt.ylabel("theta_est")
    plt.savefig("posterior_thetas2.png")

    kappas = hm.lam / (hm.n_i + hm.lam)
