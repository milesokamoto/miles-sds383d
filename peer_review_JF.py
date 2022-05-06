import numpy as np
from scipy.stats import multivariate_normal, gamma, norm
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.style as style
import pymc3 as pm

plt.rcParams.update({"axes.labelsize": 16})
plt.rcParams.update({"axes.titlesize": 16})
plt.rcParams.update({"legend.fontsize": 16})
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["lines.linewidth"] = 4
style.use("ggplot")


class Initializer:
    def __init__(self, df, theta=None, mu=None, sigma_squared=None, tau_squared=None):
        self.df = df
        self.P = self._calculate_number_of_groups()
        self.n_i = self._calculate_number_of_people_per_group()
        self.mean_per_group = self._calculate_mean_per_group()
        self.theta = self._initialize_theta(theta)
        self.mu = self._initialize_mu(mu)
        self.sigma_squared = self._initialize_sigma_squared(sigma_squared)
        self.tau_squared = self._initialize_tau_squared(tau_squared)
        self.total_people = self.n_i.sum()

    def _calculate_mean_per_group(self):
        return self.df.groupby("group")["values"].mean().to_numpy().flatten()

    def _initialize_theta(self, theta):
        if theta is not None:
            return theta
        return np.zeros(self.P)

    def _initialize_mu(self, mu):
        if mu is not None:
            return mu
        return 0

    @staticmethod
    def _initialize_sigma_squared(sigma_squared):
        if sigma_squared is not None:
            return sigma_squared
        return 1

    @staticmethod
    def _initialize_tau_squared(tau_squared):
        if tau_squared is not None:
            return tau_squared
        return 1

    def _calculate_number_of_groups(self):
        P = self.df.groupby("group").ngroups
        return P

    def _calculate_number_of_people_per_group(self):
        n_i = self.df.groupby("group").size().to_numpy()
        return n_i


class GibbsSampler(Initializer):
    def __init__(
        self,
        df,
        theta=None,
        mu=None,
        sigma_squared=None,
        tau_squared=None,
        n_iter=5000,
        burn=100,
    ):
        super().__init__(
            df=df,
            theta=theta,
            mu=mu,
            sigma_squared=sigma_squared,
            tau_squared=tau_squared,
        )
        self.n_iter = n_iter
        self.burn = burn
        self.traces = {
            "sigma_squared": np.zeros(self.n_iter),
            "tau_squared": np.zeros(self.n_iter),
            "mu": np.zeros(self.n_iter),
            "theta": np.zeros((self.n_iter, self.P)),
        }

    def _update_tau_squared(self, group_means):
        self.tau_squared = 1 / gamma.rvs(
            a=(self.P + 1) / 2,
            scale=1
            / (
                1
                / (2 * self.sigma_squared)
                * (((self.theta - group_means) ** 2).sum() + 1)
            ),
            size=1,
        )

    def _update_sigma_squared(self, group_means):
        self.sigma_squared = 1 / gamma.rvs(
            a=(self.total_people + self.P) / 2,
            scale=1
            / 0.5
            * (
                (0.5 / self.tau_squared * ((self.theta - group_means) ** 2).sum())
                + (
                    (self.theta[self.df["group"].values - 1] - self.df["values"]) ** 2
                ).sum()
            ),
            size=1,
        )

    def _update_mu(self):
        self.mu = norm.rvs(
            loc=self.theta.mean(),
            scale=np.sqrt(self.sigma_squared * self.tau_squared / self.P),
        )

    def _update_theta(self, group_means):
        self.theta = multivariate_normal.rvs(
            mean=(self.mean_per_group * self.tau_squared * self.n_i + group_means)
            / (self.tau_squared * self.n_i + 1),
            cov=(
                self.sigma_squared
                * self.tau_squared
                / (self.tau_squared * self.n_i + 1)
                * np.identity(self.P)
            ),
        )

    def _update_traces(self, it):
        self.traces["theta"][it, :] = self.theta
        self.traces["mu"][it] = self.mu
        self.traces["sigma_squared"][it] = self.sigma_squared
        self.traces["tau_squared"][it] = self.tau_squared

    def _remove_burn(self):
        for trace in self.traces.keys():
            self.traces[trace] = self.traces[trace][self.burn :]

    def fit(self):
        for it in tqdm(range(self.n_iter)):
            self._update_mu()
            self._update_sigma_squared(self.mu)
            self._update_tau_squared(self.mu)
            self._update_theta(self.mu)
            self._update_traces(it)

        self._remove_burn()

    def plot_theta_histograms(self):
        traces = self.traces["theta"]
        for group in range(self.P):
            plt.hist(traces[:, group], density=True, alpha=0.5, bins=50)
        plt.title("theta posteriors")
        plt.figure()

    def plot_other_histograms(self, variable):
        trace = self.traces[variable]
        plt.hist(trace, density=True, alpha=0.5, bins=50)
        plt.title(f"{variable} posterior")
        plt.figure()

    def plot_all_posteriors(self):
        self.plot_theta_histograms()
        keys = list(self.traces.keys())
        keys.remove("theta")
        for trace in keys:
            self.plot_other_histograms(trace)


def main():
    """main method"""
    df = pd.read_csv("./data/mathtest.csv")
    df.columns = ["group", "values"]

    stdev = 5
    group_means = norm.rvs(50, stdev, size=500)
    classes = np.concatenate([[x] * 100 for x in np.arange(len(group_means))])
    samples = np.concatenate([norm.rvs(i, 5, size=100) for i in group_means])
    df = pd.DataFrame({"group": classes, "values": samples})
    gibbs = GibbsSampler(df, n_iter=2000, burn=1000)

    gibbs.fit()

    theta_means = []

    for school in tqdm(range(gibbs.P)):
        data = df[df["group"] == school + 1]
        traces = {
            "sigma": np.zeros(2000),
            "tau": np.zeros(2000),
            "mu": np.zeros(2000),
            "theta": np.zeros(2000),
        }

    with pm.Model() as model:
        sigma = pm.Normal("sigma", mu=1e-6, sigma=1e3)
        mu = pm.Normal("mu", mu=1e-6, sigma=1e3)
        tau = pm.InverseGamma("tau", 0.5, 0.5)
        theta = pm.Normal("theta", mu=mu, sd=tau * sigma)
        y = pm.Normal("y", mu=theta, sd=sigma, observed=df["values"])
        # try:
        trace = pm.sample(
            2000, tune=1000, chains=2, cores=1, return_inferencedata=False
        )
        theta_means.append(trace.get_values("theta").mean())

    with pm.Model(coords={"school": df.group.values}) as model:
        mu = pm.Normal("mu", mu=1e-6, sigma=1e3)
        sigma = pm.Normal("sigma", mu=1e-6, sigma=1e3)
        tau = pm.InverseGamma("tau", 0.5, 0.5)
        theta = pm.Normal("theta", mu=mu, sd=tau * sigma, dims="school")

        y = pm.Normal("y", mu=theta, sd=sigma, observed=df["values"].T, dims="school")

    with model:
        trace = pm.sample(2000, tune=1000, return_inferencedata=False)


if __name__ == "__main__":
    main()
