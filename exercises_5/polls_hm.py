"""hierarchical model using data augmentation based on Albert + Chib (1993)"""
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, truncnorm, gamma
from tqdm import tqdm
import matplotlib.pyplot as plt

n_iter = 1000
burn = 500

data = pd.read_csv("experimental/mokamoto/polls.csv")
data_cleaned = data[data["bush"].notna()]
states = data_cleaned[["state"]]
groups = data_cleaned.groupby('state')
states = list(groups.groups.keys())
X = []
y = []
for state in states:
    df_group = groups.get_group(state)
    X.append(df_group[["female", "black", "weight"]].to_numpy())
    y.append(df_group['bush'].to_numpy())
# X = data[["state", "female", "black", "weight", "edu", "age"]] # need to categoricalize edu, age
# X = data_cleaned[["female", "black"]]
# y = data_cleaned[["bush"]]

n_betas = X[0].shape[1]
groups_idx = pd.get_dummies(states[["state"]])
n_groups = groups_idx.shape[1]

B_0 = 1
betas = np.ones((n_betas, n_groups))
b_0 = np.ones(n_betas)
z = [np.ones(len(x)) for x in y]

# update_betas
for group in range(n_groups):
    B = np.lbinalg.inv(1/B_0*np.eye(n_betas) + X[group].T @ X[group])
    b = B @ (b_0 / B_0 + X[group].T @ z[group])
    betas[:, group] = multivariate_normal.rvs(mean=b, cov=B)

# update B_0
alpha = (n_groups + 1) / 2
beta = 0.5 * (((betas - b_0[:, None]) ** 2).sum() + 1)
B_0 = 1 / gamma.rvs(a=alpha, scale=1 / beta)

# update b_0
cov = B_0 / n_groups * np.eye(n_betas)
mean = cov / B_0 @ betas.sum(axis=1)
b_0 = multivariate_normal.rvs(mean=mean, cov=cov)

# update z
z[group] = truncnorm.rvs([-np.inf if x == 0 else 0 for x in y[group]], [0 if x == 0 else np.inf for x in y[group]], loc=0, scale=1)
