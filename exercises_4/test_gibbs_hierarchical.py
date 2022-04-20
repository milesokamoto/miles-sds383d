import gibbs_hierarchical as test
from pytest import fixture
import numpy as np


@fixture
def init_model():
    y = np.array([1, 1, 1, 2, 2, 5, 5, 5, 6])
    cl = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
    return test.HierarchicalModel(y, cl)


def test_get_group_means(init_model):
    group_means = init_model._group_means()
    assert np.all(group_means == np.array([1.4, 5.25]))
