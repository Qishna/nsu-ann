import numpy as np


def MSE(true, predict):
    return np.sum(true - predict)**2
