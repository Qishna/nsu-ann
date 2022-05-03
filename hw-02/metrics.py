import numpy as np


def MSE(true, predict):
    return np.sum(true - predict)**2


def R2(true, predict):
    my_fitting = np.polyfit(true, predict, 1, full=True)

    SSE = my_fitting[1][0]
    SST = ((predict - predict.mean()) ** 2).sum()

    R2 = 1 - SSE/SST
    return R2
