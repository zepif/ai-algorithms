import numpy as np

def rmse(y, hx):
    return np.sqrt(np.sum(np.square(y - hx)) / y.shape[0])
