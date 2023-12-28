import numpy as np

def rmse(y, hx):
    return np.sqrt(np.sum(np.square(y - hx)) / y.shape[0])

def mse(y, hx):
    return np.sum(np.square(y - hx)) / y.shape[0]

def mae(y, hx):
    return np.sum(np.abs(y - hx)) / y.shape[0]

def mape(y, hx):
    return np.mean(np.abs((y - hx) / y)) * 100

