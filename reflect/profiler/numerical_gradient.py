import numpy as np

def num_grad(func, X, delta=1e-5):
    shape = X.shape
    grad = np.zeros(shape)
    f_X = func(X)
    for i in range(X.size):
        X.flat[i] += delta
        f_X_delta = func(X)
        grad.flat[i] = np.sum(f_X_delta - f_X) / delta
        X.flat[i] -= delta
    return grad