import numpy as np

def num_grad(func, X, delta=1e-5):
    """
    Calculates the numerical gradient 

    ∇f(X) = ( f(X + delta) - f(X) )/delta

    returns: gradient
    """

    shape = X.shape
    grad = np.zeros(shape)
    f_X = func(X)
    for i in range(X.size):
        X.flat[i] += delta
        f_X_delta = func(X)
        grad.flat[i] = np.sum(f_X_delta - f_X) / delta
        X.flat[i] -= delta
    return grad