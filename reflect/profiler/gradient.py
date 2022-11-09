import numpy as np

def num_grad(func, X, dout=1, delta=1e-5):
    """
    Calculates the numerical gradient 

    ∇f(X) = ( f(X + delta) - f(X) ) dout / delta

    returns: gradient
    """
    grad = np.zeros(X.shape)

    f_X = func(X).copy()
    buffer = np.zeros(shape=f_X.shape)
    X = np.copy(X)
    for i in range(X.size):
        X.flat[i] += delta
        f_X_delta = func(X)
        np.subtract(f_X_delta, f_X, out=buffer)
        np.multiply(buffer, dout, out=buffer)
        grad.flat[i] = np.sum(buffer) / delta
        X.flat[i] -= delta
    return grad

def check_grad(func, x, grad, dout=1, delta=1e-5):
    """
    Checks grad with numeric grad.

    return true if grad is similar with numeric grad

    returns: bool, msg
    """
    n_grad = num_grad(func, x, dout, delta)
    print(f"Hashes\n  numeric grad: {np.sum(np.abs(n_grad))}\n  grad:         {np.sum(np.abs(grad))}")

    if np.all(n_grad == grad):
        return False, "n_grad == grad strictly"
    if n_grad is  grad:
        return False, "n_grad and grad is the same instance"
    if not np.allclose(n_grad, grad, atol = delta * 10):
        return False, "n_grad and grad gradient differ"
    return True, "numeric grad and grad are similar"