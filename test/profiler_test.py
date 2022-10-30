from reflect.profiler.numerical_gradient import num_grad
import numpy as np

np.random.seed(0)



def num_grad_test():
    def squared(X):
        return X ** 2

    shape = (3, 5)
    X = np.random.uniform(size=shape)
    grad = num_grad(squared, X)
    real_grad = 2 * X
    assert np.all(grad != real_grad), "grad == real_grad strictly"
    assert grad is not real_grad, "grad and real_grad is the same instance"
    print(f"grad:\n{grad}\n\nreal_grad:\n{real_grad}")
    assert np.allclose(grad, real_grad, atol = 1e-4), "num_grad and real gradient differ"
    print("num_grad_test() passed\n")

def matmul_num_grad_test():
    matrix_shape = (3, 5)
    A = np.random.uniform(size=matrix_shape)

    def func(x):
        return np.dot(A, x)

    x = np.random.uniform(size=matrix_shape[1])
    grad = num_grad(func, x)
    real_grad = np.dot(A.T, np.ones(matrix_shape[0]))
    assert np.all(grad != real_grad), "grad == real_grad strictly"
    assert grad is not real_grad, "grad and real_grad is the same instance"
    print(f"grad:\n{grad}\n\nreal_grad:\n{real_grad}")
    assert np.allclose(grad, real_grad, atol = 1e-4), "num_grad and real gradient differ"
    print("matmul_num_grad_test() passed\n")


if __name__ == "__main__":
    num_grad_test()
    matmul_num_grad_test()