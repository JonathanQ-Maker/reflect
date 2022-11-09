from reflect.profiler.gradient import num_grad
import numpy as np
import unittest

np.random.seed(0)


class GradientProfilerTest(unittest.TestCase):

    def test_num_grad(self):
        def squared(X):
            return X ** 2

        shape = (3, 5)
        X = np.random.uniform(size=shape)
        grad = num_grad(squared, X)
        real_grad = 2 * X
        self.assertTrue(np.all(grad != real_grad), "grad == real_grad strictly")
        self.assertTrue(grad is not real_grad, "grad and real_grad is the same instance")
        print(f"grad:\n{grad}\n\nreal_grad:\n{real_grad}")
        self.assertTrue(np.allclose(grad, real_grad, atol = 1e-4), "num_grad and real gradient differ")
        print("num_grad_test() passed\n")

    def test_matmul_num_grad(self):
        matrix_shape = (3, 5)
        A = np.random.uniform(size=matrix_shape)

        def func(x):
            return np.dot(A, x)

        x = np.random.uniform(size=matrix_shape[1])
        grad = num_grad(func, x)
        real_grad = np.dot(A.T, np.ones(matrix_shape[0]))
        self.assertTrue(np.all(grad != real_grad), "grad == real_grad strictly")
        self.assertTrue(grad is not real_grad, "grad and real_grad is the same instance")
        print(f"grad:\n{grad}\n\nreal_grad:\n{real_grad}")
        self.assertTrue(np.allclose(grad, real_grad, atol = 1e-4), "num_grad and real gradient differ")
        print("matmul_num_grad_test() passed\n")


if (__name__ == "__main__"):
    unittest.main()