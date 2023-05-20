import numpy as np
from reflect.layers import Dense, Relu
from reflect.regularizers import L1, L2, L1L2
from reflect.profiler import num_grad
import unittest

class ReluTest(unittest.TestCase):

    def test_relu_num_grad(self):
        input_size = 5
        batch_size = 3

        l = Relu()
        l.compile(input_size, batch_size)


        x = np.random.uniform(size=l.input_shape)
        target = np.random.uniform(size=l.output_shape)

        def func(X):
            return np.sum((target - l.forward(X))**2) / 2

        grad = num_grad(func, x)

        residual = target - l.forward(x)
        real_grad = -l.backprop(residual)

        self.assertTrue(np.all(grad != real_grad), "grad == real_grad strictly")
        self.assertTrue(grad is not real_grad, "grad and real_grad is the same instance")
        print(f"grad:\n{grad}\n\nreal_grad:\n{real_grad}")
        self.assertTrue(np.allclose(grad, real_grad, atol = 1e-4), "num_grad and real gradient differ")
        print("relu_num_grad() passed")

    def test_dense_relu_multi_num_grad_bias(self):
        input_size = 5
        output_size_1 = 7
        batch_size = 2


        l1 = Dense(output_size_1, "xavier")
        l2 = Relu()
        l1.compile(input_size, batch_size, gen_param=True)
        l2.compile(output_size_1, batch_size)

        x = np.random.uniform(size=l1.input_shape)
        target = np.random.uniform(size=l2.output_shape)
        bias_original = l1.param.bias
        bias = np.copy(l1.param.bias)

        def forward(b):
            np.copyto(l1.param.bias, b)
            return np.sum((target - l2.forward(l1.forward(x)))**2) / 2

        grad = num_grad(forward, bias)

        np.copyto(l1.param.bias, bias_original)
        residual = target - l2.forward(l1.forward(x))
        l1.backprop(l2.backprop(residual))
        real_grad = -l1.dldb

        self.assertTrue(grad is not real_grad, "grad and real_grad is the same instance")
        print(f"grad:\n{grad}\n\nreal_grad:\n{real_grad}")
        self.assertTrue(np.allclose(grad, real_grad, atol = 1e-4), "num_grad and real gradient differ")
        print("dense_relu_multi_num_grad_bias() passed\n")

    def test_multi_dense_relu_num_grad_weight(self):
        np.set_printoptions(precision=8)
        input_size = 5
        output_size_1 = 7
        output_size_2 = 3
        batch_size = 2


        l1 = Dense(output_size_1, "xavier")
        l2 = Relu()
        l1.compile(input_size, batch_size, gen_param=True)
        l2.compile(output_size_1, batch_size)

        x = np.random.uniform(size=l1.input_shape)
        target = np.random.uniform(size=l2.output_shape)
        weight_original = l1.param.weight
        weight = np.copy(l1.param.weight)

        def forward(W):
            np.copyto(l1.param.weight, W)
            return np.sum((target - l2.forward(l1.forward(x)))**2) / 2

        grad = num_grad(forward, weight)

        np.copyto(l1.param.weight, weight_original)
        residual = target - l2.forward(l1.forward(x))
        l1.backprop(l2.backprop(residual))
        real_grad = -l1.dldw

        self.assertTrue(grad is not real_grad, "grad and real_grad is the same instance")
        print(f"grad:\n{grad}\n\nreal_grad:\n{real_grad}")
        self.assertTrue(np.allclose(grad, real_grad, atol = 1e-4), "num_grad and real gradient differ")
        print("multi_dense_relu_num_grad_weight() passed\n")

    def test_relu_shape(self):
        input_size = 2

        l = Relu()
        l.compile(input_size, 3)

        self.assertTrue(l.output_shape == l.input_shape, "x, output shape differ")
        self.assertTrue(l.output_shape == (3, 2), "Expected output shape differ")

        l.compile((4, 5), 3)

        self.assertTrue(l.output_shape == l.input_shape, "x, output shape differ")
        self.assertTrue(l.output_shape == (3, 4, 5), "Expected output shape differ")
        print("relu_shape() passed\n")

    def test_relu_num_grad_cache(self):
        input_size = 5
        batch_size = 3

        l = Relu()
        l.compile(input_size, batch_size)


        x = np.random.uniform(size=l.input_shape)
        x2 = np.random.uniform(size=l.input_shape)
        target = np.random.uniform(size=l.output_shape)

        def func(X):
            return np.sum((target - l.forward(X))**2) / 2

        grad = num_grad(func, x)

        cache = l.create_cache()

        residual = target - l.forward(x)
        l.forward(x2, out_cache=cache)
        real_grad = -l.backprop(residual)

        self.assertTrue(np.all(grad != real_grad), "grad == real_grad strictly")
        self.assertTrue(grad is not real_grad, "grad and real_grad is the same instance")
        print(f"grad:\n{grad}\n\nreal_grad:\n{real_grad}")
        self.assertTrue(np.allclose(grad, real_grad, atol = 1e-4), "num_grad and real gradient differ")
        print("relu_num_grad() passed")


if __name__ == "__main__":
    unittest.main()
