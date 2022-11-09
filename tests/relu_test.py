import numpy as np
from reflect.layers import Dense, Relu
from reflect.regularizers import L1, L2, L1L2
from reflect.profiler import num_grad
import unittest

class ReluTest(unittest.TestCase):

    def test_relu_num_grad(self):
        input_size = 5
        batch_size = 3

        l = Relu(input_size, batch_size)
        l.compile()


        input = np.random.uniform(size=l.input_shape)
        target = np.random.uniform(size=l.output_shape)

        def func(X):
            return np.sum((target - l.forward(X))**2) / 2

        grad = num_grad(func, input)

        residual = target - l.forward(input)
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


        l1 = Dense(input_size, output_size_1, batch_size, "xavier")
        l2 = Relu(output_size_1, batch_size)
        l1.compile(gen_param=True)
        l2.compile()

        input = np.random.uniform(size=l1.input_shape)
        target = np.random.uniform(size=l2.output_shape)
        bias_original = l1.param.bias
        bias = np.copy(l1.param.bias)

        def forward(b):
            l1.param.bias = b
            return np.sum((target - l2.forward(l1.forward(input)))**2) / 2

        grad = num_grad(forward, bias)

        l1.param.bias = bias_original
        residual = target - l2.forward(l1.forward(input))
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


        l1 = Dense(input_size, output_size_1, batch_size, "xavier")
        l2 = Relu(output_size_1, batch_size)
        l1.compile(gen_param=True)
        l2.compile()

        input = np.random.uniform(size=l1.input_shape)
        target = np.random.uniform(size=l2.output_shape)
        weight_original = l1.param.weight
        weight = np.copy(l1.param.weight)

        def forward(W):
            l1.param.weight = W
            return np.sum((target - l2.forward(l1.forward(input)))**2) / 2

        grad = num_grad(forward, weight)

        l1.param.weight = weight_original
        residual = target - l2.forward(l1.forward(input))
        l1.backprop(l2.backprop(residual))
        real_grad = -l1.dldw

        self.assertTrue(grad is not real_grad, "grad and real_grad is the same instance")
        print(f"grad:\n{grad}\n\nreal_grad:\n{real_grad}")
        self.assertTrue(np.allclose(grad, real_grad, atol = 1e-4), "num_grad and real gradient differ")
        print("multi_dense_relu_num_grad_weight() passed\n")

    def test_relu_shape(self):
        input_size = 2

        l = Relu(input_size, 3)
        l.compile()

        self.assertTrue(l.output_shape == l.input_shape, "Input, output shape differ")
        self.assertTrue(l.output_shape == (3, 2), "Expected output shape differ")

        l.input_size = (4, 5)
        l.compile()

        self.assertTrue(l.output_shape == l.input_shape, "Input, output shape differ")
        self.assertTrue(l.output_shape == (3, 4, 5), "Expected output shape differ")
        print("relu_shape() passed\n")


if __name__ == "__main__":
    unittest.main()