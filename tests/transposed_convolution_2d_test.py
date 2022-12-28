import numpy as np
from reflect.layers import TransposedConv2D, Convolve2DParam
from reflect.profiler import num_grad, check_grad
from reflect.optimizers import GradientDescent
import time
import unittest

class TransposedConvolve2DTest(unittest.TestCase):

    def test_grad_match(self):
        B, H, W, C = 5, 2, 2, 2
        K, h, w = 4, 3, 3
        s_h, s_w = 1, 1

        input_size = (H, W, C)
        input_shape = (B, H, W, C)
        kernel_size = (h, w)
        strides = (s_h, s_w)

        input = np.random.normal(size=input_shape)


        l = TransposedConv2D(kernel_size, K, strides, "xavier", 
                       kernel_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        l.compile(input_size, B, gen_param=True)

        dldz = np.random.normal(size=l.output_shape)

        l.forward(input)
        l.backprop(dldz)

        passed, msg = check_grad(l.forward, input, l.dldx, dldz)
        self.assertTrue(passed, msg)
        
        original_k = l.param.kernel
        kernel = np.copy(l.param.kernel)
        def func(k):
            l.param.kernel = k
            return l.forward(input)
        l.param.kernel = original_k
        l.forward(input)
        l.backprop(dldz)

        passed, msg = check_grad(func, kernel, l.dldk, dldz)
        self.assertTrue(passed, msg)

        original_b = l.param.bias
        bias = np.copy(l.param.bias)
        def func(b):
            l.param.bias = b
            return l.forward(input)
        l.param.bias = original_b
        l.forward(input)
        l.backprop(dldz)

        passed, msg = check_grad(func, bias, l.dldb, dldz)
        self.assertTrue(passed, msg)

    def test_grad_match_jagged(self):
        B, H, W, C = 10, 5, 4, 7
        K, h, w = 6, 2, 3
        s_h, s_w = 2, 3

        input_size = (H, W, C)
        input_shape = (B, H, W, C)
        kernel_size = (h, w)
        strides = (s_h, s_w)

        input = np.random.normal(size=input_shape)


        l = TransposedConv2D(kernel_size, K, strides, "xavier",
                       kernel_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        l.compile(input_size, B, gen_param=True)

        self.assertTrue(l.is_compiled(), "is not compiled")

        dldz = np.random.normal(size=l.output_shape)

        l.forward(input)
        l.backprop(dldz)

        passed, msg = check_grad(l.forward, input, l.dldx, dldz)
        self.assertTrue(passed, msg)
        
        original_k = l.param.kernel
        kernel = np.copy(l.param.kernel)
        def func(k):
            l.param.kernel = k
            return l.forward(input)
        l.param.kernel = original_k
        l.forward(input)
        l.backprop(dldz)

        passed, msg = check_grad(func, kernel, l.dldk, dldz)
        self.assertTrue(passed, msg)

        original_b = l.param.bias
        bias = np.copy(l.param.bias)
        def func(b):
            l.param.bias = b
            return l.forward(input)
        l.param.bias = original_b
        l.forward(input)
        l.backprop(dldz)

        passed, msg = check_grad(func, bias, l.dldb, dldz)
        self.assertTrue(passed, msg)


if __name__ == '__main__':
    unittest.main()
