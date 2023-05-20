import numpy as np
from reflect.layers import Convolve2D, Convolve2DParam
from reflect.profiler import num_grad, check_grad
from reflect.optimizers import GradientDescent
import time
import unittest

class Convolve2DTest(unittest.TestCase):

    def test_grad_match(self):
        B, H, W, C = 5, 7, 7, 2
        K, h, w = 4, 3, 3
        s_h, s_w = 1, 1

        input_size = (H, W, C)
        input_shape = (B, H, W, C)
        kernel_size = (h, w)
        strides = (s_h, s_w)

        x = np.random.normal(size=input_shape)


        l = Convolve2D(kernel_size, K, strides, "xavier", 
                       kernel_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        l.compile(input_size, B, gen_param=True)

        dldz = np.random.normal(size=l.output_shape)

        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(l.forward, x, l.dldx, dldz)
        self.assertTrue(passed, msg)
        
        original_k = l.param.kernel
        kernel = np.copy(l.param.kernel)
        def func(k):
            np.copyto(l.param.kernel, k)
            return l.forward(x)
        np.copyto(l.param.kernel, original_k)
        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(func, kernel, l.dldk, dldz)
        self.assertTrue(passed, msg)

        original_b = l.param.bias
        bias = np.copy(l.param.bias)
        def func(b):
            np.copyto(l.param.bias, b)
            return l.forward(x)
        np.copyto(l.param.bias, original_b)
        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(func, bias, l.dldb, dldz)
        self.assertTrue(passed, msg)

    def test_grad_match_padded(self):
        B, H, W, C = 10, 9, 8, 7
        K, h, w = 6, 5, 4
        s_h, s_w = 2, 3

        input_size = (H, W, C)
        input_shape = (B, H, W, C)
        kernel_size = (h, w)
        strides = (s_h, s_w)

        x = np.random.normal(size=input_shape)


        l = Convolve2D(kernel_size, K, strides, "xavier", pad=True,
                       kernel_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        l.compile(input_size, B, gen_param=True)

        self.assertTrue(l.output.shape[:-1] == l.input_shape[:-1], 
                        f"output and x shape differ {l.output.shape[:-1]} != {l.input_shape[:-1]}")

        dldz = np.random.normal(size=l.output_shape)

        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(l.forward, x, l.dldx, dldz)
        self.assertTrue(passed, msg)
        
        original_k = l.param.kernel
        kernel = np.copy(l.param.kernel)
        def func(k):
            np.copyto(l.param.kernel, k)
            return l.forward(x)
        np.copyto(l.param.kernel, original_k)
        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(func, kernel, l.dldk, dldz)
        self.assertTrue(passed, msg)

        original_b = l.param.bias
        bias = np.copy(l.param.bias)
        def func(b):
            np.copyto(l.param.bias, b)
            return l.forward(x)
        np.copyto(l.param.bias, original_b)
        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(func, bias, l.dldb, dldz)
        self.assertTrue(passed, msg)

    def test_grad_match_jagged(self):
        B, H, W, C = 5, 8, 7, 2
        K, h, w = 4, 2, 3
        s_h, s_w = 2, 1

        input_size = (H, W, C)
        input_shape = (B, H, W, C)
        kernel_size = (h, w)
        strides = (s_h, s_w)

        x = np.random.normal(size=input_shape)


        l = Convolve2D(kernel_size, K, strides, "xavier", 
                       kernel_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        l.compile(input_size, B, gen_param=True)

        dldz = np.random.normal(size=l.output_shape)

        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(l.forward, x, l.dldx, dldz)
        self.assertTrue(passed, msg)
        
        original_k = l.param.kernel
        kernel = np.copy(l.param.kernel)
        def func(k):
            np.copyto(l.param.kernel, k)
            return l.forward(x)
        np.copyto(l.param.kernel, original_k)
        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(func, kernel, l.dldk, dldz)
        self.assertTrue(passed, msg)

        original_b = l.param.bias
        bias = np.copy(l.param.bias)
        def func(b):
            np.copyto(l.param.bias, b)
            return l.forward(x)
        np.copyto(l.param.bias, original_b)
        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(func, bias, l.dldb, dldz)
        self.assertTrue(passed, msg)

    def test_grad_match_cache(self):
        B, H, W, C = 5, 7, 7, 2
        K, h, w = 4, 3, 3
        s_h, s_w = 1, 1

        input_size = (H, W, C)
        input_shape = (B, H, W, C)
        kernel_size = (h, w)
        strides = (s_h, s_w)

        x = np.random.normal(size=input_shape)
        x2 = np.random.normal(size=input_shape)


        l = Convolve2D(kernel_size, K, strides, "xavier", 
                       kernel_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        l.compile(input_size, B, gen_param=True)

        dldz = np.random.normal(size=l.output_shape)

        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(l.forward, x, l.dldx, dldz)
        self.assertTrue(passed, msg)
        
        original_k = l.param.kernel
        kernel = np.copy(l.param.kernel)
        def func(k):
            np.copyto(l.param.kernel, k)
            return l.forward(x)
        np.copyto(l.param.kernel, original_k)
        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(func, kernel, l.dldk, dldz)
        self.assertTrue(passed, msg)

        original_b = l.param.bias
        bias = np.copy(l.param.bias)
        def func(b):
            np.copyto(l.param.bias, b)
            return l.forward(x)
        np.copyto(l.param.bias, original_b)

        cache = l.create_cache()
        l.forward(x)
        l.forward(x2, out_cache=cache)
        l.backprop(dldz)

        passed, msg = check_grad(func, bias, l.dldb, dldz)
        self.assertTrue(passed, msg)

    def test_grad_match_padded_cache(self):
        B, H, W, C = 10, 9, 8, 7
        K, h, w = 6, 5, 4
        s_h, s_w = 2, 3

        input_size = (H, W, C)
        input_shape = (B, H, W, C)
        kernel_size = (h, w)
        strides = (s_h, s_w)

        x = np.random.normal(size=input_shape)
        x2 = np.random.normal(size=input_shape)


        l = Convolve2D(kernel_size, K, strides, "xavier", pad=True,
                       kernel_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        l.compile(input_size, B, gen_param=True)

        self.assertTrue(l.output.shape[:-1] == l.input_shape[:-1], 
                        f"output and x shape differ {l.output.shape[:-1]} != {l.input_shape[:-1]}")

        dldz = np.random.normal(size=l.output_shape)

        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(l.forward, x, l.dldx, dldz)
        self.assertTrue(passed, msg)
        
        original_k = l.param.kernel
        kernel = np.copy(l.param.kernel)
        def func(k):
            np.copyto(l.param.kernel, k)
            return l.forward(x)
        np.copyto(l.param.kernel, original_k)
        l.forward(x)
        l.backprop(dldz)

        passed, msg = check_grad(func, kernel, l.dldk, dldz)
        self.assertTrue(passed, msg)

        original_b = l.param.bias
        bias = np.copy(l.param.bias)
        def func(b):
            np.copyto(l.param.bias, b)
            return l.forward(x)
        np.copyto(l.param.bias, original_b)

        cache = l.create_cache()

        l.forward(x)
        l.forward(x2, out_cache=cache)
        l.backprop(dldz)

        passed, msg = check_grad(func, bias, l.dldb, dldz)
        self.assertTrue(passed, msg)


if __name__ == '__main__':
    unittest.main()
