import numpy as np
from reflect.layers import AvgPool2D
from reflect.profiler import num_grad, check_grad
import unittest

class AvgPool2DTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(312)

    def test_grad_match(self):
        B, H, W, C = 5, 7, 7, 2
        h, w = 3, 3
        s_h, s_w = 1, 1

        input_size = (H, W, C)
        input_shape = (B, H, W, C)
        pool_size = (h, w)
        strides = (s_h, s_w)

        input   = np.random.normal(size=input_shape)

        l = AvgPool2D(pool_size, strides)        
        self.assertFalse(l.is_compiled(), "should not be compiled")

        l.compile(input_size, B)
        self.assertTrue(l.is_compiled(), "should be compiled")

        dldz = np.random.normal(size=l.output_shape)
        dldz = np.ones(dldz.shape)

        l.forward(input)
        l.backprop(dldz)

        passed, msg = check_grad(l.forward, input, l.dldx, dldz)
        self.assertTrue(passed, msg)

    def test_jagged_grad_match(self):
        B, H, W, C  = 9, 8, 7, 6
        h, w        = 5, 4
        s_h, s_w    = 3, 2

        input_size = (H, W, C)
        input_shape = (B, H, W, C)
        pool_size = (h, w)
        strides = (s_h, s_w)

        input   = np.random.normal(size=input_shape)

        l = AvgPool2D(pool_size, strides)        
        self.assertFalse(l.is_compiled(), "should not be compiled")

        l.compile(input_size, B)
        self.assertTrue(l.is_compiled(), "should be compiled")

        dldz = np.random.normal(size=l.output_shape)
        dldz = np.ones(dldz.shape)

        l.forward(input)
        l.backprop(dldz)

        passed, msg = check_grad(l.forward, input, l.dldx, dldz)
        self.assertTrue(passed, msg)



if __name__ == '__main__':
    unittest.main()
