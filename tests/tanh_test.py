import numpy as np
from reflect.layers import Tanh
from reflect.profiler import check_grad
import unittest

class TanhTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(312)

    def test_gradient(self):

        x = np.random.randn(3, 7)
        dout = np.random.randn(3, 7)
        
        l = Tanh()
        self.assertFalse(l.is_compiled(), "layer should not be compiled")
        l.compile(7, 3)
        self.assertTrue(l.is_compiled(), "layer should be compiled")

        l.forward(x)
        l.backprop(dout)
        passed, msg = check_grad(l.forward, x, l.dldx, dout)
        self.assertTrue(passed, msg)


if __name__ == "__main__":
    unittest.main()
