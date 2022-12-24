import unittest
from reflect.layers import Flatten
from reflect.profiler import check_grad
import numpy as np

class FlattenTest(unittest.TestCase):

    def test_output(self):
        batch_size = 2
        input_size = (3, 2, 1)
        input_shape = (batch_size, ) + input_size
        expected_shape = (batch_size, np.prod(input_size))

        input = np.random.normal(size=input_shape)


        l = Flatten()
        l.compile(input_size, batch_size)

        self.assertTrue(l.is_compiled(), "layer is not compiled")

        output = l.forward(input)
        self.assertTrue(output.shape == expected_shape, 
                        f"expected shape and output shape mismatch. {output.shape} != {expected_shape}")

    def test_grad(self):
        batch_size = 2
        input_size = (3, 2, 1)
        input_shape = (batch_size, ) + input_size
        expected_shape = (batch_size, np.prod(input_size))

        input = np.random.normal(size=input_shape)
        dldz = np.random.normal(size=expected_shape)


        l = Flatten()
        l.compile(input_size, batch_size)
        self.assertTrue(l.is_compiled(), "layer is not compiled")

        l.forward(input)
        l.backprop(dldz)
        passed, msg = check_grad(l.forward, input, l.dldx, dldz)
        self.assertTrue(passed, msg)



if __name__ == '__main__':
    unittest.main()
