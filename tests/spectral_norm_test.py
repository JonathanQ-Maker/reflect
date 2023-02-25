from reflect.layers import DenseSN, ConvolveSN2D, TransposedConvSN2D
import unittest
import numpy as np


class SNTest(unittest.TestCase):
    
    def test_DenseSN(self):
        lip_const = 3.17
        layer = DenseSN(units=5, lip_const=lip_const)
        input_size = 3
        batch_size = 2
        count = 20
        step = 0.001
        
        X = np.random.randn(batch_size, input_size)
        dout = np.random.randn(batch_size, 5)

        self.assertFalse(layer.is_compiled(), "layer should not be compiled")
        layer.compile(input_size=input_size, batch_size=batch_size)
        self.assertTrue(layer.is_compiled(), "layer should be compiled")

        for i in range(count):
            layer.forward(X)
            layer.backprop(dout)
            layer.apply_grad(step)

        sigma = np.linalg.svd(layer.param.weight)[1].max()

        print(f"DenseSN sigma: {round(sigma, 3)} Expected: {lip_const}")
        self.assertAlmostEqual(sigma, lip_const, 3, msg=f"{sigma} != {lip_const} Max singlular value differ from expected")

    def test_ConvolveSN2D(self):
        lip_const = 3.17
        layer = ConvolveSN2D(filter_size=(3, 3), kernels=2, lip_const=lip_const)
        input_size = (5, 5, 2)
        batch_size = 2
        count = 20
        step = 0.001
        
        X = np.random.randn(batch_size, input_size[0], input_size[1], input_size[2])

        self.assertFalse(layer.is_compiled(), "layer should not be compiled")
        layer.compile(input_size=input_size, batch_size=batch_size)
        self.assertTrue(layer.is_compiled(), "layer should be compiled")

        dout = np.random.normal(size=layer.output_shape)

        for i in range(count):
            layer.forward(X)
            layer.backprop(dout)
            layer.apply_grad(step)

        kernel = layer.param.kernel.copy()
        kernel.shape = (layer.kernels, -1)
        sigma = np.linalg.svd(kernel)[1].max()
        
        print(f"ConvolveSN2D sigma: {round(sigma, 3)} Expected: {lip_const}")
        self.assertAlmostEqual(sigma, lip_const, 3, msg=f"{sigma} != {lip_const} Max singlular value differ from expected")

    def test_TransposedConvSN2D(self):
        lip_const = 3.17
        layer = TransposedConvSN2D(filter_size=(3, 3), kernels=2, lip_const=lip_const)
        input_size = (5, 5, 2)
        batch_size = 2
        count = 20
        step = 0.001
        
        X = np.random.randn(batch_size, input_size[0], input_size[1], input_size[2])

        self.assertFalse(layer.is_compiled(), "layer should not be compiled")
        layer.compile(input_size=input_size, batch_size=batch_size)
        self.assertTrue(layer.is_compiled(), "layer should be compiled")

        dout = np.random.normal(size=layer.output_shape)

        for i in range(count):
            layer.forward(X)
            layer.backprop(dout)
            layer.apply_grad(step)

        kernel = layer.param.kernel.copy()
        kernel.shape = (layer.kernels, -1)
        sigma = np.linalg.svd(kernel)[1].max()
        
        print(f"TransposedConvSN2D sigma: {round(sigma, 3)} Expected: {lip_const}")
        self.assertAlmostEqual(sigma, lip_const, 3, msg=f"{sigma} != {lip_const} Max singlular value differ from expected")



if __name__ == "__main__":
    unittest.main()