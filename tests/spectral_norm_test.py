from reflect.layers import DenseSN, ConvolveSN2D, TransposedConvSN2D
from reflect.profiler import check_grad, num_grad
from reflect.constraints import Clip
import unittest
import numpy as np


class SNTest(unittest.TestCase):


    def setUp(self):
        np.random.seed(312)

    def spectal_norm(self, W, u, v, lip_const):
        sigma = np.dot(u.T, np.dot(W, v))
        return (lip_const * W / sigma, sigma)
    
    def test_lip_const_DenseSN(self):
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

        expected_weight_sigma = np.linalg.svd(layer.param.weight)[1].max()

        norm, weight_sigma = self.spectal_norm(layer.param.weight.T, layer._u, layer._v, lip_const)
        sigma = np.linalg.svd(norm)[1].max()

        print(f"Weight sigma: {round(weight_sigma, 3)} Expected: {round(expected_weight_sigma, 3)}")
        self.assertAlmostEqual(weight_sigma, expected_weight_sigma, 3, 
                               msg=f"{weight_sigma} != {expected_weight_sigma} Max singlular value differ from expected")

        print(f"DenseSN sigma: {round(sigma, 3)} Expected: {lip_const}")
        self.assertAlmostEqual(sigma, lip_const, 3, msg=f"{sigma} != {lip_const} Max singlular value differ from expected")

    def test_lip_const_ConvolveSN2D(self):
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

        expected_kernel_sigma = np.linalg.svd(kernel)[1].max()
        norm, kernel_sigma = self.spectal_norm(kernel, layer._u, layer._v, lip_const)
        sigma = np.linalg.svd(norm)[1].max()

        print(f"kernel sigma: {round(kernel_sigma, 3)} Expected: {round(expected_kernel_sigma, 3)}")
        self.assertAlmostEqual(kernel_sigma, expected_kernel_sigma, 3, 
                               msg=f"{kernel_sigma} != {expected_kernel_sigma} Max singlular value differ from expected")

        
        print(f"ConvolveSN2D sigma: {round(sigma, 3)} Expected: {lip_const}")
        self.assertAlmostEqual(sigma, lip_const, 3, msg=f"{sigma} != {lip_const} Max singlular value differ from expected")

    def test_lip_const_TransposedConvSN2D(self):
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

        expected_kernel_sigma = np.linalg.svd(kernel)[1].max()
        norm, kernel_sigma = self.spectal_norm(kernel, layer._u, layer._v, lip_const)
        sigma = np.linalg.svd(norm)[1].max()

        print(f"kernel sigma: {round(kernel_sigma, 3)} Expected: {round(expected_kernel_sigma, 3)}")
        self.assertAlmostEqual(kernel_sigma, expected_kernel_sigma, 3, 
                               msg=f"{kernel_sigma} != {expected_kernel_sigma} Max singlular value differ from expected")

        
        print(f"TransposedConvSN2D sigma: {round(sigma, 3)} Expected: {lip_const}")
        self.assertAlmostEqual(sigma, lip_const, 3, msg=f"{sigma} != {lip_const} Max singlular value differ from expected")

    def test_gradient_DenseSN(self):
        lip_const = 3.17
        layer = DenseSN(units=3, lip_const=lip_const)
        input_size = 2
        batch_size = 2

        # expected_grad = [[ 1.1535988,  -1.29080936,  0.21015461],
        #                 [-1.27672926, -1.20568211, -2.8557631 ]]
        
        X = np.random.randn(batch_size, input_size)
        dout = np.random.randn(batch_size, layer.units)

        self.assertFalse(layer.is_compiled(), "layer should not be compiled")
        layer.compile(input_size=input_size, batch_size=batch_size)
        self.assertTrue(layer.is_compiled(), "layer should be compiled")

        layer.forward(X)
        layer.backprop(dout)

        def forward(W):
            original = layer.param.weight.copy()
            np.copyto(layer.param.weight, W)
            np.dot(layer.param.weight.T, layer._v, out=layer._Wv)
            layer.forward(X)
            np.copyto(layer.param.weight, original)
            return layer.output
        
        W = layer.param.weight.copy()
        
        passed, msg = check_grad(forward, W, layer.dldw, dout)
        self.assertTrue(passed, msg)

    def test_gradient_TransposedConvSN2D(self):
        lip_const = 3.17
        layer = TransposedConvSN2D(filter_size=(3, 3), kernels=2, lip_const=lip_const)
        input_size = (5, 5, 2)
        batch_size = 2
        
        X = np.random.randn(batch_size, input_size[0], input_size[1], input_size[2])

        self.assertFalse(layer.is_compiled(), "layer should not be compiled")
        layer.compile(input_size=input_size, batch_size=batch_size)
        self.assertTrue(layer.is_compiled(), "layer should be compiled")

        dout = np.random.normal(size=layer.output_shape)

        layer.forward(X)
        layer.backprop(dout)

        def forward(K):
            original = layer.param.kernel.copy()
            np.copyto(layer.param.kernel, K)
            np.dot(layer._kernel_2d_view, layer._v, out=layer._Kv)
            layer.forward(X)
            np.copyto(layer.param.kernel, original)
            return layer.output
        
        K = layer.param.kernel.copy()
        
        passed, msg = check_grad(forward, K, layer.dldk, dout)
        self.assertTrue(passed, msg)

    def test_gradient_ConvolveSN2D(self):
        lip_const = 3.17
        layer = ConvolveSN2D(filter_size=(3, 3), kernels=2, lip_const=lip_const)
        input_size = (5, 5, 2)
        batch_size = 2
        
        X = np.random.randn(batch_size, input_size[0], input_size[1], input_size[2])

        self.assertFalse(layer.is_compiled(), "layer should not be compiled")
        layer.compile(input_size=input_size, batch_size=batch_size)
        self.assertTrue(layer.is_compiled(), "layer should be compiled")

        dout = np.random.normal(size=layer.output_shape)

        layer.forward(X)
        layer.backprop(dout)

        def forward(K):
            original = layer.param.kernel.copy()
            np.copyto(layer.param.kernel, K)
            np.dot(layer._kernel_2d_view, layer._v, out=layer._Kv)
            layer.forward(X)
            np.copyto(layer.param.kernel, original)
            return layer.output
        
        K = layer.param.kernel.copy()
        
        passed, msg = check_grad(forward, K, layer.dldk, dout)
        self.assertTrue(passed, msg)

    def test_limit(self):
       
        def calc_lip_const(f, x):
            return np.linalg.norm(-f, axis=1) / np.linalg.norm(x, axis=1)

        lip_const = 3.476
        layer = DenseSN(units=1, lip_const=lip_const, bias_constraint=Clip(0))
        input_size = 4
        batch_size = 1
        
        
        X = np.random.randn(batch_size, input_size)

        self.assertFalse(layer.is_compiled(), "layer should not be compiled")
        layer.compile(input_size=input_size, batch_size=batch_size)
        self.assertTrue(layer.is_compiled(), "layer should be compiled")

        dout = np.ones(layer.output_shape)

        for i in range(1000):
            output = layer.forward(X)
            layer.backprop(dout)
            layer.apply_grad(0.01)

            if (i % 100 == 0):
                print(f"output:\n{output}")
                print(f"lip_const:\n{calc_lip_const(output, X)}")
                print()

        self.assertAlmostEqual(calc_lip_const(layer.output, X)[0], lip_const, 
                               msg="expected and actual lip_const differ!")

    def test_conv_lip_const(self):
       
        def calc_lip_const(f, x):
            f = f.reshape((f.shape[0], -1))
            x = x.reshape((x.shape[0], -1))
            return np.linalg.norm(f, axis=1) / np.linalg.norm(x, axis=1)

        lip_const = 2.47463
        layer = ConvolveSN2D(kernels=1, filter_size=3, lip_const=lip_const, bias_constraint=Clip(0))
        input_size = (3, 3, 1)
        batch_size = 1

        
        X = np.random.randn(batch_size, input_size[0], input_size[1], input_size[2])

        self.assertFalse(layer.is_compiled(), "layer should not be compiled")
        layer.compile(input_size=input_size, batch_size=batch_size)
        self.assertTrue(layer.is_compiled(), "layer should be compiled")

        dout = np.ones(layer.output_shape)

        for i in range(1000):
            output = layer.forward(X)
            layer.backprop(dout)
            layer.apply_grad(0.01)

            if (i % 100 == 0):
                print(f"output:\n{output}")
                print(f"lip_const:\n{calc_lip_const(output, X)}")
                print()

        self.assertAlmostEqual(calc_lip_const(layer.output, X)[0], lip_const, 
                               msg="expected and actual lip_const differ!")






if __name__ == "__main__":
    unittest.main()