import numpy as np
from reflect.layers import BatchNorm
from reflect.profiler import num_grad, check_grad
from reflect.optimizers import GradientDescent
import time
import unittest

class BatchNormTest(unittest.TestCase):

    def test_batch_norm_grad(self):
        np.random.seed(231)
        N, D = 5, 3
        x = 5 * np.random.randn(N, D) + 12
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)

        bn = BatchNorm()
        self.assertFalse(bn.is_compiled(), "is compiled before compiling")

        bn.compile(D, N, gen_param=True)
        self.assertTrue(bn.is_compiled(), "is not compiled after compiling")

        np.copyto(bn.param.beta, beta)
        np.copyto(bn.param.gamma, gamma)

        def forward(x):
            return bn.forward(x)

        def forwardGamma(g):
            np.copyto(bn.param.gamma, g)
            return bn.forward(x)

        def forwardBeta(b):
            np.copyto(bn.param.beta, b)
            return bn.forward(x)


        bn.forward(x)
        real_grad = bn.backprop(dout)

    
        passed, msg = check_grad(forward, x, real_grad, dout)
        self.assertTrue(passed, msg)

        bn.forward(x)
        bn.backprop(dout)
        passed, msg = check_grad(forwardGamma, gamma, bn.dldg.copy(), dout)
        self.assertTrue(passed, msg)

        bn.forward(x)
        bn.backprop(dout)
        passed, msg = check_grad(forwardBeta, beta, bn.dldb.copy(), dout)
        self.assertTrue(passed, msg)

        print("batch_norm_grad_test_2D() passed")

    def test_batch_norm_grad_4D(self):
        np.random.seed(231)
        B, H, W, C = 3, 4, 4, 2
        x = 5 * np.random.randn(B, H, W, C) + 12
        gamma = np.random.randn(C)
        beta = np.random.randn(C)
        dout = np.random.randn(B, H, W, C)

        bn = BatchNorm(gamma_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        self.assertFalse( bn.is_compiled(), "is compiled before compiling")

        bn.compile((H, W, C), B, gen_param=True)
        self.assertTrue(bn.is_compiled(), "is not compiled after compiling")

        np.copyto(bn.param.beta, beta)
        np.copyto(bn.param.gamma, gamma)

        def forward(x):
            return bn.forward(x)

        def forwardGamma(g):
            np.copyto(bn.param.gamma, g)
            return bn.forward(x)

        def forwardBeta(b):
            np.copyto(bn.param.beta, b)
            return bn.forward(x)

        bn.forward(x)
        real_grad = bn.backprop(dout)


        passed, msg = check_grad(forward, x, real_grad, dout)
        self.assertTrue(passed, msg)

        bn.forward(x)
        bn.backprop(dout)
        passed, msg = check_grad(forwardGamma, gamma, bn.dldg.copy(), dout)
        self.assertTrue(passed, msg)

        bn.forward(x)
        bn.backprop(dout)
        passed, msg = check_grad(forwardBeta, beta, bn.dldb.copy(), dout)
        self.assertTrue(passed, msg)

        print("batch_norm_grad_test_4D() passed")

    def test_batch_norm_one_batch_grad_2D(self):
        np.random.seed(231)
        N, D = 1, 1
        x = 5 * np.random.randn(N, D) + 12
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)

        bn = BatchNorm(gamma_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        self.assertFalse(bn.is_compiled(), "is compiled before compiling")

        bn.compile(D, N, gen_param=True)
        self.assertTrue(bn.is_compiled(), "is not compiled after compiling")

        np.copyto(bn.param.beta, beta)
        np.copyto(bn.param.gamma, gamma)

        bn.forward(x)
        real_grad = bn.backprop(dout)

        self.assertTrue(np.all(np.isfinite(bn.dldg)), "inf exists dldg")
        self.assertTrue(np.all(np.isfinite(bn.dldb)), "inf exists dldb")
        self.assertTrue(np.all(np.isfinite(real_grad)), "inf exists dldx")

        print("batch_norm_one_batch_grad_test_2D() passed")

    def test_batch_norm_one_batch_grad_4D(self):
        np.random.seed(231)
        B, H, W, C = 1, 1, 1, 1
        x = 5 * np.random.randn(B, H, W, C) + 12
        gamma = np.random.randn(C)
        beta = np.random.randn(C)
        dout = np.random.randn(B, H, W, C)

        bn = BatchNorm(gamma_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        self.assertFalse( bn.is_compiled(), "is compiled before compiling")

        bn.compile((H, W, C), B, gen_param=True)
        self.assertTrue(bn.is_compiled(), "is not compiled after compiling")

        np.copyto(bn.param.beta, beta)
        np.copyto(bn.param.gamma, gamma)

        bn.forward(x)
        real_grad = bn.backprop(dout)

        self.assertTrue(np.all(np.isfinite(bn.dldg)), "inf exists dldg")
        self.assertTrue(np.all(np.isfinite(bn.dldb)), "inf exists dldb")
        self.assertTrue(np.all(np.isfinite(real_grad)), "inf exists dldx")

        print("batch_norm_one_batch_grad_test_4D() passed")

    def test_batch_norm_dout_2D(self):
        np.random.seed(231)
        N, D = 1500, 3
        x = 5 * np.random.randn(N, D) + 12
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)
        count = 500
        step = 0.01

        new_std = 2
        new_mean = -5

        bn = BatchNorm(gamma_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        self.assertFalse( bn.is_compiled(), "is compiled before compiling")

        bn.compile(D, N, gen_param=True)
        self.assertTrue(bn.is_compiled(), "is not compiled after compiling")

        np.copyto(bn.param.beta, beta)
        np.copyto(bn.param.gamma, gamma)

        target = (x - 12) * new_std / 5 + new_mean
        out = bn.forward(x)
        for i in range(count):
            bn.forward(x)
            residual = out - target
            bn.backprop(residual)
            bn.apply_grad(step)

        mean = np.mean(out, axis=bn.axis)
        std = np.std(out, axis=bn.axis)
        expected_std = np.full(std.shape, new_std) 
        expected_mean = np.full(mean.shape, new_mean)

        self.assertTrue(np.allclose(std, expected_std, atol = 0.5), "output std does not match expected")
        self.assertTrue(np.allclose(mean, expected_mean, atol = 0.5), "output mean does not match expected")
        print("batch_norm_dout_2D_test() passed")

    def test_batch_norm_dout_4D(self):
        np.random.seed(231)
        B, H, W, C = 100, 10, 10, 4
        x = 5 * np.random.randn(B, H, W, C) + 12
        gamma = np.random.randn(C)
        beta = np.random.randn(C)
        dout = np.random.randn(B, H, W, C)
        count = 500
        step = 0.01

        new_std = 2
        new_mean = -5

        bn = BatchNorm(gamma_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        self.assertFalse( bn.is_compiled(), "is compiled before compiling")

        bn.compile((H, W, C), B, gen_param=True)
        self.assertTrue(bn.is_compiled(), "is not compiled after compiling")

        np.copyto(bn.param.beta, beta)
        np.copyto(bn.param.gamma, gamma)

        target = (x - 12) * new_std / 5 + new_mean
        out = bn.forward(x)
        for i in range(count):
            bn.forward(x)
            residual = (out - target) / B
            bn.backprop(residual)
            bn.apply_grad(step)

        mean = np.mean(out, axis=bn.axis)
        std = np.std(out, axis=bn.axis)
        expected_std = np.full(std.shape, new_std) 
        expected_mean = np.full(mean.shape, new_mean)

        self.assertTrue(np.allclose(std, expected_std, atol = 0.5), 
                        "output std does not match expected")
        self.assertTrue(np.allclose(mean, expected_mean, atol = 0.5), 
                        "output mean does not match expected")
        print("batch_norm_dout_4D_test() passed")

    def test_batch_norm_dldx_approx_2D(self):
        np.random.seed(231)
        N, D = 55000, 3
        x = 5 * np.random.randn(N, D) + 12
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)

        bn_a = BatchNorm(approx_dldx=True,
                         gamma_optimizer=GradientDescent(), bias_optimizer=GradientDescent())
        bn = BatchNorm(approx_dldx=False,
                       gamma_optimizer=GradientDescent(), bias_optimizer=GradientDescent())

        self.assertFalse( bn_a.is_compiled(), "is compiled before compiling")
        self.assertFalse( bn.is_compiled(), "is compiled before compiling")

        bn_a.compile(D, N, gen_param=True)
        bn.compile(D, N, gen_param=True)
        self.assertTrue(bn_a.is_compiled(), "is not compiled after compiling")
        self.assertTrue(bn.is_compiled(), "is not compiled after compiling")

        np.copyto(bn_a.param.beta, beta)
        np.copyto(bn_a.param.gamma, gamma)
        np.copyto(bn.param.beta, beta)
        np.copyto(bn.param.gamma, gamma)



        bn_a.forward(x)
        bn.forward(x)

        start = time.time()
        bn_a.backprop(dout)
        approx_time = time.time() - start

        start = time.time()
        bn.backprop(dout)
        dldx_time = time.time() - start

        print(f"approx time: {approx_time}  dldx time: {dldx_time}")
        print(f"approx: {bn_a.dldx[0, 0:D]}\ndldx: {bn.dldx[0, 0:D]}")

        self.assertTrue(np.allclose(bn_a.dldx, bn.dldx, atol = 0.1), "dldx approximation differ")

        print("batch_norm_dldx_approx_test_2D() passed")

    def test_batch_norm_grad_cache(self):
        np.random.seed(231)
        N, D = 5, 3
        x = 5 * np.random.randn(N, D) + 12
        x2 = 5 * np.random.randn(N, D) + 12
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)

        bn = BatchNorm()
        self.assertFalse(bn.is_compiled(), "is compiled before compiling")

        bn.compile(D, N, gen_param=True)
        self.assertTrue(bn.is_compiled(), "is not compiled after compiling")

        np.copyto(bn.param.beta, beta)
        np.copyto(bn.param.gamma, gamma)

        def forward(x):
            return bn.forward(x)

        def forwardGamma(g):
            np.copyto(bn.param.gamma, g)
            return bn.forward(x)

        def forwardBeta(b):
            np.copyto(bn.param.beta, b)
            return bn.forward(x)


        bn.forward(x)
        real_grad = bn.backprop(dout)

    
        passed, msg = check_grad(forward, x, real_grad, dout)
        self.assertTrue(passed, msg)

        bn.forward(x)
        bn.backprop(dout)
        passed, msg = check_grad(forwardGamma, gamma, bn.dldg.copy(), dout)
        self.assertTrue(passed, msg)

        cache = bn.create_cache()

        bn.forward(x)
        bn.forward(x2, out_cache=cache)
        bn.backprop(dout)
        passed, msg = check_grad(forwardBeta, beta, bn.dldb.copy(), dout)
        self.assertTrue(passed, msg)

        print("batch_norm_grad_test_2D() passed")

    def test_batch_norm_scale(self):
        np.random.seed(231)
        D = 3
        x = 5 * np.random.randn(2, D) + 12
        x4 = np.tile(x, (2,1))
        x8 = np.tile(x, (4,1))
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(2, D)
        dout4 = np.tile(dout, (2,1))
        dout8 = np.tile(dout, (4,1))

        bn4 = BatchNorm()
        bn4.compile(D, 4, gen_param=True)
        self.assertTrue(bn4.is_compiled(), "is not compiled after compiling")
        np.copyto(bn4.param.beta, beta)
        np.copyto(bn4.param.gamma, gamma)
        bn4.forward(x4)
        bn4.backprop(dout4)

        bn8 = BatchNorm()
        bn8.compile(D, 8, gen_param=True)
        self.assertTrue(bn8.is_compiled(), "is not compiled after compiling")
        np.copyto(bn8.param.beta, beta)
        np.copyto(bn8.param.gamma, gamma)
        bn8.forward(x8)
        bn8.backprop(dout8)

        print(f"batchnorm dldg sum")
        print(bn4.dldg)
        print(bn8.dldg)
        self.assertAlmostEqual(bn4.dldg.sum()*2, bn8.dldg.sum(), msg="gradient scale is not as expected")


        



if __name__ == '__main__':
    unittest.main()