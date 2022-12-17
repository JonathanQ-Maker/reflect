from reflect.optimizers.momentum import Momentum
import numpy as np

import unittest

class MomentumOptimizerTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(312)

    def test_momentum(self):
        friction    = 0.1
        step        = 1
        shape       = (2,)
        count       = 10
        beta        = 1.0 - friction

        opt     = Momentum(friction)
        opt.compile(shape)

        self.assertTrue(opt.is_compiled(), "optimizer not compiled")

        v = np.zeros(shape)
        v_ = np.zeros(shape)
        for i in range(count):
            grad    = np.random.normal(size=shape)
            opt_grad = opt.gradient(step, grad)
            print(f"[{i}]:\t{opt_grad} {grad}")

            # test if match complex but efficient formula
            v = (1.0 - friction) * v + grad
            unbiased = v / (1 - (1 - friction)**(i + 1))
            expected_grad = (step * friction) * unbiased

            self.assertTrue(np.allclose(opt_grad, expected_grad, atol = 1e-4), 
                            f"expected gradient differ, {opt_grad} != {expected_grad}")

            v_ = beta * v_ + (1.0 - beta) * grad
            unbiased = v_ / (1 - beta**(i+1))
            expected_grad = step * unbiased

            # test if match intuitive formula
            self.assertTrue(np.allclose(opt_grad, expected_grad, atol = 1e-4), 
                            f"expected gradient differ, {opt_grad} != {expected_grad}")

    


if __name__ == "__main__":
    unittest.main()


