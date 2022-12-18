from reflect.optimizers.adam import Adam
import numpy as np

import unittest

class MomentumOptimizerTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(312)

    def test_adam(self):
        friction    = 0.1
        decay       = 0.01
        step        = 1
        shape       = (2,)
        count       = 10
        beta_m      = 1.0 - friction
        beta_rms    = 1.0 - decay

        opt     = Adam(friction, decay)
        opt.compile(shape)

        self.assertTrue(opt.is_compiled(), "optimizer not compiled")

        v               = np.zeros(shape)
        v_              = np.zeros(shape)
        grad_squared    = np.zeros(shape)
        grad_squared_   = np.zeros(shape)
        for i in range(count):
            grad    = np.random.normal(size=shape)
            opt_grad = opt.gradient(step, grad)
            print(f"[{i}]:\t{opt_grad} {grad}")

            # test if match complex but efficient formula
            v = (1.0 - friction) * v + grad
            unbiased_m = v / (1.0 - (1.0 - friction)**(i+1))

            grad_squared = (1.0 - decay) * grad_squared + grad**2
            unbiased_rms = decay / (1.0 - (1.0 - decay)**(i+1)) * grad_squared
            expected_grad = step * friction * unbiased_m / (np.sqrt(unbiased_rms) + opt.epsilon)

            self.assertTrue(np.allclose(opt_grad, expected_grad, atol = 1e-4), 
                            f"expected gradient differ, {opt_grad} != {expected_grad}")

            v_ = beta_m * v_ + (1.0 - beta_m) * grad
            grad_squared_ = beta_rms * grad_squared_ + (1.0 - beta_rms) * grad**2
            unbiased_m = v_ / (1.0 - beta_m**(i+1))
            unbiased_rms = grad_squared_ / (1.0 - beta_rms**(i+1))
            expected_grad = step * unbiased_m / (np.sqrt(unbiased_rms) + opt.epsilon)


            # test if match intuitive formula
            self.assertTrue(np.allclose(opt_grad, expected_grad, atol = 1e-4), 
                            f"expected gradient differ, {opt_grad} != {expected_grad}")

    


if __name__ == "__main__":
    unittest.main()


