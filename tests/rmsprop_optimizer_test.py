from reflect.optimizers.rmsprop import RMSprop
import numpy as np

import unittest

class RMSpropOptimizerTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(312)

    def test_momentum(self):
        decay   = 0.1
        step    = 1
        shape   = (2,)
        count   = 10
        beta    = 1.0 - decay

        opt     = RMSprop(decay)
        opt.compile(shape)

        self.assertTrue(opt.is_compiled(), "optimizer not compiled")

        grad_sqrd   = np.zeros(shape)
        grad_sqrd_  = np.zeros(shape)
        for i in range(count):
            grad    = np.random.normal(size=shape)
            opt_grad = opt.gradient(step, grad)
            print(f"[{i}]:\t{opt_grad} {grad}")

            # test if match complex but efficient formula
            grad_sqrd = (1.0 - decay) * grad_sqrd + grad**2
            unbiased = decay / (1.0 - (1.0 - decay)**(i+1)) * grad_sqrd
            expected_grad = step * grad / (np.sqrt(unbiased) + opt.epsilon)

            self.assertTrue(np.allclose(opt_grad, expected_grad, atol = 1e-4), 
                            f"expected gradient differ, {opt_grad} != {expected_grad}")

            grad_sqrd_ = beta * grad_sqrd_ + (1.0 - beta) * grad**2
            unbiased = grad_sqrd_ / (1.0 - beta**(i+1))
            expected_grad = step * grad / (np.sqrt(unbiased) + opt.epsilon)

            # test if match intuitive formula
            self.assertTrue(np.allclose(opt_grad, expected_grad, atol = 1e-4), 
                            f"expected gradient differ, {opt_grad} != {expected_grad}")

    


if __name__ == "__main__":
    unittest.main()



