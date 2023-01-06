import unittest
from reflect.layers import Recurrent
import numpy as np
from reflect.profiler import num_grad, check_grad

class RecurrentTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(312)

    def test_dldx(self):
        units           = 3
        truncate_length = 3
        input_size      = 4
        batch_size      = 2
        timesteps       = 4

        l = Recurrent(units = units, truncate_length=truncate_length)
        l.compile(input_size, batch_size, timesteps, gen_param=True)

        X = np.random.randn(timesteps, batch_size, input_size)
        initial_state = np.random.randn(batch_size, units)

        # normally, timesteps = truncate_lengtth in dldz
        dldz = np.random.randn(timesteps, batch_size, units)

        # dldx
        def forward(X):
            return l.forward(X, initial_state)

        l.forward(X, initial_state)
        l.backprop(dldz)

        delta = 1e-5
        n_dldx = num_grad(forward, X, dldz, delta)
        print(n_dldx)
        print("==================")
        print(l.dldx)
        print("\n\n\n")

        self.assertTrue(np.allclose(n_dldx[-l.truncate_length:], l.dldx, atol = delta * 10), 
                        "numeric grad and grad differ")

    def test_grad(self):
        units           = 3
        truncate_length = 5
        input_size      = 4
        batch_size      = 2
        timesteps       = truncate_length

        l = Recurrent(units = units, truncate_length=truncate_length)
        l.compile(input_size, batch_size, timesteps, gen_param=True)

        X = np.random.randn(timesteps, batch_size, input_size) * 2 # x2 to fix numerical percision issues 
        initial_state = np.random.randn(batch_size, units)

        # normally, timesteps = truncate_lengtth in dldz
        dldz = np.random.randn(timesteps, batch_size, units)

        # dldw
        original_weight = np.copy(l.param.weight)
        def forward(weight):
            l.param.weight = weight
            return l.forward(X, initial_state)

        l.forward(X, initial_state)
        l.backprop(dldz)

        passed, msg = check_grad(forward, original_weight, l.dldw, dldz)
        self.assertTrue(passed, msg)

        # dldh
        original_hidden = np.copy(l.param.hidden_weight)
        def forward(hidden_weight):
            l.param.hidden_weight = hidden_weight
            return l.forward(X, initial_state)

        l.forward(X, initial_state)
        l.backprop(dldz)

        passed, msg = check_grad(forward, original_hidden, l.dldh, dldz)
        self.assertTrue(passed, msg)

        # dldb
        original_bias = np.copy(l.param.bias)
        def forward(bias):
            l.param.bias = bias
            return l.forward(X, initial_state)

        l.forward(X, initial_state)
        l.backprop(dldz)

        passed, msg = check_grad(forward, original_bias, l.dldb, dldz)
        self.assertTrue(passed, msg)


if __name__ == '__main__':
    unittest.main()
