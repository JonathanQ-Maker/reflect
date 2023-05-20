import unittest
from reflect.layers import Recurrent, Relu
from reflect.regularizers import L2
import numpy as np
from reflect.profiler import num_grad, check_grad

class RecurrentTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(312)
        pass

    def test_dldx(self):
        units           = 3
        truncate_length = 3
        input_size      = 4
        batch_size      = 2
        timesteps       = 4

        l = Recurrent(units = units, truncate_length=truncate_length)
        self.assertFalse(l.is_compiled(), "layer should not be compiled")
        l.compile(input_size, batch_size, timesteps, gen_param=True)
        self.assertTrue(l.is_compiled(), "layer should be compiled")


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
        self.assertFalse(l.is_compiled(), "layer should not be compiled")
        l.compile(input_size, batch_size, timesteps, gen_param=True)
        self.assertTrue(l.is_compiled(), "layer should be compiled")

        X = np.random.randn(timesteps, batch_size, input_size) * 2 # x2 to fix numerical percision issues 
        initial_state = np.random.randn(batch_size, units)

        # normally, timesteps = truncate_lengtth in dldz
        dldz = np.random.randn(timesteps, batch_size, units)

        # dldw
        original_weight = np.copy(l.param.weight)
        def forward(weight):
            np.copyto(l.param.weight, weight)
            return l.forward(X, initial_state)

        l.forward(X, initial_state)
        l.backprop(dldz)

        passed, msg = check_grad(forward, original_weight, l.dldw, dldz)
        self.assertTrue(passed, msg)
        np.copyto(l.param.weight, original_weight)

        # dldh
        original_hidden = np.copy(l.param.hidden_weight)
        def forward(hidden_weight):
            np.copyto(l.param.hidden_weight, hidden_weight)
            return l.forward(X, initial_state)

        l.forward(X, initial_state)
        l.backprop(dldz)

        passed, msg = check_grad(forward, original_hidden, l.dldh, dldz, delta=1e-10)
        self.assertTrue(passed, msg)
        np.copyto(l.param.hidden_weight, original_hidden)

        # dldb
        original_bias = np.copy(l.param.bias)
        def forward(bias):
            np.copyto(l.param.bias, bias)
            return l.forward(X, initial_state)

        l.forward(X, initial_state)
        l.backprop(dldz)

        passed, msg = check_grad(forward, original_bias, l.dldb, dldz)
        self.assertTrue(passed, msg)

    # TODO: XOR test
    def test_sequence(self):
        """
        Test if model is able to predict next item 
        in sequence given last item in sequence
        """

        sequence = [-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1]

        units           = 100
        truncate_length = 15
        input_size      = 1
        batch_size      = 1
        timesteps       = 1
        epoch           = 100
        initial_state   = np.zeros((batch_size, units))
        initial_state2  = np.zeros((batch_size, 1))
        predicted_seq   = -np.ones(len(sequence))
        step_size       = 0.0001

        l = Recurrent(units = units, truncate_length=truncate_length, activation=Relu(), weight_reg=L2(), hidden_reg=L2())
        l2 = Recurrent(units = 1, truncate_length=truncate_length, activation=None, weight_reg=L2(), hidden_reg=L2())

        self.assertFalse(l.is_compiled(), "layer should not be compiled")
        l.compile(input_size, batch_size, timesteps, gen_param=True)
        self.assertTrue(l.is_compiled(), "layer should be compiled")

        self.assertFalse(l2.is_compiled(), "layer should not be compiled")
        l2.compile(units, batch_size, timesteps, gen_param=True)
        self.assertTrue(l2.is_compiled(), "layer should be compiled")

        def loss(output, label):
            return np.log(1 + np.exp(-label * output))

        def dldz(output, label):
            return (np.exp(-label * output) * -label) / (1.0 + np.exp(-label * output))


        initial_loss = 0
        for i in range(len(sequence)-1):
            x = np.array([[[sequence[i]]]]) # shape = (timesteps, batch size, input size)
            state = None
            state2 = None
            if (i == 0):
                state = initial_state
                state2 = initial_state2
            l.forward(x, initial_state=state)
            l2.forward(l.output, initial_state=state2)
            if (l2.output[0, 0, 0] > 0):
                predicted_seq[i+1] = 1
            else:
                predicted_seq[i+1] = -1
            initial_loss += loss(l2.output[-1], sequence[i+1])[0, 0]
        initial_errors = np.sum(sequence != predicted_seq)

        print(f"Ground Truth Seqeuence: {np.maximum(sequence, 0)}")
        for t in range(epoch):
            total_loss = 0
            dldz_arry = [] # 1
            for i in range(len(sequence)-1):
                x = np.array([[[sequence[i]]]]) # shape = (timesteps, batch size, input size)
                state = None
                state2 = None
                if (i == 0):
                    state = initial_state
                    state2 = initial_state2
                l.forward(x, initial_state=state)
                l2.forward(l.output, initial_state=state2)
                if (l2.output[0, 0, 0] > 0):
                    predicted_seq[i+1] = 1
                else:
                    predicted_seq[i+1] = -1
                total_loss += loss(l2.output[-1], sequence[i+1])[0, 0]
                dldz_arry.append(dldz(l2.output[-1], sequence[i+1])) # 1
                
                l2.backprop(dldz_arry)
                l.backprop(l2.dldx)

                l2.apply_grad(step_size)
                l.apply_grad(step_size)
            print(f"loss: {total_loss}")
            print(f"predicted seq: {np.maximum(predicted_seq, 0).astype(int)}")
            errors = np.sum(sequence != predicted_seq)
            print(f"Errors: {errors}")
            if (errors == 0):
                break
        final_loss = 0
        for i in range(len(sequence)-1):
            x = np.array([[[sequence[i]]]]) # shape = (timesteps, batch size, input size)
            state = None
            state2 = None
            if (i == 0):
                state = initial_state
                state2 = initial_state2
            l.forward(x, initial_state=state)
            l2.forward(l.output, initial_state=state2)
            if (l2.output[0, 0, 0] > 0):
                predicted_seq[i+1] = 1
            else:
                predicted_seq[i+1] = -1
            final_loss += loss(l2.output[-1], sequence[i+1])[0, 0]
        final_errors = np.sum(sequence != predicted_seq)

        print(f"inital loss: {initial_loss}")
        print(f"inital errors: {initial_errors}")
        print(f"final loss: {final_loss}")
        print(f"final errors: {final_errors}")

        self.assertTrue(final_loss < initial_loss, "final loss is > than inital loss")
        self.assertTrue(final_errors < initial_errors, "final error >= inital error")

        
        





if __name__ == '__main__':
    unittest.main()
