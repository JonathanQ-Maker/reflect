from reflect.layers.absrtact_layer import AbstractLayer
from reflect import np

class Relu(AbstractLayer):
    """
    Rectified Linear Activation Unit (RELU)
    activation layer
    """

    _input = None

    @property
    def input(self):
        if (self._input is None):
            return None
        view = self._input.view()
        view.flags.writeable = False
        return view

    def __init__(self, input_size=1, batch_size=1):
        super().__init__(input_size, input_size, batch_size)


    def compile(self):
        self._output_size = self._input_size
        super().compile()
    
    def is_compiled(self):
        dldx_ok = self._dldx is not None and self._dldx.shape == self._input_shape
        return super().is_compiled() and dldx_ok

    def forward(self, X):
        """
        forward pass with input

        Args:
            X: input

        Returns: 
            output

        Make copy of output if intended to be modified
        Input instance will be kept and expected not to be modified between forward and backward pass
        """
        self._input = X
        return np.maximum(X, 0, out=self._output)

    def backprop(self, dldz):
        """
        backward pass to compute the gradients

        Args:
            dldz: gradient of loss with respect to output

        Returns: 
            dldx, gradient of loss with respect to input

        Note:
            expected to execute only once after forward
        """
        return np.multiply(np.greater(self._input, 0, out=self._dldx), dldz, out=self._dldx)
        