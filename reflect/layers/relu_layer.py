from reflect.layers.absrtact_layer import AbstractLayer
from reflect import np

class Relu(AbstractLayer):
    """
    Rectified Linear Activation Unit (RELU)

    Shape:
        input:  (batch size, ...)
        output: (batch size, ...)
    """

    _input = None

    @property
    def input(self):
        if (self._input is None):
            return None
        view = self._input.view()
        view.flags.writeable = False
        return view

    def __init__(self):
        super().__init__()


    def compile(self, input_size, batch_size):
        self._output_size = input_size
        super().compile(input_size, batch_size)
        self.name = "Relu"
    
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
        np.maximum(X, 0, out=self._output)
        return self._readonly_output

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
        np.multiply(np.greater(self._input, 0, out=self._dldx), dldz, out=self._dldx)
        return self._readonly_dldx
        