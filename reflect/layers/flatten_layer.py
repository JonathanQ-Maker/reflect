from reflect.layers.absrtact_layer import AbstractLayer
from reflect import np

class Flatten(AbstractLayer):
    """
    Flatten layer, flattens input

    Shape:
        input:  (batch size, ...)
        output: (batch size, prod(input_size))
    """

    _flat_output    = None
    _flat_dldx      = None

    def __init__(self, input_size=1, batch_size=1):
        super().__init__(input_size, input_size, batch_size)


    def compile(self):
        self._output_size = np.prod(self._input_size)
        super().compile()
        self._flat_output   = self._output.ravel()
        self._flat_dldx     = self._dldx.ravel()
    
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

        np.copyto(self._flat_output, X.ravel())
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
        np.copyto(self._flat_dldx, dldz.ravel())
        return self._readonly_dldx
        