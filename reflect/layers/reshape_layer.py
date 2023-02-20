from reflect.layers.abstract_layer import AbstractLayer
from reflect import np

class Reshape(AbstractLayer):

    """
    Reshape layer, reshape input

    Shape:
        input:  (batch size, ...)
        output: (batch size, input_size)
    """

    _flat_output    = None
    _flat_dldx      = None

    def __init__(self, output_size):
        self._output_size = output_size
        super().__init__()


    def compile(self, input_size, batch_size=1):
        super().compile(input_size, batch_size)
        self._flat_output   = self._output.ravel()
        self._flat_dldx     = self._dldx.ravel()
        self.name           = f"Reshape {self._output_size}"
    
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
