from abc import ABC, abstractmethod
from reflect.compiled_object import CompiledObject
from reflect import np
from reflect.utils.misc import to_tuple

class AbstractLayer(CompiledObject):
    """
    Base layer class
    """

    # public variables
    _input_size         = None
    _input_shape        = None # (batch size, ) + input_size

    _output_size        = None
    _output_shape       = None # (batch size, ) + output_size
    _output             = None
    _readonly_output    = None # readonly output view

    _dldx               = None # gradient of loss with respect to input
    _readonly_dldx      = None # readonly dldx view

    _batch_size         = None # mini-batch size
    name                = None # non-unique name


    @property
    def output(self):
        return self._readonly_output

    @property
    def dldx(self):
        return self._readonly_dldx

    @property
    def input_size(self):
        return self._input_size

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_size(self):
        return self._output_size

    @output_size.setter
    def output_size(self, size):
        self._output_size = size

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def batch_size(self):
        return self._batch_size

    def compile(self, input_size, batch_size):
        super().compile()
        self._input_size = input_size
        self._batch_size = batch_size
        # compile shapes
        self._input_shape   = (self._batch_size, ) + to_tuple(self._input_size)
        self._output_shape  = (self._batch_size, ) + to_tuple(self._output_size)

        # compile arrays
        self._output    = np.zeros(shape=self._output_shape)
        self._dldx      = np.zeros(shape=self._input_shape)

        # compile read only views
        self._readonly_output = self._output.view()
        self._readonly_dldx   = self._dldx.view()
        self._readonly_output.flags.writeable = False
        self._readonly_dldx.flags.writeable = False

        self.name = "UNKNOWN"

    def is_compiled(self):
        """
        Check if layer is up-to-date with layer arguments

        # of check items should be the same as # of compile items
        """
        input_size_match = self._input_shape == (self._batch_size, ) + to_tuple(self._input_size)
        output_size_match = self._output_shape == (self._batch_size, ) + to_tuple(self._output_size)
        dldx_ok = self._dldx is not None and self._dldx.shape == self._input_shape
        output_ok = self._output is not None and self._output.shape == self._output_shape

        return (self.name is not None
                and input_size_match
                and output_size_match
                and dldx_ok
                and output_ok)

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
        return 0

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
        return 1

    def __str__(self):
        return self.attribute_to_str()

    def attribute_to_str(self):
        return (f"name:           {self.name}\n" 
        + f"batch size:     {self._batch_size}\n"
        + f"compiled:       {self.is_compiled()}\n"
        + f"input_shape:    {self._input_shape}\n"
        + f"output_shape:   {self._output_shape}\n")
