from abc import ABC, abstractmethod
from reflect.compiled_object import CompiledObject
from reflect import np
from reflect.utils.misc import to_tuple

class AbstractLayer(CompiledObject):
    """
    Base layer class
    """

    input_size = None
    input_shape = None

    output_size = None
    output_shape = None

    output = None
    dldx = None         # gradient of loss with respect to input
    batch_size = None
    name = None         # not unique name

    def __init__(self, input_size, output_size, batch_size):
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size

    def compile(self):
        super().compile()
        # compile shapes
        self.input_shape = (self.batch_size, ) + to_tuple(self.input_size)
        self.output_shape = (self.batch_size, ) + to_tuple(self.output_size)

        # compile output
        self.output = np.zeros(shape=self.output_shape)
        self.dldx = np.zeros(shape=self.input_shape)

        self.name = "UNKNOWN"

    def is_compiled(self):
        """
        Check if layer is up-to-date with layer arguments

        # of check items should be the same as # of compile items
        """
        input_size_match = self.input_shape == (self.batch_size, ) + to_tuple(self.input_size)
        output_size_match = self.output_shape == (self.batch_size, ) + to_tuple(self.output_size)
        dldx_ok = self.dldx is not None and self.dldx.shape == self.input_shape
        output_ok = self.output is not None and self.output.shape == self.output_shape

        return (self.name is not None
                and input_size_match
                and output_size_match
                and dldx_ok
                and output_ok)

    def forward(self, X):
        return 0

    def backprop(self, dldz):
        return 1

    def attribute_to_str(self):
        return (f"name:           {self.name}\n" 
        + f"batch size:     {self.batch_size}\n"
        + f"compiled:       {self.is_compiled()}\n"
        + f"output size:    {self.output_size}\n"
        + f"output_shape:   {self.output_shape}\n"
        + f"input size:     {self.input_size}\n"
        + f"input_shape:    {self.input_shape}\n")
