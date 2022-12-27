from abc import ABC, abstractmethod
from reflect.utils.misc import to_tuple
from reflect.compiled_object import CompiledObject

class AbstractModel(CompiledObject):
    """Abstract class of Model"""
    
    _input_size     = None
    _input_shape    = None
    _output_shape   = None
    _batch_size     = None
    _output         = None
    _dldx           = None

    @property
    def output(self):
        return self._output

    @property
    def dldx(self):
        return self._dldx

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, size):
        self._input_size = size

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
        self._batch_size = size

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def total_params(self):
        return 0

    def compile(self):
        super().compile()
        self._input_shape = (self._batch_size, ) + to_tuple(self._input_size)

    @abstractmethod
    def forward(self, X):
        return

    @abstractmethod
    def backprop(self, dldz):
        return

    def __str__(self):
        return (f"Type:           {self.__class__.__name__}\n"
                + f"Total params:   {self.total_params}\n\n\nLayers:\n\n")

    def print_summary(self):
        print(f"{'='*50}\nModel Summary\n")
        print(self.__str__())
        print(f"{'='*50}\n")
