from abc import ABC, abstractmethod
from reflect.compiled_object import CompiledObject

class AbstractModel(CompiledObject):
    """Abstract class of Model"""
    
    _input_size = None
    _batch_size = None
    _output     = None
    _dldx       = None

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

    @abstractmethod
    def forward(self, X):
        return
