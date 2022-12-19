from abc import ABC, abstractmethod
from reflect.compiled_object import CompiledObject

class AbstractModel(CompiledObject):
    """Abstract class of Model"""

    _output = None
    _dldx   = None

    @property
    def output(self):
        return self._output

    @property
    def dldx(self):
        return self._dldx

    @abstractmethod
    def forward(self, X):
        return
