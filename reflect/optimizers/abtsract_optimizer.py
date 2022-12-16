from abc import ABC, abstractmethod
from reflect.compiled_object import CompiledObject
from reflect import np

class AbstractOptimizer(CompiledObject):
    
    _shape = None

    @property
    def shape(self):
        return _shape

    @property
    def grad(self):
        """
        calculated gradient
        """
        return 1

    @abstractmethod
    def gradient(self, step, grad):
        """
        Calculate optimizer processed gradient

        Args:
            step: gradient descent step size
            grad: vanilla gradient to be processed

        Returns:
            optimizer processed gradient

        NOTE: grad matrix must match size and must be compiled

        see class doc for more info
        """
        pass

    def compile(self, shape):
        self._shape = shape

    def is_compiled(self):
        return self._shape is not None




