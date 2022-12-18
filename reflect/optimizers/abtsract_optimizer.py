from abc import ABC, abstractmethod
from reflect.compiled_object import CompiledObject
from reflect import np

class AbstractOptimizer(CompiledObject):
    
    _shape          = None
    _grad           = None
    _readonly_grad  = None

    @property
    def shape(self):
        return _shape

    @property
    def grad(self):
        """
        calculated gradient
        """
        return self._readonly_grad

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
        self._shape         = shape
        self._grad          = np.zeros(self._shape)
        self._readonly_grad = self._grad.view()

        self._readonly_grad.flags.writeable = False

    def is_compiled(self):
        grad_ok = (self._grad is not None 
                       and self._grad.shape == self._shape
                       and self._readonly_grad is not None
                       and self._readonly_grad.shape == self._shape)
        return (self._shape is not None 
                and grad_ok)




