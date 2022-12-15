from reflect import np
from reflect.compiled_object import CompiledObject

class L2(CompiledObject):
    """
    L2 weight regularization
    """


    reg_coeff       = 0
    _grad           = None
    _readonly_grad  = None
    _shape          = None

    @property
    def shape(self):
        return self._shape
            
    @property
    def grad(self):
        return self._readonly_grad

    def __init__(self, reg_coeff=0.001):
        self.reg_coeff = reg_coeff

    def gradient(self, weight):
        """
        calculate gradient of L2 norm with respect to weight

        Args:
            weight: weight matrix

        NOTE: weight matrix must match size and must be compiled
        """
        np.multiply(weight, self.reg_coeff, out=self._grad)
        return self._readonly_grad

    def compile(self, shape: tuple):
        self._shape = shape
        self._grad = np.zeros(self._shape)
        self._readonly_grad = self._grad.view()
        self._readonly_grad.flags.writeable = False

    def is_compiled(self):
        return self._shape is not None and self._grad is not None and self._grad.shape == self._shape
