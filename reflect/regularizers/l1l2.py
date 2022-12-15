from reflect import np
from reflect.compiled_object import CompiledObject

class L1L2(CompiledObject):
    """
    L1 + L2 weight regularization
    """


    reg_coeff_l1    = 0     # lambda for l1 
    reg_coeff_l2    = 0     # lambda for l2
    _grad           = None
    _buffer         = None
    _readonly_grad  = None
    _shape          = None

    @property
    def shape(self):
        return self._shape
            
    @property
    def grad(self):
        return self._readonly_grad

    def __init__(self, reg_coeff_l1=0.001, reg_coeff_l2=0.001):
        self.reg_coeff_l1 = reg_coeff_l1
        self.reg_coeff_l2 = reg_coeff_l2

    def gradient(self, weight):
        """
        calculate gradient of L1 + L2 norm with respect to weight

        Args:
            weight: weight matrix

        NOTE: weight matrix must match size and must be compiled
        """


        # g: grad, r: coeff, sgn(): sign function, w: weight
        # g = rw + rsgn(w)
        # g = r(w + sgn(w))
        if (self.reg_coeff_l1 == self.reg_coeff_l2):
            np.add(weight, np.sign(weight, out=self._grad), out=self._grad)
            np.multiply(self.reg_coeff_l1, self._grad, out=self._grad)

        # L2
        np.multiply(self.reg_coeff_l1, weight, out=self._grad)

        # L1
        np.sign(weight, out=self._buffer)
        np.multiply(self.reg_coeff_l2, self._buffer, out=self._buffer)

        np.add(self._grad, self._buffer, out=self._grad)
        return self._readonly_grad

    def compile(self, shape: tuple):
        self._shape = shape
        self._grad = np.zeros(self._shape)
        self._buffer = np.zeros(self._shape)
        self._readonly_grad = self._grad.view()
        self._readonly_grad.flags.writeable = False

    def is_compiled(self):
        return self._shape is not None and self._grad is not None and self._grad.shape == self._shape
