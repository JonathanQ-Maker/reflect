from reflect import np
from reflect.compiled_object import CompiledObject

class L1L2(CompiledObject):
    """
    L2 weight regularization
    """


    reg_coeff_l1 = 0
    reg_coeff_l2 = 0
    grad = None
    shape = None

    def __init__(self, reg_coeff_l1, reg_coeff_l2):
        self.reg_coeff_l1 = reg_coeff_l1
        self.reg_coeff_l2 = reg_coeff_l2

    def gradient(self, weight):
        # g: grad, r: coeff, sgn(): sign function, w: weight
        # g = rw + rsgn(w)
        # g = r(w + sgn(w))
        if (self.reg_coeff_l1 == self.reg_coeff_l2):
            return np.multiply(self.reg_coeff_l1, np.add(weight, np.sign(weight, out=self.grad), out=self.grad), out=self.grad)

    def compile(self):
        self.grad = np.zeros(self.shape)

    def is_compiled(self):
        return self.shape is not None and self.grad is not None and self.grad.shape == self.shape
