import numpy as np
from reflect.compiled_object import CompiledObject

class L2(CompiledObject):
    """
    L2 weight regularization
    """


    reg_coeff = 0
    grad = None
    shape = None

    def __init__(self, reg_coeff):
        self.reg_coeff = reg_coeff

    def gradient(self, weight):
        return np.multiply(weight, self.reg_coeff, out=self.grad)

    def compile(self):
        self.grad = np.zeros(self.shape)

    def is_compiled(self):
        return self.shape is not None and self.grad is not None and self.grad.shape == self.shape
