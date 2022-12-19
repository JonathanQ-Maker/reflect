from reflect.optimizers.abtsract_optimizer import AbstractOptimizer
from reflect import np

class GradientDescent(AbstractOptimizer):
    """
    Vanilla Gradient Descent Optimizer 

    grad_t = step * grad
    """

    def __init__(self):
        pass

    def compile(self, shape):
        super().compile(shape)

    def is_compiled(self):
        return super().is_compiled()

    def gradient(self, step: float, grad):
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
        np.multiply(step, grad, out=self._grad)
        return self._readonly_grad

    

