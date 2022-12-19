from reflect.optimizers.abtsract_optimizer import AbstractOptimizer
from reflect import np

class Momentum(AbstractOptimizer):
    """
    Momentum Optimizer 

    v_t = (1 - friction) * v_(t-1) + grad
    unbiased = v_t / (1 - (1 - friction)**t)
    grad_t = (step * friction) * unbiased

    NOTE: 
        sometimes velocity equation is 

            v_t = (1 - friction) * v_(t-1) + friction * grad

        but this version was used

            v_t = (1 - friction) * v_(t-1) + grad

        because it is computationally more efficient and use less memory.
        The first equation is the same as the second if the
        first equation was scaled by a factor of 1 / friction.
        See: 
        https://ai.stackexchange.com/questions/25152/how-are-these-equations-of-sgd-with-momentum-equivalent
    """

    _velocity           = None
    friction            = 0.0   # percent to decay/remove of old velocity, [0, 1)
    _correction         = 1.0   # correction term for unbiased/early gradients

    def __init__(self, friction=0.1):
        self.friction = friction

    def compile(self, shape):
        super().compile(shape)
        self._velocity  = np.zeros(self._shape)

    def is_compiled(self):
        velocity_ok = (self._velocity is not None 
                       and self._velocity.shape == self._shape)
        return (super().is_compiled()
                and velocity_ok)

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

        self._correction *= 1.0 - self.friction
        np.multiply(1.0-self.friction, self._velocity, out=self._velocity)
        np.add(self._velocity, grad, out=self._velocity)

        np.multiply(self.friction * step / (1.0 - self._correction), self._velocity, out=self._grad)
        return self._readonly_grad

    
