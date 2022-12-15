from reflect.optimizers.abtsract_optimizer import AbstractOptimizer
from reflect import np

class Momentum(AbstractOptimizer):
    """
    Momentum Optimizer 

    v_t = (1 - friction) * v_(t-1) + grad
    grad_t = step * v_t

    NOTE: 
        real velocity equation is 

            v_t = (1 - friction) * v_(t-1) + friction * grad

        but was reduced to 

            v_t = (1 - friction) * v_(t-1) + grad

        such that it is computationally more efficient
    """

    _velocity           = None
    _readonly_velocity  = None
    friction            = 0

    @property
    def grad(self):
        """
        calculated gradient
        """
        return self._readonly_velocity

    def __init__(self, friction=0.01):
        self.friction = friction

    def compile(self, shape):
        super().compile(shape)

        self._velocity = np.zeros(self._shape)
        self._readonly_velocity = self._velocity.view()
        self._readonly_velocity.flags.writeable = False

    def is_compiled(self):
        velocity_ok = (self._velocity is not None 
                       and self._velocity.shape == self._shape)
        return (super().is_compiled
                and velocity_ok)

    def gradient(self, grad):
        """
        Optimizer processed gradient

        Args:
            grad: vanilla gradient to be processed

        Returns:
            optimizer processed gradient

        NOTE: grad matrix must match size and must be compiled

        see class doc for more info
        """

        np.multiply(1.0-self.friction, self._velocity, out=self._velocity)
        np.add(self._velocity, grad, out=self._velocity)
        return self._readonly_velocity

    