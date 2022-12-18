from reflect.optimizers.abtsract_optimizer import AbstractOptimizer
from reflect import np

class Adam(AbstractOptimizer):
    """
    Adam Optimizer

    v_t = (1 - friction) * v_(t-1) + grad
    grad_sqrd_t = (1 - decay) * grad_sqrd_(t-1) + grad**2

    unbiased_m = friction * v_t / (1 - (1 - friction)**t)
    unbiased_rms = decay / (1 - (1 - decay)**t) * grad_sqrd_t
    grad_t = step * unbiased_m / (sqrt(unbiased_rms) + epsilon)

    NOTE: 
        (1) sometimes velocity equation is 

            v_t = (1 - friction) * v_(t-1) + friction * grad

        but this version was used

            v_t = (1 - friction) * v_(t-1) + grad

        because it is computationally more efficient and use less memory.
        The first equation is the same as the second if the
        first equation was scaled by a factor of 1 / friction.
        See: 
        https://ai.stackexchange.com/questions/25152/how-are-these-equations-of-sgd-with-momentum-equivalent

        (2) original grad_sqrd_t equation is 

            grad_sqrd_t = (1 - decay) * grad_sqrd_(t-1) + decay * grad**2

        but was reduced to 

            grad_sqrd_t = (1 - decay) * grad_sqrd_(t-1) + grad**2

        such that it is more memory efficient.
        The first equation is the same as the second if the
        first equation was scaled by a factor of 1 / decay.
    """

    _velocity       = None
    friction        = 0.0   # percent to decay/remove of old velocity, [0, 1)
    _correction_m   = 1.0   # correction term for unbiased/early momentum gradients

    _grad_squared   = None  # running average gradient squared
    decay           = 0.0   # percent to decay/remove of old gradients, [0, 1)
    epsilon         = 1e-7  # numerical stability coefficient, (0, inf)
    _correction_rms = 1.0   # correction term for unbiased/early RMS gradients

    def __init__(self, friction=0.1, decay=0.01, epsilon=1e-7):
        self.friction   = friction
        self.decay      = decay
        self.epsilon    = epsilon

    def compile(self, shape):
        super().compile(shape)
        self._velocity  = np.zeros(self._shape)
        self._grad_squared  = np.zeros(self._shape)

    def is_compiled(self):
        velocity_ok = (self._velocity is not None 
                       and self._velocity.shape == self._shape)
        grad_squared_ok = (self._grad_squared is not None 
                           and self._grad_squared.shape == self._shape)
        return (super().is_compiled
                and velocity_ok
                and grad_squared_ok)

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

        # momentum
        self._correction_m *= 1.0 - self.friction
        np.multiply(1.0-self.friction, self._velocity, out=self._velocity)
        np.add(self._velocity, grad, out=self._velocity)

        # root mean squared
        self._correction_rms *= 1.0 - self.decay
        np.square(grad, out=self._grad)
        np.multiply(1.0 - self.decay, self._grad_squared, out=self._grad_squared)
        np.add(self._grad_squared, self._grad, out=self._grad_squared)

        np.multiply(self.decay / (1.0 - self._correction_rms), self._grad_squared, out=self._grad)
        np.sqrt(self._grad, out=self._grad)
        np.add(self._grad, self.epsilon, out=self._grad)

        np.divide(self._velocity, self._grad, out=self._grad)
        np.multiply(step * self.friction / (1.0 - self._correction_m), self._grad, out=self._grad)
        return self._readonly_grad

