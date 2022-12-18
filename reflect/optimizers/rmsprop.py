from reflect.optimizers.abtsract_optimizer import AbstractOptimizer
from reflect import np

class RMSprop(AbstractOptimizer):
    """
    Root Mean Squared Propagation Optimizer

    grad_sqrd_t = (1 - decay) * grad_sqrd_(t-1) + grad**2
    unbiased = decay / (1 - (1 - decay)**t) * grad_sqrd_t
    grad_t = step * grad / (sqrt(unbiased) + epsilon)


    NOTE: 
        real grad_sqrd_t equation is 

            grad_sqrd_t = (1 - decay) * grad_sqrd_(t-1) + decay * grad**2

        but was reduced to 

            grad_sqrd_t = (1 - decay) * grad_sqrd_(t-1) + grad**2

        such that it is more memory efficient.
        The first equation is the same as the second if the
        first equation was scaled by a factor of 1 / decay.
        See:
        https://ai.stackexchange.com/questions/25152/how-are-these-equations-of-sgd-with-momentum-equivalent
    """

    _grad           = None  # 
    _readonly_grad  = None
    _grad_squared   = None  # running average gradient squared
    decay           = 0.0   # percent to decay/remove of old gradients, [0, 1)
    epsilon         = 1e-7  # numerical stability coefficient, (0, inf)
    _correction     = 1.0   # correction term for unbiased/early gradients

    @property
    def grad(self):
        """
        calculated gradient
        """
        return self._readonly_grad

    def __init__(self, decay=0.01, epsilon=1e-7):
        self.decay      = decay
        self.epsilon    = epsilon

    def compile(self, shape):
        super().compile(shape)
        self._grad_squared  = np.zeros(self._shape)
        self._grad          = np.zeros(self._shape)
        self._readonly_grad = self._grad.view()
        self._readonly_grad.flags.writeable = False

    def is_compiled(self):
        grad_squared_ok = (self._grad_squared is not None 
                           and self._grad_squared.shape == self._shape)
        grad_ok = (self._grad is not None 
                       and self._grad.shape == self._shape
                       and self._readonly_grad is not None
                       and self._readonly_grad.shape == self._shape)
        return (super().is_compiled
                and grad_ok)

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

        self._correction *= 1.0 - self.decay
        np.square(grad, out=self._grad)
        np.multiply(1.0 - self.decay, self._grad_squared, out=self._grad_squared)
        np.add(self._grad_squared, self._grad, out=self._grad_squared)

        np.multiply(self.decay / (1.0 - self._correction), self._grad_squared, out=self._grad)
        np.sqrt(self._grad, out=self._grad)
        np.add(self._grad, self.epsilon, out=self._grad)
        np.divide(grad, self._grad, out=self._grad)
        np.multiply(step, self._grad, out=self._grad)
        return self._readonly_grad

    
