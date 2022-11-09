from reflect.layers.absrtact_layer import AbstractLayer
from abc import abstractmethod

class ParametricLayer(AbstractLayer):
    param = None

    def __init__(self, input_size, output_size, batch_size):
        super().__init__(input_size, output_size, batch_size)
    
    @abstractmethod
    def create_param(self):
        """
        Creates a new object with parameters.
        Must be compiled
        """
        assert self.is_compiled(), "Cannot create param without compiling"
        pass

    def apply_param(self, param):
        if (self.param_compatible(param)):
            self.param = param
        else:
            raise AttributeError("Parameter to be applied is not compatible to this layer")

    @abstractmethod
    def param_compatible(self, param):
        """
        Check if parameter applied is compatible to this layer
        """
        pass

    def compile(self, gen_param=True):
        super().compile()

    def apply_grad(self, step):
        """
        Applies graident to params.

        Note: gradients over batch axis is summed
        """
        pass