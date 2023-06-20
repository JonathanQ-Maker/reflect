from __future__ import annotations
from reflect.layers.abstract_layer import AbstractLayer
from abc import abstractmethod

class ParametricLayer(AbstractLayer):
    param: Parameter = None

    @property
    def total_params(self):
        return 0

    def __init__(self):
        super().__init__()
    
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

    def compile(self, input_size, batch_size, gen_param=True):
        super().compile(input_size, batch_size)

    @abstractmethod
    def apply_grad(self, step):
        """
        Applies graident to params.

        Note: gradients over batch axis is summed
        """
        pass

    def attribute_to_str(self):
        return (super().attribute_to_str()
                + f"params:         {self.total_params}\n")
    

class Parameter():
    _data: dict = None

    def __init__(self) -> None:
        self._data = {}

    def set_weight(self, name: str, weight):
        '''
        Sets weight to parameter. 
        '''
        self._data[name] = weight

    def get_weight(self, name: str):
        return self._data[name]
    
    def serialize(self):
        '''
        Returns shallow copy of data dict
        '''
        return self._data.copy()
    
    def populate(self, data: dict):
        self._data = data

