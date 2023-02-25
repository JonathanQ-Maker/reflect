from reflect import np
from abc import abstractmethod
from reflect.compiled_object import CompiledObject

class AbstractConstraint(CompiledObject):

    @abstractmethod
    def constrain(self, weight):
        """
        Applies constraints inplace on weight
        """
        pass

    def compile(self, shape: tuple):
        pass
