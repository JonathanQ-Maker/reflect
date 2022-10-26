from abc import ABC, abstractmethod
from reflect.compiled_object import CompiledObject

class AbstractLayer(CompiledObject):
    """
    Base layer class
    """

    output = None
    dldx = None         # gradient of loss with respect to input
    batch_size = None
    name = None

    def __init__(self, batch_size = 1):
        self.batch_size = batch_size

    def compile(self):
        super().compile()
        self.name = "UNKNOWN"

    def is_compiled(self):
        """
        Check if layer is up-to-date with layer arguments
        """
        return self.output is not None

    def forward(self, X):
        return 0

    def backprop(self, dldz):
        return 1

    def attribute_to_str(self):
        return (f"name:           {self.name}\n" 
        + f"batch size:     {self.batch_size}\n"
        + f"compiled:       {self.is_compiled()}\n")
