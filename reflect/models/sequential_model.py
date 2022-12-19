from reflect.models.abstract_model import AbstractModel
from reflect.layers.parametric_layer import ParametricLayer

class SequentialModel(AbstractModel):
    """Sequential forward model"""


    _layers = None

    def __init__(self, *layers):
        self._layers = layers

    def compile(self):
        for layer in self._layers:
            layer.compile()

    def is_compiled(self):
        for layer in self._layers:
            if (not layer.is_compiled()):
                return False
        return True

    def forward(self, X):
        input = X
        for layer in self._layers:
            input = layer.forward(input)
        self._output = input
        return self._output

    def backprop(self, dldz):
        for i in range(len(self._layers) - 1, -1, -1):
            dldz = self._layers[i].backprop(dldz)
        self._dldx = dldz
        return self._dldx

    def apply_grad(self, step):
        for layer in self._layers:
            if (isinstance(layer, ParametricLayer)):
                layer.apply_grad(step)