from reflect.layers.absrtact_layer import AbstractLayer
from reflect import np

class Relu(AbstractLayer):

    input = None

    def __init__(self, input_size=1, batch_size=1):
        super().__init__(input_size, input_size, batch_size)


    def compile(self):
        super().compile()
        self.dldx = np.zeros(self.input_shape)
    
    def is_compiled(self):
        return super().is_compiled()

    def forward(self, X):
        self.input = X
        return np.maximum(X, 0, out=self.output)

    def backprop(self, dldz):
        return np.multiply(np.greater(self.input, 0, out=self.dldx), dldz, out=self.dldx)
        