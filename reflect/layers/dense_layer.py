from reflect.layers.absrtact_layer import AbstractLayer
import numpy as np

class Dense(AbstractLayer):

    output_size = None
    output_shape = None

    weight = None
    weight_type = None
    dldw = None

    bias = None
    dldb = None

    input_size = None
    input_shape = None
    input = None

    regularizer = None


    def __init__(self, input_size = 1, output_size = 1, batch_size = 1, weight_type = "he", regularizer=None):
        super().__init__(batch_size)
        self.input_size     = input_size
        self.output_size    = output_size
        self.weight_type    = weight_type
        self.regularizer    = regularizer

    def compile(self):
        super().compile()
        self.init_weight(self.weight_type)
        self.bias = np.zeros(self.output_size)

        # compile output
        self.output_shape = (self.batch_size, self.output_size)
        self.output = np.zeros(shape=self.output_shape)

        # compile gradient
        self.dldw = np.zeros(shape=self.weight.shape)

        self.input_shape = (self.batch_size, self.input_size)
        self.dldx = np.zeros(shape=self.input_shape)

        self.dldb = np.zeros(shape=self.output_size)

        # compule regularizer
        if (self.regularizer != None):
            self.regularizer.shape = self.weight.shape
            self.regularizer.compile()

    def is_compiled(self):
        output_size_match = self.output_shape == (self.batch_size, self.output_size)
        bias_compiled = (self.bias is not None) and self.bias.shape[0] == self.output_size
        regularizer_ok = True
        if (self.regularizer is not None):
            regularizer_ok = self.regularizer.is_compiled()

        return super().is_compiled() and output_size_match and bias_compiled and regularizer_ok
        

    def init_weight(self, type):
        """
        Params:
            type: weight initalization type
                [he, xavier]
        """


        scale = 1
        if  (type == "xavier"):
            scale = 1 / np.sqrt(self.input_size) # Xavier init
        elif (type == "he"):
            scale = np.sqrt(2 / self.input_size) # he init, for relus



        shape = (self.input_size, self.output_size)
        bias = 0
        self.weight = np.random.normal(loc=bias, scale=scale, size=shape)

    
    def forward(self, X):
        """
        return: output

        Make copy of output if intended to be modified
        """
        self.input = X
        return np.add(np.dot(X, self.weight, out=self.output), self.bias, out=self.output)

    def backprop(self, dldz, step=None):
        """
        return: dldx, gradient of loss with respect to input

        Make copy of dldw, dldx if intended to be modified
        """
        np.dot(self.input.T, dldz, out=self.dldw)
        np.sum(dldz, axis=0, out=self.dldb)
        if (step != None):
            if (self.regularizer != None):
                np.subtract(self.dldw, self.regularizer.gradient(self.weight), out=self.dldw)
            np.add(self.weight, step * self.dldw, out=self.weight)  # weight update
            np.add(self.bias, step * self.dldb, out=self.bias)      # bias update
        return np.dot(dldz, self.weight.T, out=self.dldx)           

