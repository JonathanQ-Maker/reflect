from reflect.layers.absrtact_layer import AbstractLayer
import numpy as np

class Dense(AbstractLayer):

    output_size = None
    output_shape = None

    input_size = None
    input_shape = None
    input = None

    weight = None
    weight_type = None
    dldw = None

    bias = None
    dldb = None

    regularizer = None


    def __init__(self, input_size = 1, output_size = 1, batch_size = 1, weight_type = "he", 
                 regularizer=None):
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

        self.name = f"Dense {self.output_size}"

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
        Input instance will be kept and expected not to be modified between forward and backward pass
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

    def __str__(self):
        return self.attribute_to_str()

    def attribute_to_str(self):
        return (super().attribute_to_str()
        + f"output size:    {self.output_size}\n"
        + f"output_shape:   {self.output_shape}\n"
        + f"input size:     {self.input_size}\n"
        + f"input_shape:    {self.input_shape}\n"
        + f"weight init:    {self.weight_type}\n"
        + f"max weight:     {self.weight.max()}\n"
        + f"min weight:     {self.weight.min()}\n"
        + f"weight std:     {np.std(self.weight)}\n"
        + f"weight mean:    {np.mean(self.weight)}\n")
