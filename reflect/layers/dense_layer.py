from reflect.layers.parametric_layer import ParametricLayer
import numpy as np
import copy

class Dense(ParametricLayer):

    input = None

    dldw = None
    dldb = None

    weight_shape = None
    weight_type = None
    regularizer = None


    def __init__(self, input_size = 1, output_size = 1, batch_size = 1, weight_type = "he", 
                 regularizer=None):
        super().__init__(input_size, output_size, batch_size)
        self.weight_type  = weight_type
        self.regularizer  = regularizer

    def compile(self, gen_param=True):
        super().compile(gen_param)
        self.weight_shape = (self.input_size, self.output_size)

        # compile gradient
        self.dldw = np.zeros(shape=self.weight_shape)
        self.dldx = np.zeros(shape=self.input_shape)
        self.dldb = np.zeros(shape=self.output_size)

        # compule regularizer
        if (self.regularizer is not None):
            self.regularizer.shape = self.weight_shape
            self.regularizer.compile()

        self.name = f"Dense {self.output_size}"
        self.apply_param(self.create_param())

    def is_compiled(self):
        dldw_ok = self.dldw is not None and self.dldw.shape == self.weight_shape
        dldb_ok = self.dldb is not None and self.dldb.shape[0] == self.output_size
        return super().is_compiled() and dldw_ok and dldb_ok
        

    def init_weight(self, param, type, weight_bias = 0):
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



        param.weight = np.random.normal(loc=weight_bias, scale=scale, size=self.weight_shape)
        param.weight_type = self.weight_type

    def create_param(self):
        super().create_param()
        param = DenseParam()
        self.init_weight(param, self.weight_type, 0)
        param.bias = np.zeros(self.output_size)
        param.regularizer = copy.deepcopy(self.regularizer)
        return param

    def param_compatible(self, param):
        bias_ok = (param.bias is not None) and param.bias.shape[0] == self.output_size
        weight_ok = (param.weight is not None) and param.weight.shape == self.weight_shape
        regularizer_ok = True
        if (param.regularizer is not None):
            regularizer_ok = (param.regularizer.shape == self.weight_shape 
                              and param.regularizer.is_compiled())

        return bias_ok and weight_ok and regularizer_ok
    
    def forward(self, X):
        """
        return: output

        Make copy of output if intended to be modified
        Input instance will be kept and expected not to be modified between forward and backward pass
        """
        self.input = X
        return np.add(np.dot(X, self.param.weight, out=self.output), self.param.bias, out=self.output)

    def backprop(self, dldz):
        """
        return: dldx, gradient of loss with respect to input

        Make copy of dldw, dldx if intended to be modified
        """
        np.dot(self.input.T, dldz, out=self.dldw)
        np.sum(dldz, axis=0, out=self.dldb)
        if (self.regularizer is not None):
            np.subtract(self.dldw, self.regularizer.gradient(self.param.weight), out=self.dldw)
        return np.dot(dldz, self.param.weight.T, out=self.dldx)

    def apply_grad(self, step, dldw=None, dldb=None):
        if (dldw is None):
            dldw = self.dldw
        if (dldb is None):
            dldb = self.dldb
        np.add(self.param.weight, step * dldw, out=self.param.weight)  # weight update
        np.add(self.param.bias, step * dldb, out=self.param.bias)      # bias update

    def __str__(self):
        return self.attribute_to_str()

    def attribute_to_str(self):
        return (super().attribute_to_str()
        + f"output size:    {self.output_size}\n"
        + f"output_shape:   {self.output_shape}\n"
        + f"input size:     {self.input_size}\n"
        + f"input_shape:    {self.input_shape}\n"
        + f"weight init:    {self.weight_type}\n"
        + f"max weight:     {self.param.weight.max()}\n"
        + f"min weight:     {self.param.weight.min()}\n"
        + f"weight std:     {np.std(self.param.weight)}\n"
        + f"weight mean:    {np.mean(self.param.weight)}\n")







class DenseParam():
    weight = None
    weight_type = None

    bias = None
    regularizer = None