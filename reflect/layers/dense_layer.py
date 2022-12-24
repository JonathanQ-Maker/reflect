from __future__ import annotations
from reflect.layers.parametric_layer import ParametricLayer
from reflect.optimizers import Adam
from reflect import np

class Dense(ParametricLayer):

    """
    Dense layer

    Shape:
        input:  (batch size, input size)
        output: (batch size, output size)
    """

    _input              = None

    _dldw               = None # gradient of loss with respect to weights
    _dldb               = None # gradient of loss with respect to bias
    _readonly_dldw      = None # read only dldw view
    _readonly_dldb      = None # read only dldw view

    _weight_shape       = None # (num input, num output)
    weight_type         = None # weight initialization type
    _regularizer        = None # weight regularizer

    weight_optimizer    = None
    bias_optimizer      = None

    @property
    def dldw(self):
        return self._readonly_dldw

    @property
    def dldb(self):
        return self._readonly_dldb

    @property
    def weight_shape(self):
        return self._weight_shape

    @property
    def regularizer(self):
        return self._regularizer

    @regularizer.setter
    def regularizer(self, regularizer):
        self._regularizer = regularizer

    @property
    def input(self):
        if (self._input is None):
            return None
        view = self._input.view()
        view.flags.writeable = False
        return view

    @property
    def units(self):
        return self._output_size

    @units.setter
    def units(self, units):
        self._output_size = units

    def __init__(self, 
                 units,
                 weight_type        = "he", 
                 regularizer        = None, 
                 weight_optimizer   = None, 
                 bias_optimizer     = None):


        super().__init__()
        self._output_size       = units
        self.weight_type        = weight_type
        self._regularizer       = regularizer
        self.weight_optimizer   = weight_optimizer
        self.bias_optimizer     = bias_optimizer
        if weight_optimizer is None:
            self.weight_optimizer   = Adam()
        if bias_optimizer is None:
            self.bias_optimizer     = Adam()

    def compile(self, input_size, batch_size=1, gen_param=True):
        super().compile(input_size, batch_size, gen_param)
        self._weight_shape = (self._input_size, self._output_size)

        # compile gradient
        self._dldw = np.zeros(shape=self._weight_shape)
        self._dldb = np.zeros(shape=self._output_size)
        self._readonly_dldw = self._dldw.view()
        self._readonly_dldb = self._dldb.view()
        self._readonly_dldw.flags.writeable = False
        self._readonly_dldb.flags.writeable = False

        # compile regularizer
        if (self._regularizer is not None):
            self._regularizer.compile(self._weight_shape)

        # compile optimizers
        self.weight_optimizer.compile(self._weight_shape)
        self.bias_optimizer.compile(self._output_size)

        self.name = f"Dense {self._output_size}"
        if (gen_param):
            self.apply_param(self.create_param())

    def is_compiled(self):
        weight_shape_match = self._weight_shape == (self._input_size, self._output_size)
        dldw_ok = self._dldw is not None and self._dldw.shape == self._weight_shape
        dldb_ok = self._dldb is not None and self._dldb.shape[0] == self._output_size

        regularizer_ok = True
        if (self._regularizer is not None):
            regularizer_ok = self._regularizer.is_compiled()

        optimizers_ok = (self.weight_optimizer is not None 
                         and self.bias_optimizer is not None
                         and self.weight_optimizer.is_compiled() 
                         and self.bias_optimizer.is_compiled())

        return (super().is_compiled() 
                and weight_shape_match 
                and regularizer_ok 
                and dldw_ok 
                and dldb_ok
                and optimizers_ok)
        

    def init_weight(self, param, type, weight_bias = 0):
        """
        Initialize weight in param object

        Args:
            type: weight initalization type
                [he, xavier]
            weight_bias: mean of weight
            param: param object to initalize weight to/store
        """


        scale = 1
        if  (type == "xavier"):
            scale = 1 / np.sqrt(self._input_size) # Xavier init
        elif (type == "he"):
            scale = np.sqrt(2 / self._input_size) # he init, for relus
        else:
            raise ValueError(f'no such weight type "{type}"')



        param.weight = np.random.normal(loc=weight_bias, scale=scale, size=self._weight_shape)
        param.weight_type = self.weight_type

    def create_param(self):
        super().create_param()
        param = DenseParam()
        self.init_weight(param, self.weight_type, 0)
        param.bias = np.zeros(self._output_size)
        return param

    def param_compatible(self, param: DenseParam):
        """
        Check if parameter is compatible

        Args:
            param: parameter to check

        Returns:
            is compatible
        """

        bias_ok = (param.bias is not None) and param.bias.shape[0] == self._output_size
        weight_ok = (param.weight is not None) and param.weight.shape == self._weight_shape
        return bias_ok and weight_ok
    
    def forward(self, X):
        """
        forward pass with input

        Args:
            X: input

        Returns: 
            output

        Make copy of output if intended to be modified
        Input instance will be kept and expected not to be modified between forward and backward pass
        """
        self._input = X
        return np.add(np.dot(X, self.param.weight, out=self._output), self.param.bias, out=self._output)

    def backprop(self, dldz):
        """
        backward pass to compute the gradients

        Args:
            dldz: gradient of loss with respect to output

        Returns: 
            dldx, gradient of loss with respect to input

        Note:
            expected to execute only once after forward
        """
        np.dot(self._input.T, dldz, out=self._dldw)
        np.sum(dldz, axis=0, out=self._dldb)
        if (self._regularizer != None):
            np.add(self._dldw, self._regularizer.gradient(self.param.weight), out=self._dldw)
        return np.dot(dldz, self.param.weight.T, out=self._dldx)

    def apply_grad(self, step, dldw=None, dldb=None):
        """
        Applies layer param

        Args:
            param: parameter to be applied
        """

        if (dldw is None):
            dldw = self._dldw
        if (dldb is None):
            dldb = self._dldb

        # average batch gradient
        step = step / self._batch_size

        # weight update
        np.subtract(self.param.weight, self.weight_optimizer.gradient(step, dldw), 
               out=self.param.weight)

        # bias update
        np.subtract(self.param.bias, self.bias_optimizer.gradient(step, dldb), 
               out=self.param.bias)

    def __str__(self):
        return self.attribute_to_str()

    def attribute_to_str(self):
        return (super().attribute_to_str()
        + f"weight init:    {self.weight_type}\n"
        + f"max weight:     {self.param.weight.max()}\n"
        + f"min weight:     {self.param.weight.min()}\n"
        + f"weight std:     {np.std(self.param.weight)}\n"
        + f"weight mean:    {np.mean(self.param.weight)}\n")







class DenseParam():
    weight = None
    weight_type = None

    bias = None