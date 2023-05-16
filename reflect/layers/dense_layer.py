from __future__ import annotations
from reflect.layers.parametric_layer import ParametricLayer
from reflect.layers.cached_layer import CachedLayer, LayerCache
from reflect.optimizers import Adam
from reflect.utils.misc import to_tuple
from reflect import np

class Dense(CachedLayer, ParametricLayer):

    """
    Dense layer

    Shape:
        input:  (batch size, input size)
        output: (batch size, output size)
    """

    _dldw               = None # gradient of loss with respect to weights
    _dldb               = None # gradient of loss with respect to bias
    _readonly_dldw      = None # read only dldw view
    _readonly_dldb      = None # read only dldw view

    _weight_shape       = None # (num input, num output)    NOTE: transposed
    weight_type         = None # weight initialization type
    weight_reg          = None # weight regularizer
    bias_reg            = None # bias regularizer

    weight_optimizer    = None
    bias_optimizer      = None

    weight_constraint   = None
    bias_constraint     = None

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
    def units(self):
        return self._output_size

    @units.setter
    def units(self, units):
        self._output_size = units

    @property
    def total_params(self):
        return self.param.weight.size + self.param.bias.size

    def __init__(self, 
                 units,
                 weight_type        = "he", 
                 weight_reg         = None,
                 bias_reg           = None,
                 weight_optimizer   = None, 
                 bias_optimizer     = None,
                 weight_constraint  = None,
                 bias_constraint    = None):


        super().__init__()
        self._output_size       = units
        self.weight_type        = weight_type
        self.weight_reg         = weight_reg
        self.bias_reg           = bias_reg
        self.weight_optimizer   = weight_optimizer
        self.bias_optimizer     = bias_optimizer
        self.weight_constraint  = weight_constraint
        self.bias_constraint    = bias_constraint

        # init optimizers
        if weight_optimizer is None:
            self.weight_optimizer   = Adam()
        if bias_optimizer is None:
            self.bias_optimizer     = Adam()

        # init cache
        self._cache = DenseCache()

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
        if (self.weight_reg is not None):
            self.weight_reg.compile(self._weight_shape)
        if (self.bias_reg is not None):
            self.bias_reg.compile(to_tuple(self._output_size))

        # compile optimizers
        self.weight_optimizer.compile(self._weight_shape)
        self.bias_optimizer.compile(self._output_size)

        # compile constraints
        if (self.weight_constraint is not None):
            self.weight_constraint.compile(self._weight_shape)
        if (self.bias_constraint is not None):
            self.bias_constraint.compile(to_tuple(self._output_size))

        self.name = f"Dense {self._output_size}"
        if (gen_param):
            self.apply_param(self.create_param())

        self._cache = self.create_cache()

    def is_compiled(self):
        weight_shape_match = self._weight_shape == (self._input_size, self._output_size)
        dldw_ok = self._dldw is not None and self._dldw.shape == self._weight_shape
        dldb_ok = self._dldb is not None and self._dldb.shape[0] == self._output_size

        regularizer_ok = True
        if (self.weight_reg is not None):
            regularizer_ok = self.weight_reg.is_compiled()

        weight_optimizer_ok = (self.weight_optimizer is not None
                         and self.weight_optimizer.is_compiled()
                         and self.weight_optimizer.shape == self._weight_shape)

        bias_optimizer_ok = (self.bias_optimizer is not None
                         and self.bias_optimizer.is_compiled()
                         and self.bias_optimizer.shape == to_tuple(self._output_size))

        weight_constraint_ok = True
        if (self.weight_constraint is not None):
            weight_constraint_ok = self.weight_constraint.is_compiled()
        bias_constraint_ok = True
        if (self.bias_constraint is not None):
            bias_constraint_ok = self.bias_constraint.is_compiled()
        constraints_ok = weight_constraint_ok and bias_constraint_ok

        return (super().is_compiled() 
                and weight_shape_match 
                and regularizer_ok 
                and dldw_ok 
                and dldb_ok
                and weight_optimizer_ok
                and bias_optimizer_ok
                and constraints_ok)

    def init_weight(self, param, type):
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
            scale = np.sqrt(2.0 / (self._input_size + self._output_size)) # Xavier init
        elif (type == "he"):
            scale = np.sqrt(2.0 / self._input_size) # he init, for relus
        elif (type == "xavier_uniform"):
            scale = np.sqrt(6.0 / (self._input_size + self._output_size))
            param.weight = np.random.uniform(low=-scale, high=scale, size=self._weight_shape)
            return
        else:
            raise ValueError(f'no such weight type "{type}"')



        param.weight = np.random.normal(loc=0, scale=scale, size=self._weight_shape)

    def create_param(self):
        super().create_param()
        param = DenseParam()
        self.init_weight(param, self.weight_type)
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

    def create_cache(self):
        """
        Create and return empty cache

        Return:
            cache
        """

        cache = DenseCache()
        cache._owner    = self
        cache._X        = np.zeros(self._input_shape)
        cache._weight   = np.zeros(shape=self._weight_shape)
        return cache
    
    def forward(self, X, out_cache: DenseCache=None):
        """
        forward pass with input, write to out_cache

        Args:
            X:  input

            out_cache:  
                cache object to be filled with forward cache for backprop, 
                if None writes to default cache

        Returns: 
            output
        """
        
        if (out_cache is None):
            out_cache = self._cache
        
        if (out_cache._owner is not self):
            raise ValueError("out_cache does not belong to this layer")

        np.copyto(out_cache._X, X)
        np.copyto(out_cache._weight, self.param.weight)

        np.dot(X, self.param.weight, out=self._output)
        np.add(self._output, self.param.bias, out=self._output)
        return self._readonly_output

    def backprop(self, dldz, cache: DenseCache=None):
        """
        backward pass to compute the gradients

        Args:
            dldz:   
                gradient of loss with respect to output
            cache:  
                cache from forward() to use for backprop,
                if None default cache will be used for backprop

        Returns: 
            dldx: gradient of loss with respect to input
        """
        
        if (cache is None):
            cache = self._cache
        
        if (cache._owner is not self):
            raise ValueError("cache does not belong to this layer")

        np.dot(cache._X.T, dldz, out=self._dldw)
        np.sum(dldz, axis=0, out=self._dldb)
        if (self.weight_reg is not None): 
            np.add(self._dldw, self.weight_reg.gradient(self.param.weight), out=self._dldw)
        if (self.bias_reg is not None):
            np.add(self._dldb, self.bias_reg.gradient(self.param.bias), out=self._dldb)
        np.dot(dldz, cache._weight.T, out=self._dldx)
        return self._readonly_dldx

    def apply_grad(self, step, dldw=None, dldb=None):
        """
        Applies layer gradient

        NOTE: None gradients default to gradient computed in backprop()

        Args:
            step: gradient step size
            dldw: gradient of loss with respect to weight
            dldb: gradient of loss with respect to bias
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
        if (self.weight_constraint is not None):
            self.weight_constraint.constrain(self.param.weight)


        # bias update
        np.subtract(self.param.bias, self.bias_optimizer.gradient(step, dldb), 
               out=self.param.bias)
        if (self.bias_constraint is not None):
            self.bias_constraint.constrain(self.param.bias)


    def attribute_to_str(self):
        return (super().attribute_to_str()
        + f"weight init:    {self.weight_type}\n"
        + f"max weight:     {self.param.weight.max()}\n"
        + f"min weight:     {self.param.weight.min()}\n"
        + f"weight std:     {np.std(self.param.weight)}\n"
        + f"weight mean:    {np.mean(self.param.weight)}\n")





class DenseParam():
    weight  = None
    bias    = None

class DenseCache(LayerCache):
    _X      = None
    _weight = None
