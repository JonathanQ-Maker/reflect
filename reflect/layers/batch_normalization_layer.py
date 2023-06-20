from __future__ import annotations
from reflect.layers.parametric_layer import ParametricLayer, Parameter
from reflect.layers.cached_layer import CachedLayer, LayerCache
from reflect import np
from reflect.optimizers import Adam

class BatchNorm(CachedLayer, ParametricLayer):
    """
    Batch Normalization layer
    normalize over all except last axis

    Shape:
        input:  (batch size, ...)
        output: (batch size, ...)
    """

    _dldg           = None # gradient of loss with respect to gamma also output std
    _dldb           = None # gradient of loss with respect to beta also output mean
    _readonly_dldg  = None
    _readonly_dldb  = None

    _axis           = None # axis to normalized over
    momentum        = 0.9  # momentum for expoential moving averaging, (0, 1]
    epsilon         = 1e-5 # numerical stabilizer, (0, inf)
    gamma_optimizer = None
    bias_optimizer  = None
    
    # intermediate terms, name not nessesarily accurate. 
    # maybe more accurate to call intermediate buffers
    _offset         = None
    _residual       = None
    _residual_mean  = None
    _n              = None


    _param_shape    = None # shared shape between gamma and beta
    _approx_dldx    = None # bool, use approx dldx. Reduceses backprop complexity and more accurate n -> inf



    @property
    def dldg(self):
        return self._readonly_dldg

    @property
    def dldb(self):
        return self._readonly_dldb

    @property
    def axis(self):
        return self._axis

    @property
    def total_params(self):
        return self.param.gamma.size + self.param.beta.size


    def __init__(self, 
                 momentum           = 0.9, 
                 epsilon            = 1e-5, 
                 approx_dldx        = False,
                 gamma_optimizer    = None,
                 bias_optimizer     = None):

        super().__init__()
        self.momentum           = momentum
        self.epsilon            = epsilon
        self._approx_dldx       = approx_dldx
        self.gamma_optimizer    = gamma_optimizer
        self.bias_optimizer     = bias_optimizer
        if gamma_optimizer is None:
            self.gamma_optimizer    = Adam()
        if bias_optimizer is None:
            self.bias_optimizer     = Adam()

    def compile(self, input_size, batch_size=1, gen_param=True):
        self._output_size = input_size
        super().compile(input_size, batch_size, gen_param)
        self._axis = tuple(i for i in range(len(self._input_shape)-1))
        self._param_shape = (self._output_shape[-1], )

        # compile intermediate terms
        self._offset        = np.zeros(shape=self._param_shape)
        self._residual      = np.zeros(shape=self._input_shape)
        self._residual_mean = np.zeros(shape=self._param_shape)
        self._n             = np.prod(self._input_shape[:-1])

        # compile gradient
        self._dldg          = np.zeros(shape=self._param_shape)
        self._dldb          = np.zeros(shape=self._param_shape)
        self._readonly_dldg = self._dldg.view()
        self._readonly_dldb = self._dldb.view()
        self._readonly_dldg.flags.writeable = False
        self._readonly_dldb.flags.writeable = False

        # compile optimizers
        self.gamma_optimizer.compile(self._param_shape)
        self.bias_optimizer.compile(self._param_shape)

        self.name = "BatchNorm"
        if (gen_param):
            self.apply_param(self.create_param())

        self._cache = self.create_cache()

    def is_compiled(self):
        axis_ok = (self._input_shape is not None 
                   and self._axis == tuple(i for i in range(len(self._input_shape)-1)))
        gamma_shape_match = (self._output_shape is not None 
                             and self._param_shape == (self._output_shape[-1], ))

        # intermediate terms
        offset_ok   = self._offset is not None and self._offset.shape == self._param_shape
        dldg_ok     = self._dldg is not None and self._dldg.shape == self._param_shape
        dldb_ok     = self._dldb is not None and self._dldb.shape == self._param_shape
        residual_ok = (self._residual is not None 
                       and self._residual.shape == self._input_shape)
        residual_mean_ok = (self._residual_mean is not None 
                            and self._residual_mean.shape == self._param_shape)
        gamma_optimizer_ok = (self.gamma_optimizer is not None
                         and self.gamma_optimizer.is_compiled()
                         and self.gamma_optimizer.shape == self._param_shape)

        bias_optimizer_ok = (self.bias_optimizer is not None
                         and self.bias_optimizer.is_compiled()
                         and self.bias_optimizer.shape == self._param_shape)
        return (super().is_compiled() 
                and axis_ok 
                and gamma_shape_match
                and offset_ok
                and dldg_ok
                and dldb_ok
                and residual_ok
                and residual_mean_ok
                and gamma_optimizer_ok
                and bias_optimizer_ok)

    def create_param(self):
        super().create_param()
        param = BatchNormParam()
        param.set_weight("gamma", np.ones(self._param_shape))   # std of output
        param.set_weight("beta", np.zeros(self._param_shape))   # mean of output
        param.set_weight("momentum", self.momentum)             # momentum for expoential moving averaging
        param.set_weight("epsilon", self.epsilon)               # numerical stabilizer

        param.set_weight("std", np.ones(self._param_shape))     # test time input std
        param.set_weight("mean", np.zeros(self._param_shape))   # test time input mean
        return param

    def param_compatible(self, param: BatchNormParam):
        gamma_ok = param.gamma is not None and param.gamma.shape == self._param_shape
        beta_ok = param.beta is not None and param.beta.shape == self._param_shape
        momentum_ok = (param.momentum is not None 
                       and param.momentum <= 1 
                       and param.momentum > 0)
        return gamma_ok and beta_ok and momentum_ok

    def create_cache(self):
        cache = BatchNormCache()
        cache._owner = self
        cache._X        = np.zeros(self._input_shape)
        cache._factor   = np.zeros(self._param_shape)
        cache._mean     = np.zeros(self._param_shape)
        cache._std      = np.zeros(self._param_shape)
        return cache
    
    def forward(self, X, training=True, out_cache: BatchNormCache=None):
        """
        forward pass with input, write to out_cache

        Args:
            X:  input

            out_cache:  
                cache object to be filled with forward cache for backprop, 
                if None writes to default cache

            training:
                (bool) updates std and mean statistics

        Returns: 
            output
        """

        if (out_cache is None):
            out_cache = self._cache
        
        if (out_cache._owner is not self):
            raise ValueError("out_cache does not belong to this layer")

        np.copyto(out_cache._X, X)
        std = self.param.std
        mean = self.param.mean
        if (training):
            std     = out_cache._std
            mean    = out_cache._mean
            
            np.std(out_cache._X, axis=self._axis, out=out_cache._std)
            np.mean(out_cache._X, axis=self._axis, out=out_cache._mean)
            
            # calc expoential moving average for test time statistics
            
            # running std
            np.multiply(self.param.momentum, self.param.std, out=self.param.std)
            np.multiply(1 - self.param.momentum, std, out=out_cache._factor)
            np.add(self.param.std, out_cache._factor, out=self.param.std)

            # running mean
            np.multiply(self.param.momentum, self.param.mean, out=self.param.mean)
            np.multiply(1 - self.param.momentum, mean, out=out_cache._factor)
            np.add(self.param.mean, out_cache._factor, out=self.param.mean)

        np.add(std, self.param.epsilon, out=out_cache._std)

        np.divide(self.param.gamma, out_cache._std, out=out_cache._factor)
        np.divide(self.param.beta, out_cache._factor, out=self._offset)
        np.subtract(mean, self._offset, out=self._offset)
        np.subtract(out_cache._X, self._offset, out=self._residual)
        return np.multiply(self._residual, out_cache._factor, out=self._output)

        
    # # batch normalization gradient for x of shape (n, d) and (b, h, w, c)
    # # used as reference for debugging
    #def dfdx(x, gamma, beta, dout, eps=1e-2):
    #    axis = tuple(i for i in range(len(x.shape)-1))
    #    m = np.mean(x, axis=axis)
    #    s = np.std(x, axis=axis) + eps
    #    n = np.prod(x.shape[:-1])
    #    res = x - m
    #    res_mean = np.sum(res, axis=axis) / n
    #    dx = (gamma / s) * (dout \
    #    - (np.sum(dout * (1 - res_mean/s), axis=axis) \
    #    + np.sum(dout * res, axis=axis) * (res + res_mean)/(s**2))/n)
    #    return dx

    def backprop(self, dldz, cache: BatchNormCache=None):
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

        # dldb
        np.sum(dldz, axis=self._axis, out=self._dldb)

        if (self._approx_dldx):
            # approximate dldx
            np.multiply(cache._factor, dldz, out=self._dldx)

            # dldg
            np.subtract(cache._X, cache._mean, self._residual)
            np.multiply(self._residual, dldz, out=self._residual)
            np.sum(self._residual, axis=self._axis, out=self._offset)
            np.divide(self._offset, cache._std, out=self._dldg)
            return self._dldx

        # dldx
        np.subtract(cache._X, cache._mean, out=self._residual)
        np.sum(self._residual, axis=self._axis, out=self._residual_mean)
        np.divide(self._residual_mean, self._n, out=self._residual_mean)

        np.divide(self._residual_mean, cache._std, out=self._offset)
        np.subtract(1, self._offset, out=self._offset)
        np.multiply(dldz, self._offset, out=self._dldx)
        np.sum(self._dldx, axis=self._axis, out=cache._mean)

        np.multiply(dldz, self._residual, out=self._dldx)
        np.sum(self._dldx, axis=self._axis, out=self._offset)
        np.divide(self._offset, cache._std, out=self._dldg)     # dldg
        np.divide(self._dldg, cache._std, out=self._offset)

        np.add(self._residual, self._residual_mean, self._dldx)
        np.multiply(self._offset, self._dldx, out=self._dldx)

        np.add(self._dldx, cache._mean, out=self._dldx)
        np.divide(self._dldx, self._n, out=self._dldx)
        np.subtract(dldz, self._dldx, out=self._dldx)
        np.multiply(cache._factor, self._dldx, out=self._dldx)
  
        return self._dldx

    def apply_grad(self, step, dldg=None, dldb=None):
        """
        Applies layer gradients
        
        NOTE: None gradients default to gradient computed in backprop()

        Args:
            step: gradient step size
            dldg: gradient of loss with respect to gamma
            dldb: gradient of loss with respect to bias
        """

        step = step / self._batch_size

        if (dldg is None):
            dldg = self._dldg
        if (dldb is None):
            dldb = self._dldb

        # gamma update
        np.subtract(self.param.gamma, self.gamma_optimizer.gradient(step, dldg), 
               out=self.param.gamma)

        # beta update
        np.subtract(self.param.beta, self.bias_optimizer.gradient(step, dldb), 
               out=self.param.beta)

    def attribute_to_str(self):
        return (super().attribute_to_str()
        + f"max gamma:      {self.param.gamma.max()}\n"
        + f"min gamma:      {self.param.gamma.min()}\n"
        + f"std gamma:      {self.param.gamma.std()}\n"
        + f"mean gamma:     {self.param.gamma.mean()}\n"
        + f"momentum:       {self.param.momentum}")







class BatchNormParam(Parameter):

    @property
    def gamma(self):
        # std of output
        return self.get_weight("gamma")
    
    @property
    def beta(self):
        # mean of output
        return self.get_weight("beta")

    @property
    def epsilon(self):
        # numerical stabilizer
        return self.get_weight("epsilon")
    
    @property
    def momentum(self):
        # momentum for expoential moving averaging
        return self.get_weight("momentum")

    # expoential moving average for test time statistics
    @property
    def std(self):
        # test time input std
        return self.get_weight("std")

    @property
    def mean(self):
        # test time input mean
        return self.get_weight("mean")

class BatchNormCache(LayerCache):
    _X      = None
    _factor = None
    _mean   = None
    _std    = None