from __future__ import annotations
from reflect.layers.parametric_layer import ParametricLayer
from reflect import np
from reflect.optimizers import Adam

class BatchNorm(ParametricLayer):
    """
    Batch Normalization layer
    normalize over all except last axis

    Shape:
        input:  (batch size, ...)
        output: (batch size, ...)
    """

    _input          = None

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
    _std            = None # input std along axis
    _mean           = None # input mean along axis
    _offset         = None
    _factor         = None
    _residual       = None
    _residual_mean  = None
    _n              = None


    _param_shape    = None # shared shape between gamma and beta
    _approx_dldx    = None # bool, use approx dldx. Reduceses backprop complexity and more accurate n -> inf



    
    @property
    def input(self):
        if (self._input is None):
            return None
        view = self._input.view()
        view.flags.writeable = False
        return view

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
        self._std           = np.zeros(shape=self._param_shape)
        self._mean          = np.zeros(shape=self._param_shape)
        self._offset        = np.zeros(shape=self._param_shape)
        self._factor        = np.zeros(shape=self._param_shape)
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

        self._name = f"BatchNorm"
        if (gen_param):
            self.apply_param(self.create_param())

    def is_compiled(self):
        axis_ok = (self._input_shape is not None 
                   and self._axis == tuple(i for i in range(len(self._input_shape)-1)))
        gamma_shape_match = (self._output_shape is not None 
                             and self._param_shape == (self._output_shape[-1], ))

        # intermediate terms
        std_ok      = self._std is not None and self._std.shape == self._param_shape
        mean_ok     = self._mean is not None and self._mean.shape == self._param_shape
        offset_ok   = self._offset is not None and self._offset.shape == self._param_shape
        factor_ok   = self._factor is not None and self._factor.shape == self._param_shape
        dldg_ok     = self._dldg is not None and self._dldg.shape == self._param_shape
        dldb_ok     = self._dldb is not None and self._dldb.shape == self._param_shape
        residual_ok = (self._residual is not None 
                       and self._residual.shape == self._input_shape)
        residual_mean_ok = (self._residual_mean is not None 
                            and self._residual_mean.shape == self._param_shape)
        momentum_ok = (self.momentum is not None 
                       and self.momentum <= 1 
                       and self.momentum > 0)
        optimizers_ok = (self.gamma_optimizer is not None 
                         and self.bias_optimizer is not None
                         and self.gamma_optimizer.is_compiled() 
                         and self.bias_optimizer.is_compiled())
        return (super().is_compiled() 
                and axis_ok 
                and gamma_shape_match
                and std_ok
                and mean_ok
                and offset_ok
                and factor_ok
                and dldg_ok
                and dldb_ok
                and residual_ok
                and residual_mean_ok
                and momentum_ok
                and optimizers_ok)

    def create_param(self):
        super().create_param()
        param = BatchNormParam()
        param.gamma     = np.ones(self._param_shape)
        param.beta      = np.zeros(self._param_shape)
        param.momentum  = self.momentum
        param.epsilon   = self.epsilon

        param.std       = np.ones(self._param_shape)
        param.mean      = np.zeros(self._param_shape)
        return param

    def param_compatible(self, param: BatchNormParam):
        gamma_ok = param.gamma is not None and param.gamma.shape == self._param_shape
        beta_ok = param.beta is not None and param.beta.shape == self._param_shape
        momentum_ok = (param.momentum is not None 
                       and param.momentum <= 1 
                       and param.momentum > 0)
        return gamma_ok and beta_ok and momentum_ok
    
    def forward(self, X, training=True):
        """
        return: output

        Make copy of output if intended to be modified
        Input instance will be kept and expected not to be modified between forward and backward pass
        """
        self._input = X
        std = self.param.std
        mean = self.param.mean
        if (training):
            std     = self._std
            mean    = self._mean
            
            np.std(self._input, axis=self._axis, out=self._std)
            np.mean(self._input, axis=self._axis, out=self._mean)
            
            # calc expoential moving average for test time statistics
            
            # running std
            np.multiply(self.param.momentum, self.param.std, out=self.param.std)
            np.multiply(1 - self.param.momentum, std, out=self._factor)
            np.add(self.param.std, self._factor, out=self.param.std)

            # running mean
            np.multiply(self.param.momentum, self.param.mean, out=self.param.mean)
            np.multiply(1 - self.param.momentum, mean, out=self._factor)
            np.add(self.param.mean, self._factor, out=self.param.mean)

        np.add(std, self.param.epsilon, out=self._std)

        np.divide(self.param.gamma, self._std, out=self._factor)
        np.divide(self.param.beta, self._factor, out=self._offset)
        np.subtract(mean, self._offset, out=self._offset)
        np.subtract(self._input, self._offset, out=self._residual)
        return np.multiply(self._residual, self._factor, out=self._output)

        
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

    def backprop(self, dldz):
        """
        approx: approximate dldx, good when n is large where n = prod(input_shape[-1])
        return: dldx, gradient of loss with respect to input

        Make copy of dldg, dldb, dldx if intended to be modified
        """

        # dldb
        np.sum(dldz, axis=self._axis, out=self._dldb)

        if (self._approx_dldx):
            # approximate dldx
            np.multiply(self._factor, dldz, out=self._dldx)

            # dldg
            np.subtract(self._input, self._mean, self._residual)
            np.multiply(self._residual, dldz, out=self._residual)
            np.sum(self._residual, axis=self._axis, out=self._offset)
            np.divide(self._offset, self._std, out=self._dldg)
            return self._dldx

        # dldx
        np.subtract(self._input, self._mean, out=self._residual)
        np.sum(self._residual, axis=self._axis, out=self._residual_mean)
        np.divide(self._residual_mean, self._n, out=self._residual_mean)

        np.divide(self._residual_mean, self._std, out=self._offset)
        np.subtract(1, self._offset, out=self._offset)
        np.multiply(dldz, self._offset, out=self._dldx)
        np.sum(self._dldx, axis=self._axis, out=self._mean)

        np.multiply(dldz, self._residual, out=self._dldx)
        np.sum(self._dldx, axis=self._axis, out=self._offset)
        np.divide(self._offset, self._std, out=self._dldg)     # dldg
        np.divide(self._dldg, self._std, out=self._offset)

        np.add(self._residual, self._residual_mean, self._dldx)
        np.multiply(self._offset, self._dldx, out=self._dldx)

        np.add(self._dldx, self._mean, out=self._dldx)
        np.divide(self._dldx, self._n, out=self._dldx)
        np.subtract(dldz, self._dldx, out=self._dldx)
        np.multiply(self._factor, self._dldx, out=self._dldx)
  
        return self._dldx

    def apply_grad(self, step, dldg=None, dldb=None):
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

    def __str__(self):
        return self.attribute_to_str()

    def attribute_to_str(self):
        return (super().attribute_to_str()
        + f"max gamma:      {self.param.gamma.max()}\n"
        + f"min gamma:      {self.param.gamma.min()}\n"
        + f"std gamma:      {self.param.gamma.std()}\n"
        + f"mean gamma:     {self.param.gamma.mean()}\n"
        + f"momentum:       {self.param.momentum}")







class BatchNormParam():
    gamma = None    # std of output
    beta = None     # mean of output
    epsilon = None  # numerical stabilizer
    momentum = None # momentum for expoential moving averaging

    # expoential moving average for test time statistics
    std = None      # test time input std
    mean = None     # test time input mean