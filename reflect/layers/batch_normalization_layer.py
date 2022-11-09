from __future__ import annotations
from reflect.layers.parametric_layer import ParametricLayer
from reflect import np
import time

class BatchNorm(ParametricLayer):

    input = None

    dldg = None     # gradient of loss with respect to gamma (output std)
    dldb = None     # gradient of loss with respect to beta (output mean)

    axis = None     # axis to normalized over
    momentum = None # momentum for expoential moving averaging
    epsilon = None  # numerical stabilizer
    
    # intermediate terms
    std = None      # input std along axis
    mean = None     # input mean along axis
    offset = None
    factor = None
    residual = None
    residual_mean = None
    n = None


    param_shape = None  # shared shape between gamma and beta
    approx_dldx = None


    def __init__(self, input_size = 1, batch_size = 1, momentum=0.9, epsilon=1e-5, approx_dldx=False):
        super().__init__(input_size, input_size, batch_size)
        self.momentum = momentum
        self.epsilon = epsilon
        self.approx_dldx = approx_dldx

    def compile(self, gen_param=True):
        self.output_size = self.input_size
        super().compile(gen_param)
        self.axis = tuple(i for i in range(len(self.input_shape)-1))
        self.param_shape = (self.output_shape[-1], )

        # compile intermediate terms
        self.std = np.zeros(shape=self.param_shape)
        self.mean = np.zeros(shape=self.param_shape)
        self.offset = np.zeros(shape=self.param_shape)
        self.factor = np.zeros(shape=self.param_shape)
        self.residual = np.zeros(shape=self.input_shape)
        self.residual_mean = np.zeros(shape=self.param_shape)
        self.n = np.prod(self.input_shape[:-1])

        # compile gradient
        self.dldg = np.zeros(shape=self.param_shape)
        self.dldb = np.zeros(shape=self.param_shape)

        self.name = f"BatchNorm"
        if (gen_param):
            self.apply_param(self.create_param())

    def is_compiled(self):
        axis_ok = (self.input_shape is not None 
                   and self.axis == tuple(i for i in range(len(self.input_shape)-1)))
        gamma_shape_match = (self.output_shape is not None 
                             and self.param_shape == (self.output_shape[-1], ))

        # intermediate terms
        std_ok = self.std is not None and self.std.shape == self.param_shape
        mean_ok = self.mean is not None and self.mean.shape == self.param_shape
        offset_ok = self.offset is not None and self.offset.shape == self.param_shape
        factor_ok = self.factor is not None and self.factor.shape == self.param_shape
        dldg_ok = self.dldg is not None and self.dldg.shape == self.param_shape
        dldb_ok = self.dldb is not None and self.dldb.shape == self.param_shape
        residual_ok = (self.residual is not None 
                       and self.residual.shape == self.input_shape)
        residual_mean_ok = (self.residual_mean is not None 
                            and self.residual_mean.shape == self.param_shape)
        momentum_ok = (self.momentum is not None 
                       and self.momentum <= 1 
                       and self.momentum > 0)
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
                and momentum_ok)

    def create_param(self):
        super().create_param()
        param = BatchNormParam()
        param.gamma = np.ones(self.param_shape)
        param.beta = np.zeros(self.param_shape)
        param.momentum = self.momentum
        param.epsilon = self.epsilon

        param.std = np.ones(self.param_shape)
        param.mean = np.zeros(self.param_shape)
        return param

    def param_compatible(self, param: BatchNormParam):
        gamma_ok = param.gamma is not None and param.gamma.shape == self.param_shape
        beta_ok = param.beta is not None and param.beta.shape == self.param_shape
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
        self.input = X
        std = self.param.std
        mean = self.param.mean
        if (training):
            std = self.std
            mean = self.mean
            
            start = time.time()
            np.std(self.input, axis=self.axis, out=self.std)
            delta = time.time() - start
            np.mean(self.input, axis=self.axis, out=self.mean)
            
            # calc expoential moving average for test time statistics
            
            # running std
            np.multiply(self.param.momentum, self.param.std, out=self.param.std)
            np.multiply(1 - self.param.momentum, std, out=self.factor)
            np.add(self.param.std, self.factor, out=self.param.std)

            # running mean
            np.multiply(self.param.momentum, self.param.mean, out=self.param.mean)
            np.multiply(1 - self.param.momentum, mean, out=self.factor)
            np.add(self.param.mean, self.factor, out=self.param.mean)

        np.add(std, self.param.epsilon, out=self.std)

        np.divide(self.param.gamma, self.std, out=self.factor)
        np.divide(self.param.beta, self.factor, out=self.offset)
        np.subtract(mean, self.offset, out=self.offset)
        np.subtract(self.input, self.offset, out=self.residual)
        return np.multiply(self.residual, self.factor, out=self.output)

        

    def backprop(self, dldz):
        """
        approx: approximate dldx, good when n is large where n = prod(input_shape[-1])
        return: dldx, gradient of loss with respect to input

        Make copy of dldg, dldb, dldx if intended to be modified
        """

        # dldb
        np.sum(dldz, axis=self.axis, out=self.dldb)

        if (self.approx_dldx):
            # approximate dldx
            np.multiply(self.factor, dldz, out=self.dldx)

            # dldg
            np.subtract(self.input, self.mean, self.residual)
            np.multiply(self.residual, dldz, out=self.residual)
            np.sum(self.residual, axis=self.axis, out=self.offset)
            np.divide(self.offset, self.std, out=self.dldg)
            return self.dldx

        # dldx
        np.subtract(self.input, self.mean, out=self.residual)
        np.sum(self.residual, axis=self.axis, out=self.residual_mean)
        np.divide(self.residual_mean, self.n, out=self.residual_mean)

        np.divide(self.residual_mean, self.std, out=self.offset)
        np.subtract(1, self.offset, out=self.offset)
        np.multiply(dldz, self.offset, out=self.dldx)
        np.sum(self.dldx, axis=self.axis, out=self.mean)

        np.multiply(dldz, self.residual, out=self.dldx)
        np.sum(self.dldx, axis=self.axis, out=self.offset)
        np.divide(self.offset, self.std, out=self.dldg)     # dldg
        np.divide(self.dldg, self.std, out=self.offset)

        np.add(self.residual, self.residual_mean, self.dldx)
        np.multiply(self.offset, self.dldx, out=self.dldx)

        np.add(self.dldx, self.mean, out=self.dldx)
        np.divide(self.dldx, self.n, out=self.dldx)
        np.subtract(dldz, self.dldx, out=self.dldx)
        np.multiply(self.factor, self.dldx, out=self.dldx)
  
        return self.dldx

    def apply_grad(self, step, dldg=None, dldb=None):
        if (dldg is None):
            dldg = self.dldg
        if (dldb is None):
            dldb = self.dldb
        np.add(self.param.gamma, step * dldg, out=self.param.gamma)     # gamma update
        np.add(self.param.beta, step * dldb, out=self.param.beta)       # beta update

    def __str__(self):
        return self.attribute_to_str()

    def attribute_to_str(self):
        return (super().attribute_to_str()
        + f"max gamma:      {self.param.gamma.max()}\n"
        + f"min gamma:      {self.param.gamma.min()}\n"
        + f"max beta:       {self.param.beta.max()}\n"
        + f"min beta:       {self.param.beta.min()}\n"
        + f"momentum:       {self.param.momentum}")







class BatchNormParam():
    gamma = None    # std of output
    beta = None     # mean of output
    epsilon = None  # numerical stabilizer
    momentum = None # momentum for expoential moving averaging

    # expoential moving average for test time statistics
    std = None      # test time input std
    mean = None     # test time input mean