from reflect.layers.transposed_convolution_2d import TransposedConv2D, TransposedConv2DParam
from reflect import np

class TransposedConvSN2D(TransposedConv2D):
    """
    Spectral Normalized Transposed Convolve 2D layer.

    Enforces Lipschitz continuity using Spectral Normalization
    See: https://arxiv.org/abs/1802.05957
    """

    lip_const       = 1

    _u              = None
    _v              = None
    _kernel_2d_view = None

    def __init__(self, 
                 filter_size        = (1, 1),
                 kernels            = 1,
                 strides            = (1, 1),
                 lip_const          = 1,
                 weight_type        = "he",
                 kernel_reg         = None,
                 bias_reg           = None,
                 kernel_optimizer   = None,
                 bias_optimizer     = None,
                 bias_constraint    = None):
        super().__init__(filter_size, 
                         kernels, 
                         strides, 
                         weight_type, 
                         kernel_reg, 
                         bias_reg, 
                         kernel_optimizer, 
                         bias_optimizer, 
                         None, bias_constraint)
        self.lip_const = lip_const

    def compile(self, input_size, batch_size=1, gen_param=True):
        super().compile(input_size, batch_size, gen_param)

        self._u     = np.random.normal(size=self.kernels)
        self._v     = np.zeros(self._kernel_2d_view.shape[1])
        self._Kv    = np.zeros(self.kernels)

    def apply_param(self, param: TransposedConv2D):
        super().apply_param(param)
        self._kernel_2d_view = self.param.kernel.view()
        self._kernel_2d_view.shape = (self.kernels, -1)

    def apply_grad(self, step, dldk=None, dldb=None):
        super().apply_grad(step, dldk, dldb)

        # Spectral Normalize using power iteration method
        np.dot(self._kernel_2d_view.T, self._u, out=self._v)
        np.divide(self._v, np.linalg.norm(self._v), out=self._v)

        np.dot(self._kernel_2d_view, self._v, out=self._Kv)
        sigma = np.dot(self._u.T, self._Kv)
        np.divide(self._Kv, np.linalg.norm(self._Kv), out=self._u)
        
        np.multiply(self.param.kernel, self.lip_const/sigma, out=self.param.kernel)