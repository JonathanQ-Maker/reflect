from reflect.layers.dense_layer import Dense
from reflect import np

class DenseSN(Dense):
    """
    Spectral Normalized Dense layer.

    Enforces Lipschitz continuity using Spectral Normalization
    See: https://arxiv.org/abs/1802.05957
    """

    lip_const   = 1     # Lipschitz constant to enforce
    
    _u          = None
    _v          = None

    def __init__(self, 
                 units, 
                 lip_const          = 1,
                 weight_type        = "he", 
                 weight_reg         = None, 
                 bias_reg           = None, 
                 weight_optimizer   = None, 
                 bias_optimizer     = None):
        super().__init__(units, 
                         weight_type, 
                         weight_reg, 
                         bias_reg, 
                         weight_optimizer, 
                         bias_optimizer, 
                         None, None)
        self.lip_const = lip_const

    def compile(self, input_size, batch_size=1, gen_param=True):
        super().compile(input_size, batch_size, gen_param)
        self._u     = np.random.normal(size=self._output_size)
        self._v     = np.zeros(self._input_size)
        self._Wv    = np.zeros(self._output_size)

    def apply_grad(self, step, dldw=None, dldb=None):
        super().apply_grad(step, dldw, dldb)
        
        # Spectral Normalize using power iteration method
        np.dot(self.param.weight, self._u, out=self._v)
        np.divide(self._v, np.linalg.norm(self._v), out=self._v)

        np.dot(self.param.weight.T, self._v, out=self._Wv)
        sigma = np.dot(self._u.T, self._Wv)
        np.divide(self._Wv, np.linalg.norm(self._Wv), out=self._u)
        
        np.multiply(self.param.weight, self.lip_const/sigma, out=self.param.weight)



        

        