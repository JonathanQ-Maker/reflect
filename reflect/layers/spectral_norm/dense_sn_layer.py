from __future__ import annotations
from reflect.layers.dense_layer import Dense, DenseCache
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
    _Wv         = None
    _vu         = None

    def __init__(self, 
                 units, 
                 lip_const          = 1,
                 weight_type        = "he", 
                 weight_reg         = None, 
                 bias_reg           = None, 
                 weight_optimizer   = None, 
                 bias_optimizer     = None,
                 bias_constraint    = None):
        super().__init__(units, 
                         weight_type, 
                         weight_reg, 
                         bias_reg, 
                         weight_optimizer, 
                         bias_optimizer, 
                         None, bias_constraint)
        self.lip_const = lip_const

    def create_cache(self):
        """
        Create and return empty cache

        Return:
            cache
        """

        cache = DenseSNCache()
        cache._owner    = self
        cache._X        = np.zeros(self._input_shape)
        cache._weight   = np.zeros(shape=self._weight_shape)
        cache._sn_weight= np.zeros(shape=self._weight_shape)
        cache._u        = np.random.normal(size=self._output_size)
        cache._v        = np.zeros(self._input_size)
        return cache

    def compile(self, input_size, batch_size=1, gen_param=True):
        super().compile(input_size, batch_size, gen_param)
        self.name = f"DenseSN {self._output_size} {self.lip_const}-Lipschitz"

        self._u     = np.random.normal(size=self._output_size)
        self._v     = np.zeros(self._input_size)
        self._Wv    = np.zeros(self._output_size)
        self._vu    = np.zeros(self._weight_shape)

        self.power_iteration(steps=5)

        

    def forward(self, X, out_cache: DenseSNCache = None):
        """
        forward pass with input and spectal normalized weight, write to out_cache

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
        np.copyto(out_cache._u, self._u)
        np.copyto(out_cache._v, self._v)

        # Spectral Norm
        sigma = np.dot(out_cache._u, self._Wv)
        out_cache._sigma = sigma
        np.multiply(out_cache._weight, self.lip_const/sigma, out=out_cache._sn_weight)

        # Forward
        np.dot(X, out_cache._sn_weight, out=self._output)
        np.add(self._output, self.param.bias, out=self._output)
        return self._readonly_output
    
    def backprop(self, dldz, cache: DenseSNCache = None):
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

        d = np.dot(cache._weight.flat, self._dldw.flat)
        np.outer(cache._v, cache._u, out=self._vu)
        np.divide(self._vu, cache._sigma/d, out=self._vu)
        np.subtract(self._dldw, self._vu, out=self._dldw)
        np.multiply(self._dldw, self.lip_const/cache._sigma, out=self._dldw)
        

        np.sum(dldz, axis=0, out=self._dldb)
        if (self.weight_reg is not None): 
            np.add(self._dldw, self.weight_reg.gradient(self.param.weight), out=self._dldw)
        if (self.bias_reg is not None):
            np.add(self._dldb, self.bias_reg.gradient(self.param.bias), out=self._dldb)
        np.dot(dldz, cache._sn_weight.T, out=self._dldx)
        return self._readonly_dldx
    
    def power_iteration(self, steps: int = 1):
        """
        Approximate Spectral Normalize using power iteration method

        Params:
            steps: number of iterations for approximation.
        """

        for i in range(steps):
            np.dot(self.param.weight, self._u, out=self._v)
            np.divide(self._v, np.linalg.norm(self._v), out=self._v)

            np.dot(self.param.weight.T, self._v, out=self._Wv)
            np.divide(self._Wv, np.linalg.norm(self._Wv), out=self._u)

    def apply_grad(self, step, dldw=None, dldb=None):
        super().apply_grad(step, dldw, dldb)
        self.power_iteration()


class DenseSNCache(DenseCache):
    _sn_weight  = None
    _u          = None
    _sigma      = None
    _v          = None


        

        