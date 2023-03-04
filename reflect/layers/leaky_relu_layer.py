from __future__ import annotations
from reflect.layers.abstract_layer import AbstractLayer
from reflect.layers.cached_layer import CachedLayer, LayerCache
from reflect import np

class LeakyRelu(CachedLayer, AbstractLayer):
    """
    Leaky Rectified Linear Activation Unit (RELU)

    LeakyRelu(x) = max(0,x) + negative_slope * min(0,x)

    Shape:
        input:  (batch size, ...)
        output: (batch size, ...)
    """

    slope = 0.1
    _leakage = None

    def __init__(self, slope=0.1):
        super().__init__()
        self.slope = slope


    def compile(self, input_size, batch_size):
        self._output_size = input_size
        super().compile(input_size, batch_size)
        self._leakage = np.zeros(self._output_shape)
        self.name = f"LeakyRelu {self.slope}"

        self._cache = self.create_cache()
    
    def is_compiled(self):
        dldx_ok = self._dldx is not None and self._dldx.shape == self._input_shape
        return super().is_compiled() and dldx_ok

    def create_cache(self):
        cache = LeakyReluCache()
        cache._owner    = self
        cache._output   = np.zeros(self._output_shape)
        return cache

    def forward(self, X, out_cache: LeakyReluCache=None):
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

        np.maximum(X, 0, out=self._output)
        np.minimum(X, 0, out=self._leakage)
        np.multiply(self._leakage, self.slope, out=self._leakage)
        np.add(self._output, self._leakage, out=self._output)
        np.copyto(out_cache._output, self._output)
        return self._readonly_output

    def backprop(self, dldz, cache: LeakyReluCache=None):
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
        np.greater(cache._output, 0, out=self._dldx)
        np.less(cache._output, 0, out=self._leakage)
        np.multiply(self._leakage, self.slope, out=self._leakage)
        np.add(self._leakage, self._dldx, out=self._dldx)
        np.multiply(self._dldx, dldz, out=self._dldx)
        return self._readonly_dldx
        

class LeakyReluCache(LayerCache):
    _output = None