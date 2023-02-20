from __future__ import annotations
from reflect.layers.abstract_layer import AbstractLayer
from reflect.layers.cached_layer import CachedLayer, LayerCache
from reflect import np

class Tanh(CachedLayer, AbstractLayer):
    """
    Tanh Activation

    Shape:
        input:  (batch size, ...)
        output: (batch size, ...)
    """

    def __init__(self):
        super().__init__()


    def compile(self, input_size, batch_size):
        self._output_size = input_size
        super().compile(input_size, batch_size)
        self.name = "Tanh"

        self._cache = self.create_cache()
    
    def is_compiled(self):
        dldx_ok = self._dldx is not None and self._dldx.shape == self._input_shape
        return super().is_compiled() and dldx_ok

    def create_cache(self):
        cache = TanhCache()
        cache._owner    = self
        cache._X        = np.zeros(self._input_shape)
        return cache

    def forward(self, X, out_cache: TanhCache=None):
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
        np.tanh(out_cache._X, out=self._output)
        return self._readonly_output

    def backprop(self, dldz, cache: TanhCache=None):
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

        np.cosh(cache._X, out=self._dldx)
        np.square(self._dldx, out=self._dldx)
        np.divide(dldz, self._dldx, out=self._dldx)
        return self._readonly_dldx
        

class TanhCache(LayerCache):
    _X = None