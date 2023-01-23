from __future__ import annotations
from reflect.layers.abstract_layer import AbstractLayer
from reflect.layers.cached_layer import CachedLayer, LayerCache
from reflect import np

class Relu(CachedLayer, AbstractLayer):
    """
    Rectified Linear Activation Unit (RELU)

    Shape:
        input:  (batch size, ...)
        output: (batch size, ...)
    """

    _input = None

    @property
    def input(self):
        if (self._input is None):
            return None
        view = self._input.view()
        view.flags.writeable = False
        return view

    def __init__(self):
        super().__init__()


    def compile(self, input_size, batch_size):
        self._output_size = input_size
        super().compile(input_size, batch_size)
        self.name = "Relu"

        self._cache = self.create_cache()
    
    def is_compiled(self):
        dldx_ok = self._dldx is not None and self._dldx.shape == self._input_shape
        return super().is_compiled() and dldx_ok

    def create_cache(self):
        cache = ReluCache()
        cache._owner    = self
        cache._output   = np.zeros(self._output_shape)
        return cache

    def forward(self, X, out_cache: ReluCache=None):
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
        np.copyto(out_cache._output, self._output)
        return self._readonly_output

    def backprop(self, dldz, cache: ReluCache=None):
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

        np.multiply(np.greater(cache._output, 0, out=self._dldx), dldz, out=self._dldx)
        return self._readonly_dldx
        

class ReluCache(LayerCache):
    _output = None