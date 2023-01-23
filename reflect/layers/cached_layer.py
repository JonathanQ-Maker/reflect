from __future__ import annotations
from abc import abstractmethod, ABC

class CachedLayer(ABC):
    """
    Cached layer

    Abstract layer that caches intermediary terms 
    during forward for ease of backprop
    """
    
    # internal variables
    _cache              = None # default cache

    @abstractmethod
    def create_cache(self):
        """
        Create and return empty cache

        Return:
            cache
        """
        pass

    @abstractmethod
    def forward(self, X, out_cache: LayerCache):
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
        pass

    @abstractmethod
    def backprop(self, dldz, cache: LayerCache):
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
        pass


class LayerCache():
    _owner = None

    @property
    def owner(self):
        return self._owner