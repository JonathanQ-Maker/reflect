from __future__ import annotations
from reflect.layers.convolution_2d_layer import Convolve2D, Convolve2DParam, Convolve2DCache
from reflect import np

class ConvolveSN2D(Convolve2D):
    """
    Spectral Normalized Convolve 2D layer.

    Enforces Lipschitz continuity using Spectral Normalization
    See: https://arxiv.org/abs/1802.05957
    """
    
    lip_const       = 1

    _u              = None
    _v              = None
    _uv             = None
    _Kv             = None
    _kernel_2d_view = None
    _uv_2d_view     = None
    

    def __init__(self, 
                 filter_size        = (1, 1),
                 kernels            = 1,
                 strides            = (1, 1),
                 lip_const          = 1,
                 weight_type        = "he",
                 pad                = False,
                 kernel_reg         = None,
                 bias_reg           = None,
                 kernel_optimizer   = None,
                 bias_optimizer     = None,
                 bias_constraint    = None):
        super().__init__(filter_size, 
                         kernels, 
                         strides, 
                         weight_type, 
                         pad, 
                         kernel_reg, 
                         bias_reg, 
                         kernel_optimizer, 
                         bias_optimizer, 
                         None, bias_constraint)
        self.lip_const = lip_const

    def compile(self, input_size, batch_size=1, gen_param=True):
        super().compile(input_size, batch_size, gen_param)
        self.name = f"{self.kernels} ConvolveSN2D {self._filter_size[0]}x{self._filter_size[1]} {self.lip_const}-Lipschitz"

        self._u     = np.random.normal(size=self.kernels)
        self._v     = np.zeros(self._kernel_2d_view.shape[1])
        self._uv    = np.zeros(self._kernel_shape)
        self._Kv    = np.zeros(self.kernels)

        self._uv_2d_view = self._uv.view()
        self._uv_2d_view.shape = self._kernel_2d_view.shape
        self.power_iteration(5)

    def create_cache(self):
        cache = ConvolveSN2DCache()
        cache._owner    = self
        cache._X        = np.zeros(self._input_shape)
        cache._kernel   = np.zeros(self._kernel_shape)

        cache._u        = np.zeros(self.kernels)
        cache._v        = np.zeros(self._kernel_2d_view.shape[1])
        cache._sn_kernel= np.zeros(self._kernel_shape)
        return cache
    
    def forward(self, X, out_cache: ConvolveSN2DCache=None):
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
        np.copyto(out_cache._kernel, self.param.kernel)
        np.copyto(out_cache._u, self._u)
        np.copyto(out_cache._v, self._v)

        # Spectral Norm
        sigma = np.dot(out_cache._u, self._Kv)
        out_cache._sigma = sigma
        np.multiply(out_cache._kernel, self.lip_const/sigma, out=out_cache._sn_kernel)

        X = out_cache._X
        if (self.pad):
            np.copyto(self._padded_input_view, X)
            X = self._padded_input
        strides = self._window_stride * X.itemsize
        view = np.lib.stride_tricks.as_strided(X, 
                                               shape=self._window_shape, 
                                               strides=strides,
                                               writeable=False)
        np.copyto(self._output, np.einsum('BHWhwC,khwC->BHWk', view, out_cache._sn_kernel, 
                  optimize="optimal"))
        np.add(self._output, self.param.bias, out=self._output)
        return self._output
    
    def backprop(self, dldz, cache: ConvolveSN2DCache=None):
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

        # compute dldx gradient
        # NOTE: convolution on stride spaced dldz with 180 rotated kernel computes dldx
        kernel_rot180 = np.rot90(cache._sn_kernel, k=2, axes=(1, 2))
        np.copyto(self._base_view, dldz)
        dldx = np.einsum('BHWhwK,KhwC->BHWC', self._base_window_view, 
                         kernel_rot180, optimize="optimal")

        # compute dldk gradient

        X = cache._X
        if (self.pad):
            np.copyto(self._padded_input_view, X)
            X = self._padded_input
        strides = self._dldz_window_stride * X.itemsize
        view = np.lib.stride_tricks.as_strided(X, 
                                               shape=self._dldz_window_shape, 
                                               strides=strides,
                                               writeable=False)
        # NOTE: stride 1 convolution on input with modified kernel shape computes dldk
        dldk = np.einsum('BHWhwC,BKhw->KHWC', view, 
                            self._dldz_kernel_view, optimize="optimal")
        
        # Spectral Norm backprop
        d = np.dot(cache._kernel.flat, dldk.flat)
        np.outer(cache._u, cache._v, out=self._uv_2d_view)
        np.divide(self._uv, cache._sigma/d, out=self._uv)
        np.subtract(dldk, self._uv, out=dldk)
        np.multiply(dldk, self.lip_const/cache._sigma, out=dldk)

        if (self.pad):
            _, H, W, _ = self._padded_input_shape
            pad_H_top, pad_H_bot, pad_W_left, pad_W_right = self._pad_size
            dldx = dldx[:, pad_H_top:H-pad_H_bot, pad_W_left:W-pad_W_right, :]
        np.copyto(self._dldx, dldx)
        np.copyto(self._dldk, dldk)

        # compute dldb gradient
        np.sum(dldz, axis=(0, 1, 2), out=self._dldb)

        # add regularizer
        if (self.kernel_reg is not None):
            np.add(self._dldk, self.kernel_reg.gradient(self.param.kernel), out=self._dldk)
        if (self.bias_reg is not None):
            np.add(self._dldb, self.bias_reg.gradient(self.param.bias), out=self._dldb)

        return self._dldx

    def apply_param(self, param: Convolve2DParam):
        super().apply_param(param)
        self._kernel_2d_view = self.param.kernel.view()
        self._kernel_2d_view.shape = (self.kernels, -1)

    def power_iteration(self, steps: int = 1):
        """
        Approximate Spectral Normalize using power iteration method

        Params:
            steps: number of iterations for approximation.
        """

        for i in range(steps):
            np.dot(self._kernel_2d_view.T, self._u, out=self._v)
            np.divide(self._v, np.linalg.norm(self._v), out=self._v)

            np.dot(self._kernel_2d_view, self._v, out=self._Kv)
            np.divide(self._Kv, np.linalg.norm(self._Kv), out=self._u)

    def apply_grad(self, step, dldk=None, dldb=None):
        super().apply_grad(step, dldk, dldb)
        self.power_iteration()
        

class ConvolveSN2DCache(Convolve2DCache):
    _sn_kernel  = None
    _u          = None
    _sigma      = None
    _v          = None