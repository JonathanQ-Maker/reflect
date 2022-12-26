from __future__ import annotations
from reflect.layers.absrtact_layer import AbstractLayer
from reflect.optimizers import Adam
from reflect.utils.misc import to_tuple, conv_size, in_conv_size
from reflect import np

class AvgPool2D(AbstractLayer):

    """
    2D Average Pool layer

    Shape:
        input:  (batch size, height, width, channels)
        output: (batch size, 
                    conv_size(height, pool_size, stride), 
                    conv_size(width, pool_size, stride), 
                    channels)
    """
    
    _pool_size  = (1, 1) # (pool height, pool width)
    _strides    = (1, 1) # (y stride, x stride) pool strides

    # internal variables
    _window_stride      = None  # input window view stride. ndarray type, used to multiply with input item size.
    _window_shape       = None  # input window view shape
    _base               = None  # array used for backprop
    _base_view          = None  # base viewed as dldz matrix
    _base_window_view   = None  # base viewed as windows 
    _scaled_dldz        = None  # scaled dldz for backprop

    @property
    def pool_size(self):
        return self._pool_size

    @pool_size.setter
    def pool_size(self, size):
        self._pool_size = size
        if isinstance(size, int):
            self._pool_size = (size, size)

    @property
    def strides(self):
        """
        (y stride, x stride) pool strides
        """
        return self._strides

    @strides.setter
    def strides(self, strides):
        self._strides = strides
        if isinstance(strides, int):
            self._strides = (strides, strides)

    def __init__(self, 
                 pool_size=1, 
                 strides=None):
        super().__init__()
        self.pool_size  = pool_size
        self.strides    = strides
        if strides is None:
            self.strides = pool_size

    def compute_output_shape(self):
        H, W, C             = self._input_size
        h, w                = self._pool_size
        stride_h, stride_w  = self._strides
        return (self._batch_size, 
                conv_size(H, h, stride_h),
                conv_size(W, w, stride_w), C)

    def compute_view_attr(self):
        """
        Computes window view attributes for input
        returns window_stride(ndarray), window_shape
        """

        # forward view attributes
        B, H, W, C  = self._input_shape
        h, w        = self._pool_size
        stride_h, stride_w = self._strides

        # ndarray to multiply with itemsize
        window_stride = np.asarray((H*W*C, stride_h*W*C, stride_w*C, W*C, C, 1))
        window_shape = (B, 
                        conv_size(H, h, stride_h),
                        conv_size(W, w, stride_w), 
                        h, w, C)
        return window_stride, window_shape

    def init_base(self):
        B, H, W, C          = self._input_shape
        h, w                = self._pool_size
        stride_h, stride_w  = self._strides
        pad_h               = h - 1
        pad_w               = w - 1

        in_h        = in_conv_size(H, h, 1)
        in_w        = in_conv_size(W, w, 1)
        self._base  = np.zeros((B, in_h, in_w, C))

        size        = self._base.itemsize
        strides     = (in_h*in_w*C*size, stride_h*in_w*C*size, stride_w*C*size, size)
        base_core   = self._base[:, pad_h:in_h-pad_h,
                                    pad_w:in_w-pad_w, :]
        self._base_view = np.lib.stride_tricks.as_strided(base_core, 
                                                    shape=self._output_shape, 
                                                    strides=strides)

        # base window view
        base_window_shape = (B, 
                                 conv_size(in_h, h, 1),
                                 conv_size(in_w, w, 1), 
                                 h, w, C)
        base_window_stride = (in_h*in_w*C*size, 
                                in_w*C*size, 
                                C*size, in_w*C*size, C*size, size)
        self._base_window_view = np.lib.stride_tricks.as_strided(self._base, 
                                                            shape=base_window_shape, 
                                                            strides=base_window_stride,
                                                            writeable=False)

    def compile(self, input_size, batch_size=1):
        self._input_size = input_size
        self._batch_size = batch_size
        # compile shapes
        self._input_shape   = (self._batch_size, ) + to_tuple(self._input_size)
        self._output_shape  = self.compute_output_shape()
        self._output_size   = self._output_shape[1:]

        # compile arrays
        self._output        = np.zeros(shape=self._output_shape)
        self._dldx          = np.zeros(shape=self._input_shape)
        self._scaled_dldz   = np.zeros(shape=self._output_shape)

        # compile read only views
        self._readonly_output = self._output.view()
        self._readonly_dldx   = self._dldx.view()
        self._readonly_output.flags.writeable = False
        self._readonly_dldx.flags.writeable = False

        # compile misc attributes
        self.init_base()
        self._window_stride, self._window_shape = self.compute_view_attr()


        self.name = f"AvgPool2D {self._pool_size[0]}x{self._pool_size[1]}"

    def is_compiled(self):
        window_view_ok = self._input_shape is not None
        if window_view_ok:
            stride, shape = self.compute_view_attr()
            window_view_ok = (np.all(stride == self._window_stride) 
                              and shape == self._window_shape)
        base_ok = (self._base_view is not None 
                   and self._base_view.shape == self._output_shape)
        scaled_dldz_ok = (self._scaled_dldz is not None
                          and self._scaled_dldz.shape == self._output_shape)

        return (super().is_compiled() 
                and window_view_ok
                and scaled_dldz_ok)
    
    def forward(self, X):
        """
        forward pass with input

        Args:
            X: input

        Returns: 
            output

        Make copy of output if intended to be modified
        Input instance will be kept and expected not to be modified between forward and backward pass
        """
        strides = self._window_stride * X.itemsize
        view = np.lib.stride_tricks.as_strided(X, shape=self._window_shape, 
                                               strides=strides)
        np.mean(view, axis=(3, 4), out=self._output)
        return self._readonly_output

    def backprop(self, dldz):
        """
        backward pass to compute the gradients

        Args:
            dldz: gradient of loss with respect to output

        Returns: 
            dldx, gradient of loss with respect to input

        Note:
            expected to execute only once after forward
        """
        h, w, = self._pool_size
        np.multiply(dldz, 1.0/(h*w), out=self._scaled_dldz)
        np.copyto(self._base_view, self._scaled_dldz)
        np.sum(self._base_window_view, axis=(3, 4), out=self._dldx)
        return self._readonly_dldx







class DenseParam():
    weight = None
    weight_type = None

    bias = None