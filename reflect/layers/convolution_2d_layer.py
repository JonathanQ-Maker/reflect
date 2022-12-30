from __future__ import annotations
from reflect.layers.parametric_layer import ParametricLayer
from reflect import np
from reflect.utils.misc import to_tuple, conv_size, in_conv_size
from reflect.optimizers import Adam
from math import ceil

class Convolve2D(ParametricLayer):
    """
    2D Convolution layer

    Shape:
        input:  (batch size, height, wdith, channels)
        
        No pad
            output: (batch size, 
                    conv_size(height, filter, stride), 
                    conv_size(width, filter, stride), 
                    kernels)
        Pad
            output: (batch size, height, width, kernels)
    """
    
    # public variables
    _filter_size        = None  # filter size, (height, width) or scaler
    kernels             = 1     # number of kernels
    _kernel_shape       = None  # (num kernel, height, width, channel)
    pad                 = False # zero pad input to maintain spatial dimention
    weight_type         = None  # weight(kernel) initialization type
    _strides            = (1, 1)# (stride y, stride x), convolution strides
    kernel_reg  = None
    bias_reg    = None
    _input              = None
    _padded_input_shape = None  # input shape after padded
    _pad_size           = None  # (pad_H_top, pad_H_bot, pad_W_left, pad_W_right) amount padded along H and W axis

    _dldk               = None  # gradient with respect to kernel
    _dldb               = None  # gradient with respect to bias
    _readonly_dldk      = None  # readonly dldk view
    _readonly_dldb      = None  # readonly dldb view
    kernel_optimizer    = None
    bias_optimizer      = None

    # internal variables
    _window_stride      = None  # input window view stride. ndarray type, used to multiply with input item size.
    _window_shape       = None  # input window view shape
    _base               = None  # array used for backprop
    _base_view          = None  # base viewed as dldz matrix
    _base_window_view   = None  # base viewed as windows 
    _dldz_kernel_view   = None  # base viewed as kernel dldz matrix for dldk
    _dldz_window_stride = None  # input window view stride for dldz_kernel_view
    _dldz_window_shape  = None  # input window view shape for dldz_kernel_view

    _padded_input       = None  # array used as input for padded inputs
    _padded_input_view  = None  # padded_input view without padding, used for input paste

    @property
    def filter_size(self):
        """
        filter size, (height, width) or scaler
        """
        return self._filter_size

    @filter_size.setter
    def filter_size(self, filter_size):
        self._filter_size = filter_size
        if isinstance(filter_size, int):
            self._filter_size = (filter_size, filter_size)

    @property
    def strides(self):
        """
        (stride y, stride x), convolution strides
        """
        return self._strides

    @strides.setter
    def strides(self, strides):
        self._strides = strides
        if isinstance(strides, int):
            self._strides = (strides, strides)

    @property
    def input(self):
        if (self._input is None):
            return None
        view = self._input.view()
        view.flags.writeable = False
        return view

    @property
    def dldk(self):
        """
        gradient with respect to kernel
        """
        return self._readonly_dldk

    @property
    def dldb(self):
        """
        gradient with respect to bias
        """
        return self._readonly_dldb

    @property
    def kernel_shape(self):
        """
        kernel shape, (num kernel, height, width, channels)
        """
        return self._kernel_shape

    @property
    def padded_input_shape(self):
        """
        input shape after padded
        """
        return self._padded_input_shape

    @property
    def pad_size(self):
        """
        (pad_H_top, pad_H_bot, pad_W_left, pad_W_right) amount padded along H and W axis
        """
        return self._pad_size

    @property
    def total_params(self):
        return self.param.kernel.size + self.param.bias.size


    def __init__(self, 
                 filter_size        = (1, 1),
                 kernels            = 1,
                 strides            = (1, 1),
                 weight_type        = "he",
                 pad                = False,
                 kernel_reg         = None,
                 bias_reg           = None,
                 kernel_optimizer   = None,
                 bias_optimizer     = None):

        super().__init__()
        self.weight_type            = weight_type
        self.kernel_reg             = kernel_reg
        self.bias_reg               = bias_reg
        self.strides                = strides
        self.pad                    = pad
        self.filter_size            = filter_size
        self.kernels                = kernels
        self.kernel_optimizer       = kernel_optimizer
        self.bias_optimizer         = bias_optimizer
        if kernel_optimizer is None:
            self.kernel_optimizer   = Adam()
        if bias_optimizer is None:
            self.bias_optimizer     = Adam()

    def compile(self, input_size, batch_size=1, gen_param=True):
        self._input_size    = input_size
        self._batch_size    = batch_size
        self._kernel_shape  = self.compute_kernel_shape()
        self._input_shape   = (self._batch_size, ) + self._input_size
        if (self.pad):
            self._padded_input_shape, self._pad_size = self.compute_padded_input()
            self._padded_input = np.zeros(shape=self._padded_input_shape)
            _, H, W, _ = self._padded_input_shape
            pad_H_top, pad_H_bot, pad_W_left, pad_W_right = self._pad_size
            self._padded_input_view = self._padded_input[:, 
                                                         pad_H_top:H-pad_H_bot, 
                                                         pad_W_left:W-pad_W_right, :]
        self._output_size = self.compute_output_shape()[1:]
        self._output_shape = (self._batch_size, ) + to_tuple(self._output_size)

        # compile output
        self._output            = np.zeros(shape=self._output_shape)
        self._dldx              = np.zeros(shape=self._input_shape)
        self._readonly_output   = self._output.view()
        self._readonly_dldx     = self._dldx.view()
        self._readonly_output.flags.writeable   = False
        self._readonly_dldx.flags.writeable     = False

        # compile gradient
        self._dldk = np.zeros(shape=self._kernel_shape)
        self._dldb = np.zeros(shape=self.kernels)
        self._readonly_dldk = self._dldk.view()
        self._readonly_dldb = self._dldb.view()
        self._readonly_dldk.flags.writeable = False
        self._readonly_dldb.flags.writeable = False

        self.init_regularizers()
        self.init_base()
        self._window_stride, self._window_shape = self.compute_view_attr()

        # compile optimizers
        self.kernel_optimizer.compile(self._kernel_shape)
        self.bias_optimizer.compile(self.kernels)

        self.name = f"{self.kernels} Convolve2D {self._filter_size[0]}x{self._filter_size[1]}"
        if (gen_param):
            self.apply_param(self.create_param())

    def is_compiled(self):
        kernel_shape_match = self._kernel_shape == self.compute_kernel_shape()

        dldk_ok = (self._dldk is not None 
                   and self._dldk.shape == self._kernel_shape)
        dldb_ok = (self._dldb is not None 
                   and self._dldb.shape[0] == self.kernels)

        regularizer_ok = self.regularizers_ok(self.kernel_reg, 
                                              self.bias_reg)
        window_view_ok = self._input_shape is not None
        if window_view_ok:
            stride, shape = self.compute_view_attr()
            window_view_ok = (np.all(stride == self._window_stride) 
                              and shape == self._window_shape)
        base_ok = (self._base_view is not None 
                   and self._base_view.shape == self._output_shape)

        pad_ok = not self.pad
        if (self.pad):
            padded_input_shape, pad_size = self.compute_padded_input()
            pad_ok = (self._pad_size == pad_size
                      and self._padded_input_shape == padded_input_shape
                      and self._padded_input is not None 
                      and self._padded_input.shape == self._padded_input_shape
                      and self._padded_input_view is not None)

        kernel_optimizer_ok = (self.kernel_optimizer is not None
                         and self.kernel_optimizer.is_compiled()
                         and self.kernel_optimizer.shape == self._kernel_shape)

        bias_optimizer_ok = (self.bias_optimizer is not None
                         and self.bias_optimizer.is_compiled()
                         and self.bias_optimizer.shape == to_tuple(self.kernels))

        return (super().is_compiled() 
                and kernel_shape_match 
                and regularizer_ok 
                and dldk_ok and dldb_ok
                and window_view_ok
                and base_ok
                and pad_ok
                and kernel_optimizer_ok
                and bias_optimizer_ok)

    def compute_output_shape(self):
        if (self.pad):
            return (self._batch_size, 
                    conv_size(self._padded_input_shape[1], self._filter_size[0], self._strides[0]),
                    conv_size(self._padded_input_shape[2], self._filter_size[1], self._strides[1]),
                    self.kernels)
        return (self._batch_size, 
                conv_size(self._input_size[0], self._filter_size[0], self._strides[0]),
                conv_size(self._input_size[1], self._filter_size[1], self._strides[1]),
                self.kernels)

    def compute_kernel_shape(self):
            return (self.kernels, ) + self._filter_size + (self._input_size[2], )

    def compute_view_attr(self):
        """
        Computes window view attributes for input
        returns window_stride(ndarray), window_shape
        """

        # forward view attributes
        B, H, W, C = self._input_shape
        if (self.pad):
            B, H, W, C = self._padded_input_shape
        K, h, w, _ = self._kernel_shape
        stride_h, stride_w = self._strides

        # ndarray to multiply with itemsize
        window_stride = np.asarray((H*W*C, stride_h*W*C, stride_w*C, W*C, C, 1))
        window_shape = (B, 
                        conv_size(H, h, stride_h),
                        conv_size(W, w, stride_w), 
                        h, w, C)
        return window_stride, window_shape

    def compute_padded_input(self):
        B, H, W, C = self._input_shape
        _, h, w, _ = self._kernel_shape
        stride_h, stride_w = self._strides
        in_H = in_conv_size(H, h, stride_h)
        in_W = in_conv_size(W, w, stride_w)
        padded_input_shape = (B, in_H, in_W, C)
        
        # (pad_H_top, pad_H_bot, pad_W_left, pad_W_right)
        pad_size = ((in_H - H)//2, ceil((in_H - H)/2), (in_W - W)//2, ceil((in_W - W)/2))
        return padded_input_shape, pad_size

    def init_regularizers(self):
        # kernel
        if (self.kernel_reg is not None):
            self.kernel_reg.compile(self._kernel_shape)

        # bias
        if (self.bias_reg is not None):
            self.bias_reg.compile(to_tuple(self.kernels))

    def init_kernel(self, param, type, weight_bias = 0):
        """
        Initialize kernel in param object

        Args:
            type: weight initalization type
                [he, xavier]
            weight_bias: mean of weight
            param: param object to initalize weight to/store
        """

        K, h, w, C = self._kernel_shape
        scale = 1
        input_size = h*w*C
        if  (type == "xavier"):
            scale = 1 / np.sqrt(input_size) # Xavier init
        elif (type == "he"):
            scale = np.sqrt(2 / input_size) # he init, for relus
        else:
            raise ValueError(f'no such weight type "{type}"')

        param.kernel = np.random.normal(loc=weight_bias, scale=scale, size=self._kernel_shape)

    def init_base(self):
        """
        base matrix used to compute dldx, dldk
        """
        B, H, W, C = self._input_shape
        if (self.pad):
            B, H, W, C = self._padded_input_shape
        K, h, w, _ = self._kernel_shape
        stride_h, stride_w = self._strides
        pad_h   = h - 1
        pad_w   = w - 1
        in_h    = in_conv_size(H, h, 1)
        in_w    = in_conv_size(W, w, 1)
        out_h   = conv_size(H, h, stride_h)
        out_w   = conv_size(W, w, stride_w)

        # base
        base_shape = (B, in_h, in_w, K)
        self._base = np.zeros(base_shape)

        size        = self._base.itemsize
        strides     = (in_h*in_w*K*size, stride_h*in_w*K*size, stride_w*K*size, size)
        base_core   = self._base[:, pad_h:in_h-pad_h,
                                 pad_w:in_w-pad_w, :]
        self._base_view = np.lib.stride_tricks.as_strided(base_core, 
                                                         shape=self._output_shape, 
                                                         strides=strides)
        base_window_shape = (B, 
                             conv_size(in_h, h, 1),
                             conv_size(in_w, w, 1), 
                             h, w, K)
        base_window_stride = (in_h*in_w*K*size, 
                              in_w*K*size, 
                              K*size, in_w*K*size, K*size, size)
        self._base_window_view = np.lib.stride_tricks.as_strided(self._base, 
                                                                shape=base_window_shape, 
                                                                strides=base_window_stride,
                                                                writeable=False)

        # dldz kernel
        dldz_kernel_shape   = (B, K, in_h - 2*pad_h, in_w - 2*pad_w)
        dldz_kernel_stride  = (K*in_h*in_w*size, size, in_w*K*size, K*size)
        self._dldz_kernel_view = np.lib.stride_tricks.as_strided(base_core, 
                                                  shape=dldz_kernel_shape, 
                                                  strides=dldz_kernel_stride,
                                                  writeable=False)

        # input window view for dldz kernel
        _, _, h, w = dldz_kernel_shape
        self._dldz_window_stride = np.asarray((H*W*C, W*C, C, W*C, C, 1))
        self._dldz_window_shape = (B, 
                                  conv_size(H, h, 1),
                                  conv_size(W, w, 1), 
                                  h, w, C)

    def create_param(self):
        super().create_param()
        param = Convolve2DParam()

        self.init_kernel(param, self.weight_type, 0)
        param.bias  = np.zeros(self.kernels)
        return param

    def regularizers_ok(self, kernel_reg, bias_reg):
        kernel_regularizer_ok = True
        if (kernel_reg is not None):
            kernel_regularizer_ok = kernel_reg.is_compiled()

        bias_regularizer_ok = True
        if (bias_reg is not None):
            bias_regularizer_ok = bias_reg.is_compiled()

        return kernel_regularizer_ok and bias_regularizer_ok

    def param_compatible(self, param: Convolve2DParam):
        """
        Check if parameter is compatible

        Args:
            param: parameter to check

        Returns:
            is compatible
        """

        bias_ok = (param.bias is not None) and param.bias.shape[0] == self.kernels
        kernel_ok = (param.kernel is not None) and param.kernel.shape == self._kernel_shape

        return bias_ok and kernel_ok

    def apply_param(self, param: Convolve2DParam):
        """
        Applies layer param

        Args:
            param: parameter to be applied
        """

        super().apply_param(param)
    
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
        self._input = X

        if (self.pad):
            np.copyto(self._padded_input_view, X)
            X = self._padded_input
        strides = self._window_stride * X.itemsize
        view = np.lib.stride_tricks.as_strided(X, 
                                               shape=self._window_shape, 
                                               strides=strides,
                                               writeable=False)
        np.copyto(self._output, np.einsum('BHWhwC,khwC->BHWk', view, self.param.kernel, 
                  optimize="optimal"))
        np.add(self._output, self.param.bias, out=self._output)
        return self._output

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
        # compute dldx gradient
        # NOTE: convolution on stride spaced dldz with 180 rotated kernel computes dldx
        kernel_rot180 = np.rot90(self.param.kernel, k=2, axes=(1, 2))
        np.copyto(self._base_view, dldz)
        dldx = np.einsum('BHWhwK,KhwC->BHWC', self._base_window_view, 
                         kernel_rot180, optimize="optimal")

        # compute dldk gradient
        input = self._input
        if (self.pad):
            input = self._padded_input
        strides = self._dldz_window_stride * input.itemsize
        view = np.lib.stride_tricks.as_strided(input, 
                                               shape=self._dldz_window_shape, 
                                               strides=strides,
                                               writeable=False)
        # NOTE: stride 1 convolution on input with modified kernel shape computes dldk
        dldk = np.einsum('BHWhwC,BKhw->KHWC', view, 
                            self._dldz_kernel_view, optimize="optimal")
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

    def apply_grad(self, step, dldk=None, dldb=None):
        """
        Applies layer gradients
        
        NOTE: None gradients default to gradient computed in backprop()

        Args:
            step: gradient step size
            dldk: gradient of loss with respect to kernel
            dldb: gradient of loss with respect to bias
        """

        if (dldk is None):
            dldk = self._dldk
        if (dldb is None):
            dldb = self._dldb

        # average batch gradient
        step = step / self._batch_size

        # kernel update
        np.subtract(self.param.kernel, self.kernel_optimizer.gradient(step, dldk),
              out=self.param.kernel)

        # bias update
        np.subtract(self.param.bias, self.bias_optimizer.gradient(step, dldb), out=self.param.bias)

    def attribute_to_str(self):
        return (super().attribute_to_str()
        + f"weight init:    {self.weight_type}\n"
        + f"max kernel:     {self.param.kernel.max()}\n"
        + f"min kernel:     {self.param.kernel.min()}\n"
        + f"kernel std:     {np.std(self.param.kernel)}\n"
        + f"kernel mean:    {np.mean(self.param.kernel)}\n"
        + f"pad:            {self.pad}\n")







class Convolve2DParam():
    kernel  = None
    bias    = None
