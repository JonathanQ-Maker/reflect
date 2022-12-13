from __future__ import annotations
from reflect.layers.parametric_layer import ParametricLayer
from reflect import np
from reflect.utils.misc import to_tuple, conv_size, in_conv_size
import copy
from math import ceil

class Convolve2D(ParametricLayer):
    
    # public variables
    _filter_size        = None  # filter size, (height, width) or scaler
    _kernels            = 1     # number of kernels
    _kernel_shape       = None  # (num kernel, height, width, channel)
    _pad                = False # zero pad input to maintain spatial dimention
    _weight_type        = None  # weight(kernel) initialization type
    _strides            = (1, 1)# (stride y, stride x), convolution strides
    _kernel_regularizer = None
    _bias_regularizer   = None
    _input              = None
    _padded_input_shape = None  # input shape after padded
    _pad_size           = None  # (pad_H_top, pad_H_bot, pad_W_left, pad_W_right) amount padded along H and W axis

    _dldk               = None  # gradient with respect to kernel
    _dldb               = None  # gradient with respect to bias
    _readonly_dldk      = None  # readonly dldk view
    _readonly_dldb      = None  # readonly dldb view

    # internal variables
    _window_stride      = None  # input window view stride. ndarray type, used to multiply with input item size.
    _window_shape       = None  # input window view shape
    _base               = None  # array used for backprop
    _base_view          = None  # base viewed as dldz matrix
    _base_window_view   = None  # base viewed as windows 
    _dldz_kernel_view   = None  # base viewed as kernel dldz matrix for dldk
    _dldz_window_stride = None  # input window view stride for dldz_kernel_view
    _dldz_window_shape  = None  # input window view shape for dldz_kernel_view

    _kernel_rot180      = None  # view object of kernel at 180 degrees
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

    @property
    def kernels(self):
        """
        number of kernels
        """
        return self._kernels

    @kernels.setter
    def kernels(self, kernels):
        self._kernels = kernels

    @property
    def pad(self):
        """
        zero pad input to maintain spatial dimention
        """
        return self._pad

    @pad.setter
    def pad(self, pad):
        self._pad = pad

    @property
    def weight_type(self):
        """
        weight initialization type
        """
        return self._weight_type

    @weight_type.setter
    def weight_type(self, type):
        self._weight_type = type

    @property
    def strides(self):
        """
        (stride y, stride x), convolution strides
        """
        return self._strides

    @strides.setter
    def strides(self, strides):
        self._strides = strides

    @property
    def kernel_regularizer(self):
        return self._kernel_regularizer

    @kernel_regularizer.setter
    def kernel_regularizer(self, regularizer):
        self._kernel_regularizer = regularizer

    @property
    def bias_regularizer(self):
        return self._kernel_regularizer

    @kernel_regularizer.setter
    def bias_regularizer(self, regularizer):
        self._bias_regularizer = regularizer

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


    def __init__(self, input_size = (1, 1, 1), filter_size = (1, 1),
                 kernels = 1,
                 batch_size = 1, 
                 strides = (1, 1),
                 weight_type = "he",
                 pad = False,
                 kernel_regularizer=None,
                 bias_regularizer=None):
        super().__init__(input_size, None, batch_size)
        self._weight_type            = weight_type
        self._kernel_regularizer     = kernel_regularizer
        self._bias_regularizer       = bias_regularizer
        self._strides                = strides
        self._pad                    = pad
        self._filter_size            = filter_size
        self._kernels                = kernels

    def compile(self, gen_param=True):
        self._kernel_shape  = self.compute_kernel_shape()
        self._input_shape   = (self._batch_size, ) + to_tuple(self._input_size)
        if (self._pad):
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
        self._dldb = np.zeros(shape=self._kernels)
        self._readonly_dldk = self._dldk.view()
        self._readonly_dldb = self._dldb.view()
        self._readonly_dldk.flags.writeable = False
        self._readonly_dldb.flags.writeable = False

        self.init_regularizers()
        self.init_base()
        self._window_stride, self._window_shape = self.compute_view_attr()

        self.name = f"Dense {self._output_size}"
        if (gen_param):
            self.apply_param(self.create_param())

    def is_compiled(self):
        kernel_shape_match = self._kernel_shape == self.compute_kernel_shape()

        dldk_ok = (self._dldk is not None 
                   and self._dldk.shape == self._kernel_shape)
        dldb_ok = (self._dldb is not None 
                   and self._dldb.shape[0] == self._kernels)

        regularizer_ok = self.regularizers_ok(self._kernel_regularizer, 
                                              self._bias_regularizer)
        stride, shape = self.compute_view_attr()
        window_view_ok = (np.all(stride == self._window_stride) 
                          and shape == self._window_shape)
        base_ok = (self._base_view is not None 
                   and self._base_view.shape == self._output_shape)

        pad_ok = not self._pad
        if (self._pad):
            padded_input_shape, pad_size = self.compute_padded_input()
            pad_ok = (self._pad_size == pad_size
                      and self._padded_input_shape == padded_input_shape
                      and self._padded_input is not None 
                      and self._padded_input.shape == self._padded_input_shape
                      and self._padded_input_view is not None)

        return (super().is_compiled() 
                and kernel_shape_match 
                and regularizer_ok 
                and dldk_ok and dldb_ok
                and window_view_ok
                and base_ok
                and pad_ok)

    def compute_output_shape(self):
        if (self._pad):
            return (self._batch_size, 
                    conv_size(self._padded_input_shape[1], self._filter_size[0], self._strides[0]),
                    conv_size(self._padded_input_shape[2], self._filter_size[1], self._strides[1]),
                    self._kernels)
        return (self._batch_size, 
                conv_size(self._input_size[0], self._filter_size[0], self._strides[0]),
                conv_size(self._input_size[1], self._filter_size[1], self._strides[1]),
                self._kernels)

    def compute_kernel_shape(self):
        if isinstance(self._filter_size, int):
            return (self._kernels, 
                    self._filter_size, 
                    self._filter_size, 
                    self._input_size[2])
        else:
            return (self._kernels, ) + self._filter_size + (self._input_size[2], )

    def compute_view_attr(self):
        """
        Computes window view attributes for input
        returns window_stride(ndarray), window_shape
        """

        # forward view attributes
        B, H, W, C = self._input_shape
        if (self._pad):
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
        if (self._kernel_regularizer is not None):
            self._kernel_regularizer.shape = self._kernel_shape

        # bias
        if (self._bias_regularizer is not None):
            self._bias_regularizer.shape = to_tuple(self._kernels)

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
        param.weight_type = self._weight_type

    def init_base(self):
        """
        base matrix used to compute dldx, dldk
        """
        B, H, W, C = self._input_shape
        if (self._pad):
            B, H, W, C = self._padded_input_shape
        K, h, w, _ = self._kernel_shape
        pad_h   = h - 1
        pad_w   = w - 1
        in_h    = in_conv_size(H, h, 1)
        in_w    = in_conv_size(W, w, 1)
        out_h   = conv_size(H, h, self._strides[0])
        out_w   = conv_size(W, w, self._strides[1])
        stride_h, stride_w = self._strides

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

        self.init_kernel(param, self._weight_type, 0)
        param.bias  = np.zeros(self._kernels)
        param.pad   = self._pad

        if (self._kernel_regularizer is not None):
            param.kernel_regularizer = copy.deepcopy(self._kernel_regularizer)
            param.kernel_regularizer.compile()
        if (self._bias_regularizer is not None):
            param.bias_regularizer = copy.deepcopy(self._bias_regularizer)
            param.bias_regularizer.compile()
        return param

    def regularizers_ok(self, kernel_regularizer, bias_regularizer):
        kernel_regularizer_ok = True
        if (kernel_regularizer is not None):
            kernel_regularizer_ok = kernel_regularizer.shape == self._kernel_shape

        bias_regularizer_ok = True
        if (bias_regularizer is not None):
            bias_regularizer_ok = bias_regularizer.shape == (self._kernels, )

        return kernel_regularizer_ok and bias_regularizer_ok

    def param_compatible(self, param: Convolve2DParam):
        """
        Check if parameter is compatible

        Args:
            param: parameter to check

        Returns:
            is compatible
        """

        bias_ok = (param.bias is not None) and param.bias.shape[0] == self._kernels
        kernel_ok = (param.kernel is not None) and param.kernel.shape == self._kernel_shape

        regularizer_ok = self.regularizers_ok(param.kernel_regularizer, 
                                              param.bias_regularizer)

        pad_ok = param.pad is not None and param.pad == self._pad

        return bias_ok and kernel_ok and regularizer_ok and pad_ok

    def apply_param(self, param: Convolve2DParam):
        """
        Applies layer param

        Args:
            param: parameter to be applied
        """

        super().apply_param(param)
        self._kernel_rot180 = np.rot90(param.kernel, k=2, axes=(1, 2))
    
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

        if (self.param.pad):
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
        np.copyto(self._base_view, dldz)
        dldx = np.einsum('BHWhwK,KhwC->BHWC', self._base_window_view, 
                         self._kernel_rot180, optimize="optimal")

        # compute dldk gradient
        input = self._input
        if (self._pad):
            input = self._padded_input
        strides = self._dldz_window_stride * input.itemsize
        view = np.lib.stride_tricks.as_strided(input, 
                                               shape=self._dldz_window_shape, 
                                               strides=strides,
                                               writeable=False)
        dldk = np.einsum('BHWhwC,BKhw->KHWC', view, 
                            self._dldz_kernel_view, optimize="optimal")
        if (self._pad):
            _, H, W, _ = self._padded_input_shape
            pad_H_top, pad_H_bot, pad_W_left, pad_W_right = self._pad_size
            dldx = dldx[:, pad_H_top:H-pad_H_bot, pad_W_left:W-pad_W_right, :]
        np.copyto(self._dldx, dldx)
        np.copyto(self._dldk, dldk)

        # compute dldb gradient
        np.sum(dldz, axis=(0, 1, 2), out=self._dldb)

        return self._dldx

    def apply_grad(self, step, dldk=None, dldb=None):
        """
        Applies gradients

        Args:
            step: gradient step
            dldk: gradient of loss with respect to kernel
            dldb: gradient of loss with respect to bias
        """

        if (dldk is None):
            dldk = self._dldk
        if (dldb is None):
            dldb = self._dldb
        # weight update
        np.add(self.param.kernel, step * dldk, out=self.param.kernel)
        # bias update
        np.add(self.param.bias, step * dldb, out=self.param.bias)

    def __str__(self):
        return self.attribute_to_str()

    def attribute_to_str(self):
        return (super().attribute_to_str()
        + f"weight init:    {self._weight_type}\n"
        + f"max weight:     {self.param.weight.max()}\n"
        + f"min weight:     {self.param.weight.min()}\n"
        + f"weight std:     {np.std(self.param.weight)}\n"
        + f"weight mean:    {np.mean(self.param.weight)}\n")







class Convolve2DParam():
    kernel = None
    weight_type = None
    pad = None

    bias = None
    kernel_regularizer = None
    bias_regularizer = None
