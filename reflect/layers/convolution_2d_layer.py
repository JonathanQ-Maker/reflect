from __future__ import annotations
from reflect.layers.parametric_layer import ParametricLayer
from reflect import np
from reflect.utils.misc import to_tuple, conv_size, in_conv_size
import copy
from math import ceil

class Convolve2D(ParametricLayer):
    
    # public variables
    __filter_size = None        # filter size, (height, width) or scaler
    __kernels = 1               # number of kernels
    __kernel_shape = None       # (num kernel, height, width, channel)
    __pad = False               # zero pad input to maintain spatial dimention
    __weight_type = None        # weight initialization type
    __strides = (1, 1)          # (stride y, stride x), convolution strides
    __kernel_regularizer = None
    __bias_regularizer = None
    __input = None
    __padded_input_shape = None # input shape after padded
    __pad_size = None           # (pad_H_top, pad_H_bot, pad_W_left, pad_W_right) amount padded along H and W axis

    __dldk = None               # gradient with respect to kernel
    __dldb = None               # gradient with respect to bias
    __readonly_dldk = None
    __readonly_dldb = None

    # internal variables
    __window_stride = None      # input window view stride. ndarray type, used to multiply with input item size.
    __window_shape = None       # input window view shape
    __base = None               # array used for backprop
    __base_view = None          # base viewed as dldz matrix
    __base_window_view = None   # base viewed as windows 
    __dldz_kernel_view = None   # base viewed as kernel dldz matrix for dldk
    __dldz_window_stride = None # input window view stride for dldz_kernel_view
    __dldz_window_shape = None  # input window view shape for dldz_kernel_view

    __kernel_rot180 = None      # view object of kernel at 180 degrees
    __padded_input = None       # array used as input for padded inputs
    __padded_input_view = None  # padded_input view without padding, used for input paste

    @property
    def filter_size(self):
        """
        filter size, (height, width) or scaler
        """
        return self.__filter_size

    @filter_size.setter
    def filter_size(self, filter_size):
        self.__filter_size = filter_size

    @property
    def kernels(self):
        """
        number of kernels
        """
        return self.__kernels

    @kernels.setter
    def kernels(self, kernels):
        self.__kernels = kernels

    @property
    def pad(self):
        """
        zero pad input to maintain spatial dimention
        """
        return self.__pad

    @pad.setter
    def pad(self, pad):
        self.__pad = pad

    @property
    def weight_type(self):
        """
        weight initialization type
        """
        return self.__weight_type

    @weight_type.setter
    def weight_type(self, type):
        self.__weight_type = type

    @property
    def strides(self):
        """
        (stride y, stride x), convolution strides
        """
        return self.__strides

    @strides.setter
    def strides(self, strides):
        self.__strides = strides

    @property
    def kernel_regularizer(self):
        return self.__kernel_regularizer

    @kernel_regularizer.setter
    def kernel_regularizer(self, regularizer):
        self.__kernel_regularizer = regularizer

    @property
    def bias_regularizer(self):
        return self.__kernel_regularizer

    @kernel_regularizer.setter
    def bias_regularizer(self, regularizer):
        self.__bias_regularizer = regularizer

    @property
    def input(self):
        if (self.__input is None):
            return None
        view = self.__input.view()
        view.flags.writeable = False
        return view

    @property
    def dldk(self):
        """
        gradient with respect to kernel
        """
        return self.__readonly_dldk

    @property
    def dldb(self):
        """
        gradient with respect to bias
        """
        return self.__readonly_dldb

    @property
    def kernel_shape(self):
        """
        kernel shape, (num kernel, height, width, channels)
        """
        return self.__kernel_shape

    @property
    def padded_input_shape(self):
        """
        input shape after padded
        """
        return self.__padded_input_shape

    @property
    def pad_size(self):
        """
        (pad_H_top, pad_H_bot, pad_W_left, pad_W_right) amount padded along H and W axis
        """
        return self.__pad_size


    def __init__(self, input_size = (1, 1, 1), filter_size = (1, 1),
                 kernels = 1,
                 batch_size = 1, 
                 strides = (1, 1),
                 weight_type = "he",
                 pad = False,
                 kernel_regularizer=None,
                 bias_regularizer=None):
        super().__init__(input_size, None, batch_size)
        self.__weight_type            = weight_type
        self.__kernel_regularizer     = kernel_regularizer
        self.__bias_regularizer       = bias_regularizer
        self.__strides                = strides
        self.__pad                    = pad
        self.__filter_size            = filter_size
        self.__kernels                = kernels

    def compile(self, gen_param=True):
        self.__kernel_shape = self.compute_kernel_shape()
        self.input_shape = (self.batch_size, ) + to_tuple(self.input_size)
        if (self.__pad):
            self.__padded_input_shape, self.__pad_size = self.compute_padded_input()
            self.__padded_input = np.zeros(shape=self.__padded_input_shape)
            _, H, W, _ = self.__padded_input_shape
            pad_H_top, pad_H_bot, pad_W_left, pad_W_right = self.__pad_size
            self.__padded_input_view = self.__padded_input[:, pad_H_top:H-pad_H_bot, pad_W_left:W-pad_W_right, :]
        self.output_size = self.compute_output_shape()[1:]
        self.output_shape = (self.batch_size, ) + to_tuple(self.output_size)

        # compile output
        self.output = np.zeros(shape=self.output_shape)
        self.dldx = np.zeros(shape=self.input_shape)

        # compile gradient
        self.__dldk = np.zeros(shape=self.__kernel_shape)
        self.__dldb = np.zeros(shape=self.__kernels)
        self.__readonly_dldk = self.__dldk.view()
        self.__readonly_dldb = self.__dldb.view()
        self.__readonly_dldk.flags.writeable = False
        self.__readonly_dldb.flags.writeable = False

        self.init_regularizers()
        self.init_base()
        self.__window_stride, self.__window_shape = self.compute_view_attr()

        self.name = f"Dense {self.output_size}"
        if (gen_param):
            self.apply_param(self.create_param())

    def is_compiled(self):
        kernel_shape_match = self.__kernel_shape == self.compute_kernel_shape()

        dldk_ok = self.__dldk is not None and self.__dldk.shape == self.__kernel_shape
        dldb_ok = self.__dldb is not None and self.__dldb.shape[0] == self.__kernels

        regularizer_ok = self.regularizers_ok(self.__kernel_regularizer, 
                                              self.__bias_regularizer)
        stride, shape = self.compute_view_attr()
        window_view_ok = np.all(stride == self.__window_stride) and shape == self.__window_shape
        base_ok = self.__base_view is not None and self.__base_view.shape == self.output_shape

        pad_ok = not self.__pad
        if (self.__pad):
            padded_input_shape, pad_size = self.compute_padded_input()
            pad_ok = (self.__pad_size == pad_size
                      and self.__padded_input_shape == padded_input_shape
                      and self.__padded_input is not None 
                      and self.__padded_input.shape == self.__padded_input_shape
                      and self.__padded_input_view is not None)

        return (super().is_compiled() 
                and kernel_shape_match 
                and regularizer_ok 
                and dldk_ok and dldb_ok
                and window_view_ok
                and base_ok
                and pad_ok)

    def compute_output_shape(self):
        if (self.__pad):
            return (self.batch_size, 
                    conv_size(self.__padded_input_shape[1], self.__filter_size[0], self.__strides[0]),
                    conv_size(self.__padded_input_shape[2], self.__filter_size[1], self.__strides[1]),
                    self.__kernels)
        return (self.batch_size, 
                conv_size(self.input_size[0], self.__filter_size[0], self.__strides[0]),
                conv_size(self.input_size[1], self.__filter_size[1], self.__strides[1]),
                self.__kernels)

    def compute_kernel_shape(self):
        if isinstance(self.__filter_size, int):
            return (self.__kernels, 
                    self.__filter_size, 
                    self.__filter_size, 
                    self.input_size[2])
        else:
            return (self.__kernels, ) + self.__filter_size + (self.input_size[2], )

    def compute_view_attr(self):
        """
        Computes window view attributes for input
        returns window_stride(ndarray), window_shape
        """

        # forward view attributes
        B, H, W, C = self.input_shape
        if (self.__pad):
            B, H, W, C = self.__padded_input_shape
        K, h, w, _ = self.__kernel_shape
        stride_h, stride_w = self.__strides

        # ndarray to multiply with itemsize
        window_stride = np.asarray((H*W*C, stride_h*W*C, stride_w*C, W*C, C, 1))
        window_shape = (B, 
                        conv_size(H, h, stride_h),
                        conv_size(W, w, stride_w), 
                        h, w, C)
        return window_stride, window_shape

    def compute_padded_input(self):
        B, H, W, C = self.input_shape
        _, h, w, _ = self.__kernel_shape
        stride_h, stride_w = self.__strides
        in_H = in_conv_size(H, h, stride_h)
        in_W = in_conv_size(W, w, stride_w)
        padded_input_shape = (B, in_H, in_W, C)
        
        # (pad_H_top, pad_H_bot, pad_W_left, pad_W_right)
        pad_size = ((in_H - H)//2, ceil((in_H - H)/2), (in_W - W)//2, ceil((in_W - W)/2))
        return padded_input_shape, pad_size

    def init_regularizers(self):
        # kernel
        if (self.__kernel_regularizer is not None):
            self.__kernel_regularizer.shape = self.__kernel_shape

        # bias
        if (self.__bias_regularizer is not None):
            self.__bias_regularizer.shape = to_tuple(self.__kernels)

    def init_kernel(self, param, type, weight_bias = 0):
        """
        Params:
            type: weight initalization type
                [he, xavier]
        """

        K, h, w, C = self.__kernel_shape
        scale = 1
        input_size = h*w*C
        if  (type == "xavier"):
            scale = 1 / np.sqrt(input_size) # Xavier init
        elif (type == "he"):
            scale = np.sqrt(2 / input_size) # he init, for relus
        else:
            raise ValueError(f'no such weight type "{type}"')

        param.kernel = np.random.normal(loc=weight_bias, scale=scale, size=self.__kernel_shape)
        param.weight_type = self.__weight_type

    def init_base(self):
        """
        base matrix used to compute dldx, dldk
        """
        B, H, W, C = self.input_shape
        if (self.__pad):
            B, H, W, C = self.__padded_input_shape
        K, h, w, _ = self.__kernel_shape
        pad_h = h - 1
        pad_w = w - 1
        in_h = in_conv_size(H, h, 1)
        in_w = in_conv_size(W, w, 1)
        out_h = conv_size(H, h, self.__strides[0])
        out_w = conv_size(W, w, self.__strides[1])
        stride_h, stride_w = self.__strides

        # base
        base_shape = (B, in_h, in_w, K)
        self.__base = np.zeros(base_shape)

        size = self.__base.itemsize
        strides = (in_h*in_w*K*size, stride_h*in_w*K*size, stride_w*K*size, size)
        base_core = self.__base[:, pad_h:in_h-pad_h,
                              pad_w:in_w-pad_w, :]
        self.__base_view = np.lib.stride_tricks.as_strided(base_core, 
                                                         shape=self.output_shape, 
                                                         strides=strides)
        base_window_shape = (B, 
                             conv_size(in_h, h, 1),
                             conv_size(in_w, w, 1), 
                             h, w, K)
        base_window_stride = (in_h*in_w*K*size, 
                              in_w*K*size, 
                              K*size, in_w*K*size, K*size, size)
        self.__base_window_view = np.lib.stride_tricks.as_strided(self.__base, 
                                                                shape=base_window_shape, 
                                                                strides=base_window_stride,
                                                                writeable=False)

        # dldz kernel
        dldz_kernel_shape = (B, K, in_h - 2*pad_h, in_w - 2*pad_w)
        dldz_kernel_stride = (K*in_h*in_w*size, size, in_w*K*size, K*size)
        self.__dldz_kernel_view = np.lib.stride_tricks.as_strided(base_core, 
                                                  shape=dldz_kernel_shape, 
                                                  strides=dldz_kernel_stride,
                                                  writeable=False)

        # input window view for dldz kernel
        _, _, h, w = dldz_kernel_shape
        self.__dldz_window_stride = np.asarray((H*W*C, W*C, C, W*C, C, 1))
        self.__dldz_window_shape = (B, 
                                  conv_size(H, h, 1),
                                  conv_size(W, w, 1), 
                                  h, w, C)

    def create_param(self):
        super().create_param()
        param = Convolve2DParam()
        self.init_kernel(param, self.__weight_type, 0)
        param.bias = np.zeros(self.__kernels)
        param.pad = self.__pad
        if (self.__kernel_regularizer is not None):
            param.kernel_regularizer = copy.deepcopy(self.__kernel_regularizer)
            param.kernel_regularizer.compile()
        if (self.__bias_regularizer is not None):
            param.bias_regularizer = copy.deepcopy(self.__bias_regularizer)
            param.bias_regularizer.compile()
        return param

    def regularizers_ok(self, kernel_regularizer, bias_regularizer):
        kernel_regularizer_ok = True
        if (kernel_regularizer is not None):
            kernel_regularizer_ok = kernel_regularizer.shape == self.__kernel_shape

        bias_regularizer_ok = True
        if (bias_regularizer is not None):
            bias_regularizer_ok = bias_regularizer.shape == (self.__kernels, )

        return kernel_regularizer_ok and bias_regularizer_ok

    def param_compatible(self, param: Convolve2DParam):
        """
        Check if parameter is compatible

        Args:
            param: parameter to check

        Returns:
            is compatible
        """

        bias_ok = (param.bias is not None) and param.bias.shape[0] == self.__kernels
        kernel_ok = (param.kernel is not None) and param.kernel.shape == self.__kernel_shape

        regularizer_ok = self.regularizers_ok(param.kernel_regularizer, 
                                              param.bias_regularizer)

        pad_ok = param.pad is not None and param.pad == self.__pad

        return bias_ok and kernel_ok and regularizer_ok and pad_ok

    def apply_param(self, param: Convolve2DParam):
        """
        Applies layer param

        Args:
            param: parameter to be applied
        """

        super().apply_param(param)
        self.__kernel_rot180 = np.rot90(param.kernel, k=2, axes=(1, 2))
    
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
        self.__input = X

        if (self.param.pad):
            np.copyto(self.__padded_input_view, X)
            X = self.__padded_input
        strides = self.__window_stride * X.itemsize
        view = np.lib.stride_tricks.as_strided(X, 
                                               shape=self.__window_shape, 
                                               strides=strides,
                                               writeable=False)
        np.copyto(self.output, np.einsum('BHWhwC,khwC->BHWk', view, self.param.kernel, 
                  optimize="optimal"))
        np.add(self.output, self.param.bias, out=self.output)
        return self.output

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
        np.copyto(self.__base_view, dldz)
        dldx = np.einsum('BHWhwK,KhwC->BHWC', self.__base_window_view, 
                         self.__kernel_rot180, optimize="optimal")

        # compute dldk gradient
        input = self.__input
        if (self.__pad):
            input = self.__padded_input
        strides = self.__dldz_window_stride * input.itemsize
        view = np.lib.stride_tricks.as_strided(input, 
                                               shape=self.__dldz_window_shape, 
                                               strides=strides,
                                               writeable=False)
        dldk = np.einsum('BHWhwC,BKhw->KHWC', view, 
                            self.__dldz_kernel_view, optimize="optimal")
        if (self.__pad):
            _, H, W, _ = self.__padded_input_shape
            pad_H_top, pad_H_bot, pad_W_left, pad_W_right = self.__pad_size
            dldx = dldx[:, pad_H_top:H-pad_H_bot, pad_W_left:W-pad_W_right, :]
        np.copyto(self.dldx, dldx)
        np.copyto(self.__dldk, dldk)

        # compute dldb gradient
        np.sum(dldz, axis=(0, 1, 2), out=self.__dldb)

        return self.dldx

    def apply_grad(self, step, dldk=None, dldb=None):
        """
        Applies gradients

        Args:
            step: gradient step
            dldk: gradient of loss with respect to kernel
            dldb: gradient of loss with respect to bias
        """

        if (dldk is None):
            dldk = self.__dldk
        if (dldb is None):
            dldb = self.__dldb
        # weight update
        np.add(self.param.kernel, step * dldk, out=self.param.kernel)
        # bias update
        np.add(self.param.bias, step * dldb, out=self.param.bias)

    def __str__(self):
        return self.attribute_to_str()

    def attribute_to_str(self):
        return (super().attribute_to_str()
        + f"weight init:    {self.__weight_type}\n"
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
