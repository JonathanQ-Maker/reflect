from __future__ import annotations
from reflect.layers.parametric_layer import ParametricLayer
from reflect import np
from reflect.utils.misc import to_tuple, conv_size, in_conv_size
import copy
from math import ceil

class Convolve2D(ParametricLayer):

    input = None


    window_stride = None        # ndarray type, used to multiply with input item size
    window_shape = None
    base = None
    base_view = None            # base viewed as dldz matrix
    base_window_view = None     # base viewed as windows 
    dldz_kernel_view = None          # base viewed as kernel dldz matrix for dldk
    dldz_window_stride = None   # input window view stride for dldz_kernel_view
    dldz_window_shape = None    # input window view shape for dldz_kernel_view

    dldk = None
    dldb = None

    kernel_rot180 = None        # view object of kernel at 180 degrees
    kernel_shape = None         # shape: kernel, height, width, channel
    kernel_size = None          # single filter size, (height, width)
    kernels = 1                 # number of kernels
    weight_type = None          # weight initialization type
    strides = (1, 1)            # (stride y, stride x), convolution strides
    pad = False                 # zero pad input to maintain spatial dimention
    padded_input_shape = None   # input shape after padded
    padded_input = None         # array used as input for padded inputs
    pad_size = None             # (pad_H_top, pad_H_bot, pad_W_left, pad_W_right) amount padded along H and W axis
    padded_input_view = None    # padded_input view without padding, used for input paste

    kernel_regularizer = None
    bias_regularizer = None


    def __init__(self, input_size = (1, 1, 1), kernel_size = (1, 1),
                 kernels = 1,
                 batch_size = 1, 
                 strides = (1, 1),
                 weight_type = "he",
                 pad = False,
                 kernel_regularizer=None,
                 bias_regularizer=None):
        super().__init__(input_size, None, batch_size)
        self.weight_type            = weight_type
        self.kernel_regularizer     = kernel_regularizer
        self.bias_regularizer       = bias_regularizer
        self.strides                = strides
        self.pad                    = pad
        self.kernel_size            = kernel_size
        self.kernels                = kernels

    def compile(self, gen_param=True):
        self.kernel_shape = self.comp_kernel_shape()
        self.input_shape = (self.batch_size, ) + to_tuple(self.input_size)
        if (self.pad):
            self.padded_input_shape, self.pad_size = self.comp_padded_input()
            self.padded_input = np.zeros(shape=self.padded_input_shape)
            _, H, W, _ = self.padded_input_shape
            pad_H_top, pad_H_bot, pad_W_left, pad_W_right = self.pad_size
            self.padded_input_view = self.padded_input[:, pad_H_top:H-pad_H_bot, pad_W_left:W-pad_W_right, :]
        self.output_size = self.comp_output_shape()[1:]
        self.output_shape = (self.batch_size, ) + to_tuple(self.output_size)

        # compile output
        self.output = np.zeros(shape=self.output_shape)
        self.dldx = np.zeros(shape=self.input_shape)

        # compile gradient
        self.dldk = np.zeros(shape=self.kernel_shape)
        self.dldb = np.zeros(shape=self.kernels)

        self.init_regularizers()
        self.init_base()
        self.window_stride, self.window_shape = self.comp_view_attr()

        self.name = f"Dense {self.output_size}"
        if (gen_param):
            self.apply_param(self.create_param())

    def is_compiled(self):
        kernel_shape_match = self.kernel_shape == self.comp_kernel_shape()

        dldk_ok = self.dldk is not None and self.dldk.shape == self.kernel_shape
        dldb_ok = self.dldb is not None and self.dldb.shape[0] == self.kernels

        regularizer_ok = self.regularizers_ok(self.kernel_regularizer, 
                                              self.bias_regularizer)
        stride, shape = self.comp_view_attr()
        window_view_ok = np.all(stride == self.window_stride) and shape == self.window_shape
        base_ok = self.base_view is not None and self.base_view.shape == self.output_shape

        return (super().is_compiled() 
                and kernel_shape_match 
                and regularizer_ok 
                and dldk_ok and dldb_ok
                and window_view_ok
                and base_ok)

    def comp_output_shape(self):
        if (self.pad):
            return (self.batch_size, 
                    conv_size(self.padded_input_shape[1], self.kernel_size[0], self.strides[0]),
                    conv_size(self.padded_input_shape[2], self.kernel_size[1], self.strides[1]),
                    self.kernels)
        return (self.batch_size, 
                conv_size(self.input_size[0], self.kernel_size[0], self.strides[0]),
                conv_size(self.input_size[1], self.kernel_size[1], self.strides[1]),
                self.kernels)

    def comp_kernel_shape(self):
        if isinstance(self.kernel_size, int):
            return (self.kernels, 
                    self.kernel_size, 
                    self.kernel_size, 
                    self.input_size[2])
        else:
            return (self.kernels, ) + self.kernel_size + (self.input_size[2], )

    def comp_view_attr(self):
        """
        Computes window view attributes for input
        returns window_stride(ndarray), window_shape
        """

        # forward view attributes
        B, H, W, C = self.input_shape
        if (self.pad):
            B, H, W, C = self.padded_input_shape
        K, h, w, _ = self.kernel_shape
        stride_h, stride_w = self.strides

        # ndarray to multiply with itemsize
        window_stride = np.asarray((H*W*C, stride_h*W*C, stride_w*C, W*C, C, 1))
        window_shape = (B, 
                        conv_size(H, h, stride_h),
                        conv_size(W, w, stride_w), 
                        h, w, C)
        return window_stride, window_shape

    def comp_padded_input(self):
        B, H, W, C = self.input_shape
        _, h, w, _ = self.kernel_shape
        stride_h, stride_w = self.strides
        in_H = in_conv_size(H, h, stride_h)
        in_W = in_conv_size(W, w, stride_w)
        padded_input_shape = (B, in_H, in_W, C)
        
        # (pad_H_top, pad_H_bot, pad_W_left, pad_W_right)
        pad_size = ((in_H - H)//2, ceil((in_H - H)/2), (in_W - W)//2, ceil((in_W - W)/2))
        return padded_input_shape, pad_size

    def init_regularizers(self):
        # kernel
        if (self.kernel_regularizer is not None):
            self.kernel_regularizer.shape = self.kernel_shape

        # bias
        if (self.bias_regularizer is not None):
            self.bias_regularizer.shape = to_tuple(self.kernels)

    def init_kernel(self, param, type, weight_bias = 0):
        """
        Params:
            type: weight initalization type
                [he, xavier]
        """

        K, h, w, C = self.kernel_shape
        scale = 1
        input_size = h*w*C
        if  (type == "xavier"):
            scale = 1 / np.sqrt(input_size) # Xavier init
        elif (type == "he"):
            scale = np.sqrt(2 / input_size) # he init, for relus
        else:
            raise ValueError(f'no such weight type "{type}"')

        param.kernel = np.random.normal(loc=weight_bias, scale=scale, size=self.kernel_shape)
        param.weight_type = self.weight_type

    def init_base(self):
        """
        base matrix used to compute dldx, dldk
        """
        B, H, W, C = self.input_shape
        if (self.pad):
            B, H, W, C = self.padded_input_shape
        K, h, w, _ = self.kernel_shape
        pad_h = h - 1
        pad_w = w - 1
        in_h = in_conv_size(H, h, 1)
        in_w = in_conv_size(W, w, 1)
        out_h = conv_size(H, h, self.strides[0])
        out_w = conv_size(W, w, self.strides[1])
        stride_h, stride_w = self.strides

        # base
        base_shape = (B, in_h, in_w, K)
        self.base = np.zeros(base_shape)

        size = self.base.itemsize
        strides = (in_h*in_w*K*size, stride_h*in_w*K*size, stride_w*K*size, size)
        base_core = self.base[:, pad_h:in_h-pad_h,
                              pad_w:in_w-pad_w, :]
        self.base_view = np.lib.stride_tricks.as_strided(base_core, 
                                                         shape=self.output_shape, 
                                                         strides=strides)
        base_window_shape = (B, 
                             conv_size(in_h, h, 1),
                             conv_size(in_w, w, 1), 
                             h, w, K)
        base_window_stride = (in_h*in_w*K*size, 
                              in_w*K*size, 
                              K*size, in_w*K*size, K*size, size)
        self.base_window_view = np.lib.stride_tricks.as_strided(self.base, 
                                                                shape=base_window_shape, 
                                                                strides=base_window_stride,
                                                                writeable=False)

        # dldz kernel
        dldz_kernel_shape = (B, K, in_h - 2*pad_h, in_w - 2*pad_w)
        dldz_kernel_stride = (K*in_h*in_w*size, size, in_w*K*size, K*size)
        self.dldz_kernel_view = np.lib.stride_tricks.as_strided(base_core, 
                                                  shape=dldz_kernel_shape, 
                                                  strides=dldz_kernel_stride,
                                                  writeable=False)

        # input window view for dldz kernel
        _, _, h, w = dldz_kernel_shape
        self.dldz_window_stride = np.asarray((H*W*C, W*C, C, W*C, C, 1))
        self.dldz_window_shape = (B, 
                                  conv_size(H, h, 1),
                                  conv_size(W, w, 1), 
                                  h, w, C)

    def create_param(self):
        super().create_param()
        param = Convolve2DParam()
        self.init_kernel(param, self.weight_type, 0)
        param.bias = np.zeros(self.kernels)
        param.pad = self.pad
        if (self.kernel_regularizer is not None):
            param.kernel_regularizer = copy.deepcopy(self.kernel_regularizer)
            param.kernel_regularizer.compile()
        if (self.bias_regularizer is not None):
            param.bias_regularizer = copy.deepcopy(self.bias_regularizer)
            param.bias_regularizer.compile()
        return param

    def regularizers_ok(self, kernel_regularizer, bias_regularizer):
        kernel_regularizer_ok = True
        if (kernel_regularizer is not None):
            kernel_regularizer_ok = kernel_regularizer.shape == self.kernel_shape

        bias_regularizer_ok = True
        if (bias_regularizer is not None):
            bias_regularizer_ok = bias_regularizer.shape == (self.kernels, )

        return kernel_regularizer_ok and bias_regularizer_ok

    def param_compatible(self, param: Convolve2DParam):
        bias_ok = (param.bias is not None) and param.bias.shape[0] == self.kernels
        kernel_ok = (param.kernel is not None) and param.kernel.shape == self.kernel_shape

        regularizer_ok = self.regularizers_ok(param.kernel_regularizer, 
                                              param.bias_regularizer)

        pad_ok = param.pad is not None and param.pad == self.pad

        return bias_ok and kernel_ok and regularizer_ok and pad_ok

    def apply_param(self, param):
        super().apply_param(param)
        self.kernel_rot180 = np.rot90(param.kernel, k=2, axes=(1, 2))
    
    def forward(self, X):
        """
        return: output

        Make copy of output if intended to be modified
        Input instance will be kept and expected not to be modified between forward and backward pass
        """
        self.input = X
        strides = self.window_stride * self.input.itemsize

        if (self.param.pad):
            np.copyto(self.padded_input_view, X)
            X = self.padded_input
        view = np.lib.stride_tricks.as_strided(X, 
                                               shape=self.window_shape, 
                                               strides=strides,
                                               writeable=False)
        np.copyto(self.output, np.einsum('BHWhwC,khwC->BHWk', view, self.param.kernel, 
                  optimize="optimal"))
        np.add(self.output, self.param.bias, out=self.output)
        return self.output

    def backprop(self, dldz):
        """
        return: dldx, gradient of loss with respect to input

        Make copy of dldk, dldb, dldx if intended to be modified
        """
        # compute dldx gradient
        np.copyto(self.base_view, dldz)
        dldx = np.einsum('BHWhwK,KhwC->BHWC', self.base_window_view, 
                         self.kernel_rot180, optimize="optimal")

        # compute dldk gradient
        strides = self.dldz_window_stride * self.input.itemsize
        view = np.lib.stride_tricks.as_strided(self.padded_input, 
                                               shape=self.dldz_window_shape, 
                                               strides=strides,
                                               writeable=False)
        dldk = np.einsum('BHWhwC,BKhw->KHWC', view, 
                            self.dldz_kernel_view, optimize="optimal")
        if (self.pad):
            _, H, W, _ = self.padded_input_shape
            pad_H_top, pad_H_bot, pad_W_left, pad_W_right = self.pad_size
            dldx = dldx[:, pad_H_top:H-pad_H_bot, pad_W_left:W-pad_W_right, :]
        np.copyto(self.dldx, dldx)
        np.copyto(self.dldk, dldk)

        # compute dldb gradient
        np.sum(dldz, axis=(0, 1, 2), out=self.dldb)

        return self.dldx

    def apply_grad(self, step, dldw=None, dldb=None):
        if (dldw is None):
            dldw = self.dldw
        if (dldb is None):
            dldb = self.dldb
        # weight update
        np.add(self.param.weight, step * dldw, out=self.param.weight)
        # bias update
        np.add(self.param.bias, step * dldb, out=self.param.bias)

    def __str__(self):
        return self.attribute_to_str()

    def attribute_to_str(self):
        return (super().attribute_to_str()
        + f"weight init:    {self.weight_type}\n"
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
