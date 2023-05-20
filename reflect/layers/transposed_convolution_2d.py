from __future__ import annotations
from reflect.layers.parametric_layer import ParametricLayer, Parameter
from reflect.layers.cached_layer import CachedLayer, LayerCache
from reflect.optimizers import Adam
from reflect import np
from reflect.utils.misc import to_tuple, conv_size, in_conv_size

class TransposedConv2D(CachedLayer, ParametricLayer):

    """
    Transpose 2D Convolution layer

    Shape:
        input:  (batch size, height, wdith, channels)
        
        output: (batch size, 
                in_conv_size(height, filter, stride), 
                in_conv_size(width, filter, stride), 
                kernels)
    """

    # public variables
    _filter_size        = None  # filter size, (height, width) or scaler
    kernels             = 1     # number of kernels
    _kernel_shape       = None  # (num kernel, height, width, channel)
    weight_type         = None  # weight initialization type
    _strides            = (1, 1)# (stride y, stride x), convolution strides

    _dldk               = None  # gradient of loss with respect to kernel
    _dldb               = None  # gradient of loss with respect to bias
    _readonly_dldk      = None  # read only dldk view
    _readonly_dldb      = None  # read only dldw view

    kernel_reg          = None  # kernel regularizer
    bias_reg            = None  # bias regularizer

    kernel_optimizer    = None
    bias_optimizer      = None

    kernel_constraint   = None
    bias_constraint     = None

    # internal variables
    _base_shape         = None
    _pad_h              = None
    _pad_w              = None
    _H_b                = None
    _W_b                = None
    _core_stride        = None

    _base_kernel_window_shape   = None
    _base_window_stride         = None
    _base_dldz_window_shape     = None


    # dldz-kernel view attributes
    _dldz_kernel_shape  = None
    _dldz_kernel_stride = None

    # dldz & kernel window view attributes
    _dldz_window_shape  = None
    _dldz_window_stride = None

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
    def total_params(self):
        return self.param.kernel.size + self.param.bias.size

    def __init__(self, 
                 filter_size        = (1, 1),
                 kernels            = 1,
                 strides            = (1, 1),
                 weight_type        = "he",
                 kernel_reg         = None,
                 bias_reg           = None,
                 kernel_optimizer   = None,
                 bias_optimizer     = None,
                 kernel_constraint  = None,
                 bias_constraint    = None):


        super().__init__()
        self.weight_type            = weight_type
        self.strides                = strides
        self.filter_size            = filter_size
        self.kernels                = kernels
        self.kernel_reg             = kernel_reg
        self.bias_reg               = bias_reg
        self.kernel_optimizer       = kernel_optimizer
        self.bias_optimizer         = bias_optimizer
        self.kernel_constraint      = kernel_constraint
        self.bias_constraint        = bias_constraint

        if kernel_optimizer is None:
            self.kernel_optimizer   = Adam()
        if bias_optimizer is None:
            self.bias_optimizer     = Adam()

    def compile(self, input_size, batch_size=1, gen_param=True):
        self._input_size    = input_size
        self._batch_size    = batch_size
        self._kernel_shape  = self.compute_kernel_shape()
        self._input_shape   = (self._batch_size, ) + self._input_size
        self._output_shape = self.compute_output_shape()
        self._output_size  = self._output_shape[1:]

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

        attributes = self.compute_view_attr()
        self._dldz_kernel_shape     = attributes[0]
        self._dldz_kernel_stride    = attributes[1]
        self._dldz_window_shape     = attributes[2]
        self._dldz_window_stride    = attributes[3]

        # compile base attributes
        base_attr = self.compute_base_attr()
        self._base_shape = base_attr[0]
        # base_input_view attrbutes, view used to copy input into base
        self._pad_h, self._pad_w, self._H_b, self._W_b, self._core_stride = base_attr[1:6]
        # base_kernel_window_view attribute, base & kernel window view
        self._base_kernel_window_shape = base_attr[6]
        # shared stride for base_kernel_window_view and 
        self._base_window_stride = base_attr[7]
        # base_dldz_window_view attribute, base & dldz-kernel window view
        self._base_dldz_window_shape = base_attr[8]


        # compile regularizers
        if (self.kernel_reg is not None):
            self.kernel_reg.compile(self._kernel_shape)
        if (self.bias_reg is not None):
            self.bias_reg.compile(to_tuple(self.kernels))

        # compile constraints
        if (self.kernel_constraint is not None):
            self.kernel_constraint.compile(self._kernel_shape)
        if (self.bias_constraint is not None):
            self.bias_constraint.compile(to_tuple(self.kernels))

        # compile optimizers
        self.kernel_optimizer.compile(self._kernel_shape)
        self.bias_optimizer.compile(self.kernels)

        self.name = f"{self.kernels} TransposedConv2D {self._filter_size[0]}x{self._filter_size[1]}"
        if (gen_param):
            self.apply_param(self.create_param())

        self._cache = self.create_cache()

    def is_compiled(self):
        if (not super().is_compiled()):
            return False
        kernel_shape_match = self._kernel_shape == self.compute_kernel_shape()

        dldk_ok = (self._dldk is not None 
                   and self._dldk.shape == self._kernel_shape)
        dldb_ok = (self._dldb is not None 
                   and self._dldb.shape[0] == self.kernels)

        attributes = self.compute_view_attr()
        attributes_ok = (np.array_equal(self._dldz_kernel_shape, attributes[0])
                         and np.array_equal(self._dldz_kernel_stride, attributes[1])
                         and np.array_equal(self._dldz_window_shape, attributes[2])
                         and np.array_equal(self._dldz_window_stride, attributes[3]))

        kernel_optimizer_ok = (self.kernel_optimizer is not None
                         and self.kernel_optimizer.is_compiled()
                         and self.kernel_optimizer.shape == self._kernel_shape)

        bias_optimizer_ok = (self.bias_optimizer is not None
                         and self.bias_optimizer.is_compiled()
                         and self.bias_optimizer.shape == to_tuple(self.kernels))

        kernel_constrain_ok = True
        if (self.kernel_constraint is not None):
            kernel_constrain_ok = self.kernel_constraint.is_compiled()
        bias_constraint_ok = True
        if (self.bias_constraint is not None):
            bias_constraint_ok = self.bias_constraint.is_compiled()
        constraints_ok = kernel_constrain_ok and bias_constraint_ok


        return (kernel_shape_match 
                and dldk_ok and dldb_ok
                and attributes_ok
                and kernel_optimizer_ok
                and bias_optimizer_ok
                and constraints_ok)

    def compute_kernel_shape(self):
        return (self.kernels, ) + self._filter_size + (self._input_size[2], )

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
        output_size = K
        if  (type == "xavier"):
            scale = np.sqrt(2.0 / (input_size + output_size)) # Xavier init
        elif (type == "he"):
            scale = np.sqrt(2.0 / input_size) # he init, for relus
        elif (type == "xavier_uniform"):
            scale = np.sqrt(6.0 / (input_size + output_size))
            param.add_weight("kernel", np.random.uniform(low=-scale, high=scale, size=self._kernel_shape))
            return
        else:
            raise ValueError(f'no such weight type "{type}"')

        param.add_weight("kernel", np.random.normal(loc=weight_bias, scale=scale, size=self._kernel_shape))

    def compute_output_shape(self):
        return (self._batch_size, 
                in_conv_size(self._input_size[0], self._filter_size[0], self._strides[0]),
                in_conv_size(self._input_size[1], self._filter_size[1], self._strides[1]),
                self.kernels)

    def compute_base_attr(self):
        B, H, W, C  = self._input_shape
        _, h, w, _  = self._kernel_shape
        stride_h = self._strides[0]
        stride_w = self._strides[1]

        # NOTE: stride 1 convolution on input with modified shape 
        # and stride spaced dldz-kernel computes dldk

        in_h       = in_conv_size(H, h, stride_h)
        in_w       = in_conv_size(W, w, stride_w)

        H_b = in_conv_size(in_h, h, 1)
        W_b = in_conv_size(in_w, w, 1)

        base_shape = (B, H_b, W_b, C)
        size = 8

        pad_h           = h - 1
        pad_w           = w - 1
        core_stride     = (H_b*W_b*C*size, stride_h*W_b*C*size, stride_w*C*size, size)

        # base & kernel window view
        base_kernel_window_shape       = (B, in_h, in_w, h, w, C)
        base_window_stride      = (H_b*W_b*C*size, 
                                   W_b*C*size, C*size, 
                                   W_b*C*size, C*size, 
                                   size)

        # base & dldz-kernel window view
        base_dldz_window_shape       = (B, h, w, in_h, in_w, C)

        return (base_shape,
                pad_h, pad_w, H_b, W_b, core_stride,
                base_kernel_window_shape, 
                base_window_stride, 
                base_dldz_window_shape)

    def compute_view_attr(self):
        B, H, W, C  = self._input_shape
        K, h, w, _  = self._kernel_shape
        stride_h = self._strides[0]
        stride_w = self._strides[1]

        in_h       = in_conv_size(H, h, stride_h)
        in_w       = in_conv_size(W, w, stride_w)

        # dldz-kernel view attributes
        dldz_kernel_shape   = (B, K, in_h, in_w)
        dldz_kernel_stride  = np.asarray((K*in_h*in_w, 1, in_w*K, K))

        # dldz window view attributes
        window_shape    = (B, conv_size(in_h, h, stride_h), conv_size(in_w, w, stride_w), h, w, K)
        window_stride   = np.asarray((in_h*in_w*K, stride_h*in_w*K, stride_w*K, in_w*K, K, 1))

        return dldz_kernel_shape, dldz_kernel_stride, window_shape, window_stride

    def create_param(self):
        super().create_param()
        param = TransposedConv2DParam()

        self.init_kernel(param, self.weight_type, 0)
        param.add_weight("bias", np.zeros(self.kernels))
        return param

    def param_compatible(self, param: TransposedConv2DParam):
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

    def apply_param(self, param: TransposedConv2DParam):
        """
        Applies layer param

        Args:
            param: parameter to be applied
        """

        super().apply_param(param)

    def create_cache(self):
        """
        Create and return empty cache

        Return:
            cache
        """
        cache = TransposedConv2DCache()
        cache._owner = self
        cache._base  = np.zeros(self._base_shape)
        base_core = cache._base[:, self._pad_h:self._H_b-self._pad_h, 
                                self._pad_w:self._W_b-self._pad_w, :]

        # base input view
        cache._base_input_view = np.lib.stride_tricks.as_strided(base_core, 
            shape=self._input_shape, strides=self._core_stride)

        # base & kernel window view
        cache._base_kernel_window_view = np.lib.stride_tricks.as_strided(cache._base, 
            shape=self._base_kernel_window_shape, strides=self._base_window_stride,
            writeable=False)

        # base & dldz-kernel window view
        cache._base_dldz_window_view = np.lib.stride_tricks.as_strided(cache._base, 
            shape=self._base_dldz_window_shape, strides=self._base_window_stride,
            writeable=False)

        cache._kernel = np.zeros(self._kernel_shape)
        return cache
    
    def forward(self, X, out_cache: TransposedConv2DCache=None):
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

        # NOTE: stride 1 convolution on stride spaced input with
        # 180 rotated kernel computes output
        np.copyto(out_cache._base_input_view, X)
        np.copyto(out_cache._kernel, self.param.kernel)

        kernel_rot180 = np.rot90(self.param.kernel, k=2, axes=(1, 2))
        output = np.einsum('BHWhwc,khwc->BHWk', out_cache._base_kernel_window_view, 
                           kernel_rot180, optimize="optimal")
        np.copyto(self._output, output)
        np.add(self._output, self.param.bias, out=self._output)
        return self._readonly_output

    def backprop(self, dldz, cache: TransposedConv2DCache=None):
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

        size = dldz.itemsize

        # NOTE: strided convolution on dldz with kernel
        # computes dldx
        dldz_window_view = np.lib.stride_tricks.as_strided(dldz, shape=self._dldz_window_shape, 
                                                          strides=self._dldz_window_stride*size,
                                                          writeable=False)
        dldx = np.einsum('BHWhwK,KhwC->BHWC', dldz_window_view, 
                         cache._kernel, optimize="optimal")

        # NOTE: stride 1 convolution on input with modified shape 
        # dldz-kernel computes dldk
        dldz_kernel_view = np.lib.stride_tricks.as_strided(dldz, shape=self._dldz_kernel_shape, 
                                                          strides=self._dldz_kernel_stride*size,
                                                          writeable=False)
        dldk = np.einsum('BHWhwC,BKhw->KHWC', cache._base_dldz_window_view, 
                            dldz_kernel_view, optimize="optimal")
        dldk = np.rot90(dldk, k=2, axes=(1, 2))

        # dldb
        np.sum(dldz, axis=(0, 1, 2), out=self._dldb)

        np.copyto(self._dldx, dldx)
        np.copyto(self._dldk, dldk)

        # add regularizer
        if (self.kernel_reg is not None):
            np.add(self._dldk, self.kernel_reg.gradient(self.param.kernel), out=self._dldk)
        if (self.bias_reg is not None):
            np.add(self._dldb, self.bias_reg.gradient(self.param.bias), out=self._dldb)
        return self._readonly_dldx

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
        if (self.kernel_constraint is not None):
            self.kernel_constraint.constrain(self.param.kernel)

        # bias update
        np.subtract(self.param.bias, self.bias_optimizer.gradient(step, dldb), out=self.param.bias)
        if (self.bias_constraint is not None):
            self.bias_constraint.constrain(self.param.bias)


    def attribute_to_str(self):
        return (super().attribute_to_str()
        + f"weight init:    {self.weight_type}\n"
        + f"max kernel:     {self.param.kernel.max()}\n"
        + f"min kernel:     {self.param.kernel.min()}\n"
        + f"kernel std:     {np.std(self.param.kernel)}\n"
        + f"kernel mean:    {np.mean(self.param.kernel)}\n")







class TransposedConv2DParam(Parameter):
    @property
    def kernel(self):
        return self.get_weight("kernel")
    
    @property
    def bias(self):
        return self.get_weight("bias")

class TransposedConv2DCache(LayerCache):
    _base                   = None
    _base_input_view        = None
    _base_kernel_window_view= None 
    _base_dldz_window_view  = None

    _kernel                 = None