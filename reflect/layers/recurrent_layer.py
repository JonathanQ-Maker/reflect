from __future__ import annotations
from reflect.layers.abstract_rnn import AbstractRNN, Parameter
from reflect.layers.cached_layer import CachedLayer
from reflect.layers.relu_layer import Relu
from reflect.optimizers.adam import Adam
from reflect.utils.misc import to_tuple
from reflect import np

class Recurrent(AbstractRNN):

    weight_type         = None # weight initialization type
    _weight_shape       = None # (input size, units)                    NOTE: transposed
    _hidden_shape       = None # hidden_weight shape, (units, units)    NOTE: transposed
    _state_view         = None # current temporal state, or last time step output
    _state_shape        = None # (batch size, units)

    _dldw               = None # gradient of loss with respect to weight
    _dldh               = None # gtaident of loss with respect to hidden_weight
    _dldb               = None # graident of loss with respect to bias
    _dlds               = None # graident of loss with respect to state

    # readonly views of arrays
    _readonly_dldw      = None
    _readonly_dldh      = None
    _readonly_dldb      = None

    activation          = None # hidden activation layer
    
    # optimizers
    weight_optimizer    = None
    bias_optimizer      = None
    hidden_optimizer    = None

    # regularizers
    weight_reg          = None # weight regularizer
    bias_reg            = None # bias regularizer
    hidden_reg          = None # hidden weight regularizer

    # internal variables
    _step_dldw          = None # gradient of loss with respect to weight at time step
    _step_dldh          = None # gradient of loss with respect to hidden weight at time step
    _step_dldb          = None # gradient of loss with respect to bias at time step

    _hidden_result      = None # buffer to store result of Z@H

    @property
    def units(self):
        return self._output_size

    @units.setter
    def units(self, units):
        self._output_size = units

    @property
    def state(self):
        return self._state_view

    @property
    def dldw(self):
        return self._readonly_dldw

    @property
    def dldh(self):
        return self._readonly_dldh

    @property
    def dldb(self):
        return self._readonly_dldb

    def __init__(self, 
                    units, 
                    truncate_length     = 5,
                    activation          = None,
                    weight_type         = "he",
                    weight_optimizer    = None,
                    bias_optimizer      = None,
                    hidden_optimizer    = None,
                    weight_reg          = None,
                    bias_reg            = None,
                    hidden_reg          = None):
        super().__init__(truncate_length)
        self._output_size       = units
        self.activation         = activation
        self.weight_type        = weight_type
        self.weight_optimizer   = weight_optimizer
        self.bias_optimizer     = bias_optimizer
        self.hidden_optimizer   = hidden_optimizer
        self.weight_reg         = weight_reg
        self.bias_reg           = bias_reg
        self.hidden_reg         = hidden_reg

        #if (self.activation is None):
        #    self.activation = Relu()

        if (self.weight_optimizer is None):
            self.weight_optimizer   = Adam()
        if (self.bias_optimizer is None):
            self.bias_optimizer     = Adam()
        if (self.hidden_optimizer is None):
            self.hidden_optimizer   = Adam()

    def compile(self, input_size, batch_size, time_steps, gen_param=True):
        super().compile(input_size, batch_size, time_steps, gen_param)
        
        # compile shapes
        self._weight_shape  = (self._input_size, self._output_size)
        self._hidden_shape  = (self._output_size, self._output_size)
        self._state_shape   = (self._batch_size, self._output_size)

        # compile arrays
        self._state = np.zeros(self._state_shape)
        self._dlds  = np.zeros(self._state_shape)
        self._dldw  = np.zeros(self._weight_shape)
        self._dldh  = np.zeros(self._hidden_shape)
        self._dldb  = np.zeros(self._output_size)

        self._state_view        = self._output[-1]
        self._readonly_dldw     = self._dldw.view()
        self._readonly_dldh     = self._dldh.view()
        self._readonly_dldb     = self._dldb.view()

        self._state_view.flags.writeable = False
        self._readonly_dldw.flags.writeable = False
        self._readonly_dldh.flags.writeable = False
        self._readonly_dldb.flags.writeable = False

        self._step_dldw  = np.zeros(self._weight_shape)
        self._step_dldh  = np.zeros(self._hidden_shape)
        self._step_dldb  = np.zeros(self._output_size)

        self._hidden_result = np.zeros(self._state_shape)

        if (self.activation is not None):
            self.activation.compile(self._output_size, self._batch_size)

        # compile regularizers
        if (self.weight_reg is not None):
            self.weight_reg.compile(self._weight_shape)
        if (self.bias_reg is not None):
            self.bias_reg.compile(to_tuple(self._output_size))
        if (self.hidden_reg is not None):
            self.hidden_reg.compile(self._hidden_shape)

        # compile optimizers
        self.weight_optimizer.compile(self._weight_shape)
        self.bias_optimizer.compile(self._output_size)
        self.hidden_optimizer.compile(self._hidden_shape)

        # compile history buffer
        for i in range(self.truncate_length):
            self._history_buffer[i] = RecurrentStep(self._input_shape[1:], self._state_shape)

        if (isinstance(self.activation, CachedLayer)):
            for i in range(self.truncate_length):
                self._history_buffer[i].activation_cache = self.activation.create_cache()

        # compile misc
        self.name = f"Recurrent {self._output_size}"

        if (gen_param):
            self.apply_param(self.create_param())

    def is_compiled(self):
        weight_shape_ok = (self._weight_shape is not None 
                           and self._weight_shape == (self._input_size, self._output_size))
        hidden_shape_ok = (self._hidden_shape is not None
                         and self._hidden_shape == (self._output_size, self._output_size))
        state_shape_ok = (self._state_shape is not None
                          and self._state_shape == (self._batch_size, self._output_size))

        return (super().is_compiled()
                and weight_shape_ok
                and hidden_shape_ok
                and state_shape_ok)

    def init_weight(self, param: RecurrentParam):
        """
        Initialize weight in param object

        Args:
            type: weight initalization type
                [he, xavier]
            weight_bias: mean of weight
            param: param object to initalize weight to/store
        """


        weight_scale = 1
        hidden_scale = 1
        if  (self.weight_type == "xavier"):
            weight_scale = 1 / np.sqrt(self._input_size) # Xavier init
            hidden_scale = 1 / np.sqrt(self._output_size)
        elif (self.weight_type == "he"):
            weight_scale = np.sqrt(2 / self._input_size) # he init, for relus
            hidden_scale = np.sqrt(2 / self._output_size)
        else:
            raise ValueError(f'no such weight type "{self.weight_type}"')

        param.set_weight("weight", np.random.normal(loc=0, scale=weight_scale, size=self._weight_shape))
        param.set_weight("hidden_weight", np.random.normal(loc=0, scale=hidden_scale, 
                                               size=self._hidden_shape))

    def create_param(self):
        """
        creates a new parameter object for this layer
        
        Returns: 
            param object
        """

        super().create_param()
        param = RecurrentParam()
        self.init_weight(param)
        param.set_weight("bias", np.zeros(self._output_size))
        return param

    def param_compatible(self, param: RecurrentParam):
        """
        Check if parameter is compatible

        Args:
            param: parameter to check

        Returns:
            is compatible
        """

        weight_ok = (param.weight is not None and param.weight.shape == self._weight_shape)
        hidden_ok = (param.hidden_weight is not None and param.hidden_weight.shape == self._hidden_shape)
        bias_ok = (param.bias is not None and param.bias.shape[0] == self._output_size)
        return weight_ok and hidden_ok and bias_ok

    def forward(self, X, initial_state=None):
        if (initial_state is not None):
            state = self._history_buffer[self._current].state
            np.copyto(state, initial_state)
            self._valid = 0
        else:
            state = self._history_buffer[self._current].state
            np.copyto(state, self._output[-1])

        for i in range(self._timesteps):
            step        = self._history_buffer[self._current]
            step_output = self._output[i]
            np.copyto(step.X, X[i]) # potential issue if input is view of weird stride

            if (i >= 1):
                np.copyto(step.state, self._output[i-1])

            np.dot(step.X, self.param.weight, out=step_output)
            np.dot(step.state, self.param.hidden_weight, out=self._hidden_result)
            np.add(step_output, self._hidden_result, out=step_output)
            np.add(step_output, self.param.bias, out=step_output)

            if (self.activation is not None):
                if (isinstance(self.activation, CachedLayer)):
                    self.activation.forward(step_output, out_cache=step.activation_cache)
                else:
                    self.activation.forward(step_output)
                np.copyto(step_output, self.activation._output)
            self.update_current()
        return self._readonly_output

    def backprop(self, dldz):
        # https://machinelearningmastery.com/gentle-introduction-backpropagation-time/

        # clear gradients
        np.copyto(self._dlds, 0)
        np.copyto(self._dldw, 0)
        np.copyto(self._dldh, 0)
        np.copyto(self._dldb, 0)

        for i in range(self._valid):
            index   = (self._current - i - 1) % self.truncate_length

            step        = self._history_buffer[index]
            np.add(self._dlds, dldz[-1-i], out=self._dlds)

            if (self.activation is not None):
                if (isinstance(self.activation, CachedLayer)):
                    self.activation.backprop(self._dlds, cache=step.activation_cache)
                else:
                    self.activation.backprop(self._dlds)
                np.copyto(self._dlds, self.activation._dldx)

            # dldx
            np.dot(self._dlds, self.param.weight.T, out=self._dldx[-1-i])

            # dldh
            np.dot(step.state.T, self._dlds, out=self._step_dldh)
            np.add(self._step_dldh, self._dldh, out=self._dldh)

            # dldw
            np.dot(step.X.T, self._dlds, out=self._step_dldw)
            np.add(self._step_dldw, self._dldw, out=self._dldw)

            # dldb
            np.sum(self._dlds, axis=0, out=self._step_dldb)
            np.add(self._step_dldb, self._dldb, out=self._dldb)

            # dlds
            np.dot(self._dlds, self.param.hidden_weight.T, out=self._dlds)

        if (self.weight_reg is not None): 
            np.add(self._dldw, self.weight_reg.gradient(self.param.weight), out=self._dldw)
        if (self.bias_reg is not None):
            np.add(self._dldb, self.bias_reg.gradient(self.param.bias), out=self._dldb)
        if (self.hidden_reg is not None): 
            np.add(self._dldh, self.hidden_reg.gradient(self.param.hidden_weight), out=self._dldh)
        return self._readonly_dldx

    def apply_grad(self, step, dldw=None, dldb=None, dldh=None):
        """
        Applies layer gradient

        NOTE: None gradients default to gradient computed in backprop()

        Args:
            step: gradient step size
            dldw: gradient of loss with respect to weight
            dldb: gradient of loss with respect to bias
            dldh: gradient of loss with respect to hidden_weight
        """

        if (dldw is None):
            dldw = self._dldw
        if (dldb is None):
            dldb = self._dldb
        if (dldh is None):
            dldh = self._dldh

        # average batch gradient
        step = step / self._batch_size

        # weight update
        np.subtract(self.param.weight, self.weight_optimizer.gradient(step, dldw), 
               out=self.param.weight)

        # bias update
        np.subtract(self.param.bias, self.bias_optimizer.gradient(step, dldb), 
               out=self.param.bias)

        # hidden_weight update
        np.subtract(self.param.hidden_weight, self.hidden_optimizer.gradient(step, dldh), 
               out=self.param.hidden_weight)




class RecurrentStep():
    X                   = None # input of step
    state               = None # hidden state of step
    activation_cache    = None # cache of activation layer

    def __init__(self, input_shape, state_shape):
        self.X      = np.zeros(input_shape)
        self.state  = np.zeros(state_shape)


class RecurrentParam(Parameter):
    @property
    def weight(self):
        return self.get_weight("weight")
    
    @property
    def hidden_weight(self):
        return self.get_weight("hidden_weight")
    
    @property
    def bias(self):
        return self.get_weight("bias")