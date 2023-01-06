from __future__ import annotations
from reflect.layers.abstract_rnn import AbstractRNN
from reflect import np

class Recurrent(AbstractRNN):

    weight_type     = None # weight initalization type
    _weight_shape   = None # (input size, units) NOTE: tranposed
    _hidden_shape   = None # hidden_weight shape, (units, units) NOTE: transposed
    _state_view     = None # current temporal state, or last time step output
    _state_shape    = None # (batch size, units)

    _dldw           = None # gradient of loss with respect to weight
    _dldh           = None # gtaident of loss with respect to hidden_weight
    _dldb           = None # graident of loss with respect to bias
    _dlds           = None # graident of loss with respect to state

    # readonly views of arrays
    _readonly_dldw  = None
    _readonly_dldh  = None
    _readonly_dldb  = None

    # internal variables
    _step_dldw      = None # gradient of loss with respect to weight at time step
    _step_dldh      = None # gradient of loss with respect to hidden weight at time step
    _step_dldb      = None # gradient of loss with respect to bias at time step

    _hidden_result  = None # buffer to store result of Z@H

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
                    truncate_length = 5,
                    weight_type     = "he"):
        super().__init__(truncate_length)
        self._output_size   = units
        self.weight_type    = weight_type

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

        # debug activation function
        self.activate = np.random.normal(size=self._state_shape)


        # compile history buffer
        for i in range(self.truncate_length):
            self._history_buffer[i] = RecurrentStep(self._input_shape[1:], self._state_shape)

        # compile misc
        self.name = f"Recurrent {self._output_size}"

        if (gen_param):
            self.apply_param(self.create_param())

    def init_weight(self, param: RecurrentParam):
        """
        Initialize weight in param object

        Args:
            type: weight initalization type
                [he, xavier]
            weight_bias: mean of weight
            param: param object to initalize weight to/store
        """


        scale = 1
        if  (self.weight_type == "xavier"):
            scale = 1 / np.sqrt(self._input_size) # Xavier init
        elif (self.weight_type == "he"):
            scale = np.sqrt(2 / self._input_size) # he init, for relus
        else:
            raise ValueError(f'no such weight type "{self.weight_type}"')

        param.weight = np.random.normal(loc=0, scale=scale, size=self._weight_shape)
        param.hidden_weight = np.random.normal(loc=0, scale=scale, 
                                               size=self._hidden_shape)

    def create_param(self):
        """
        creates a new parameter object for this layer
        
        Returns: 
            param object
        """

        super().create_param()
        param = RecurrentParam()
        self.init_weight(param)
        param.bias = np.zeros(self._output_size)
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


    def forward(self, X, inital_state=None):
        if (inital_state is not None):
            state = self._history_buffer[self._current].state
            np.copyto(state, inital_state)
            self._valid = 0

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
            # TODO: activation 
            np.multiply(self.activate, step_output, out=step_output)
            self.update_current()
        return self._readonly_output

    def backprop(self, dldz):
        # clear gradients
        np.copyto(self._dlds, 0)
        np.copyto(self._dldw, 0)
        np.copyto(self._dldh, 0)
        np.copyto(self._dldb, 0)

        for i in range(self._valid):
            index   = (self._current - i - 1) % self.truncate_length

            step        = self._history_buffer[index]
            np.add(self._dlds, dldz[-1-i], out=self._dlds)
            np.multiply(self._dlds, self.activate, out=self._dlds)

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
        return self._readonly_dldx


class RecurrentStep():
    X = None
    state = None

    def __init__(self, input_shape, state_shape):
        self.X = np.zeros(input_shape)
        self.state = np.zeros(state_shape)


class RecurrentParam():
    weight          = None
    hidden_weight   = None
    bias            = None