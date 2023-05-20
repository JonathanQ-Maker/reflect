from __future__ import annotations
from reflect.layers.parametric_layer import ParametricLayer, Parameter # NOTE: Must import Parameter because derivative class imports it
from reflect.utils.misc import to_tuple
from reflect import np

class AbstractRNN(ParametricLayer):
    """
    TODO: description
    """

    truncate_length = 5     # max length of history stored
    _timesteps      = None  # number of input time steps

    _history_buffer = None  # array to store sequence history
    _valid          = 0     # number of valid history in buffer
    _current        = 0     # current history_buffer index
    _dldx_shape     = None  # (tuncate length, batch size, ) + output_size

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def valid(self):
        return self._valid

    def __init__(self,
                 truncate_length):
        self.truncate_length = truncate_length

    def compile(self, input_size, batch_size, timesteps, gen_param=True):
        self._timesteps     = timesteps
        self._input_size    = input_size
        self._batch_size    = batch_size

        # compile shapes
        self._input_shape   = (self._timesteps, self._batch_size) + to_tuple(self._input_size)
        self._output_shape  = (self._timesteps, self._batch_size) + to_tuple(self._output_size)
        self._dldx_shape    = (self.truncate_length, self._batch_size) + to_tuple(self._input_size)

        # compile arrays
        self._output    = np.zeros(shape=self._output_shape)
        self._dldx      = np.zeros(shape=self._dldx_shape)

        # compile read only views
        self._readonly_output = self._output.view()
        self._readonly_dldx   = self._dldx.view()
        self._readonly_output.flags.writeable = False
        self._readonly_dldx.flags.writeable = False

        self.name = "UNKNOWN"

        # instantiate history_buffer, expected to be filled in concrete class
        self._history_buffer = [None] * self.truncate_length

    def is_compiled(self):
        """
        Check if layer is up-to-date with layer arguments

        # of check items should be the same as # of compile items
        """
        history_buffer_ok = (self._history_buffer is not None
                             and len(self._history_buffer) == self.truncate_length)
        input_size_match = self._input_shape == (self._timesteps, self._batch_size) + to_tuple(self._input_size)
        output_size_match = self._output_shape == (self._timesteps, self._batch_size) + to_tuple(self._output_size)
        dldx_size_match = self._dldx_shape == (self.truncate_length, self._batch_size) + to_tuple(self._input_size)

        dldx_ok = self._dldx is not None and self._dldx.shape == self._dldx_shape
        output_ok = self._output is not None and self._output.shape == self._output_shape

        return (self.name is not None
                and input_size_match
                and output_size_match
                and dldx_size_match
                and dldx_ok
                and output_ok
                and history_buffer_ok)

    def next_current(self):
        """
        Computes the next current index for forward pass
        """

        # cycles index current for repeated history_buffer use
        return (self._current + 1) % self.truncate_length

    def update_current(self):
        """
        Updates the current history_buffer index for next time step
        """

        # keep track of valid history
        if (self._valid < self.truncate_length):
            self._valid += 1

        self._current = self.next_current()


