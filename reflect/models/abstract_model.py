from abc import ABC, abstractmethod
from reflect.utils.misc import to_tuple
from reflect.compiled_object import CompiledObject
import numpy as np
import json
import os.path

class AbstractModel(CompiledObject):
    """Abstract class of Model"""
    
    _input_size     = None
    _input_shape    = None
    _output_shape   = None
    _batch_size     = None
    _output         = None
    _dldx           = None

    # Metric tracking
    _train_steps    = 0

    @property
    def output(self):
        return self._output

    @property
    def dldx(self):
        return self._dldx

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, size):
        self._input_size = size

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
        self._batch_size = size

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def total_params(self):
        return 0
    
    @property
    def train_steps(self):
        return self._train_steps

    def compile(self):
        super().compile()
        self._input_shape = (self._batch_size, ) + to_tuple(self._input_size)

    @abstractmethod
    def forward(self, X):
        return

    @abstractmethod
    def backprop(self, dldz):
        return
    
    def apply_grad(self, step):
        self._train_steps += 1
    
    def serialize(self):
        if (not self.is_compiled()):
            raise RuntimeError("can not serialize uncompiled model")
        
        data = {"train_steps": self.train_steps}
        return data

    def populate(self, data):
        if (not self.is_compiled()):
            raise RuntimeError("can not populate uncompiled model")
        
        if ("train_steps" not in data.keys()):
            raise ValueError("Does not contain expected data")

        self._train_steps = data["train_steps"]
    
    def save_model(self, path: str, name: str, metadata: dict=None):
        name = name.replace(".npy", "")
        
        # save binary .npy file with model parameter data
        np.save(os.path.join(path, name), self.serialize())

        # save txt summary of model for easy reimplementation of model
        with open(os.path.join(path, f"{name}_summary.txt"), 'w', encoding='utf-8') as f:
           f.write(self.__str__())

        # save json meta data file
        _metadata = {"train_steps": self.train_steps}
        if isinstance(metadata, dict):
            _metadata.update(metadata)
        with open(os.path.join(path, f"{name}_metadata.json"), 'w', encoding='utf-8') as f:
           json.dump(_metadata, f, ensure_ascii=False, indent=4)

    def load_model(self, path: str):
        if path.count(".npy") < 1:
            path += ".npy"
        self.populate(np.load(path, allow_pickle=True).item())
        

    def __str__(self):
        return (f"Type:           {self.__class__.__name__}\n"
                + f"Total params:   {self.total_params}\n\n\nLayers:\n\n")

    def print_summary(self):
        print(f"{'='*50}\nModel Summary\n")
        print(self.__str__())
        print(f"{'='*50}\n")
