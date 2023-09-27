import copy
import re

from keras.models import Sequential
from keras_tuner import HyperModel

from pyMAISE.methods.nn._adam import AdamOpt
from pyMAISE.methods.nn._dense import DenseLayer
from pyMiase.methods.nn._lstm import LstmLayer
from pyMAISE.methods.nn._gru import GruLayer
from pyMAISE.methods.nn._dropout import DropoutLayer
from pyMAISE.utils.hyperparameters import Choice, HyperParameters


class nnHyperModel(HyperModel):
    def __init__(self, parameters: dict):
        # Structure/Architectural hyperparameters
        self._structural_params = parameters["structural_params"]

        # Optimizers and their hyperparameters
        if parameters["optimizer"]:
            self._optimizer = parameters["optimizer"]

            self._optimizer_params = {}
            if isinstance(self._optimizer, Choice):
                for optimizer in self._optimizer.values:
                    assert parameters[optimizer]
                    self._optimizer_params[optimizer] = parameters[optimizer]
            else:
                self._optimizer_params[self._optimizer] = parameters[self._optimizer]
        else:
            self._optimizer = "rmsprop"

        # Model compilation hyperparameters
        self._compilation_params = parameters["compile_params"]

        # Model fitting hyperparameters
        self._fitting_params = parameters["fitting_params"]

    # ==========================================================================
    # Methods
    def build(self, hp):
        # Sequential keras neural network
        model = Sequential()

        # Iterating though archetecture
        for layer_name in self._structural_params.keys():
            layer = copy.deepcopy(self._get_layer(layer_name))

            for i in range(layer.num_layers(hp)):
                model.add(layer.build(hp))
                layer.increment_layer()

            layer.reset()

        # Compile Model
        self._compilation_params["optimizer"] = self._get_optimizer().build(hp)
        model.compile(**self._compilation_params)
        return model

    # Fit function for keras-tuner to allow hyperparameter tuning of fitting parameters
    def fit(self, hp, model, x, y, **kwargs):
        for key, value in self._fitting_params.items():
            if isinstance(value, HyperParameters):
                self._fitting_params[key] = value.hp(hp, key)
        return model.fit(x, y, **self._fitting_params, **kwargs)

    # Update parameters after tuning, a common use case is increasing the number of epochs
    def set_params(self, parameters: dict = None):
        if "structural_params" in parameters:
            for key, value in parameters["structural_params"].items():
                assert key in self._structural_params
                for param_key, param_value in value.items():
                    self._structural_params[key][param_key] = param_value

        elif "optimizer" in parameters:
            self._optimizer = parameters["optimizer"]
            if parameters[self._optimizer]:
                for key, value in parameters[self._optimizer].items():
                    self._optimizer_params[self._optimizer][key] = value

        elif "compile_params" in parameters:
            for key, value in parameters["compile_params"].items():
                self._compilation_params[key] = value

        elif "fitting_params" in parameters:
            for key, value in parameters["fitting_params"].items():
                self._fitting_params[key] = value

    def _get_layer(self, layer_name):
        if bool(re.search("dense", layer_name)):
            return DenseLayer(layer_name, self._structural_params[layer_name])
        elif bool(re.search("lstm", layer_name)):
            return LstmLayer(layer_name, self._structural_params[layer_name])
        
        elif bool(re.search("gru", layer_name)):
            return GruLayer(layer_name, self.structural_params[layer_name])
        elif bool(re.search("dropout", layer_name)):
            return DropoutLayer(layer_name, self._structural_params[layer_name])
        else:
            raise Exception(
                f"Layer ({layer_name}) is either not supported or spelled incorrectly"
            )

    def _get_optimizer(self):
        optimizer = self._optimizer
        if optimizer == "adam":
            assert self._optimizer_params[optimizer]
            return AdamOpt(self._optimizer_params[optimizer])
        else:
            return optimizer
