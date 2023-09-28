import copy
import re

from keras.models import Sequential
from keras_tuner import HyperModel

from pyMAISE.methods.nn._adam import AdamOpt
from pyMAISE.methods.nn._conv1d import Conv1dLayer
from pyMAISE.methods.nn._conv2d import Conv2dLayer
from pyMAISE.methods.nn._conv3d import Conv3dLayer
from pyMAISE.methods.nn._dense import DenseLayer
from pyMAISE.methods.nn._dropout import DropoutLayer
from pyMAISE.methods.nn._flatten import FlattenLayer
from pyMAISE.methods.nn._gru import GruLayer
from pyMAISE.methods.nn._lstm import LstmLayer
from pyMAISE.methods.nn._reshape import ReshapeLayer
from pyMAISE.methods.nn._ada_delta import AdaDeltaOpt
from pyMAISE.methods.nn._ada_grad import AdaGradOpt
from pyMAISE.methods.nn._ada_max import AdaMaxOpt
from pyMAISE.methods.nn._adamw import AdamwOpt
from pyMAISE.methods.nn._ftrl import FtrlOpt
from pyMAISE.methods.nn._nadam import NadamOpt
from pyMAISE.methods.nn._rms_prop import RmsPropOpt
from pyMAISE.methods.nn._sgd import SgdOpt
from pyMAISE.utils.hyperparameters import Choice, HyperParameters
import pyMAISE.settings as settings


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
                assert parameters[self._optimizer]
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
        self._compilation_params["optimizer"] = self._get_optimizer(hp).build(hp)
        model.compile(**self._compilation_params)
        return model

    # Fit function for keras-tuner to allow hyperparameter tuning of fitting parameters
    def fit(self, hp, model, x, y, **kwargs):
        fitting_params = copy.deepcopy(self._fitting_params)
        for key, value in self._fitting_params.items():
            if isinstance(value, HyperParameters):
                fitting_params[key] = value.hp(hp, key)

        return model.fit(
            x, y, **fitting_params, **kwargs, verbose=settings.values.verbosity
        )

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
        elif bool(re.search("dropout", layer_name)):
            return DropoutLayer(layer_name, self._structural_params[layer_name])
        elif bool(re.search("lstm", layer_name)):
            return LstmLayer(layer_name, self._structural_params[layer_name])
        elif bool(re.search("gru", layer_name)):
            return GruLayer(layer_name, self._structural_params[layer_name])
        elif bool(re.search("conv1d", layer_name)):
            return Conv1dLayer(layer_name, self._structural_params[layer_name])
        elif bool(re.search("conv2d", layer_name)):
            return Conv2dLayer(layer_name, self._structural_params[layer_name])
        elif bool(re.search("conv3d", layer_name)):
            return Conv3dLayer(layer_name, self._structural_params[layer_name])
        elif bool(re.search("flatten", layer_name)):
            return FlattenLayer(layer_name, self._structural_params[layer_name])
        elif bool(re.search("reshape", layer_name)):
            return ReshapeLayer(layer_name, self._structural_params[layer_name])
        else:
            raise Exception(
                f"Layer ({layer_name}) is either not supported or spelled incorrectly"
            )

    def _get_optimizer(self, hp):
        optimizer = copy.deepcopy(self._optimizer)
        if isinstance(self._optimizer, Choice):
            optimizer = optimizer.hp(hp, "optimizer")

        assert self._optimizer_params[optimizer]

        if optimizer == "adam":
            return AdamOpt(self._optimizer_params[optimizer])
        elif optimizer == "SGD":
            return SgdOpt(self._optimizer_params[optimizer])
        elif optimizer == "RMSprop":
            return RmsPropOpt(self._optimizer_params[optimizer])
        elif optimizer == "AdamW":
            return AdamwOpt(self._optimizer_params[optimizer])
        elif optimizer == "Adadelta":
            return AdaDeltaOpt(self._optimizer_params[optimizer])
        elif optimizer == "Adagrad":
            return AdaGradOpt(self._optimizer_params[optimizer])
        elif optimizer == "Adamax":
            return AdaMaxOpt(self._optimizer_params[optimizer])
        elif optimizer == "Nadam":
            return NadamOpt(self._optimizer_params[optimizer])
        elif optimizer == "Ftrl":
            return FtrlOpt(self._optimizer_params[optimizer])
        else:
            return optimizer