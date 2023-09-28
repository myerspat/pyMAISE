import copy

from tensorflow.keras.optimizers import Adagrad

from pyMAISE.utils.hyperparameters import HyperParameters


# Adam optimizer for keras neural network
class AdaGradOpt:
    def __init__(self, parameters: dict):
        self.reset()

        for key, value in parameters.items():
            if self._data[key]:
                self._data[key] = value

    def build(self, hp):
        sampled_data = copy.deepcopy(self._data)
        for key, value in self._data.items():
            if isinstance(value, HyperParameters):
                sampled_data[key] = value.hp(hp, "_".join([self._layer_name, key]))
            else:
                sampled_data[key] = value

        return Adagrad(**sampled_data)

    def reset(self):
        self._data = {
            "learning_rate": 0.001,
            "initial_accumulator_value": 0.1,
            "epsilon": 1e-07,
            "weight_decay": None,
            "clipnorm": None,
            "clipvalue": None,
            "global_clipnorm": None,
            "use_ema": False,
            "ema_momentum": 0.99,
            "ema_overwrite_frequency": None,
            "jit_compile": True,
        }
