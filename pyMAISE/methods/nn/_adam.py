import copy

from tensorflow.keras.optimizers import Adam

from pyMAISE.utils.hyperparameters import HyperParameters


# Adam optimizer for keras neural network
class AdamOpt:
    def __init__(self, parameters: dict):
        self.reset()

        for key, value in parameters.items():
            if self._data[key]:
                self._data[key] = value

    def build(self, hp):
        sampled_data = copy.deepcopy(self._data)
        for key, value in self._data.items():
            if isinstance(value, HyperParameters):
                sampled_data[key] = value.hp(hp, "_".join([Adam.__name__, key]))
            else:
                sampled_data[key] = value

        return Adam(**sampled_data)

    def reset(self):
        self._data = {
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-07,
            "amsgrad": False,
            "clipnorm": None,
            "clipvalue": None,
        }
