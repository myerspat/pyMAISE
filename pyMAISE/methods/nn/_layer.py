import copy

from pyMAISE.utils.hyperparameters import Choice, HyperParameters


class Layer:
    def __init__(self, layer_name, parameters):
        # Set layer name and reset the layer
        self._layer_name = layer_name
        self.reset()

        # Set self._base_data values
        for key, value in parameters.items():
            if key == "sublayer":
                # Save the sublayer data and determine if multiple sublayers
                # are given for pyMAISE.Choice
                self._base_data["sublayer"] = value
                sublayer_list = value
                if isinstance(sublayer, Choice):
                    sublayer_list = value.values
                else:
                    sublayer_list = [sublayer_list]

                # We set sublayer data to none but expect it to be updated on
                # later key, value pairs in the dictionary assuming that they
                # proceed "sublayer"
                for sublayer_name in sublayer_list:
                    self._base_data[sublayer_name] = None

            elif key in self._base_data:
                self._base_data[key] = value

        # Make sure the wrapper is a tuple
        if self._base_data["wrapper"] is not None and not isinstance(
            self._base_data["wrapper"], tuple
        ):
            self._base_data["wrapper"] = (parameters["wrapper"], {})

        # Assert we have a layer name
        assert self._layer_name != None

    # ==========================================================================
    # Methods
    def build_data(self, data, parameters):
        # Get data for keys shared between data and parameters (or input_shape)
        for key, value in parameters.items():
            if key in data or key == "input_shape":
                data[key] = value
        return data

    def sample_parameters(self, data, hp):
        # Sample hyperparameter data during training
        sampled_data = copy.deepcopy(data)
        for key, value in sampled_data.items():
            if isinstance(value, HyperParameters):
                sampled_data[key] = value.hp(
                    hp,
                    "_".join([self._layer_name + str(self._current_layer), key]),
                )
        return sampled_data

    def reset(self):
        self._current_layer = 0
        self._base_data = {
            "num_layers": 1,
            "wrapper": None,
            "sublayer": None,
        }

    def increment_layer(self):
        # Increment layer by one
        self._current_layer = self._current_layer + 1

    def num_layers(self, hp):
        # Determine number of layers, if HyperParameters then sample it
        if isinstance(self._base_data["num_layers"], HyperParameters):
            return self._base_data["num_layers"].hp(
                hp, self._layer_name + "_num_layers"
            )
        else:
            return self._base_data["num_layers"]

    def sublayer(self, hp):
        # Get sublayer data, if a Choice then sample
        if isinstance(self._base_data["sublayer"], Choice):
            sublayer_name = self._base_data["sublayer"].hp(
                hp, self._layer_name + "_sublayer"
            )
            return (sublayer_name, self._base_data[sublayer_name])
        elif isinstance(self._base_data["sublayer"], str):
            sublayer_name = self._base_data["sublayer"]
            return (sublayer_name, self._base_data[sublayer_name])
        else:
            return None

    def wrapper(self):
        # Return wrapper
        return self._base_data["wrapper"]
