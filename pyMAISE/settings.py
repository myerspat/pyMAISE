import os
import random
import numpy as np
import tensorflow as tf
import warnings

# Class for global settings
class Settings:
    def __init__(self, update: dict = None):
        # Defaults
        self._verbosity = 0
        self._random_state = None
        self._test_size = 0.3
        self._num_configs_saved = 5
        self._regression = False
        self._classification = False

        # If a dictionary of key/value pairs is given,
        # update settings
        if update != None:
            for key, value in update.items():
                setattr(self, key, value)

        if self._verbosity <= 1:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

            warnings.simplefilter(action="ignore", category=Warning)
            warnings.simplefilter(action="ignore", category=FutureWarning)

        if self._random_state != None:
            os.environ["PYTHONHASHSEED"] = str(self._random_state)
            random.seed(self._random_state)
            np.random.seed(self._random_state)
            tf.compat.v1.set_random_seed(self._random_state)

            # Deterministic tensorflow
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_CUBNN_DETERMINISTIC"] = "1"

            session_conf = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
            )
            sess = tf.compat.v1.Session(
                graph=tf.compat.v1.get_default_graph(), config=session_conf
            )
            tf.compat.v1.keras.backend.set_session(sess)

            # assert (self._regression == True) | (self._classification == True)
            assert (self._regression == True & self._classification == True) != True

    # Getters
    @property
    def verbosity(self) -> int:
        return self._verbosity

    @property
    def random_state(self) -> int:
        return self._random_state

    @property
    def test_size(self) -> float:
        return self._test_size

    @property
    def num_configs_saved(self) -> int:
        return self._num_configs_saved
    
    @property
    def regression(self) -> bool:
        return self._regression

    @property
    def classification(self) -> bool:
        return self._classification

    # Setters
    @verbosity.setter
    def verbosity(self, verbosity: int):
        assert isinstance(verbosity, int)
        assert verbosity >= 0
        self._verbosity = verbosity

    @random_state.setter
    def random_state(self, random_state: int):
        assert random_state == None or random_state >= 0
        self._random_state = random_state

    @test_size.setter
    def test_size(self, test_size: float):
        assert isinstance(test_size, float)
        assert test_size >= 0.0 and test_size < 1.0
        self._test_size = test_size

    @num_configs_saved.setter
    def num_configs_saved(self, num_configs_saved: int):
        assert num_configs_saved > 0
        self._num_configs_saved = num_configs_saved

    @regression.setter
    def regression(self, regression: bool) -> bool:
        self._regression = regression

    @classification.setter
    def classification(self, classification: bool) -> bool:
        self._classification = classification

# Initialization function for global settings
def init(settings_changes: dict = None):
    global values
    values = Settings(settings_changes)
    return values
