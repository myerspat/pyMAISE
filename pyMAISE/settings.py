# Class for global settings
class Settings:
    def __init__(self, update: dict = None):
        # Defaults
        self._verbosity = 0
        self._random_state = None
        self._test_size = 0.3
        self._num_configs_saved = 5

        # If a dictionary of key/value pairs is given,
        # update settings
        if update != None:
            for key, value in update.items():
                setattr(self, key, value)

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


# Initialization function for global settings
def init(settings_changes: dict = None):
    global values
    values = Settings(settings_changes)
    return values
