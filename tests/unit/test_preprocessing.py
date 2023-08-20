import pytest
import pyMAISE as mai


def test_data_split():
    # Load PreProcessor of cross section data from xs.csv
    xs_preprocessor = mai.load_xs()

    # Init settings
    settings = {
            "verbosity": 0,
            "random_state": 0,
            "test_size": 0.3,
            "regression": True,
            "classification": False,
            }
    settings = mai.settings.init(settings_changes=settings)

# Split data
    xtrain, xtest, ytrain, ytest = xs_preprocessor.data_split()

    # Assert DataFrame dimensions and size after split
    assert settings.test_size == 0.3
    assert xtrain.shape[0] == 700 and ytrain.shape[0] == 700
    assert xtest.shape[0] == 300 and ytest.shape[0] == 300
    assert xtrain.shape[1] == 8 and xtest.shape[1] == 8

    # Change random_state and test_size
    settings_changes = {"random_state": 42, "test_size": 0.4}
    settings = mai.settings.init(settings_changes)

    # Assert settings are correct
    assert settings.verbosity == 0
    assert settings.random_state == 42
    assert settings.test_size == 0.4
    assert settings.regression == True
    assert settings.classification == False

