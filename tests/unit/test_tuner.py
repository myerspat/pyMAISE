import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from skopt.space import Categorical, Real

import pyMAISE as mai


# ================================================================
@pytest.fixture
def setup_xr():
    # Initialize pyMAISE
    settings = {
        "verbosity": 2,
        "random_state": 42,
        "num_configs_saved": 2,
        "new_nn_architecture": True,
        "test_size": 0.3,
        "regression": True,
        "classification": False,
    }
    settings = mai.settings.init(settings_changes=settings)

    # Load reactor physics PreProcessor and split data
    preprocessor = mai.load_xs()
    preprocessor.train_test_split()
    return settings, preprocessor


@pytest.fixture
def setup_fp():
    # Initialize pyMAISE
    settings = {
        "verbosity": 2,
        "random_state": 42,
        "num_configs_saved": 2,
        "new_nn_architecture": True,
        "test_size": 0.3,
        "regression": True,
        "classification": False,
    }
    settings = mai.settings.init(settings_changes=settings)

    # Load fuel performance PreProcessor and split data
    preprocessor = mai.load_fp()
    preprocessor.train_test_split()
    return settings, preprocessor


@pytest.fixture()
def setup_nn_model_settings(setup_xr, setup_fp):
    data = []
    for settings, preprocessor in [setup_xr, setup_fp]:
        # Min-max scale for some okay NN performance
        preprocessor.train_test_split(scaler=MinMaxScaler())

        # FNN structural parameters
        structural = {
            "Dense_input": {
                "units": mai.Choice([100, 200, 300]),
                "input_dim": preprocessor.inputs.shape[-1],
                "activation": "relu",
                "kernel_initializer": "normal",
                "sublayer": "Dropout",
                "Dropout": {"rate": 0.5},
            },
            "Dense_output": {
                "units": preprocessor.outputs.shape[-1],
                "activation": "linear",
                "kernel_initializer": "normal",
            },
        }

        # Training settings
        model_settings = {
            "models": ["fnn"],
            "fnn": {
                "structural_params": structural,
                "optimizer": "Adam",
                "Adam": {
                    "learning_rate": 0.001,
                },
                "compile_params": {
                    "loss": "mean_absolute_error",
                    "metrics": ["mean_absolute_error"],
                },
                "fitting_params": {
                    "batch_size": 8,
                    "epochs": 10,
                    "validation_split": 0.15,
                },
            },
        }
        data.append((settings, preprocessor, model_settings))
    return data


# ================================================================
def test_grid_search(setup_xr, setup_fp):
    for settings, preprocessor in [setup_xr, setup_fp]:
        # Build Tuner object
        model_settings = {
            "models": ["linear", "lasso"],
        }
        tuner = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

        # Run grid search over search spaces
        spaces = {
            "linear": {"fit_intercept": [True, False]},
            "lasso": {"alpha": np.linspace(0.000001, 1, 5)},
        }
        search_data = tuner.grid_search(
            param_spaces=spaces,
            cv=ShuffleSplit(
                n_splits=1, test_size=0.15, random_state=settings.random_state
            ),
        )

        # Assert that we have the model wrappers
        assert isinstance(search_data["linear"][0], pd.DataFrame)
        assert isinstance(search_data["lasso"][0], pd.DataFrame)
        assert isinstance(search_data["linear"][1], LinearRegression)
        assert isinstance(search_data["lasso"][1], Lasso)

        # Assert search data dimensions
        assert search_data.keys() == spaces.keys()
        assert search_data["linear"][0].shape == (2, 1)
        assert search_data["lasso"][0].shape == (2, 1)
        assert tuner.cv_performance_data["linear"].shape == (2, 2)
        assert tuner.cv_performance_data["lasso"].shape == (2, 5)


# ================================================================
def test_random_search(setup_xr, setup_fp):
    for settings, preprocessor in [setup_xr, setup_fp]:
        # Build Tuner object
        model_settings = {
            "models": ["linear", "lasso"],
        }
        tuner = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

        # Run random search over search spaces
        spaces = {
            "linear": {"fit_intercept": [True, False]},
            "lasso": {"alpha": np.linspace(0.000001, 1, 5)},
        }
        search_data = tuner.random_search(
            param_spaces=spaces,
            n_iter=5,
            cv=ShuffleSplit(
                n_splits=1, test_size=0.15, random_state=settings.random_state
            ),
        )

        # Assert that we have the model wrappers
        assert isinstance(search_data["linear"][0], pd.DataFrame)
        assert isinstance(search_data["lasso"][0], pd.DataFrame)
        assert isinstance(search_data["linear"][1], LinearRegression)
        assert isinstance(search_data["lasso"][1], Lasso)

        # Assert search data dimensions
        assert search_data.keys() == spaces.keys()
        assert search_data["linear"][0].shape == (2, 1)
        assert search_data["lasso"][0].shape == (2, 1)
        assert tuner.cv_performance_data["linear"].shape == (
            2,
            2,
        )  # n_iter > space size so restrict to space size
        assert tuner.cv_performance_data["lasso"].shape == (2, 5)


# ================================================================
def test_bayesian_search(setup_xr, setup_fp):
    for settings, preprocessor in [setup_xr, setup_fp]:
        # Build Tuner object
        model_settings = {
            "models": ["linear", "lasso"],
        }
        tuner = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

        # Run bayesian search over search spaces
        spaces = {
            "linear": {"fit_intercept": Categorical([True, False])},
            "lasso": {"alpha": Real(0.000001, 1)},
        }
        search_data = tuner.bayesian_search(
            param_spaces=spaces,
            n_iter=5,
            cv=ShuffleSplit(
                n_splits=1, test_size=0.15, random_state=settings.random_state
            ),
        )

        # Assert that we have the model wrappers
        assert isinstance(search_data["linear"][0], pd.DataFrame)
        assert isinstance(search_data["lasso"][0], pd.DataFrame)
        assert isinstance(search_data["linear"][1], LinearRegression)
        assert isinstance(search_data["lasso"][1], Lasso)

        # Assert search data dimensions
        assert search_data.keys() == spaces.keys()
        assert search_data["linear"][0].shape == (2, 1)
        assert search_data["lasso"][0].shape == (2, 1)
        assert tuner.cv_performance_data["linear"].shape == (2, 5)
        assert tuner.cv_performance_data["lasso"].shape == (2, 5)


# ================================================================
def test_nn_grid_search(setup_nn_model_settings):
    for settings, preprocessor, model_settings in setup_nn_model_settings:
        # Build Tuner
        tuner = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

        # Run grid search
        search_data = tuner.nn_grid_search(
            objective="r2_score",
            cv=ShuffleSplit(
                n_splits=1, test_size=0.15, random_state=settings.random_state
            ),
        )

        # Assert that we have the correct types
        assert isinstance(search_data["fnn"][0], pd.DataFrame)
        assert isinstance(search_data["fnn"][1], mai.nnHyperModel)

        # Assert search data dimensions
        assert list(search_data.keys()) == model_settings["models"]
        assert search_data["fnn"][0].shape == (2, 1)
        assert tuner.cv_performance_data["fnn"].shape == (2, 3)


# ================================================================
def test_nn_random_search(setup_nn_model_settings):
    for settings, preprocessor, model_settings in setup_nn_model_settings:
        # Build Tuner
        tuner = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

        # Run random search
        search_data = tuner.nn_random_search(
            objective="r2_score",
            max_trials=3,
            cv=ShuffleSplit(
                n_splits=1, test_size=0.15, random_state=settings.random_state
            ),
        )

        # Assert that we have the correct types
        assert isinstance(search_data["fnn"][0], pd.DataFrame)
        assert isinstance(search_data["fnn"][1], mai.nnHyperModel)

        # Assert search data dimensions
        assert list(search_data.keys()) == model_settings["models"]
        assert search_data["fnn"][0].shape == (2, 1)
        assert tuner.cv_performance_data["fnn"].shape == (2, 3)


# ================================================================
def test_nn_bayesian_search(setup_nn_model_settings):
    for settings, preprocessor, model_settings in setup_nn_model_settings:
        # Build Tuner
        tuner = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

        # Run bayesian search
        search_data = tuner.nn_bayesian_search(
            objective="r2_score",
            max_trials=3,
            cv=ShuffleSplit(
                n_splits=1, test_size=0.15, random_state=settings.random_state
            ),
        )

        # Assert that we have the correct types
        assert isinstance(search_data["fnn"][0], pd.DataFrame)
        assert isinstance(search_data["fnn"][1], mai.nnHyperModel)

        # Assert search data dimensions
        assert list(search_data.keys()) == model_settings["models"]
        assert search_data["fnn"][0].shape == (2, 1)
        assert tuner.cv_performance_data["fnn"].shape == (2, 3)


# ================================================================
def test_nn_hyperband_search(setup_nn_model_settings):
    for settings, preprocessor, model_settings in setup_nn_model_settings:
        # Build Tuner
        tuner = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

        # Run hyperband search
        search_data = tuner.nn_hyperband_search(
            objective="r2_score",
            cv=ShuffleSplit(
                n_splits=1, test_size=0.15, random_state=settings.random_state
            ),
        )

        # Assert that we have the correct types
        assert isinstance(search_data["fnn"][0], pd.DataFrame)
        assert isinstance(search_data["fnn"][1], mai.nnHyperModel)

        # Assert search data dimensions
        assert list(search_data.keys()) == model_settings["models"]
        assert search_data["fnn"][0].shape == (2, 1)
        assert tuner.cv_performance_data["fnn"].shape == (2, 3)


# ================================================================
def test_convergence_plot(setup_xr):
    settings, preprocessor = setup_xr

    # Build Tuner object
    model_settings = {
        "models": ["linear", "lasso"],
    }
    tuner = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

    # Run grid search over search spaces
    spaces = {
        "linear": {"fit_intercept": [True, False]},
        "lasso": {"alpha": np.linspace(0.000001, 1, 5)},
    }
    search_data = tuner.grid_search(
        param_spaces=spaces,
        cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=settings.random_state),
    )

    # Plot convergence of linear
    ax = tuner.convergence_plot(model_types="linear")
