import numpy as np
import pytest
from keras_tuner import HyperParameters
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai


# ================================================================
@pytest.fixture
def setup_xs_grid_search_results():
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
    preprocessor.train_test_split(scaler=MinMaxScaler())

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

    return (settings, preprocessor, search_data)


@pytest.fixture
def setup_xs_nn_grid_search_results():
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

    # Build Tuner
    tuner = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

    # Run grid search
    search_data = tuner.nn_grid_search(
        objective="r2_score",
        cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=settings.random_state),
    )

    return search_data


# ================================================================
def test_constructor(setup_xs_grid_search_results, setup_xs_nn_grid_search_results):
    # Extract data from fixtures
    setttings, preprocessor, classical_search_data = setup_xs_grid_search_results
    nn_search_data = setup_xs_nn_grid_search_results

    # Build PreProcessor
    new_model_settings = {
        "nn": {"epochs": 20},
    }
    postprocessor = mai.PostProcessor(
        data=preprocessor.split_data,
        models_list=[classical_search_data, nn_search_data],
        new_model_settings=new_model_settings,
        yscaler=preprocessor.yscaler,
    )

    # Shape and contents assertions (2 models per type)
    print(postprocessor._models)
    assert postprocessor._models.shape == (6, 6)
    for i in range(postprocessor._models.shape[0]):
        assert (
            postprocessor._models["Train Yhat"][i].shape
            == preprocessor.split_data[2].shape
        )
        assert (
            postprocessor._models["Test Yhat"][i].shape
            == preprocessor.split_data[3].shape
        )
        if i < 4:
            assert len(postprocessor._models["Parameter Configurations"][i].keys()) == 1
            assert postprocessor._models["History"][i] == None
        else:
            assert isinstance(
                postprocessor._models["Parameter Configurations"][i],
                HyperParameters,
            )
            assert isinstance(postprocessor._models["History"][i], dict)
            assert "loss" in postprocessor._models["History"][i]
            assert "val_loss" in postprocessor._models["History"][i]

    assert postprocessor._models["Model Types"][0] == "linear"
    assert postprocessor._models["Model Types"][1] == "linear"
    assert postprocessor._models["Model Types"][2] == "lasso"
    assert postprocessor._models["Model Types"][3] == "lasso"
    assert postprocessor._models["Model Types"][4] == "fnn"
    assert postprocessor._models["Model Types"][5] == "fnn"

    assert isinstance(postprocessor._models["Model Wrappers"][0], LinearRegression)
    assert isinstance(postprocessor._models["Model Wrappers"][1], LinearRegression)
    assert isinstance(postprocessor._models["Model Wrappers"][2], Lasso)
    assert isinstance(postprocessor._models["Model Wrappers"][3], Lasso)
    assert isinstance(postprocessor._models["Model Wrappers"][4], mai.nnHyperModel)
    assert isinstance(postprocessor._models["Model Wrappers"][5], mai.nnHyperModel)


# ================================================================
@pytest.fixture
def setup_preprocessor(setup_xs_grid_search_results, setup_xs_nn_grid_search_results):
    # Extract data from fixtures
    setttings, preprocessor, classical_search_data = setup_xs_grid_search_results
    nn_search_data = setup_xs_nn_grid_search_results

    # Build PreProcessor
    new_model_settings = {
        "nn": {"epochs": 20},
    }
    postprocessor = mai.PostProcessor(
        data=preprocessor.split_data,
        models_list=[classical_search_data, nn_search_data],
        new_model_settings=new_model_settings,
        yscaler=preprocessor.yscaler,
    )

    return postprocessor


# ================================================================
def test_metrics(setup_preprocessor):
    # Run metrics function
    metrics = setup_preprocessor.metrics()

    # Shape and contents assertions
    assert metrics.shape == (6, 10)
    assert list(metrics.columns) == [
        "Model Types",
        "Parameter Configurations",
        "Train R2",
        "Train MAE",
        "Train MSE",
        "Train RMSE",
        "Test R2",
        "Test MAE",
        "Test MSE",
        "Test RMSE",
    ]
    assert metrics["Test R2"].to_numpy()[0] == np.max(metrics["Test R2"].to_numpy()[0])

    # Run metrics function (specify output)
    metrics = setup_preprocessor.metrics(y="k")

    # Shape and contents assertions
    assert metrics.shape == (6, 10)
    assert list(metrics.columns) == [
        "Model Types",
        "Parameter Configurations",
        "Train R2",
        "Train MAE",
        "Train MSE",
        "Train RMSE",
        "Test R2",
        "Test MAE",
        "Test MSE",
        "Test RMSE",
    ]
    assert metrics["Test R2"].to_numpy()[0] == np.max(metrics["Test R2"].to_numpy()[0])

    # Run metrics function (specify output)
    metrics = setup_preprocessor.metrics(y=0)

    # Shape and contents assertions
    assert metrics.shape == (6, 10)
    assert list(metrics.columns) == [
        "Model Types",
        "Parameter Configurations",
        "Train R2",
        "Train MAE",
        "Train MSE",
        "Train RMSE",
        "Test R2",
        "Test MAE",
        "Test MSE",
        "Test RMSE",
    ]
    assert metrics["Test R2"].to_numpy()[0] == np.max(metrics["Test R2"].to_numpy()[0])

    # Run metrics function (only fnn)
    metrics = setup_preprocessor.metrics(model_type="fnn")

    # Shape and contents assertions
    assert metrics.shape == (2, 10)
    assert list(metrics.columns) == [
        "Model Types",
        "Parameter Configurations",
        "Train R2",
        "Train MAE",
        "Train MSE",
        "Train RMSE",
        "Test R2",
        "Test MAE",
        "Test MSE",
        "Test RMSE",
    ]
    assert metrics["Test R2"].to_numpy()[0] == np.max(metrics["Test R2"].to_numpy()[0])

    # Run metrics function (change sort by)
    metrics = setup_preprocessor.metrics(sort_by="Train MSE")

    # Shape and contents assertions
    assert metrics.shape == (6, 10)
    assert list(metrics.columns) == [
        "Model Types",
        "Parameter Configurations",
        "Train R2",
        "Train MAE",
        "Train MSE",
        "Train RMSE",
        "Test R2",
        "Test MAE",
        "Test MSE",
        "Test RMSE",
    ]
    assert metrics["Train MAE"].to_numpy()[0] == np.min(
        metrics["Train MAE"].to_numpy()[0]
    )
