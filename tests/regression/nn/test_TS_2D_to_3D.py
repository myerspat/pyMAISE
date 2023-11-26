import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai


def test_new_nn_lstm_univariate_series():
    # Initialize pyMAISE
    settings = {
        "verbosity": 1,
        "random_state": 7,
        "test_size": 0.3,
        "regression": True,
        "classification": False,
        "num_configs_saved": 1,
    }
    settings = mai.settings.init(settings_changes=settings)

    # Get univariate timeseries data (airline data)
    data = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    )

    # Define raw sequence data and pass to preprocessor
    data_xarray = xr.DataArray(
        data.iloc[:, 1].values.reshape(144, 1),
        coords={"timesteps": data.iloc[:, 0].values, "features": ["passengers"]},
    )
    preprocessor = mai.PreProcessor()
    preprocessor.set_data(data_xarray)

    # Assert data shapes
    assert preprocessor.data.shape == (144, 1)

    # Check contents prior to splitting
    np.testing.assert_array_equal(
        preprocessor.data.sel(features="passengers").values, data.iloc[:, 1]
    )

    # Split data and confirm shapes and contents of first 2 samples
    preprocessor.split_sequences(
        input_steps=1,
        output_steps=1,
        output_position=1,
        sequence_inputs=["passengers"],
        sequence_outputs=["passengers"],
    )
    assert preprocessor.inputs.shape == (
        143,
        1,
        1,
    )
    assert preprocessor.outputs.shape == (
        143,
        1,
        1,
    )

    # Initialize scaling
    preprocessor.train_test_split(scaler=MinMaxScaler())
    xtrain, xtest, ytrain, ytest = preprocessor.split_data

    # Testing if split is is 20 %
    assert xtrain.shape == (100, 1, 1)
    assert ytrain.shape == (100, 1, 1)
    assert xtest.shape == (43, 1, 1)
    assert ytest.shape == (43, 1, 1)

    # Creating RNN model
    structural_hyperparameters = {
        "LSTM_input": {
            "units": 4,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "input_shape": (1, 1),
        },
        "Dense_output": {
            "units": 1,
            "activation": "linear",
        },
    }
    model_settings = {
        "models": ["rnn"],
        "rnn": {
            "structural_params": structural_hyperparameters,
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": 1e-3,
            },
            "compile_params": {
                "loss": "mean_squared_error",
                "metrics": ["mean_squared_error"],
            },
            "fitting_params": {
                "batch_size": 1,
                "epochs": 100,
            },
        },
    }

    # Passing preprocessed data into tuner function for HP tuning
    tuner = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

    # Grid search
    grid_search_configs = tuner.nn_grid_search(
        objective="mean_squared_error",
        cv=ShuffleSplit(n_splits=1),
    )
    assert isinstance(grid_search_configs["rnn"][0], pd.DataFrame)
    assert isinstance(grid_search_configs["rnn"][1], mai.nnHyperModel)
    assert grid_search_configs["rnn"][0].shape == (1, 1)
    assert tuner.cv_performance_data["rnn"].shape == (2, 1)

    # Model post-processing
    new_model_settings = {
        "rnn": {
            "fitting_params": {
                "epochs": 100,
            },
        },
    }
    postprocessor = mai.PostProcessor(
        data=preprocessor.split_data,
        models_list=[grid_search_configs],
        new_model_settings=new_model_settings,
        yscaler=preprocessor.yscaler,
    )
    print(postprocessor.metrics()[["Train RMSE", "Test RMSE"]].values.tolist()[0])
    assert postprocessor.metrics()[["Train RMSE", "Test RMSE"]].values.tolist()[0] == [
        22.68,
        49.34,
    ]
