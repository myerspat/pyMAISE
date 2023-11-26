
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import xarray as xr


from matplotlib.gridspec import GridSpec
from scipy.stats import randint, uniform
from sklearn.model_selection import ShuffleSplit, TimeSeriesSplit

from pyMAISE.preprocessor import PreProcessor
import pyMAISE as mai
from sklearn.preprocessing import MinMaxScaler

def test_new_nn_lstm_univariate_series():

        
    settings = {
            "verbosity": 1,
            "random_state": 42,
            "test_size": 0.3,
            "regression": True,
            "classification": False,
    }
    settings = mai.settings.init(settings_changes=settings)

    data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')

    # Define raw sequence data
    print()
    num_sequences = len(data) # Number of data points
    raw_data = np.zeros((num_sequences, 3))
    raw_data = data.iloc[:, 1]
    univariate_xarray = (
            xr.Dataset(
                {
                    "sec0": (["timesteps"], data.iloc[:, 1]),
                },
                coords={
                    "timesteps": np.linspace(0, len(raw_data)-1, num_sequences),
                },
            )
            .to_array()
            .transpose(..., "variable")
    )
    print(univariate_xarray)
    print('--------------------------------------')
    print('--------------------------------------')
    print('--------------------------------------')
    print('--------------------------------------')

    # ============================================================
        # Univariate
        # Input: (samples=6, timesteps=3, features=1)
        # Output: (samples=6, timesteps=1, features=1)
        # ============================================================
        # What is defined in preprocessor.data, preprocessor.input,
        # and preprocessor.output is already given in a univariate form
        # with "sec1" and "sec2" as inputs and "sec3" as output
    preprocessor = mai.PreProcessor()
    preprocessor.set_data(univariate_xarray, inputs=["sec0"])

    # Check contents prior to splitting
    np.testing.assert_array_equal(
            preprocessor.data.sel(variable="sec0").to_numpy(), data.iloc[:, 1]
    )


    # Split data and confirm shapes and contents of first 2 samples
   # Split data and confirm shapes and contents of first 2 samples
    preprocessor.split_sequences(
        input_steps=1,
        output_steps=1,
        output_position=1,
        sequence_inputs=["sec0"],
        sequence_outputs=["sec0"],
    )

    assert preprocessor.inputs.shape == (143, 1, 1) # reshape input to be [samples, time steps, features]
    assert preprocessor.outputs.shape == (143, 1, 1) # reshape output to be [samples, time steps, features]
    print(preprocessor.inputs.shape)

    # Initialize scaling 
    preprocessor.train_test_split(scaler=MinMaxScaler())
    xtrain, xtest, ytrain, ytest = preprocessor.split_data
    print(xtrain.shape)
    print(ytrain.shape)

    # Testing if split is is 20 % 
    assert xtrain.shape == (100, 1, 1)
    assert ytrain.shape == (100, 1, 1)
    assert xtest.shape == (43, 1, 1)
    assert ytest.shape == (43, 1, 1)


    # Creating RNN model
    # pyMAISE Initialization
    structural_hyperparameters = {
        "lstm_input": {
            "units": 4,
            "activation": 'tanh',
            "recurrent_activation": "sigmoid",
            "input_shape": (1, 1),

        },
        "dense_output": {
            "units": 1,
            "activation": "linear",
        },
    }

    adam = {
        "learning_rate": 1e-3,
    }


    model_settings = {
        "models": ["rnn"],
        "rnn": {
            "structural_params": structural_hyperparameters,
            "optimizer": "adam",
            "adam": {
                "learning_rate": mai.Choice([0.0001, 0.001]),
                "clipnorm": 1.0,
                "clipvalue": 0.5,
            },
            "compile_params": {
                "loss": "mean_squared_error",
                "metrics": ["mean_squared_error"],
            },
            "fitting_params": {"batch_size": 1, "epochs": 100, "validation_split": 0.15},
        },
    }

    # passing preprocessed data into tuner function for HP tuning
    data = (xtrain, xtest, ytrain, ytest)
    tuning = mai.Tuner(data=data, model_settings=model_settings)

    tuner = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

    # Grid search
    grid_search_configs = tuner.nn_grid_search(
        objective="mean_squared_error",
        
    )
    # assert isinstance(grid_search_configs["rnn"][0], pd.DataFrame)
    # assert isinstance(grid_search_configs["rnn"][1], mai.nnHyperModel)
    # assert grid_search_configs["rnn"][0].shape == (2, 1)
    # assert tuner.cv_performance_data["rnn"].shape == (2, 2)

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
    print("METRICSSSS")
    print(postprocessor.metrics())

    assert 1==0