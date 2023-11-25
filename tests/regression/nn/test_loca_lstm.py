import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai


def test_loca_lstm():
    # Initialize pyMAISE
    settings = {
        "verbosity": 1,
        "random_state": 42,
        "test_size": 0.3,
        "num_configs_saved": 2,
        "regression": True,
        "new_nn_architecture": True,
        "cuda_visible_devices": "-1",  # Use CPUs only
    }
    global_settings = mai.settings.init(settings_changes=settings)

    # Load LOCA data and shrink dataset for test speed
    preprocessor = mai.load_loca()
    preprocessor.data = preprocessor.data.isel(samples=range(1000))

    # Split sequence data
    preprocessor.split_sequences(
        input_steps=4,
        output_steps=1,
        output_position=1,
        sequence_inputs=["PCT"],
        sequence_outputs=["PCT"],
        feature_inputs=preprocessor.data.coords["features"].values[:-1],
    )
    assert preprocessor.data.shape == (1000, 400, 41)
    assert preprocessor.inputs.shape == (1000, 396, 44)
    assert preprocessor.outputs.shape == (1000, 396, 1)

    # Train test split data
    preprocessor.train_test_split(scaler=MinMaxScaler())

    assert preprocessor.split_data[0].shape == (700, 396, 44)
    assert preprocessor.split_data[1].shape == (300, 396, 44)
    assert preprocessor.split_data[2].shape == (700, 396, 1)
    assert preprocessor.split_data[3].shape == (300, 396, 1)
    print(preprocessor.inputs.shape)

    # RNN model settings
    structural = {
        "lstm_input": {
            "units": 100,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "return_sequences": True,
            "input_shape": (396, 44),
        },
        "lstm_hidden0": {
            "units": 80,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "return_sequences": True,
        },
        "lstm_hidden1": {
            "units": 60,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "return_sequences": True,
        },
        "lstm_hidden2": {
            "units": 40,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "return_sequences": True,
        },
        "dense_output": {
            "units": preprocessor.outputs.shape[-1],
            "activation": "linear",
        },
    }
    model_settings = {
        "models": ["rnn"],
        "rnn": {
            "structural_params": structural,
            "optimizer": "adam",
            "adam": {
                "learning_rate": mai.Choice([0.0001, 0.001]),
                "clipnorm": 1.0,
                "clipvalue": 0.5,
            },
            "compile_params": {
                "loss": "mean_absolute_error",
                "metrics": ["mean_absolute_error"],
            },
            "fitting_params": {"batch_size": 16, "epochs": 5, "validation_split": 0.15},
        },
    }
    tuner = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

    # Grid search
    grid_search_configs = tuner.nn_grid_search(
        objective="r2_score",
        cv=TimeSeriesSplit(n_splits=2),
    )
    assert isinstance(grid_search_configs["rnn"][0], pd.DataFrame)
    assert isinstance(grid_search_configs["rnn"][1], mai.nnHyperModel)
    assert grid_search_configs["rnn"][0].shape == (2, 1)
    assert tuner.cv_performance_data["rnn"].shape == (2, 2)

    # Model post-processing
    new_model_settings = {
        "rnn": {
            "fitting_params": {
                "epochs": 10,
            },
        },
    }
    postprocessor = mai.PostProcessor(
        data=preprocessor.split_data,
        models_list=[grid_search_configs],
        new_model_settings=new_model_settings,
        yscaler=preprocessor.yscaler,
    )
    assert postprocessor.metrics().shape == (2, 10)
