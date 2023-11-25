import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai


def test_mitr():
    # ===========================================================================
    # Regression test parameters
    # Data set parameters
    num_observations = 1000
    num_features = 6
    num_outputs = 22

    # Expected performance metrics
    expected_metrics = pd.read_csv(
        mai.data._handler.get_full_path(
            "../tests/regression/classical_models/supporting/"
            + "mitr_testing_metrics.csv"
        )
    )

    # ===========================================================================
    # pyMAISE initialization
    settings = {
        "verbosity": 1,
        "random_state": 42,
        "test_size": 0.3,
        "num_configs_saved": 1,
        "regression": True,
        "classification": False,
        "cuda_visible_devices": "-1",  # Use CPUs only
    }
    global_settings = mai.settings.init(settings_changes=settings)

    # Assertions for global settings
    assert global_settings.verbosity == 1
    assert global_settings.random_state == 42
    assert global_settings.test_size == 0.3
    assert global_settings.num_configs_saved == 1

    # Get heat conduction preprocessor
    preprocessor = mai.load_MITR()

    # Assert inputs and outputs are the correct size
    assert (
        preprocessor.inputs.shape[0] == num_observations
        and preprocessor.inputs.shape[1] == num_features
    )
    assert (
        preprocessor.outputs.shape[0] == num_observations
        and preprocessor.outputs.shape[1] == num_outputs
    )

    # Train test split
    preprocessor.train_test_split(scaler=MinMaxScaler())
    data = preprocessor.split_data

    # Train-test split size assertions
    assert (
        data[0].shape[0] == num_observations * (1 - global_settings.test_size)
        and data[0].shape[1] == num_features
    )
    assert (
        data[1].shape[0] == num_observations * global_settings.test_size
        and data[1].shape[1] == num_features
    )
    assert (
        data[2].shape[0] == num_observations * (1 - global_settings.test_size)
        and data[2].shape[1] == num_outputs
    )
    assert (
        data[3].shape[0] == num_observations * global_settings.test_size
        and data[3].shape[1] == num_outputs
    )

    # ===========================================================================
    # Model initialization
    model_settings = {
        "models": ["linear", "lasso", "dtree", "knn", "rforest"],
    }
    tuning = mai.Tuner(data=data, model_settings=model_settings)

    # ===========================================================================
    # Hyper-parameter tuning
    grid_search_spaces = {
        "linear": {"fit_intercept": [True, False]},
        "lasso": {"alpha": np.linspace(0.000001, 1, 200)},
        "dtree": {
            "max_depth": [None, 5, 10, 25, 50],
            "max_features": [None, "sqrt", "log2", 0.2, 0.4, 0.6, 0.8, 1],
            "min_samples_leaf": [1, 2, 4, 6, 8, 10],
            "min_samples_split": [2, 4, 6, 8, 10],
        },
        "rforest": {
            "n_estimators": [50, 100, 150],
            "criterion": ["squared_error", "absolute_error"],
            "min_samples_split": [2, 4],
            "max_features": [None, "sqrt", "log2", 1],
        },
        "knn": {
            "n_neighbors": [1, 2, 4, 6, 8, 10, 14, 17, 20],
            "weights": ["uniform", "distance"],
            "leaf_size": [1, 5, 10, 15, 20, 25, 30],
        },
    }

    grid_search_configs = tuning.grid_search(
        param_spaces=grid_search_spaces,
        models=grid_search_spaces.keys(),
        cv=ShuffleSplit(
            n_splits=1, test_size=0.15, random_state=global_settings.random_state
        ),
    )

    # ===========================================================================
    # Model post-processing
    postprocessor = mai.PostProcessor(
        data=data,
        models_list=[grid_search_configs],
    )
    metrics = postprocessor.metrics()[
        [
            "Model Types",
            "Train MAE",
            "Train MSE",
            "Train RMSE",
            "Train R2",
            "Test MAE",
            "Test MSE",
            "Test RMSE",
            "Test R2",
        ]
    ]

    # Assert expected dataframe and results match
    print(
        "Expected Values\n",
        expected_metrics.sort_values(by=["Test R2"], ascending=False),
    )
    print("pyMAISE Values\n", metrics)
    pd.testing.assert_frame_equal(
        expected_metrics.sort_values(by=["Test R2"], ascending=False), metrics
    )
