import pytest
import pyMAISE as mai
import numpy as np
from sklearn.model_selection import ShuffleSplit


def test_classification():
    # ===========================================================================
    # Regression test parameters
    # Data set parameters
    num_observations = 150
    num_features = 4
    num_outputs = 1

    # Expected model test r-squared
    expected_models = {
        "dtree": 1.0,
        "rforest": 1.0,
        "knn": 1.0,
    }
    plus_minus = 0.025

    # ===========================================================================
    # pyMAISE initialization
    settings = {
        "verbosity": 0,
        "random_state": 42,
        "test_size": 0.3,
        "num_configs_saved": 1,
        "regression": False,
        "classification": True,

    }
    global_settings = mai.settings.init(settings_changes=settings)

    # Assertions for global settings
    assert global_settings.verbosity == 0
    assert global_settings.random_state == 42
    assert global_settings.test_size == 0.3
    assert global_settings.num_configs_saved == 1

    # Get heat conduction preprocessor
    preprocessor = mai.load_iris()

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
    data = preprocessor.data_split()

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

    # Assert values are between 0 and 1
    for d in data:
        assert d[(d >= 0) & (d <= 1)].size == d.size

    # ===========================================================================
    # Model initialization
    model_settings = {
        "models": ["dtree", "rforest", "knn"],
    }
    tuning = mai.Tuning(data=data, model_settings=model_settings)

    # ===========================================================================
    # Hyper-parameter tuning
    grid_search_spaces = {
        "dtree": {
            "max_depth": [None, 5, 10, 25, 50],
            "max_features": [None, "sqrt", "log2"],
            "min_samples_leaf": [1, 2, 4, 6, 8, 10],
            "min_samples_split": [2, 4, 6, 8, 10],
        },
        "rforest": {
            "n_estimators": [50, 100, 150],
            "criterion": ["gini", "entropy", "log_loss"],
            "min_samples_split": [2, 4, 6],
            "max_features": ["sqrt", "log2"],
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

    for key, value in expected_models.items():
        assert postprocessor.metrics(model_type=key)["Test Accuracy"].to_numpy()[
            0
        ] == pytest.approx(value, plus_minus / value)
