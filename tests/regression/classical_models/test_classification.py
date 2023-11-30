import pytest
from sklearn.model_selection import ShuffleSplit

import pyMAISE as mai


def test_classification():
    # ===========================================================================
    # pyMAISE initialization
    settings = {
        "verbosity": 1,
        "random_state": 42,
        "test_size": 0.3,
        "num_configs_saved": 1,
        "regression": False,
        "classification": True,
        "cuda_visible_devices": "-1",  # Use CPUs only
    }
    global_settings = mai.settings.init(settings_changes=settings)

    # Assertions for global settings
    assert global_settings.verbosity == 1
    assert global_settings.random_state == 42
    assert global_settings.test_size == 0.3
    assert global_settings.num_configs_saved == 1

    # Get iris data
    preprocessor = mai.PreProcessor()
    preprocessor.read_csv(
        "https://raw.githubusercontent.com/scikit-learn/scikit-learn/04e39db499"
        + "afab852e4e2603807384a402a871a9/sklearn/datasets/data/iris.csv",
        slice(0, 4),
        slice(4, 5),
    )
    # Change output data type and one hot encode
    preprocessor.outputs = preprocessor.outputs.astype("object")
    preprocessor.outputs = preprocessor.one_hot_encode(preprocessor.outputs)

    # Assert inputs and outputs are the correct size
    assert preprocessor.data.shape == (150, 5)
    assert preprocessor.inputs.shape == (150, 4)
    assert preprocessor.outputs.shape == (150, 3)

    # Train test split
    preprocessor.train_test_split()

    # Train-test split size assertions
    assert preprocessor.split_data[0].shape == (105, 4)
    assert preprocessor.split_data[1].shape == (45, 4)
    assert preprocessor.split_data[2].shape == (105, 3)
    assert preprocessor.split_data[3].shape == (45, 3)

    # ===========================================================================
    # Model initialization
    model_settings = {
        "models": ["dtree", "rforest", "knn"],
    }
    tuning = mai.Tuner(data=preprocessor.split_data, model_settings=model_settings)

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
        data=preprocessor.split_data,
        models_list=[grid_search_configs],
    )

    # Performance metric assertions
    expected_models = {
        "dtree": 1.0,
        "rforest": 1.0,
        "knn": 1.0,
    }
    for key, value in expected_models.items():
        assert postprocessor.metrics(model_type=key)["Test Accuracy"].to_numpy()[
            0
        ] == pytest.approx(value, 0.0001)
