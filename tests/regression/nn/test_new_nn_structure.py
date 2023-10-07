import pandas as pd
import pytest
from scipy.stats import randint, uniform
from sklearn.model_selection import ShuffleSplit

import pyMAISE as mai


def test_new_nn_structure():
    plus_minus = 0.015

    # Loop over each base benchmark data set
    load_functions = ["load_MITR", "load_fp", "load_xs"]

    for load_function in load_functions:
        # ======================================================================
        # Old Model Structure
        settings = {
            "verbosity": 0,
            "random_state": 42,
            "test_size": 0.3,
            "num_configs_saved": 1,
            "regression": True,
            "cuda_visible_devices": "-1",  # Use CPUs only
            "new_nn_architecture": False,
        }

        # Initialize pyMAISE settings
        global_settings = mai.settings.init(settings_changes=settings)

        # Initialize preprocessor and scaling
        preprocessor = eval("mai." + load_function + "()")
        data = preprocessor.min_max_scale()

        # Old NN model settings
        model_settings = {
            "models": ["nn"],
            "nn": {
                # Sequential
                "num_layers": 2,
                "dropout": True,
                "rate": 0.5,
                "validation_split": 0.15,
                "loss": "mean_absolute_error",
                "metrics": ["mean_absolute_error"],
                "batch_size": 16,
                "epochs": 50,
                # Starting Layer
                "start_num_nodes": 100,
                "start_kernel_initializer": "normal",
                "start_activation": "relu",
                "input_dim": preprocessor.inputs.shape[1],  # Number of inputs
                # Middle Layers
                "mid_num_node_strategy": "constant",  # Middle layer nodes vary linearly from 'start_num_nodes' to 'end_num_nodes'
                "mid_kernel_initializer": "normal",
                "mid_activation": "relu",
                # Ending Layer
                "end_num_nodes": preprocessor.outputs.shape[1],  # Number of outputs
                "end_activation": "linear",
                "end_kernel_initializer": "normal",
                # Optimizer
                "optimizer": "adam",
                "learning_rate": 0.0001,
            },
        }
        tuning = mai.Tuning(data=data, model_settings=model_settings)

        # Grid search space
        grid_search_space = {
            "nn": {
                # Sequantial
                "batch_size": [16, 64],
                # Starting Layer
                "start_num_nodes": [100, 400],
                # Optimizer
                "learning_rate": [0.0001, 0.001],
            }
        }

        # Grid search
        grid_search_configs = tuning.grid_search(
            param_spaces=grid_search_space,
            models=grid_search_space.keys(),
            cv=ShuffleSplit(
                n_splits=2, test_size=0.15, random_state=global_settings.random_state
            ),
        )

        new_model_settings = {"nn": {"epochs": 200}}

        # Model post-processing
        postprocessor = mai.PostProcessor(
            data=data,
            models_list=[grid_search_configs],
            new_model_settings=new_model_settings,
            yscaler=preprocessor.yscaler,
        )

        old_nn_structure_results = postprocessor.metrics()

        # ======================================================================
        # New Model Structure
        settings["new_nn_architecture"] = True

        # Initialize pyMAISE settings
        global_settings = mai.settings.init(settings_changes=settings)

        # New NN model settings
        structural = {
            "dense_input": {
                "units": mai.Choice([100, 400]),
                "input_dim": preprocessor.inputs.shape[1],
                "activation": "relu",
                "kernel_initializer": "normal",
            },
            "dropout_input": {
                "rate": 0.5,
            },
            "dense_output": {
                "units": preprocessor.outputs.shape[1],
                "activation": "linear",
                "kernel_initializer": "normal",
            },
        }
        model_settings = {
            "models": ["nn"],
            "nn": {
                "structural_params": structural,
                "optimizer": "adam",
                "adam": {
                    "learning_rate": mai.Choice([0.0001, 0.001]),
                },
                "compile_params": {
                    "loss": "mean_absolute_error",
                    "metrics": ["mean_absolute_error"],
                },
                "fitting_params": {
                    "batch_size": mai.Choice([16, 64]),
                    "epochs": 50,
                    "validation_split": 0.15,
                },
            },
        }
        tuning = mai.Tuning(data=data, model_settings=model_settings)

        # Grid search
        grid_search_configs = tuning.nn_grid_search(
            objective="r2_score",
            cv=ShuffleSplit(
                n_splits=2, test_size=0.15, random_state=global_settings.random_state
            ),
        )

        # Model post-processing
        postprocessor = mai.PostProcessor(
            data=data,
            models_list=[grid_search_configs],
            new_model_settings=new_model_settings,
            yscaler=preprocessor.yscaler,
        )

        new_nn_structure_results = postprocessor.metrics()
        print(f"Data set: {load_function}")
        print("Old Model Results\n", old_nn_structure_results.to_string())
        print("New Model Results\n", new_nn_structure_results.to_string())

        # Compare top models hyperparameters
        assert (
            old_nn_structure_results.loc[0, "Parameter Configurations"][
                "start_num_nodes"
            ]
            == new_nn_structure_results.loc[0, "Parameter Configurations"][
                "dense_input0_units"
            ]
        )
        assert (
            old_nn_structure_results.loc[0, "Parameter Configurations"]["learning_rate"]
            == new_nn_structure_results.loc[0, "Parameter Configurations"][
                "Adam_learning_rate"
            ]
        )
        assert (
            old_nn_structure_results.loc[0, "Parameter Configurations"]["batch_size"]
            == new_nn_structure_results.loc[0, "Parameter Configurations"]["batch_size"]
        )

        # Compare performance metrics
        assert (
            abs(
                old_nn_structure_results.loc[0, "Train R2"]
                - new_nn_structure_results.loc[0, "Train R2"]
            )
            / old_nn_structure_results.loc[0, "Train R2"]
            < plus_minus
        )
        assert (
            abs(
                old_nn_structure_results.loc[0, "Test R2"]
                - new_nn_structure_results.loc[0, "Test R2"]
            )
            / old_nn_structure_results.loc[0, "Test R2"]
            < plus_minus
        )
