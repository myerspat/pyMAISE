
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.gridspec import GridSpec
from scipy.stats import randint, uniform
from sklearn.model_selection import ShuffleSplit

import pyMAISE as mai



def test_benchmark_nn_old_to_new():
    # ===========================================================================
    # Regression test parameters
    # Data set parameters
     # Expected model test r-squared
    plus_minus = 0.025

    datasets = [mai.load_xs(), mai.heat]
    for data_set in datasets:
        #----------------------------------------
        #  new model generation with close to the same search space
        #----------------------------------------   
        settings = {
            "verbosity": 0,
            "random_state": 42,
            "test_size": 0.3,
            "num_configs_saved": 5,
            "regression": True,
            "cuda_visible_devices": "-1", # Use CPUs only
        }

        # Setting for new nn model info
        global_settings = mai.settings.init(settings_changes=settings)
        
        # Initializing preprocessor and scaling
        preprocessor = data_set
        data = preprocessor.min_max_scale()

        # Constructing hypermodel nn
        structural_hyperparameters = {
            "dense_input": {
                "units": mai.Int(min_value=25, max_value=300, step=10),
                "input_dim": preprocessor.inputs.shape[1],
                "activation": "relu",
                "kernel_initializer": "normal",
            },
            "dense_hidden": {
                "units": mai.Int(min_value=25, max_value=300, step=10),
                "activation": "relu",
                "kernel_initializer": "normal",
                "num_layers": mai.Int(min_value=1, max_value=4),
            },
            "dense_output": {
                "units": preprocessor.outputs.shape[1],
                "input_dim": None,
                "activation": "linear",
                "kernel_initializer": "normal",
            },
        }

        adam = {
            "learning_rate": 1e-3,
        }


        model_settings = {
            "models": ["fnn"],
            "fnn": {
                "structural_params": structural_hyperparameters,
                "optimizer": mai.Choice(["adam"]),
                "adam": adam,
                "compile_params": {
                    "loss": "mean_absolute_error",
                    "metrics": ["mean_absolute_error"],
                },
                "fitting_params": {
                    "batch_size": 8,
                    "epochs": mai.Int(8, 64, 10),
                    "validation_split": 0.15,
                },
            },
        }

        tuning = mai.Tuning(data=data, model_settings=model_settings)
        bayesian_search = tuning.nn_bayesian_search(
            models = ["fnn"],
            objective = 'r2_score',
            max_trials = 5,
            cv = 5,
        )

        new_model_settings = {
            "fnn": {
                "fitting_params":{
                    "epochs": 200,
                }
            }
        }
        new_nn_postprocessor = mai.PostProcessor(
            data=data,
            models_list=[bayesian_search],
            new_model_settings=new_model_settings,
        )

        #----------------------------------------
        # Old model generation with close to the same search space
        #----------------------------------------

        settings = {
        "verbosity": 0,
        "random_state": 42,
        "test_size": 0.3,
        "num_configs_saved": 1,
        "regression": True,
        "cuda_visible_devices": "-1", # Use CPUs only
        "new_nn_architecture": False,   
        }

        
        # Setting for new nn model info
        global_settings = mai.settings.init(settings_changes=settings)
        

        model_settings = {
            "models": ["nn"],
            "nn": {
                # Sequential
                "num_layers": 4,
                "dropout": True,
                "rate": 0.5,
                "validation_split": 0.15,
                "loss": "mean_absolute_error",
                "metrics": ["mean_absolute_error"],
                "batch_size": 8,
                "epochs": 50,
                "warm_start": True,
                "jit_compile": False,
                # Starting Layer
                "start_num_nodes": 100,
                "start_kernel_initializer": "normal",
                "start_activation": "relu",
                "input_dim": preprocessor.inputs.shape[1], # Number of inputs
                # Middle Layers
                "mid_num_node_strategy": "linear", # Middle layer nodes vary linearly from 'start_num_nodes' to 'end_num_nodes'
                "mid_kernel_initializer": "normal",
                "mid_activation": "relu",
                # Ending Layer
                "end_num_nodes": preprocessor.outputs.shape[1], # Number of outputs
                "end_activation": "linear",
                "end_kernel_initializer": "normal",
                # Optimizer
                "optimizer": "adam",
                "learning_rate": 5e-4,
            },
        }
        tuning = mai.Tuning(data=data, model_settings=model_settings)

        bayesian_search_spaces = {
            "nn": {
                "mid_num_node_strategy": ["constant", "linear"],
                "batch_size": [8, 64],
                "learning_rate": [1e-5, 0.001],
                "num_layers": [2, 6],
                "start_num_nodes": [25, 300],
            },
        }


        bayesian_search_configs = tuning.bayesian_search(
            param_spaces=bayesian_search_spaces,
            models=bayesian_search_spaces.keys(),
            n_iter=5,
            cv=5,
        )

        new_model_settings = {
            "nn": {"epochs": 200}
        }

        old_nn_postprocessor = mai.PostProcessor(
            data=data, 
            models_list=[bayesian_search_configs], 
            new_model_settings=new_model_settings,
            yscaler=preprocessor.yscaler,
        )

        # Asserting if the R2 is within a 0.02 tolerence of each other for similiar metrics
        plus_minus=0.02
        assert old_nn_postprocessor.metrics(model_type="nn")["Test R2"].to_numpy()[[0] == pytest.approx(new_nn_postprocessor.metrics(model_type="fnn")["Test R2"].to_numpy()[[0], plus_minus /  pytest.approx(new_nn_postprocessor.metrics(model_type="fnn")["Test R2"].to_numpy()[[0])
   

            