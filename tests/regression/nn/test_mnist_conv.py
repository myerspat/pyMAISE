
%load_ext autoreload
%autoreload 2
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai

import pandas as pd
import pytest
from tensorflow.keras.utils import to_categorical
import pyMAISE as mai

# Import mnist dataset
from tensorflow.keras.datasets import mnist

def test_mnist_conv():
    # Initialize pyMAISE
    settings = {
        "verbosity": 1,
        "random_state": 42,
        "test_size": 0.3,
        "num_configs_saved": 2,
        "regression": False,
        "classification": True,
        "new_nn_architecture": True,
        "cuda_visible_devices": "-1",  # Use CPUs only
    }
    global_settings = mai.settings.init(settings_changes=settings)

    # load train and test dataset
    def load_dataset():
        # load dataset
        (trainX, trainY), (testX, testY) = mnist.load_data()
        # reshape dataset to have a single channel
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))
        # one hot encode target values
        trainY = to_categorical(trainY)
        testY = to_categorical(testY)
        return trainX, trainY, testX, testY

    # scale pixels
    def prep_pixels(train, test):
        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        # return normalized images
        return train_norm, test_norm
    
    # --------------
    # Update preprocessor and add new functionality here
    #---------------

    # Using Sve Final Model archetecture
    structural = {
        "Conv2D_input": {
            "filters": 32,
            "kernel_size": (3, 3),
            "activation": "relu",
            "kernel_initializer": 'he_uniform',
            "input_shape": (28, 28, 1),
        },
        "MaxPooling2D": {
            "pool_size": (2, 2),
        },
        "Conv2D_hidden1": {
            "filters": 32,
            "kernel_size": (3, 3),
            "activation": "relu",
            "kernel_initializer": 'he_uniform',
            "input_shape": (28, 28, 1),
        },
        "Conv2D_hidden2": {
            "filters": 32,
            "kernel_size": (3, 3),
            "activation": "relu",
            "kernel_initializer": 'he_uniform',
            "input_shape": (28, 28, 1),
        },
        "MaxPooling2D_hidden1": {
           "pool_size": (2, 2),
        },
        "Flatten": {

        },
        "Dense_hidden": {
            "units": 100,
            "activation": "relu",
            "kernel_initializer": "he_uniform",
        },
        "Dense_output": {
            "units": 10,
            "activation": "softmax",
        },
    }

    model_settings = {
        "models": ["cnn"],
        "cnn": {
            "structural_params": structural,
            "optimizer": "SGD",
            "SGD": {
                "learning_rate": mai.Choice([0.0001, 0.01]),
                "momentum": 0.9,
            },
            "compile_params": {
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy"],
            },
            "fitting_params": {"batch_size": 32, "epochs": 10, "validation_split": 0.15},
        },
    }
    
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data and converting it to a xarray for pyMAISE
    trainX, testX = prep_pixels(trainX, testX)
    trainX = xr.DataArray(trainX, dims=['Number of Samples', 'Height', 'Width', 'Channels'])
    trainY = xr.DataArray(trainY, dims=['Number of Samples', 'Number of Classes'])
    testX  = xr.DataArray(testX,  dims=['Number of Samples', 'Height', 'Width', 'Channels'])
    testY  =  xr.DataArray(trainY, dims=['Number of Samples', 'Number of Classes'])    
    data=(trainX, testX, trainY, testY)
    
    
    tuner = mai.Tuner(data=data, model_settings=model_settings)
    
    # Grid search
    grid_search_configs = tuner.nn_grid_search(
        objective="accuracy_score",
        cv=5
    )

#     assert isinstance(grid_search_configs["cnn"][0], pd.DataFrame)
#     assert isinstance(grid_search_configs["cnn"][1], mai.nnHyperModel)
#     assert grid_search_configs["cnn"][0].shape == (2, 1)
#     assert tuner.cv_performance_data["cnn"].shape == (2, 2)

    # Model post-processing
    new_model_settings = {
        "cnn": {
            "fitting_params": {
                "epochs": 10,
            },
        },
    }
    postprocessor = mai.PostProcessor(
        data=data,
        models_list=[grid_search_configs],
        new_model_settings=new_model_settings,
        
    )
    # assert postprocessor.metrics().shape == (2, 10)
