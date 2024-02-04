.. _models:

==========================
Model Dictionary Templates
==========================

This page includes templates for defining models supported by the :class:`pyMAISE.Tuner` and :class:`pyMAISE.PostProcessor`. The parameters of the dictionaries are set to their defaults.

--------------------------
Classical Model Templates
--------------------------

Regression
^^^^^^^^^^

**Linear Regressor**

.. code-block:: python

   "Linear": {
        "fit_intercept" = True,
        "copy_X" = True,
        "n_jobs" = None,
        "positive" = False,
   }

**Lasso Regressor**

.. code-block:: python

   "Lasso": {
        "alpha" = 1.0,
        "fit_intercept" = True,
        "precompute" = False,
        "copy_X" = True,
        "max_iter" = 1000,
        "tol" = 1e-4,
        "warm_start" = False,
        "positive" = False,
        "selection" = "cyclic",
   }

**Support Vector Machine Regressor**

.. code-block:: python

   "SVM": {
        "kernel" = "rbf",
        "degree" = 3,
        "gamma" = "scale",
        "coef0" = 0.0,
        "tol" = 1e-3,
        "C" = 1.0,
        "epsilon" = 0.1,
        "shrinking" = True,
        "cache_size" = 200,
        "max_iter" = -1,
   }

**Decision Tree Regressor**

.. code-block:: python

   "DT": {
        "criterion" = "squared_error",
        "splitter" = "best",
        "max_depth" = None,
        "min_samples_split" = 2,
        "min_samples_leaf" = 1,
        "min_weight_fraction_leaf" = 0.0,
        "max_features" = None,
        "max_leaf_nodes" = None,
        "min_impurity_decrease" = 0.0,
        "ccp_alpha" = 0.0,
   }

**Random Forest Regressor**

.. code-block:: python

   "RF": {
        "n_estimators" = 100,
        "criterion" = "squared_error",
        "max_depth" = None,
        "min_samples_split" = 2,
        "min_samples_leaf" = 1,
        "min_weight_fraction_leaf" = 0.0,
        "max_features" = None,
        "max_leaf_nodes" = None,
        "min_impurity_decrease" = 0.0,
        "bootstrap" = True,
        "oob_score" = False,
        "n_jobs" = None,
        "warm_start" = False,
        "ccp_alpha" = 0.0,
        "max_samples" = None,
   }

**K-Nearest Neighbors Regressor**

.. code-block:: python

   "KN": {
        "n_neighbors" = 5,
        "weights" = "uniform",
        "algorithm" = "auto",
        "leaf_size" = 30,
        "p" = 2,
        "metric" = "minkowski",
        "metric_params" = None,
        "n_jobs" = None,
   }

Classification
^^^^^^^^^^^^^^

**Logistic Regression**

.. code-block:: python

   "Logistic": {
        "penalty": "l2",
        "dual": False,
        "tol": 1e-4,
        "C": 1.0,
        "fit_intercept": True,
        "intercept_scaling": 1,
        "class_weight": None,
        "solver": "lbfgs",
        "max_iter": 100,
        "multi_class": "auto",
        "verbose": 0,
        "warm_start": False,
        "n_jobs": None,
        "l1_ratio": None,
   }

**Support Vector Machine Classifier**

.. code-block:: python

   "SVM": {
        "C": 1.0,
        "kernel": "rbf",
        "degree": 3,
        "gamma": "scale",
        "coef0": 0.0,
        "shrinking": True,
        "probability": False,
        "tol": 1e-3,
        "cache_size": 200,
        "class_weight": None,
        "verbose": False,
        "max_iter": -1,
        "decision_function_shape": "ovr",
        "break_ties": False,
   }

**Decision Tree Classifier**

.. code-block:: python

   "DT": {
        "criterion": "gini",
        "spitter": "best",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": None,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "class_weight": None,
        "ccp_alpha": 0.0,
   }

**Random Forest Classifier**

.. code-block:: python

   "RF": {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": False,
        "warm_start": False,
        "class_weight": None,
        "ccp_alpha": 0.0,
        "max_samples": None,
   }

**K-Nearest Neighbors Classifier**

.. code-block:: python

   "KN": {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
        "metric_params": None,
        "n_jobs": None,
   }

.. _nn_templates:

------------------------
Neural Network Templates
------------------------

Layers
^^^^^^

**Dense**

.. code-block:: python

   "Dense": {
       "units": ,
       "activation": None,
       "use_bias": True,
       "kernel_initializer": "glorot_uniform",
       "bias_initializer": "zeros",
       "kernel_regularizer": None,
       "bias_regularizer": None,
       "activity_regularizer": None,
       "kernel_constraint": None,
       "bias_constraint": None,
   }

**Dropout**

.. code-block:: python

   "Dropout": {
       "rate": ,
       "noise_shape": None,
   }

**LSTM**

.. code-block:: python

   "LSTM": {
       "units": ,
       "activation": "tanh",
       "recurrent_activation": "sigmoid",
       "use_bias": True,
       "kernel_initializer": "glorot_uniform",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "unit_forget_bias": True,
       "kernel_regularizer": None,
       "recurrent_regularizer": None,
       "bias_regularizer": None,
       "activity_regularizer": None,
       "kernel_constraint": None,
       "recurrent_constraint": None,
       "bias_constraint": None,
       "dropout": 0.0,
       "recurrent_dropout": 0.0,
       "return_sequences": False,
       "return_state": False,
       "go_backwards": False,
       "stateful": False,
       "time_major": False,
       "unroll": False,
   }

**GRU**

.. code-block:: python

   "GRU": {
       "units": ,
       "activation": "tanh",
       "recurrent_activation": "sigmoid",
       "use_bias": True,
       "kernel_initializer": "glorot_uniform",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "kernel_regularizer": None,
       "recurrent_regularizer": None,
       "bias_regularizer": None,
       "activity_regularizer": None,
       "kernel_constraint": None,
       "recurrent_constraint": None,
       "bias_constraint": None,
       "dropout": 0.0,
       "recurrent_dropout": 0.0,
       "return_sequences": False,
       "return_state": False,
       "go_backwards": False,
       "stateful": False,
       "time_major": False,
       "unroll": False,
       "reset_after": True,
   }

**Conv1D**

.. code-block:: python

   "Conv1D": {
       "filters": ,
       "kernel_size": ,
       "strides": 1,
       "padding": "valid",
       "data_format": "channels_last",
       "dilation_rate": 1,
       "groups": 1,
       "activation": "None",
       "use_bias": True,
       "kernel_initializer": "glorot_uniform",
       "bias_initializer": "zeros",
       "kernel_regularizer": None,
       "bias_regularizer": None,
       "activity_regularizer": None,
       "kernel_constraint": None,
       "bias_constraint": None,
   }

**Conv2D**

.. code-block:: python

   "Conv2D": {
       "filters": ,
       "kernel_size": ,
       "strides": (1, 1),
       "padding": "valid",
       "data_format": None,
       "dilation_rate": (1, 1),
       "groups": 1,
       "activation": "None",
       "use_bias": True,
       "kernel_initializer": "glorot_uniform",
       "bias_initializer": "zeros",
       "kernel_regularizer": None,
       "bias_regularizer": None,
       "activity_regularizer": None,
       "kernel_constraint": None,
       "bias_constraint": None,
       "input_shape": None,
   }

**Conv3D**

.. code-block:: python

   "Conv3D": {
       "filters": ,
       "kernel_size": ,
       "strides": (1, 1, 1),
       "padding": "valid",
       "data_format": None,
       "dilation_rate": (1, 1, 1),
       "groups": 1,
       "activation": "None",
       "use_bias": True,
       "kernel_initializer": "glorot_uniform",
       "bias_initializer": "zeros",
       "kernel_regularizer": None,
       "bias_regularizer": None,
       "activity_regularizer": None,
       "kernel_constraint": None,
       "bias_constraint": None,
   }

**MaxPooling1D**

.. code-block:: python

   "MaxPooling1D": {
       "pool_size": 2,
       "strides": None,
       "padding": "valid",
       "data_format": "channels_last",
   }

**MaxPooling2D**

.. code-block:: python

   "MaxPooling2D": {
       "pool_size": (2, 2),
       "strides": None,
       "padding": "valid",
       "data_format": None,
   }

**MaxPooling3D**

.. code-block:: python

   "MaxPooling3D": {
       "pool_size": (2, 2, 2),
       "strides": None,
       "padding": "valid",
       "data_format": None,
   }

**Flatten**

.. code-block:: python

   "Flatten": {
       "data_format": None,
   }

**Reshape**

.. code-block:: python

   "Reshape": {
       "target_shape": None,
   }

Optimizers
^^^^^^^^^^

**SGD**

.. code-block:: python

   "SGD": {
       "learning_rate": 0.01,
       "momentum": 0.0,
       "nesterov": False,
       "weight_decay": None,
       "clipnorm": None,
       "clipvalue": None,
       "global_clipnorm": None,
       "use_ema": False,
       "ema_momentum": 0.99,
       "ema_overwrite_frequency": None,
   }

**RMSprop**

.. code-block:: python

   "RMSprop": {
       "learning_rate": 0.001,
       "rho": 0.9,
       "momentum": 0.0,
       "epsilon": 1e-07,
       "centered": False,
       "weight_decay": None,
       "clipnorm": None,
       "clipvalue": None,
       "global_clipnorm": None,
       "use_ema": False,
       "ema_momentum": 0.99,
       "ema_overwrite_frequency": 100,
   }

**Adam**

.. code-block:: python

   "Adam": {
       "learning_rate": 0.001,
       "beta_1": 0.9,
       "beta_2": 0.999,
       "epsilon": 1e-07,
       "amsgrad": False,
       "weight_decay": None,
       "clipnorm": None,
       "clipvalue": None,
       "global_clipnorm": None,
       "use_ema": False,
       "ema_momentum": 0.99,
       "ema_overwrite_frequency": None,
   }

**AdamW**

.. code-block:: python

   "AdamW": {
       "learning_rate": 0.001,
       "weight_decay": 0.004,
       "beta_1": 0.9,
       "beta_2": 0.999,
       "epsilon": 1e-07,
       "amsgrad": False,
       "clipnorm": None,
       "clipvalue": None,
       "global_clipnorm": None,
       "use_ema": False,
       "ema_momentum": 0.99,
       "ema_overwrite_frequency": None,
   }

**Adadelta**

.. code-block:: python

   "Adadelta": {
       "learning_rate": 0.001,
       "rho": 0.95,
       "epsilon": 1e-7,
       "weight_decay": None,
       "clipnorm": None,
       "clipvalue": None,
       "global_clipnorm": None,
       "use_ema": False,
       "ema_momentum": 0.99,
       "ema_overwrite_frequency": None,
   }

**Adagrad**

.. code-block:: python

   "Adagrad": {
       "learning_rate": 0.001,
       "initial_accumulator_value": 0.1,
       "epsilon": 1e-07,
       "weight_decay": None,
       "clipnorm": None,
       "clipvalue": None,
       "global_clipnorm": None,
       "use_ema": False,
       "ema_momentum": 0.99,
       "ema_overwrite_frequency": None,
   }

**Adamax**

.. code-block:: python

   "Adamax": {
       "learning_rate": 0.001,
       "beta_1": 0.9,
       "beta_2": 0.999,
       "epsilon": 1e-07,
       "weight_decay": None,
       "clipnorm": None,
       "clipvalue": None,
       "global_clipnorm": None,
       "use_ema": False,
       "ema_momentum": 0.99,
       "ema_overwrite_frequency": None,
   }

**Adafactor**

.. code-block:: python

   "Adafactor": {
       "learning_rate": 0.001,
       "beta_2_decay": -0.8,
       "epsilon_1": 1e-30,
       "epsilon_2": 0.001,
       "clip_threshold": 1.0,
       "relative_step": True,
       "weight_decay": None,
       "clipnorm": None,
       "clipvalue": None,
       "global_clipnorm": None,
       "use_ema": False,
       "ema_momentum": 0.99,
       "ema_overwrite_frequency": None,
   }

**FTRL**

.. code-block:: python

   "Ftrl": {
       "learning_rate": 0.001,
       "learning_rate_power": -0.5,
       "initial_accumulator_value": 0.1,
       "l1_regularization_strength": 0.0,
       "l2_regularization_strength": 0.0,
       "l2_shrinkage_regularization_strength": 0.0,
       "beta": 0.0,
       "weight_decay": None,
       "clipnorm": None,
       "clipvalue": None,
       "global_clipnorm": None,
       "use_ema": False,
       "ema_momentum": 0.99,
       "ema_overwrite_frequency": None,
   }
