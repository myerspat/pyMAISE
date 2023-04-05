.. _models:

=======================
Machine Learning Models
=======================

The machine learning (ML) models supported in pyMAISE include:

- linear regression,
- lasso regression, 
- support vector regression,
- decision tree regression,
- random forest regression,
- k-nearest neighbors regression,
- and sequential dense neural networks.

Each model is discussed in more detail below and dictionary templates are provided in the :ref:`Model Dictionary Templates <model_temp>` section.

-----------------
Linear Regression
-----------------

This model is based on ``sklearn.linear_model.LinearRegression`` which minimizes the residual sum of squares between data points assuming a linear function. For more information refer to https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html.

----------------
Lasso Regression
----------------

Lasso regression uses the ``sklearn.linear_model.Lasso`` which is a linear regression with L1 prior as regularizer. For more information on the method in Python reger to https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html.

-------------------------
Support Vector Regression
-------------------------

Support vector regression uses ``sklearn.svm.SVR`` to do epsilon-support vector regression. For more information refer to https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html.

.. caution:: Support vector regression is **only** supported for 1-dimensional outputs.

------------------------
Decision Tree Regression
------------------------

This method uses the ``sklearn.tree.DecisionTreeRegressor`` to build a decision tree. Refer to https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html for more information.

------------------------
Random Forest Regression
------------------------

Random forest regression uses the ``sklearn.ensemble.RandomForestRegressor`` to build a specified number of decision trees and uses averaging to improve accuracy and over-fitting. Refer to https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html for more information.


------------------------------
K-Nearest Neighbors Regression
------------------------------

K-nearest neighbors in pyMAISE uses ``sklearn.neighbors.KNeighborsRegressor``. For more information refer to https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html.

--------------------------------
Sequential Dense Neural networks
--------------------------------

This model is the most complicated model implemented in pyMAISE and uses Keras sequential models with dense layers. Dropout layers are also used. For hyper-parameter tuning the ``mid_num_node_strategy`` parameter is used to define the hidden layer nodes based on callable functions. Currently ``"linear"`` and ``"constant"`` are supported. ``"linear"`` calculates the number of nodes in each hidden layer assuming a line between ``start_num_nodes`` and ``end_num_nodes``. ``"constant"`` assumes each hidden layer has the same number of nodes as the input layer. User defined callables can also be pased to ``mid_num_node_strategy``. For more information refer to https://keras.io/api/.

.. _model_temp:

--------------------------
Model Dictionary Templates
--------------------------

**Linear Regression**

.. code-block:: python

   "linear": {
        "fit_intercept" = True,
        "copy_X" = True,
        "n_jobs" = None,
        "positive" = False,
   }

**Lasso Regression**

.. code-block:: python

   "lasso": {
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

**Support Vector Regression**

.. code-block:: python

   "svr": {
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

**Decision Tree Regression**

.. code-block:: python

   "dtree": {
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

**Random Forest Regression**

.. code-block:: python

   "rforest": {
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

**K-Nearest Neighbors Regression**

.. code-block:: python

   "knn": {
        "n_neighbors" = 5,
        "weights" = "uniform",
        "algorithm" = "auto",
        "leaf_size" = 30,
        "p" = 2,
        "metric" = "minkowski",
        "metric_params" = None,
        "n_jobs" = None,
   }

**Sequential Dense Neural Networks**

.. code-block:: python

   "nn": {
        # Sequential
        "num_layers" = None,
        "name" = None,
        "dropout" = False,
        "rate" = 0.5,
        "loss" = None,
        "metrics" = None,
        "loss_weights" = None,
        "weighted_metrics" = None,
        "run_eagerly" = False,
        "steps_per_execution" = None,
        "jit_compile" = True,  # Used for both compile and adam
        "batch_size" = None,
        "validation_batch_size" = None,
        "shuffle" = True,
        "callbacks" = None,
        "validation_split" = 0.0,
        "epochs" = 1,
        "warm_start" = False,

        # Starting Layer
        "start_num_nodes" = None,
        "start_activation" = None,
        "start_use_bias" = True,
        "start_kernel_initializer" = "glorot_uniform",
        "start_bias_initializer" = "zeros",
        "start_kernel_regularizer" = None,
        "start_bias_regularizer" = None,
        "start_activity_regularizer" = None,
        "start_kernel_constraint" = None,
        "start_bias_constraint" = None,
        "input_dim" = None,

        # Middle Layers
        "mid_num_node_strategy" = None,
        "mid_activation" = None,
        "mid_use_bias" = True,
        "mid_kernel_initializer" = "glorot_uniform",
        "mid_bias_initializer" = "zeros",
        "mid_kernel_regularizer" = None,
        "mid_bias_regularizer" = None,
        "mid_activity_regularizer" = None,
        "mid_kernel_constraint" = None,
        "mid_bias_constraint" = None,

        # Ending Layer
        "end_num_nodes" = None,
        "end_activation" = None,
        "end_use_bias" = True,
        "end_kernel_initializer" = "glorot_uniform",
        "end_bias_initializer" = "zeros",
        "end_kernel_regularizer" = None,
        "end_bias_regularizer" = None,
        "end_activity_regularizer" = None,
        "end_kernel_constraint" = None,
        "end_bias_constraint" = None,

        # Optimizer
        "optimizer" = "adam",

        # Adam
        "learning_rate" = 0.001,
        "beta_1" = 0.9,
        "beta_2" = 0.999,
        "epsilon" = 1e-7,
        "amsgrad" = False,
        "clipnorm" = None,
        "clipvalue" = None,
        "global_clipnorm" = None,
   }
