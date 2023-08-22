==========
User Guide
==========

ML generation in pyMAISE is done in five steps:

1. :ref:`pyMAISE initialization <pymaise_init>`, 
2. :ref:`pre-processing <preprocessing>`,
3. :ref:`Model initialization <model_init>`,
4. :ref:`hyper-parameter tuning <hyperparameter_tun>`, 
5. :ref:`post-processing <postprocessing>`.

Each step and their respective classes are discussed below.

.. _pymaise_init:

----------------------
pyMAISE Initialization
----------------------

To access pyMAISE in your Python code you must import the library:

.. code-block:: python

   import pyMAISE as mai

We shorten ``pyMAISE`` to ``mai`` for readability and convenience. 

Defining Global Settings
^^^^^^^^^^^^^^^^^^^^^^^^

Every pyMAISE job requires initialization through the ``Settings`` class. These global settings and their defaults include:

- ``verbosity = 0``: how much pyMAISE outputs to the terminal,
- ``random_state = None``: the global random state used by ML pseudo random algorithms during training,
- ``test_size = 0.3``: the fraction of data used for model testing,
- ``num_configs_saved = 5``: the number of hyper-parameter configurations saved, only the top ``num_configs_saved`` are returned,
- ``regression = False``: Boolean for regression models,
- ``classification = False``: Boolean for classification models.

Either ``regression`` and ``classification`` must be set to ``True`` depending on your problem. To initialize pyMAISE with default global variables you run:

.. code-block:: python

   global_settings = mai.settings.init()

However, these defaults can be changed by passing any changed parameters you want in a dictionary. For example, if we wanted a ``verbosity`` of 1 and a ``random_state`` of 42 we define a dictionary,

.. code-block:: python

   settings_changes = {
       "verbosity": 1,
       "random_state": 42,
   }

We can pass that to the initialization function:

.. code-block:: python

   global_settings = mai.settings.init(settings_changes)

.. _preprocessing:

--------------
Pre-processing
--------------

In this section we'll load and scale our data. Additionally, we can generate correlation matricies and retrieve properties such as ``xscaler`` and ``yscaler`` (discussed in the `Data Scaling <data_scaling>`_ section).

Loading Data
^^^^^^^^^^^^

With pyMAISE initialized with ``settings.init`` we can load our data into the ``PreProcessor`` class. For personal data in one data file, initialize the ``PreProcessor`` with

.. code-block:: python

   preprocessor = mai.PreProcessor(
      "path/to/data.csv",
      slice(0, x),
      slice(x, y)
   )

where ``x`` defines the beginning of the outputs and ``y`` defines the end +1 position of the outputs in the data file. For data with inputs and outputs in seperate files use

.. code-block:: python

   preprocessor = mai.PreProcessor(
       [
           "path/to/inputs.csv", 
           "path/to/outputs.csv"
       ]
   )

If you wish to load the benchmark specific pre-processors run the corresponding load function:

- MITR: ``mai.load_MITR()``
- Reactor physics: ``mai.load_xs()``
- Fuel performance: ``mai.load_fp()``
- Heat conduction: ``mai.load_heat()``
- BWR: ``mai.load_BWR()``

.. _data_scaling:

Data Scaling
^^^^^^^^^^^^

The performance of many ML models depends on the scaling of the data. pyMAISE offers three scaling options: none, min-max, and standard scaling. Their respective functions are

.. code-block:: python

   # No scaling
   data = preprocessor.data_split()

   # Min-max scaling
   data = preprocessor.min_max_scale()

   # Standard scaling
   data = preprocessor.std_scale()

All three methods return a tuple of training and testing data, ``xtrain, xtest, ytrain, ytest``, and both ``min_max_scale`` and ``std_scale`` can scale input and/or output data depending on how ``scale_x`` and ``scale_y`` are defined. Both ``min_max_scale`` and ``std_scale`` use scaling objects from ``sklearn.preprocessing`` that can be retrieved with the ``xscaler`` and ``yscaler`` properties. To min-max scale only the inputs run

.. code-block:: python

   data = preprocessor.min_max_scale(scale_y=False)

Generating a Correlation Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To better understand the correlation between the inputs and the outputs we can plot a correlation matrix using ``preprocessor.correlation_matrix()``. You can toggle the colorbar or annotations using the ``colorbar`` and ``anotations`` parameters. Additionally, you can pass your own matplotlib figure or axis. 

Pre-processing Properties
^^^^^^^^^^^^^^^^^^^^^^^^^

Using the ``preprocessor`` object we can retrieve several useful objects:

- ``preprocessor.data``: is the raw data, 
- ``preprocessor.inputs``: is only the input raw data,
- ``preprocessor.outputs``: is only the output raw data,
- ``preprocessor.xscaler``: scaler object used to scale the inputs,
- ``preprocessor.yscaler``: scaler object used to scale the outputs. 

.. _model_init:

--------------------
Model Initialization
--------------------

pyMAISE supports both classical ML methods and dense sequantial neural networks. Here is a list of the support ML methods and their names in pyMAISE:

- Linear regression: ``linear``,
- Lasso regression: ``lasso``,
- Support vector regression: ``svr``,
- Decision tree regression: ``dtree``,
- Random forest regression: ``rforest``,
- K-nearest neighbors regression: ``knn``,
- Sequential dense neural networks: ``nn``.

.. caution:: Support vector regression is **only** supported for 1-dimensional outputs.

The classical models use `scikit-learn <https://scikit-learn.org/stable/index.html>`_ model wrappers and the neural networks are based on `Keras <https://keras.io>`_. For more information on the models themselves refer to the :ref:`Machine Learning Models <models>` section.

To initialize each of your desired models, specify their pyMAISE specific names in the ``models`` list in a dicitonary. Then for each of the models define a dictionary of hyper-parameters you'd like to change. Only parameters with values different from their scikit-learn or Keras defaults must be specified. These parameters will define the values that remain constant throughout tuning or the initial guess for random and Bayesian search. Refer to the :ref:`Model Dictionary Templates <model_temp>` section for a dictionary of parameters for each model. As an example, lets initialize ``linear``, ``lasso``, and ``rforest`` with 200 estimators:

.. code-block:: python

   model_settings = {
      "models": ["linear", "lasso", "rforest"],
      "rforest": {
          "n_estimators": 200,
      }
   }
   
We can then initialize the ``Tuning`` class in pyMAISE with our ``data`` tuple and the ``model_settings``:

.. code-block:: python

   tuning = mai.Tuning(data=data, model_settings=model_settings)

.. _hyperparameter_tun:

----------------------
Hyper-parameter Tuning
----------------------

With all the models of interest initialized in ``tuning``, we can begin hyper-parameter tuning. pyMAISE supports three types of search methods: grid, random, or Bayesian search. Each of the search functions require the definition of parameter search spaces in dictionaries for each model type. The function then pulls the parameter space for each model and passes it to the search function.

Grid Search
^^^^^^^^^^^

Grid search evaluates all possible combinations of a given parameter space. To define the parameter search space for a grid search we define a dictionary of Numpy arrays or lists for each parameter of interest. For the models defined in the above section we can define

.. code-block:: python

   grid_search_spaces = {
      "lasso": {"alpha": np.linspace(0.0001, 5, 20)},
      "rforest": {
          "max_features": [None, "sqrt", "log2", 2, 4, 6],
      },
   }

This dictionary is then passed to the grid search tuning function:

.. code-block:: python

   grid_search_configs = tuning.grid_search(
      param_spaces=grid_search_spaces,
      models=["linear"] + list(grid_search_spaces.keys())
   )

Which will run the grid search. Notice that a ``linear`` search space was not defined; in this case a manual search is done on linear for the given initial guess (on the scikit-learn linear regression default in this example). Therefore, only one ``linear`` model is generated. 

As ``grid_search`` uses ``GridSearchCV`` from scikit-learn we can pass other supported parameters to the function such as ``cv``. Additionally, we can define a list of models from ``grid_search_spaces`` we want to run as opposed to all that were defined in the dictionary. 

Random Search
^^^^^^^^^^^^^

Random search evaluates hyper-parameter configurations from randomly sampled distributions. As this method in pyMAISE uses ``RandomizedSearchCV`` from scikit-learn, we can define the parameter spaces as dictionaries of ``scipy.stats`` distributions or lists. While the number of evaluated parameter configurations grows quickly in grid search, random search requires you to define the number of iterations to sample and train models. Here is an example with ``lasso`` and ``rforest``:

.. code-block:: python

   random_search_spaces = {
      "lasso": {
          # Uniform distribution for alpha between 0.0001 - 0.01
          "alpha": scipy.stats.uniform(loc=0.0001, scale=0.0099),
      },
      "rforest": {
          "max_features": [None, "sqrt", "log2", 2, 4, 6],
      },
   }

We can then define the models, number of iterations, cross-validation, and other parameters in ``RandomizedSearchCV`` and pass those to ``random_search``:

.. code-block:: python

   random_search_configs = tuning.random_search(
      param_spaces=random_search_spaces,
      n_iter=200,
      cv=5,
   )

Bayesian Search
^^^^^^^^^^^^^^^

Bayesian search uses prior parameter configurations results to inform the next configuration of hyper-parameters to converge on the optimal hyper-parameter set. This process uses a Gaussian process surrogate function to predict the next parameter configuration with better statistics. Under the hood pyMAISE's ``bayesian_search`` function uses ``BayesSearchCV`` from scikit-optimize. Just as in grid search, we can define the parameter space using lists of minimum and maximum values or the list of categorical strings:

.. code-block:: python

   bayesian_search_spaces = {
      "lasso": {
          "alpha": [0.0001, 0.01],
      },
      "rforest": {
          "max_features": [1, 10],
      },
   }

We can then pass this to ``bayesian_search``:

.. code-block:: python

   bayesian_search_configs = tuning.bayesian_search(
      param_spaces=bayesian_search_spaces,
      n_iter=50,
   )

where we pass the parameter spaces, the number of iterations, and other parameters specific to ``BayesSearchCV``. Bayesian search will then sample between the limits defined in ``bayesian_search_spaces``. 

Convergence Plots
^^^^^^^^^^^^^^^^^

For each of the search methods you can plot a convergence plot using the ``convergence_plot`` function; however, this is more appealing for Bayesian search as it shows how the kernel converges to the optimal hyper-parameter configuration with each step. To plot a specific model such as ``nn`` run

.. code-block:: python

   tuning.convergence_plot(model_types="nn")

.. _postprocessing:

---------------
Post-processing
---------------

With our top ``num_configs_saved`` models we can pass these to the ``PostProcessor`` class for model comparison and testing. To do so we provide the ``data`` and configuration(s):

.. code-block:: python
  
   postprocessor = mai.PostProcessor(
      data=data,
      models_list=[random_search_configs, bayesian_search_configs],
   )
   
Additionally, we can pass a dictionary similar to ``model_settings`` of updated model settings to the ``new_model_settings`` parameter. With our ``PostProcessor`` initialized we can begin evaluating our models.

Performance Metrics
^^^^^^^^^^^^^^^^^^^

The performance metrics we'll use to assess and compare each of the models are

- r-squared: :math:`R^2 = 1 - \frac{\sum_{i = 1}^{n}(y_i - \hat{y_i})^2}{\sum_{i = 1}^{n}(y_i - \bar{y_i})^2}`,
- mean absolute error: :math:`MAE = \frac{1}{n}\sum_{i = 1}^{n}|y_i - \hat{y_i}|`,
- mean squared error: :math:`MSE = \frac{1}{n}\sum_{i = 1}^n(y_i - \hat{y_i})^2`,
- root mean squared error: :math:`RMSE = \sqrt{\frac{1}{n}\sum_{i = 1}^n(y_i - \hat{y_i})^2}`,

where :math:`y` is the actual outcome, :math:`\hat{y}` is the model predicted outcome, :math:`\bar` is the average outcome, and :math:`n` is the number of observations. These metrics are computed for both the training and testing data sets and are computed through the ``metrics`` function in the ``PostProcessor``. You can choose how the DataFrame is sorted, whether the features are averaged or only the metrics for one feature is computed, and which models to show. With this information you can compare the performance of each of your models on your data set.

Diagonal Validation Plots
^^^^^^^^^^^^^^^^^^^^^^^^^

After computing the performance metrics of each model, you can create diagonal validation plots that show the models predictions versus the actual result. This is done through the ``diagonal_validation_plot`` function in the ``PostProcessor`` and you can choose the model and the label to plot. Additionally, you can pass the ``yscaler`` from the ``PreProcessor`` to get representative numbers of the output. 

Validation Plots
^^^^^^^^^^^^^^^^

Similar to the diagonal validation plots, you can also plot validation plots that show the absolute relative error of the model predictions to the correct result. This is done through the ``validation_plot`` function in the ``PostProcessor``. This function has the same capabilities as ``diagonal_validation_plot``.

Neural Network Learning Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final plotting capability of the ``PostProcessor`` is the neural network learning curves. These curves show the loss per epoch which informs the training of the neural network. With these curves you can determine if the neural network is overfit to the training data. Neural network learning curves are created through the ``nn_learning_plot`` function and you can choose with neural network model to plot.

Other Post-processing Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, the ``PreProcessor`` is equipped with several additional methods to model analysis. These include

- ``get_params``: get the parameter configurations from a specific model,
- ``get_model``: get the model wrapper,
- ``get_predictions``: get the training and testing predictions from a specific model.
