import pyMAISE.settings as settings
from pyMAISE.methods import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt.space import Integer, Real, Categorical
from skopt import BayesSearchCV

# New packages
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from kerastuner.tuners import RandomSearch, BayesianOptimization

class Tuning:
    def __init__(self, data: pd.DataFrame, model_settings: dict):
        # Extract training data
        self._xtrain = data[0]
        self._ytrain = data[2]

        # Extract target models from dictionary
        models_str = model_settings["models"]

        # Tuning loss for convergence plots
        self._tuning = {}

        # Initialize models of interest
        self._models = {}
        for model in models_str:
            # Pull parameters provided
            parameters = None
            if model in model_settings:
                parameters = model_settings[model]

            # Add model object to dictionary
            if model == "linear":
                self._models[model] = LinearRegression(parameters=parameters)
            elif model == "lasso":
                self._models[model] = LassoRegression(parameters=parameters)
            elif model == "logistic":
                self._models[model] = Logistic_Regression(parameters=parameters)
            elif model == "svr":
                if self._ytrain.shape[1] > 1:
                    raise Exception("SVR does not support multi-output data sets")
                self._models[model] = SVRegression(parameters=parameters)
            elif model == "dtree":
                self._models[model] = DecisionTree(parameters=parameters)
            elif model == "rforest":
                self._models[model] = RandomForest(parameters=parameters)
            elif model == "knn":
                self._models[model] = KNeighbors(parameters=parameters)
            elif model == "nn":
                self._models[model] = NeuralNetsRegression(parameters=parameters)
            elif model == "nn_new":
                self._models[model] = nnHyperModel(structural_hyperparameters=model_settings["nn_new"][0], 
                                                   optimizer_hyperparameters=model_settings["nn_new"][1])

                print("model = ", model, self._models[model])

                print("Checking if model is initialized")
                print("optimizer = ", self._models["nn_new"].optimizer)
                print("loss = ", self._models["nn_new"].loss)
            else:
                raise Exception(
                    "The model requested ("
                    + model
                    + ") is either misspelled or not supported"
                )

        # For single input or outputs the data must be converted to a 1D array to
        # get rid of sklearn warnings
        if self._xtrain.shape[1] == 1:
            self._xtrain = self._xtrain.iloc[:, 0]
        if self._ytrain.shape[1] == 1:
            self._ytrain = self._ytrain.iloc[:, 0]

    # ===========================================================
    # Methods
    def manual_search(self, models: list = None, model_settings=None):
        # Get model types if not provided
        if models == None:
            models = list(self._models.keys())

        data = {}
        for model in models:
            if settings.values.verbosity > 0:
                print("-- " + model)

            # Run model
            estimator = self._models[model].regressor()
            if model_settings != None and model in model_settings:
                estimator.set_params(model_settings)

            resulting_model = estimator.fit(self._xtrain, self._ytrain)

            # Place model in
            data[model] = (
                pd.DataFrame({"params": [resulting_model.get_params()]}),
                resulting_model,
            )

        return data

    def grid_search(
        self,
        param_spaces: dict,
        scoring=None,
        models: list = None,
        n_jobs: int = None,
        refit=True,
        cv=None,
        pre_dispatch="2*n_jobs",
    ):
        if models == None:
            models = list(self._models.keys())

        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning with grid search")

        data = {}
        for model in models:
            if model in param_spaces:
                if settings.values.verbosity > 0:
                    print("-- " + model)

                # Run grid search
                search = GridSearchCV(
                    estimator=self._models[model].regressor(),
                    param_grid=param_spaces[model],
                    scoring=scoring,
                    n_jobs=n_jobs,
                    refit=refit,
                    cv=cv,
                    verbose=settings.values.verbosity,
                    pre_dispatch=pre_dispatch,
                )
                resulting_models = search.fit(self._xtrain, self._ytrain)

                # Add tuning results for convergence plot
                cv_results = pd.DataFrame(resulting_models.cv_results_)
                self._tuning[model] = cv_results["mean_test_score"]

                # Place parameter configurations in DataFrame and sort based on rank,
                # save the top num_configs_saved to the data dictionary
                # Exclude timing data from DataFrame
                top_configs = cv_results.sort_values("rank_test_score").iloc[
                    : settings.values.num_configs_saved, :
                ]

                if settings.values.verbosity > 1:
                    print(top_configs)

                data[model] = (top_configs, resulting_models.best_estimator_)

            else:
                print(
                    "Hyper-parameter tuning search space was not provided for "
                    + model
                    + ", doing manual fit"
                )
                data = {**data, **self.manual_search(models=[model])}

        return data

    def random_search(
        self,
        param_spaces: dict,
        scoring=None,
        models: list = None,
        n_iter: int = 10,
        n_jobs: int = None,
        refit=True,
        cv=None,
        pre_dispatch="2*n_jobs",
    ):
        if models == None:
            models = list(self._models.keys())

        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning with random search")

        data = {}

        for model in models:
            if model in param_spaces:
                if settings.values.verbosity > 0:
                    print("-- " + model)

                # Run random search
                search = RandomizedSearchCV(
                    estimator=self._models[model].regressor(),
                    param_distributions=param_spaces[model],
                    scoring=scoring,
                    n_iter=n_iter,
                    n_jobs=n_jobs,
                    refit=refit,
                    cv=cv,
                    verbose=settings.values.verbosity,
                    random_state=settings.values.random_state,
                    pre_dispatch=pre_dispatch,
                )
                resulting_models = search.fit(self._xtrain, self._ytrain)

                # Add tuning results for convergence plot
                cv_results = pd.DataFrame(resulting_models.cv_results_)
                self._tuning[model] = cv_results["mean_test_score"]

                # Place parameter configurations in DataFrame and sort based on rank,
                # save the top num_configs_saved to the data dictionary
                # Exclude timing data from DataFrame
                top_configs = cv_results.sort_values("rank_test_score").iloc[
                    : settings.values.num_configs_saved, :
                ]

                if settings.values.verbosity > 1:
                    print(top_configs)

                data[model] = (top_configs, resulting_models.best_estimator_)

            else:
                print(
                    "Hyper-parameter tuning search space was not provided for "
                    + model
                    + ", doing manual fit"
                )
                data = {**data, **self.manual_search(models=[model])}

        return data

    def bayesian_search(
        self,
        param_spaces: dict,
        scoring=None,
        models: list = None,
        n_iter: int = 50,
        optimizer_kwargs: dict = None,
        fit_params: dict = None,
        n_jobs: int = None,
        n_points: int = 1,
        refit=True,
        cv=None,
        pre_dispatch="2*n_jobs",
    ):
        if models == None:
            models = list(self._models.keys())

        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning with bayesian search")

        data = {}
        for model in models:
            if model in param_spaces:
                if settings.values.verbosity > 0:
                    print("-- " + model)

                # Convert list of values to search space dimensions
                for key, value in param_spaces[model].items():
                    if isinstance(value[0], int):
                        param_spaces[model][key] = Integer(
                            low=np.min(value), high=np.max(value), name=key
                        )
                    elif isinstance(value[0], float):
                        param_spaces[model][key] = Real(
                            low=np.min(value), high=np.max(value), name=key
                        )
                    elif isinstance(value[0], str):
                        param_spaces[model][key] = Categorical(
                            categories=value, name=key
                        )

                # Run Bayesian search
                search = BayesSearchCV(
                    estimator=self._models[model].regressor(),
                    search_spaces=param_spaces[model],
                    n_iter=n_iter,
                    optimizer_kwargs=optimizer_kwargs,
                    scoring=scoring,
                    fit_params=fit_params,
                    n_jobs=n_jobs,
                    n_points=n_points,
                    pre_dispatch=pre_dispatch,
                    cv=cv,
                    refit=refit,
                    verbose=settings.values.verbosity,
                    random_state=settings.values.random_state,
                )
                resulting_models = search.fit(self._xtrain, self._ytrain)

                # Add tuning results for convergence plot
                cv_results = pd.DataFrame(resulting_models.cv_results_)
                self._tuning[model] = cv_results["mean_test_score"]

                # Place parameter configurations in DataFrame and sort based on rank,
                # save the top num_configs_saved to the data dictionary
                # Exclude timing data from DataFrame
                top_configs = cv_results.sort_values("rank_test_score").iloc[
                    : settings.values.num_configs_saved, :
                ]

                if settings.values.verbosity > 1:
                    print(top_configs)

                data[model] = (top_configs, resulting_models.best_estimator_)

            else:
                print(
                    "Hyper-parameter tuning search space was not provided for "
                    + model
                    + ", doing manual fit"
                )
                data = {**data, **self.manual_search(models=[model])}

        return data

    def convergence_plot(self, ax=None, model_types=None):
        # If no models are provided fit all
        if model_types == None:
            model_types = self._tuning.keys()
        elif isinstance(model_types, str):
            model_types = [model_types]

        if ax == None:
            ax = plt.gca()

        for model in model_types:
            ax.plot(
                np.linspace(
                    1, self._tuning[model].shape[0], self._tuning[model].shape[0]
                ),
                self._tuning[model],
                linestyle="-",
                marker="o",
                label=model,
            )

        # Show legend if length of models is more than one
        if len(model_types) > 1:
            ax.legend()

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean Test Score")

        return ax



    def bayesian_search_hypermodel(self, 
                                   objective,
                                   max_trials, 
                                   directory, 
                                   project_name,
                                   num_folds,
                                   shuffle,
                                   # callback_loss
                                   ):

            print("Checking if model is initialized")
            print("optimizer = ", self._models["nn_new"].optimizer)
            print("loss = ", self._models["nn_new"].loss)

                
            tuner = BayesianOptimization(self._models["nn_new"], 
                                        objective=objective,
                                        max_trials=max_trials,
                                        directory=directory,
                                        project_name=project_name,
                                        )

                
            # ------------------------- For classiification Problem ----------------------------

            # Define cross-validation using Stratified K-Fold
            if settings.values.classification:
                kf = StratifiedKFold(n_splits=num_folds,
                                shuffle=shuffle, 
                                random_state=settings.values.random_state
                                )

                # Perform hyperparameter tuning with cross validation
                for train_index, val_index in kf.split(self._xtrain, self._ytrain):
                    x_train_fold, x_test_fold = x[train_index], x[val_index]
                    y_train_fold, y_test_fold = y[train_index], y[val_index]

            if settings.values.regression:
                kf = KFold(n_splits=num_folds, 
                           shuffle=shuffle, 
                           random_state=settings.values.random_state
                           )
                print("Tuning")
                # Perform hyperparameter tuning with cross validation
                for train_index, test_index in kf.split(self._xtrain):
                    xtrain_fold, xtest_fold = self._xtrain.iloc[train_index], self._xtrain.iloc[test_index]
                    ytrain_fold, ytest_fold = self._ytrain.iloc[train_index], self._ytrain.iloc[test_index]

                    # hp tuning on kFolds
                    tuner.search(x=xtrain_fold,
                                 y=ytrain_fold, 
                                 epochs=self._models["nn_new"].epochs,
                                 batch_size=self._models["nn_new"].batch_size,
                                 validation_data=(xtest_fold, ytest_fold),
                                 # callbacks=[tf.keras.callbacks.EarlyStopping(monitor=callback_loss, patience=3, restore_best_weights=True)]
                                )

                # Get Hyperparameters
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                print("-- Best Hyperparameters")
                print(best_hps)

                #Build model with the optimal hyperparameters and train it on the data for 50 epochs
                model=tuner.hypermodel.build(best_hps)
                history = model.fit(self._xtrain, self._ytrain, epochs=50, validation_split=0.2)

                val_acc_per_epoch = history.history['mean_absolute_error']
                best_epochs = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
                print('Best Epochs: %d' % (best_epochs,))

                # Re-instantiate the hypermoel and train it with the optimal number of epochs from above
                hypermodel = tuner.hypermodel.build(best_hps)
                hypermodel.fit(self._xtrain, self._ytrain, epochs=best_epochs, validation=0.2)

                # next step in preprocessing eval_results = hypermodel.evaluate()





