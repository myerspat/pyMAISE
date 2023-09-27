import copy
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras_tuner.tuners import BayesianOptimization, RandomSearch

# New packages
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

import pyMAISE.settings as settings
from pyMAISE.methods import *


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
            elif bool(re.search("nn", model)):
                if settings.values.new_nn_architecture == True:
                    self._models[model] = nnHyperModel(parameters=parameters)
                else:
                    self._models[model] = NeuralNetsRegression(parameters=parameters)

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

                if bool(re.search("nn", model)) and settings.values.new_nn_architecture:
                    continue

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

    def nn_bayesian_search(
        self,
        models=None,
        objective=None,
        max_trials=10,
        num_initial_points=None,
        alpha=0.0001,
        beta=2.6,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        overwrite=True,
        directory="./",
        project_name="best_hp",
        cv=5,
        shuffle=True,
    ):
        if models == None:
            models = []
            for model in self._models.keys():
                if re.search("nn", model):
                    models.append(model)

        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning neural networks with bayesian search")

        if isinstance(cv, int):
            if settings.values.regression:
                cv = KFold(
                    n_splits=cv,
                    shuffle=shuffle,
                    random_state=settings.values.random_state,
                )
            else:
                cv = StratifiedKFold(
                    n_splits=cv,
                    shuffle=shuffle,
                    random_state=settings.values.random_state,
                )

        data = {}
        for model in models:
            # Initialize keras-tuner tuner
            tuner = BayesianOptimization(
                hypermodel=self._models[model],
                objective=objective,
                max_trials=max_trials,
                num_initial_points=num_initial_points,
                alpha=alpha,
                beta=beta,
                seed=settings.values.random_state,
                hyperparameters=hyperparameters,
                tune_new_entries=tune_new_entries,
                allow_new_entries=allow_new_entries,
                max_retries_per_trial=max_retries_per_trial,
                overwrite=overwrite,
                directory=directory,
                project_name=project_name,
            )

            if settings.values.verbosity > 0:
                print("-- " + model)
                print(tuner.search_space_summary())

            tuner.search(x=self._xtrain, y=self._ytrain)

            if settings.values.verbosity > 0:
                print("-- " + model)
                print(tuner.search_space_summary())
                print(tuner.results_summary())

            best_hps = tuner.get_best_hyperparameters(settings.values.num_configs_saved)

            top_configs = pd.DataFrame({"params": best_hps})

            data[model] = (top_configs, tuner.hypermodel)

        return data
