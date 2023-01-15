import pyMAISE.settings as settings
from pyMAISE.methods import *

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import BayesSearchCV


class Tuning:
    def __init__(self, data: pd.DataFrame, model_settings: dict):
        # Extract data
        self._xtrain = data[0]
        self._xtest = data[1]
        self._ytrain = data[2]
        self._ytest = data[3]

        # Extract target models from dictionary
        models_str = model_settings["models"]

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
            elif model == "dtree":
                self._models[model] = DecisionTreeRegression(parameters=parameters)
            elif model == "rforest":
                self._models[model] = RandomForestRegression(parameters=parameters)
            else:
                raise Exception(
                    "The model requested ("
                    + model
                    + ") is either misspelled or not supported"
                )

    # ===========================================================
    # Methods
    def grid_search(
        self,
        param_grids: dict,
        scoring=None,
        models: list = None,
        n_jobs: int = None,
        refit=True,
        cv: int = None,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score: bool = False,
    ):
        if models == None:
            models = list(self._models.keys())

        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning with grid search")

        data = {}

        for model in models:
            if model in param_grids:
                if settings.values.verbosity > 0:
                    print("-- " + model)

                # Run grid search
                search = GridSearchCV(
                    estimator=self._models[model].regressor(),
                    param_grid=param_grids[model],
                    scoring=scoring,
                    n_jobs=n_jobs,
                    refit=refit,
                    cv=cv,
                    verbose=settings.values.verbosity,
                    pre_dispatch=pre_dispatch,
                    error_score=error_score,
                    return_train_score=return_train_score,
                )
                resulting_models = search.fit(self._xtrain, self._ytrain)

                # Place parameter configurations in DataFrame and sort based on rank,
                # save the top num_configs_saved to the data dictionary
                # Exclude timing data from DataFrame
                data[model] = (
                    pd.DataFrame(resulting_models.cv_results_)
                    .sort_values("rank_test_score")
                    .iloc[: settings.values.num_configs_saved, :]
                )

                if settings.values.verbosity > 1:
                    print(data[model])

            else:
                # If model settings are not given by user throw an exception
                raise Exception(
                    model + " tuning settings were not provided for grid search"
                )

        return data

    def random_search(
        self,
        param_distributions: dict,
        scoring=None,
        models: list = None,
        n_iter: int = 10,
        n_jobs: int = None,
        refit=True,
        cv: int = None,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score: bool = False,
    ):
        if models == None:
            models = list(self._models.keys())

        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning with random search")

        data = {}

        for model in models:
            if model in param_distributions:
                if settings.values.verbosity > 0:
                    print("-- " + model)

                # Run random search
                search = RandomizedSearchCV(
                    estimator=self._models[model].regressor(),
                    param_distributions=param_distributions[model],
                    scoring=scoring,
                    n_iter=n_iter,
                    n_jobs=n_jobs,
                    refit=refit,
                    cv=cv,
                    verbose=settings.values.verbosity,
                    random_state=settings.values.random_state,
                    pre_dispatch=pre_dispatch,
                    error_score=error_score,
                    return_train_score=return_train_score,
                )
                resulting_models = search.fit(self._xtrain, self._ytrain)

                # Place parameter configurations in DataFrame and sort based on rank,
                # save the top num_configs_saved to the data dictionary
                # Exclude timing data from DataFrame
                data[model] = (
                    pd.DataFrame(resulting_models.cv_results_)
                    .sort_values("rank_test_score")
                    .iloc[: settings.values.num_configs_saved, :]
                )

                if settings.values.verbosity > 1:
                    print(data[model])

            else:
                # If model settings are not given by user throw an exception
                raise Exception(
                    model + " tuning settings were not provided for grid search"
                )

        return data

    def bayesian_search(
        self,
        search_spaces: dict,
        scoring=None,
        models: list = None,
        n_iter: int = 50,
        optimizer_kwargs: dict = None,
        fit_params: dict = None,
        n_jobs: int = None,
        n_points: int = 1,
        refit=True,
        cv: int = None,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score: bool = False,
    ):
        if models == None:
            models = list(self._models.keys())

        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning with bayesian search")

        data = {}

        for model in models:
            if model in search_spaces:
                if settings.values.verbosity > 0:
                    print("-- " + model)

                # Convert list of values to search space dimensions
                for key, value in search_spaces[model].items():
                    if isinstance(value[0], int):
                        search_spaces[model][key] = Integer(
                            low=np.min(value), high=np.max(value), name=key
                        )
                    elif isinstance(value[0], float):
                        search_spaces[model][key] = Real(
                            low=np.min(value), high=np.max(value), name=key
                        )
                    elif isinstance(value[0], str):
                        search_spaces[model][key] = Categorical(
                            categories=value, name=key
                        )

                # Run Bayesian search
                search = BayesSearchCV(
                    estimator=self._models[model].regressor(),
                    search_spaces=search_spaces[model],
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
                    error_score=error_score,
                    return_train_score=return_train_score,
                )
                resulting_models = search.fit(self._xtrain, self._ytrain)

                # Place parameter configurations in DataFrame and sort based on rank,
                # save the top num_configs_saved to the data dictionary
                # Exclude timing data from DataFrame
                data[model] = (
                    pd.DataFrame(resulting_models.cv_results_)
                    .sort_values("rank_test_score")
                    .iloc[: settings.values.num_configs_saved, :]
                )

                if settings.values.verbosity > 1:
                    print(data[model])

            else:
                # If model settings are not given by user throw an exception
                raise Exception(
                    model + " tuning settings were not provided for grid search"
                )

        return data
