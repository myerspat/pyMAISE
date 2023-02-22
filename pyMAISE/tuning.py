import pyMAISE.settings as settings
from pyMAISE.methods import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt.space import Integer, Real, Categorical
from skopt import BayesSearchCV


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
            elif model == "svr":
                if self._ytrain.shape[1] > 1:
                    raise Exception("SVR does not support multi-output data sets")
                self._models[model] = SVRegression(parameters=parameters)
            elif model == "dtree":
                self._models[model] = DecisionTreeRegression(parameters=parameters)
            elif model == "rforest":
                self._models[model] = RandomForestRegression(parameters=parameters)
            elif model == "knn":
                self._models[model] = KNeighborsRegression(parameters=parameters)
            elif model == "nn":
                self._models[model] = NeuralNetsRegression(parameters=parameters)
            else:
                raise Exception(
                    "The model requested ("
                    + model
                    + ") is either misspelled or not supported"
                )

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
        param_grids: dict,
        scoring=None,
        models: list = None,
        n_jobs: int = None,
        refit=True,
        cv: int = None,
        pre_dispatch="2*n_jobs",
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
        param_distributions: dict,
        scoring=None,
        models: list = None,
        n_iter: int = 10,
        n_jobs: int = None,
        refit=True,
        cv: int = None,
        pre_dispatch="2*n_jobs",
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
                )
                resulting_models = search.fit(self._xtrain, self._ytrain)

                # Add tuning results for convergence plot
                cv_results = pd.DataFrame(resulting_models.cv_results_)
                self._tuning[model] = cv_results[["mean_test_score", "std_test_score"]]

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

    def convergence_plot(self, ax=None, models=None):
        # If no models are provided fit all
        if models == None:
            models = self._tuning.keys()
        elif isinstance(models, str):
            models = [models]

        if ax == None:
            ax = plt.gca()

        for model in models:
            ax.plot(
                np.linspace(
                    1, self._tuning[model].shape[0], self._tuning[model].shape[0]
                ),
                self._tuning[model]["mean_test_score"],
                linestyle="-",
                marker="o",
                label=model,
            )

        # Show legend if length of models is more than one
        if len(models) > 1:
            ax.legend()

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean Test Score")

        return ax
