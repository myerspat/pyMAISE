import copy
import re

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras_tuner.oracles import (
    BayesianOptimizationOracle,
    GridSearchOracle,
    HyperbandOracle,
    RandomSearchOracle,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

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
from pyMAISE.utils import CVTuner


class Tuner:
    def __init__(self, data: list, model_settings: dict):
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
            elif model == "svm":
                if self._ytrain.shape[-1] > 1 and settings.values.regression:
                    raise Exception("SVM does not support multi-output data sets")
                else:
                    self._models[model] = SVM(parameters=parameters)
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
                raise Exception("The model requested (" + model + ") is not supported")

    # ===========================================================
    # Methods
    def manual_search(self, models=None, model_settings=None):
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

            # Save model hyperparameters and the model itself
            data[model] = (
                pd.DataFrame({"params": [resulting_model.get_params()]}),
                resulting_model,
            )

        return data

    def _run_search(self, spaces, search_method, search_kwargs, models=None):
        if models == None:
            models = list(self._models.keys())

        search_data = {}
        for model in models:
            if model in spaces:
                if settings.values.verbosity > 0:
                    print(f"-- {model}")

                # Run search method
                search = search_method(
                    self._models[model].regressor(), spaces[model], **search_kwargs
                )
                resulting_models = search.fit(
                    self._xtrain.values, self._ytrain.values
                )

                # Save tuning results
                cv_results = pd.DataFrame(resulting_models.cv_results_)
                self._tuning[model] = np.array(
                    [
                        cv_results["mean_test_score"],
                        cv_results["std_test_score"],
                    ]
                )

                # Place parameter configurations in DataFrame and sort based on rank,
                # save the top num_configs_saved to the data dictionary
                top_configs = pd.DataFrame(
                    cv_results.sort_values("rank_test_score")["params"]
                )

                search_data[model] = (
                    top_configs[: settings.values.num_configs_saved],
                    resulting_models.best_estimator_,
                )

            else:
                print(f"Search space was not provided for {model}, doing manual fit")
                search_data = {**search_data, **self.manual_search(models=[model])}

        return search_data

    def grid_search(
        self,
        param_spaces: dict,
        scoring=None,
        models=None,
        n_jobs=None,
        refit=True,
        cv=None,
        pre_dispatch="2*n_jobs",
    ):
        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning with grid search")

        return self._run_search(
            spaces=param_spaces,
            search_method=GridSearchCV,
            search_kwargs={
                "scoring": scoring,
                "n_jobs": n_jobs,
                "refit": refit,
                "cv": cv,
                "verbose": settings.values.verbosity,
                "pre_dispatch": pre_dispatch,
            },
            models=models,
        )

    def random_search(
        self,
        param_spaces: dict,
        scoring=None,
        models=None,
        n_iter=10,
        n_jobs: int = None,
        refit=True,
        cv=None,
        pre_dispatch="2*n_jobs",
    ):
        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning with random search")

        return self._run_search(
            spaces=param_spaces,
            search_method=RandomizedSearchCV,
            search_kwargs={
                "scoring": scoring,
                "n_iter": n_iter,
                "n_jobs": n_jobs,
                "refit": refit,
                "cv": cv,
                "verbose": settings.values.verbosity,
                "random_state": settings.values.random_state,
                "pre_dispatch": pre_dispatch,
            },
        )

    def bayesian_search(
        self,
        param_spaces: dict,
        scoring=None,
        models=None,
        n_iter=50,
        optimizer_kwargs=None,
        fit_params=None,
        n_jobs=None,
        n_points=1,
        refit=True,
        cv=None,
        pre_dispatch="2*n_jobs",
    ):
        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning with bayesian search")

        return self._run_search(
            spaces=param_spaces,
            search_method=BayesSearchCV,
            search_kwargs={
                "n_iter": n_iter,
                "optimizer_kwargs": optimizer_kwargs,
                "scoring": scoring,
                "fit_params": fit_params,
                "n_jobs": n_jobs,
                "n_points": n_points,
                "pre_dispatch": pre_dispatch,
                "cv": cv,
                "refit": refit,
                "verbose": settings.values.verbosity,
                "random_state": settings.values.random_state,
            },
        )

    def nn_grid_search(
        self,
        models=None,
        objective=None,
        max_trials=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        overwrite=True,
        directory="./",
        project_name="best_hp",
        cv=5,
        shuffle=True,
    ):
        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning neural networks with grid search")

        kt_objective = self._determine_kt_objective(objective)
        oracle = GridSearchOracle(
            objective=kt_objective[0],
            max_trials=max_trials,
            seed=settings.values.random_state,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        return self._nn_tuning(
            models=models,
            objective=objective,
            cv=cv,
            shuffle=shuffle,
            oracle=oracle,
            metrics=kt_objective[1],
            overwrite=overwrite,
            directory=directory,
            project_name=project_name,
        )

    def nn_random_search(
        self,
        models=None,
        objective=None,
        max_trials=10,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        overwrite=True,
        directory="./",
        project_name="best_hp",
        cv=5,
        shuffle=True,
    ):
        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning neural networks with random search")

        kt_objective = self._determine_kt_objective(objective)
        oracle = RandomSearchOracle(
            objective=kt_objective[0],
            max_trials=max_trials,
            seed=settings.values.random_state,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        return self._nn_tuning(
            models=models,
            objective=objective,
            cv=cv,
            shuffle=shuffle,
            oracle=oracle,
            metrics=kt_objective[1],
            overwrite=overwrite,
            directory=directory,
            project_name=project_name,
        )

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
        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning neural networks with bayesian search")

        kt_objective = self._determine_kt_objective(objective)
        oracle = BayesianOptimizationOracle(
            objective=kt_objective[0],
            max_trials=max_trials,
            num_initial_points=num_initial_points,
            alpha=alpha,
            beta=beta,
            seed=settings.values.random_state,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        return self._nn_tuning(
            models=models,
            objective=objective,
            cv=cv,
            shuffle=shuffle,
            oracle=oracle,
            metrics=kt_objective[1],
            overwrite=overwrite,
            directory=directory,
            project_name=project_name,
        )

    def nn_hyperband_search(
        self,
        models=None,
        objective=None,
        cv=5,
        shuffle=True,
        max_epochs=100,
        factor=3,
        hyperband_iterations=1,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        overwrite=True,
        directory="./",
        project_name="best_hp",
    ):
        if settings.values.verbosity > 0:
            print("Hyper-parameter tuning neural networks with hyperband search")

        kt_objective = self._determine_kt_objective(objective)
        oracle = HyperbandOracle(
            objective=kt_objective[0],
            max_epochs=max_epochs,
            factor=factor,
            hyperband_iterations=hyperband_iterations,
            seed=settings.values.random_state,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        return self._nn_tuning(
            models=models,
            objective=objective,
            cv=cv,
            shuffle=shuffle,
            oracle=oracle,
            metrics=kt_objective[1],
            overwrite=overwrite,
            directory=directory,
            project_name=project_name,
        )

    def _nn_tuning(
        self,
        models,
        objective,
        cv,
        shuffle,
        oracle,
        metrics,
        overwrite,
        directory,
        project_name,
    ):
        # Find all NN models if none are given by user
        if models == None:
            models = []
            for model in self._models.keys():
                if re.search("nn", model):
                    models.append(model)

        data = {}
        for model in models:
            # Initialize keras-tuner tuner
            tuner = CVTuner(
                objective=objective,
                cv=cv,
                shuffle=shuffle,
                hypermodel=self._models[model],
                oracle=oracle,
                metrics=metrics,
                overwrite=overwrite,
                directory=directory,
                project_name=project_name,
            )

            # Run search
            tuner.search(x=self._xtrain.to_numpy(), y=self._ytrain.to_numpy())

            if settings.values.verbosity > 0:
                print("-- " + model)
                print(tuner.search_space_summary())
                print(tuner.results_summary())

            # Get best hyperparameters
            best_hps = tuner.get_best_hyperparameters(settings.values.num_configs_saved)
            top_configs = pd.DataFrame({"params": best_hps})

            # Save test scores
            self._tuning[model] = np.array(
                [tuner.mean_test_score, tuner.std_test_score]
            )

            data[model] = (top_configs, tuner.hypermodel)

        return data

    # Determine objective from sklearn and make it compatible with keras_tuner
    def _determine_kt_objective(self, objective):
        if objective in ["r2_score", "accuracy_score"]:
            return (kt.Objective(objective, direction="max"), eval(objective))
        elif objective in [
            "f1_score",
            "mean_absolute_error",
            "mean_squared_error",
            "precision_score",
            "recall_score",
        ]:
            return (kt.Objective(objective, direction="min"), eval(objective))
        else:
            return (objective, None)

    def convergence_plot(self, ax=None, model_types=None):
        # If no models are provided fit all
        if model_types == None:
            model_types = self._tuning.keys()
        elif isinstance(model_types, str):
            model_types = [model_types]

        # Make axis if not given one
        if ax == None:
            ax = plt.gca()

        # For each model assert the performance metrics are the same size
        assert_shape = self._tuning[model_types[0]].shape

        for model in model_types:
            assert assert_shape == self._tuning[model].shape
            print(self._tuning[model][0,])
            print(self._tuning[model][1,])
            x = np.linspace(
                1, self._tuning[model].shape[0], self._tuning[model].shape[0]
            )
            ax.plot(
                x,
                self._tuning[model][0, :],
                linestyle="-",
                marker="o",
                label=model,
            )
            ax.fill_between(
                x, -2 * self._tuning[model][1, :], 2 * self._tuning[model][1, :]
            )

        # Show legend if length of models is more than one
        if len(model_types) > 1:
            ax.legend()

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean Test Score")

        return ax

    # Getters
    @property
    def cv_performance_data(self):
        return self._tuning
