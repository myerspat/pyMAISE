import copy
import math
import re

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

import pyMAISE.settings as settings
from pyMAISE.methods import *


class PostProcessor:
    def __init__(
        self,
        data: pd.DataFrame,
        models_list,
        new_model_settings: dict = None,
        yscaler=None,
    ):
        # Extract data
        self._xtrain = data[0]
        self._xtest = data[1]
        self._ytrain = data[2]
        self._ytest = data[3]

        # Initialize models of interest
        model_types = []
        params = []
        model_wrappers = []

        if isinstance(models_list, dict):
            models_list = [models_list]

        # Extract models and the DataFrame of hyper-parameter configurations
        for models in models_list:
            for model, configs in models.items():
                # Fill model types list with string of model type
                model_types = model_types + [model] * settings.values.num_configs_saved

                # Fil parameter string with hyper-parameter configurations for each type
                params = params + configs[0]["params"].tolist()

                # Get all model wrappers and update parameter configurations if needed
                estimator = configs[1]
                if new_model_settings != None and model in new_model_settings:
                    if (
                        not re.search("nn", model)
                        or model == "knn"
                        or not settings.values.new_nn_architecture
                    ):
                        estimator = estimator.set_params(**new_model_settings[model])

                    else:
                        estimator.set_params(new_model_settings[model])

                model_wrappers = (
                    model_wrappers + [estimator] * settings.values.num_configs_saved
                )

        # Create models DataFrame
        self._models = pd.DataFrame(
            {
                "Model Types": model_types,
                "Parameter Configurations": params,
                "Model Wrappers": model_wrappers,
            }
        )

        # Fit each model to training data and get predicted training
        # and testing from each model
        yhat_train, yhat_test, histories = self._fit()

        if yscaler != None:
            self._yscaler = yscaler
            self._ytrain = pd.DataFrame(
                yscaler.inverse_transform(self._ytrain),
                index=self._ytrain.index,
                columns=self._ytrain.columns,
            )
            self._ytest = pd.DataFrame(
                yscaler.inverse_transform(self._ytest),
                index=self._ytest.index,
                columns=self._ytest.columns,
            )

            for i in range(len(yhat_train)):
                yhat_train[i] = yscaler.inverse_transform(yhat_train[i])
                yhat_test[i] = yscaler.inverse_transform(yhat_test[i])

        self._models = pd.concat(
            [
                self._models,
                pd.DataFrame(
                    {
                        "Train Yhat": yhat_train,
                        "Test Yhat": yhat_test,
                        "History": histories,
                    }
                ),
            ],
            axis=1,
        )

    # ===========================================================
    # Methods
    def _fit(self):
        # Array for trainig and testing prediceted outcomes
        yhat_train = []
        yhat_test = []
        histories = []

        ytrain = copy.deepcopy(self._ytrain)
        if ytrain.shape[1] == 1:
            ytrain = self._ytrain.values.ravel()

        # Fit each model and predict outcomes
        for i in range(self._models.shape[0]):
            # Estract regressor for the configuration
            regressor = None
            if (
                not re.search("nn", self._models["Model Types"][i])
                or self._models["Model Types"][i] == "knn"
                or not settings.values.new_nn_architecture
            ):
                regressor = self._models["Model Wrappers"][i].set_params(
                    **self._models["Parameter Configurations"][i]
                )
            else:
                regressor = self._models["Model Wrappers"][i].build(
                    self._models["Parameter Configurations"][i]
                )

            # Append learning curve history of neural networks and run fit for all
            if (
                not re.search("nn", self._models["Model Types"][i])
                or self._models["Model Types"][i] == "knn"
            ):
                regressor.fit(self._xtrain, ytrain)
                histories.append(None)
            else:
                if not settings.values.new_nn_architecture:
                    histories.append(
                        regressor.fit(self._xtrain, ytrain).model.history.history
                    )
                else:
                    histories.append(
                        self._models["Model Wrappers"][i]
                        .fit(
                            self._models["Parameter Configurations"][i],
                            regressor,
                            self._xtrain,
                            ytrain,
                        )
                        .model.history.history
                    )

            # Append training and testing predictions
            yhat_train.append(
                regressor.predict(self._xtrain).reshape(self._ytrain.shape)
            )
            yhat_test.append(regressor.predict(self._xtest).reshape(self._ytest.shape))

        return (yhat_train, yhat_test, histories)

    def metrics(self, sort_by=None, model_type: str = None, y=None):
        # Initialize metrics dictionary
        metrics = {}
        for split in ["Train", "Test"]:
            if settings.values.regression:
                for metric in ["R2", "MAE", "MSE", "RMSE"]:
                    metrics[f"{split} {metric}"] = []

            else:
                for metric in ["Accuracy", "Recall", "Precision", "F1"]:
                    metrics[f"{split} {metric}"] = []

        # Get the list of y if not provided
        if y == None:
            y = slice(0, len(self._ytrain.columns) + 1)
        elif isinstance(y, str):
            y = self._ytrain.columns.get_loc(y)

        for i in range(self._models.shape[0]):
            # Get predictions
            yhat_train = self._models["Train Yhat"][i]
            yhat_test = self._models["Test Yhat"][i]

            data = {
                "Train": [self._ytrain, self._models["Train Yhat"][i]],
                "Test": [self._ytest, self._models["Test Yhat"][i]],
            }

            for split in ["Train", "Test"]:
                if settings.values.regression:
                    # Regression performance metrics
                    metrics[f"{split} R2"].append(
                        r2_score(
                            data[split][0][data[split][0].columns[y]],
                            data[split][1][:, y],
                        )
                    )
                    metrics[f"{split} MAE"].append(
                        mean_absolute_error(
                            data[split][0][data[split][0].columns[y]],
                            data[split][1][:, y],
                        )
                    )
                    metrics[f"{split} MSE"].append(
                        mean_squared_error(
                            data[split][0][data[split][0].columns[y]],
                            data[split][1][:, y],
                        )
                    )
                    metrics[f"{split} RMSE"].append(
                        math.sqrt(metrics[f"{split} MSE"][i])
                    )
                else:
                    # Classification performance metrics
                    metrics[f"{split} Accuracy"].append(
                        accuracy_score(
                            data[split][0][data[split][0].columns[y]],
                            data[split][1][:, y],
                        )
                    )
                    metrics[f"{split} Recall"].append(
                        recall_score(
                            data[split][0][data[split][0].columns[y]],
                            data[split][1][:, y],
                            average="micro",
                        )
                    )
                    metrics[f"{split} Precision"].append(
                        precision_score(
                            data[split][0][data[split][0].columns[y]],
                            data[split][1][:, y],
                            average="micro",
                        )
                    )
                    metrics[f"{split} F1"].append(
                        f1_score(
                            data[split][0][data[split][0].columns[y]],
                            data[split][1][:, y],
                            average="micro",
                        )
                    )

        # Determine sort_by depending on problem
        if sort_by is None:
            if settings.values.regression:
                sort_by = "Test R2"
            else:
                sort_by = "Test Accuracy"

        ascending = True
        if (
            sort_by == "Test R2"
            or sort_by == "Train R2"
            or sort_by == "Test Accuracy"
            or sort_by == "Train Accuracy"
        ):
            ascending = False

        for key, value in metrics.items():
            self._models[key] = value

        models = copy.deepcopy(
            self._models[
                ["Model Types", "Parameter Configurations"] + list(metrics.keys())
            ]
        )

        for i in range(models.shape[0]):
            if isinstance(models["Parameter Configurations"][i], kt.HyperParameters):
                models["Parameter Configurations"][i] = models[
                    "Parameter Configurations"
                ][i].values.copy

        if model_type == None:
            return models.sort_values(sort_by, ascending=[ascending])
        else:
            return models[models["Model Types"] == model_type].sort_values(
                sort_by, ascending=[ascending]
            )

    def _get_idx(self, idx: int = None, model_type: str = None, sort_by=None):
        if sort_by is None:
            if settings.values.regression == True:
                sort_by = "Test R2"
            else:
                sort_by = "Test Accuracy"

        # Determine the index of the model in the DataFrame
        if idx == None:
            if model_type != None:
                # If an index is not given but a model type is, get index based on sort_by
                if (
                    sort_by == "Test R2"
                    or sort_by == "Train R2"
                    or sort_by == "Test Accuracy"
                    or sort_by == "Train Accuracy"
                ):
                    idx = self._models[self._models["Model Types"] == model_type][
                        sort_by
                    ].idxmax()
                else:
                    idx = self._models[self._models["Model Types"] == model_type][
                        sort_by
                    ].idxmin()
            else:
                # If an index is not given and the model type is not given, return index
                # of best in sort_by
                if sort_by == "Test R2" or sort_by == "Train R2":
                    idx = self._models[sort_by].idxmax()
                else:
                    idx = self._models[sort_by].idxmin()

        return idx

    def get_predictions(self, idx: int = None, model_type: str = None, sort_by=None):
        if sort_by is None:
            if settings.values.regression:
                sort_by = "Test R2"
            else:
                sort_by = "Test Accuracy"

        # Determine the index of the model in the DataFrame
        idx = self._get_idx(idx=idx, model_type=model_type, sort_by=sort_by)

        yhat_train = self._models["Train Yhat"][idx]
        yhat_test = self._models["Test Yhat"][idx]

        return (yhat_train, yhat_test)

    def get_params(
        self,
        idx: int = None,
        model_type: str = None,
        sort_by=None,
        full: bool = False,
    ):
        if sort_by is None:
            if settings.values.regression:
                sort_by = "Test R2"
            else:
                sort_by = "Test Accuracy"

        # Determine the index of the model in the DataFrame
        idx = self._get_idx(idx=idx, model_type=model_type, sort_by=sort_by)

        parameters = copy.deepcopy(self._models["Parameter Configurations"][idx])
        if (
            re.search("nn", self._models["Model Types"][idx])
            and settings.values.new_nn_architecture
        ):
            parameters = parameters.values
        model_type = self._models["Model Types"][idx]

        return pd.DataFrame({"Model Types": [model_type], **parameters})

    def get_model(self, idx: int = None, model_type: str = None, sort_by=None):
        if sort_by is None:
            if settings.values.regression:
                sort_by = "Test R2"
            else:
                sort_by = "Test Accuracy"

        # Determine the index of the model in the DataFrame
        idx = self._get_idx(idx=idx, model_type=model_type, sort_by=sort_by)

        if self._yscaler:
            ytrain = self._yscaler.transform(self._ytrain)

        # Get regressor and fit the model
        regressor = self._models["Model Wrappers"][idx].set_params(
            **self._models["Parameter Configurations"][idx]
        )
        regressor.fit(self._xtrain, ytrain)

        return regressor

    def diagonal_validation_plot(
        self,
        ax=None,
        idx: int = None,
        model_type: str = None,
        sort_by=None,
        y: list = None,
    ):
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(idx=idx, model_type=model_type, sort_by=sort_by)

        # Get the list of y if not provided
        if y == None:
            y = list(range(self._ytrain.shape[1]))

        # Get predicted and actual outputs
        yhat_train = self._models["Train Yhat"][idx]
        yhat_test = self._models["Test Yhat"][idx]
        ytrain = self._ytrain.to_numpy()
        ytest = self._ytest.to_numpy()

        if ax == None:
            ax = plt.gca()

        for y_idx in y:
            if isinstance(y_idx, str):
                y_idx = self._ytest.columns.get_loc(y_idx)

            ax.scatter(yhat_train[:, y_idx], ytrain[:, y_idx], c="b", marker="o")
            ax.scatter(yhat_test[:, y_idx], ytest[:, y_idx], c="r", marker="o")

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]

        ax.plot(lims, lims, "k--")
        ax.set_aspect("equal")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend(["Training Data", "Testing Data"])
        ax.set_xlabel("Predicted Outcome")
        ax.set_ylabel("Actual Outcome")

        return ax

    def validation_plot(
        self,
        ax=None,
        idx: int = None,
        model_type: str = None,
        sort_by="Test R2",
        y: list = None,
    ):
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(idx=idx, model_type=model_type, sort_by=sort_by)

        # Get the list of y if not provided
        if y == None:
            y = list(range(self._ytest.shape[1]))

        # Get prediected and actual outputs
        yhat_test = self._models["Test Yhat"][idx]
        ytest = self._ytest.to_numpy()

        if ax == None:
            ax = plt.gca()

        for y_idx in y:
            # If the column name is given as opposed to the position,
            # find the position
            if isinstance(y_idx, str):
                y_idx = self._ytest.columns.get_loc(y_idx)

            ax.plot(
                np.linspace(1, ytest.shape[0], ytest.shape[0]),
                np.abs((ytest[:, y_idx] - yhat_test[:, y_idx]) / ytest[:, y_idx]) * 100,
                "-o",
                label=self._ytest.columns[y_idx],
            )

        if len(y) > 1:
            ax.legend()

        ax.set_xlabel("Testing Data Index")
        ax.set_ylabel("Absolute Relative Error (%)")

        return ax

    def nn_learning_plot(
        self,
        idx=None,
        ax=None,
        sort_by=None,
    ):
        nn_models = []
        ax = ax or plt.gca()
        for model_type in self._models["Model Types"]:
            if (
                re.search("nn", model_type)
                and model_type != "knn"
                and not any(model_type in x for x in nn_models)
            ):
                # Determine the index of the model in the DataFrame
                idx = self._get_idx(idx=idx, model_type=model_type, sort_by=sort_by)
                nn_models = nn_models + [model_type]

                history = self._models["History"][idx]

                ax.plot(history["loss"], label=model_type + " Training")
                ax.plot(history["val_loss"], label=model_type + " Validation")
                ax.legend()
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")

        return ax

    def confusion_matrix(
        self,
        ax=None,
        idx: int = None,
        model_type: str = None,
        sort_by=None,
    ):
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(idx=idx, model_type=model_type, sort_by=sort_by)

        # Get predicted and actual outputs
        yhat_train = self._models["Train Yhat"][idx]  # predicted
        yhat_test = self._models["Test Yhat"][idx]  # predicted

        ytrain = self._ytrain.to_numpy()  # actual
        ytest = self._ytest.to_numpy()  # actual

        cm = confusion_matrix(ytest, yhat_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(self._models["Model Types"][idx])
        plt.show()
