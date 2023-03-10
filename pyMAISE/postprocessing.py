from pyMAISE.methods import *

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class PostProcessor:
    def __init__(
        self, data: pd.DataFrame, models_list, new_model_settings: dict = None
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
                model_types = model_types + [model] * configs[0].shape[0]

                # Fil parameter string with hyper-parameter configurations for each type
                params = params + configs[0]["params"].tolist()

                # Get all model wrappers and update parameter configurations if needed
                estimator = configs[1]
                if new_model_settings != None and model in new_model_settings:
                    estimator = estimator.set_params(**new_model_settings[model])
                model_wrappers = model_wrappers + [estimator] * configs[0].shape[0]

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

        # Fit each model and predict outcomes
        for i in range(self._models.shape[0]):
            # Estract regressor for the configuration
            regressor = self._models["Model Wrappers"][i].set_params(
                **self._models["Parameter Configurations"][i]
            )

            # Append learning curve history of neural networks and run fit for all
            if self._models["Model Types"][i] != "nn":
                regressor.fit(self._xtrain, self._ytrain)
                histories.append(None)
            else:
                histories.append(
                    regressor.fit(self._xtrain, self._ytrain).model.history.history
                )

            # Append training and testing predictions
            yhat_train.append(regressor.predict(self._xtrain))
            yhat_test.append(regressor.predict(self._xtest))

        return (yhat_train, yhat_test, histories)

    def metrics(self, sort_by="Test R2"):
        if {
            "Train R2",
            "Train MAE",
            "Train MSE",
            "Train RMSE",
            "Test R2",
            "Test MAE",
            "Test MSE",
            "Test RMSE",
        }.issubset(self._models) == False:
            train_r2 = []
            train_mae = []
            train_mse = []
            train_mse = []
            train_rmse = []
            test_r2 = []
            test_mae = []
            test_mse = []
            test_mse = []
            test_rmse = []

            for i in range(self._models.shape[0]):
                # Get predictions
                yhat_train = self._models["Train Yhat"][i]
                yhat_test = self._models["Test Yhat"][i]

                # Calculate performance metrics and append to lists
                train_r2.append(r2_score(self._ytrain, yhat_train))
                train_mae.append(mean_absolute_error(self._ytrain, yhat_train))
                train_mse.append(mean_squared_error(self._ytrain, yhat_train))
                train_rmse.append(math.sqrt(train_mse[i]))
                test_r2.append(r2_score(self._ytest, yhat_test))
                test_mae.append(mean_absolute_error(self._ytest, yhat_test))
                test_mse.append(mean_squared_error(self._ytest, yhat_test))
                test_rmse.append(math.sqrt(test_mse[i]))

            # If the sort_by is anything other than R2, metrics is ascending
            ascending = True
            if sort_by == "Test R2" or sort_by == "Train R2":
                ascending = False

            # Combine models and performance metrics
            self._models = pd.concat(
                [
                    self._models,
                    pd.DataFrame(
                        {
                            "Train R2": train_r2,
                            "Train MAE": train_mae,
                            "Train MSE": train_mse,
                            "Train RMSE": train_rmse,
                            "Test R2": test_r2,
                            "Test MAE": test_mae,
                            "Test MSE": test_mse,
                            "Test RMSE": test_rmse,
                        }
                    ),
                ],
                axis=1,
            ).sort_values(sort_by, ascending=[ascending])

        return self._models[
            [
                "Model Types",
                "Parameter Configurations",
                "Train R2",
                "Train MAE",
                "Train MSE",
                "Train RMSE",
                "Test R2",
                "Test MAE",
                "Test MSE",
                "Test RMSE",
            ]
        ]

    def _get_idx(self, idx: int = None, model_type: str = None, sort_by="Test R2"):
        # Determine the index of the model in the DataFrame
        if idx == None:
            if model_type != None:
                # If an index is not given but a model type is, get index based on sort_by
                if sort_by == "Test R2" or sort_by == "Train R2":
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

    def get_predictions(
        self, idx: int = None, model_type: str = None, sort_by="Test R2"
    ):
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(idx=idx, model_type=model_type, sort_by=sort_by)

        yhat_train = self._models["Train Yhat"][idx]
        yhat_test = self._models["Test Yhat"][idx]

        return (yhat_train, yhat_test)

    def get_params(
        self,
        idx: int = None,
        model_type: str = None,
        sort_by="Test R2",
        full: bool = False,
    ):
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(idx=idx, model_type=model_type, sort_by=sort_by)

        parameters = self._models["Parameter Configurations"][idx]
        model_type = self._models["Model Types"][idx]

        if full == True:
            estimator = self._models["Model Wrappers"][idx].set_params(**parameters)
            parameters = estimator.get_params()

        return pd.DataFrame({"Model Types": [model_type], **parameters})

    def get_model(self, idx: int = None, model_type: str = None, sort_by="Test R2"):
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(idx=idx, model_type=model_type, sort_by=sort_by)

        # Get regressor and fit the model
        regressor = self._models["Model Wrappers"][idx].set_params(
            **self._models["Parameter Configurations"][idx]
        )
        regressor.fit(self._xtrain, self._ytrain)

        return regressor

    def diagonal_validation_plot(
        self,
        ax=None,
        idx: int = None,
        model_type: str = None,
        sort_by="Test R2",
        yscaler=None,
        y: list = None,
    ):
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(idx=idx, model_type=model_type, sort_by=sort_by)

        # Get the list of y if not provided
        if y == None:
            y = list(range(self._ytrain.shape[1]))

        # Get prediected and actual outputs
        yhat_train = self._models["Train Yhat"][idx]
        yhat_test = self._models["Test Yhat"][idx]
        ytrain = self._ytrain
        ytest = self._ytest

        # Scale outputs if scaler is given
        if yscaler != None:
            yhat_train = yscaler.inverse_transform(yhat_train)
            yhat_test = yscaler.inverse_transform(yhat_test)
            ytrain = yscaler.inverse_transform(ytrain)
            ytest = yscaler.inverse_transform(ytest)

        if ax == None:
            ax = plt.gca()

        for y_idx in y:
            ax.scatter(yhat_train[y_idx], ytrain[y_idx], c="b", marker="o")
            ax.scatter(yhat_test[y_idx], ytest[y_idx], c="r", marker="o")

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
        yscaler=None,
        y: list = None,
    ):
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(idx=idx, model_type=model_type, sort_by=sort_by)

        # Get the list of y if not provided
        if y == None:
            y = list(range(self._ytest.shape[1]))

        # Get prediected and actual outputs
        yhat_test = self._models["Test Yhat"][idx]
        ytest = self._ytest

        # Scale outputs if scaler is given
        if yscaler != None:
            yhat_test = yscaler.inverse_transform(yhat_test)
            ytest = yscaler.inverse_transform(ytest)

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
        ax=None,
        idx: int = None,
        sort_by="Test R2",
    ):
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(idx=idx, model_type="nn", sort_by=sort_by)

        if ax == None:
            ax = plt.gca()

        history = self._models["History"][idx]

        ax.plot(history["loss"], label="Training")
        ax.plot(history["val_loss"], label="Validation")
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        return ax
