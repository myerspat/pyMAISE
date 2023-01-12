import pyMAISE.settings as settings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class PreProcesser:
    def __init__(
        self,
        path: str,
        x_idx: slice,
        y_idx: slice,
    ):
        # Read in data from path
        self._data = pd.read_csv(path)
        self._x_idx = x_idx
        self._y_idx = y_idx

        # Separate data based on x and y indicies
        self._x = self._data.iloc[:, x_idx]
        self._y = self._data.iloc[:, y_idx]

        if settings.values.verbosity > 0:
            print("Reading data from " + path)

            if settings.values.verbosity == 2:
                print(self._data.head(5))

    def data_split(self):
        return train_test_split(
            self._x,
            self._y,
            test_size=settings.values.test_size,
            random_state=settings.values.random_state,
        )

    def min_max_scale(self, scale_x: bool = True, scale_y: bool = True):
        xtrain, xtest, ytrain, ytest = self.data_split()

        if scale_x:
            xscaler = MinMaxScaler()
            xtrain = xscaler.fit_transform(xtrain)
            xtest = xscaler.transform(xtest)

        if scale_y:
            yscaler = MinMaxScaler()
            ytrain = yscaler.fit_transform(ytrain)
            ytest = yscaler.transform(ytest)

        if settings.values.verbosity > 0:
            print("MinMax scaling data")

        return (xtrain, xtest, ytrain, ytest)

    def std_scale(self, scale_x: bool = True, scale_y: bool = True):
        xtrain, xtest, ytrain, ytest = self.data_split()

        if scale_x:
            xscaler = StandardScaler()
            xtrain = xscaler.fit_transform(xtrain)
            xtest = xscaler.transform(xtest)

        if scale_y:
            yscaler = StandardScaler()
            ytrain = yscaler.fit_transform(ytrain)
            ytest = yscaler.transform(ytest)

        if settings.values.verbosity > 0:
            print("MinMax scaling data")

        return (xtrain, xtest, ytrain, ytest)
