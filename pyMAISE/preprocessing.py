import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pyMAISE.settings as settings


class PreProcesser:
    def __init__(
        self,
        path,
        input_slice: slice = None,
        output_slice: slice = None,
        header="infer",
    ):
        if isinstance(path, str):
            # Read in data from path
            self._data = pd.read_csv(path, header=header)
            self._input_slice = input_slice
            self._output_slice = output_slice

            # Separate data based on x and y indicies
            self._inputs = self._data.iloc[:, input_slice]
            self._outputs = self._data.iloc[:, output_slice]
        elif isinstance(path, list):
            # Make sure only two paths are given
            assert len(path) == 2

            # Assuming the first path is the input and second is
            # output data
            self._inputs = pd.read_csv(path[0], header=header)
            self._outputs = pd.read_csv(path[1], header=header)

            # Join input and output DataFrames
            self._data = pd.concat([self._inputs, self._outputs], axis=1)

            # Fill slicing information
            self._input_slice = slice(0, self._inputs.shape[1])
            self._output_slice = slice(
                self._inputs.shape[1] - 1, self._outputs.shape[1]
            )

        # Scalers
        self._xscaler = None
        self._yscaler = None

        if settings.values.verbosity > 0:
            print("Reading data from", path)

            if settings.values.verbosity == 2:
                print(self._data.head(5))

    def data_split(self):
        return train_test_split(
            self._inputs,
            self._outputs,
            test_size=settings.values.test_size,
            random_state=settings.values.random_state,
        )

    def min_max_scale(self, scale_x: bool = True, scale_y: bool = True):
        # Split data
        xtrain, xtest, ytrain, ytest = self.data_split()

        # Scale x
        if scale_x:
            self._xscaler = MinMaxScaler()
            xtrain = pd.DataFrame(
                self._xscaler.fit_transform(xtrain),
                index=xtrain.index,
                columns=xtrain.columns,
            )
            xtest = pd.DataFrame(
                self._xscaler.transform(xtest),
                index=xtest.index,
                columns=xtest.columns,
            )

        # Scale y
        if scale_y:
            self._yscaler = MinMaxScaler()
            ytrain = pd.DataFrame(
                self._yscaler.fit_transform(ytrain),
                index=ytrain.index,
                columns=ytrain.columns,
            )
            ytest = pd.DataFrame(
                self._yscaler.transform(ytest),
                index=ytest.index,
                columns=ytest.columns,
            )

        if settings.values.verbosity > 0:
            print("MinMax scaling data")

        return (xtrain, xtest, ytrain, ytest)

    def std_scale(self, scale_x: bool = True, scale_y: bool = True):
        # Split data
        xtrain, xtest, ytrain, ytest = self.data_split()

        # Scale x
        if scale_x:
            self._xscaler = StandardScaler()
            xtrain = pd.DataFrame(
                self._xscaler.fit_transform(xtrain),
                index=xtrain.index,
                columns=xtrain.columns,
            )
            xtest = pd.DataFrame(
                self._xscaler.transform(xtest),
                index=xtest.index,
                columns=xtest.columns,
            )

        # Scale y
        if scale_y:
            self._yscaler = StandardScaler()
            ytrain = pd.DataFrame(
                self._yscaler.fit_transform(ytrain),
                index=ytrain.index,
                columns=ytrain.columns,
            )
            ytest = pd.DataFrame(
                self._yscaler.transform(ytest),
                index=ytest.index,
                columns=ytest.columns,
            )

        if settings.values.verbosity > 0:
            print("MinMax scaling data")

        return (xtrain, xtest, ytrain, ytest)

    def correlation_matrix(
        self,
        method="pearson",
        min_periods=1,
        fig=None,
        ax=None,
        colorbar=True,
        annotations=False,
    ):
        # Create correlation matrix DataFrame
        corr = self._data.corr(method, min_periods)

        # Get figure and axis object if not provided
        if ax == None and fig == None:
            fig, ax = plt.subplots()

        # Create heatmap
        im = ax.imshow(corr)

        # Set x and y ticks to column headers
        ax.set_xticks(np.arange(corr.shape[0]), labels=corr.columns, rotation=65)
        ax.set_yticks(np.arange(corr.shape[1]), labels=corr.columns, rotation=0)

        # Add colorbar if colorbar == True
        if colorbar:
            fig.colorbar(im)

        # Add annotations of each value in square if annotations == True
        if annotations:
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    ax.text(
                        j,
                        i,
                        round(corr.to_numpy()[i, j], 2),
                        ha="center",
                        va="center",
                        color="black",
                    )

        return (fig, ax)

    # Getters
    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def inputs(self) -> pd.DataFrame:
        return self._inputs

    @property
    def outputs(self) -> pd.DataFrame:
        return self._outputs

    @property
    def xscaler(self):
        return self._xscaler

    @property
    def yscaler(self):
        return self._yscaler
