import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split

import pyMAISE.settings as settings


class PreProcessor:
    def read_csv(
        self,
        path,
        input_slice: slice = None,
        output_slice: slice = None,
        header="infer",
    ):
        """
        Read comma-separated values (csv) file into `xarray.DataArray` for
        `PreProcessor.data`, `PreProcessor.inputs`, and `PreProcessor.outputs`.

        Parameters
        ----------
        path : str or list of str
            The path to the single input file containing both inputs and outputs
            or a list of two paths, the first to the inputs and the second to
            the outputs.
        input_slice : slice
            Indice slice of inputs (required if only a single path was given in `path`).
        output_slice : slice
            Indice slice of outputs (required if only a single path was given in `path`).
        header : int, 'infer', or None
            Row number(s) containing column labels and marking the start of the data.
            Used in pandas.read_csv.
        """
        if settings.values.verbosity > 0:
            print("Reading data from", path)

        if isinstance(path, str):
            self._input_slice = input_slice
            self._output_slice = output_slice

            # Read in data from path
            self._data = (
                pd.read_csv(path, header=header)
                .to_xarray()
                .to_array()
                .transpose(..., "variable")
            )

            # Separate data based on slices
            self._inputs = self._data.isel(variable=self._input_slice)
            self._outputs = self._data.isel(variable=self._output_slice)

        elif isinstance(path, list):
            # Make sure only two paths are given
            assert len(path) == 2

            # Assuming the first path is the input and second is
            # output data
            self._inputs = (
                pd.read_csv(path[0], header=header)
                .to_xarray()
                .to_array()
                .transpose(..., "variable")
            )
            self._outputs = (
                pd.read_csv(path[1], header=header)
                .to_xarray()
                .to_array()
                .transpose(..., "variable")
            )

            # Join input and output
            self._data = self._inputs.combine_first(self._outputs)

            # Fill slicing information
            self._input_slice = slice(0, self._inputs.shape[1])
            self._output_slice = slice(
                self._inputs.shape[1] - 1, self._outputs.shape[1]
            )

    def set_data(self, data: xr.DataArray, inputs: list, outputs: list = None):
        """
        Set `PreProcessor.data`, `PreProcessor.inputs`, and `PreProcessor.outputs`.

        Parameters
        ----------
        data : xarray.DataArray or list of xarray.DataArray
            Raw data shaped with the feature dimension as the last dimension.
        inputs : list
            List of names or indices for each feature to be used in the inputs.
        outputs : list
            List of names or indices for each feature to be used in the outputs. If none
            is given and `inputs` does not include all features in `data` then the
            outputs are se to the remaining features. If none is given and `inputs`
            includes all features in `data` then a parallel series is assumed and
            `outputs = inputs`.
        """
        if isinstance(data, xr.DataArray):
            feature_dim = data.dims[-1]
            self._data = data.to_dataset(dim=feature_dim)
            # Restrict input to the features given by the user
            self._inputs = self._data[inputs]

            # If outputs are given then use them, if no outputs but inputs
            # has all features in the data set then assume parallel series,
            # and if none are true then drop the input features from the
            # data set to create outputs
            if outputs:
                self._outputs = self._data[outputs]
            elif set(self._inputs.keys()) == set(self._data.keys()):
                self._outputs = self._data[inputs]
            else:
                self._outputs = self._data.drop_vars(inputs)

        elif isinstance(data, list):
            self._inputs = data[0]
            self._outputs = data[1]

            # Join input and output
            self._data = self._inputs.combine_first(self._outputs)

        else:
            raise RuntimeError("Data type not supported for data")

        # Ensure the feature dimension is last so numpy arrays are correct
        self._data = self._data.to_array().transpose(..., feature_dim)
        self._inputs = self._inputs.to_array().transpose(..., feature_dim)
        self._outputs = self._outputs.to_array().transpose(..., feature_dim)

    def split_sequences(
        self,
        input_steps: int,
        output_steps: int,
        output_lookback: int = 1,
        dim: str = None,
    ):
        """
        Split time series data. Populates input and output data with `xarrays.DataArray`
        that are dimensions of (samples, time steps, features). These can be accesed
        through `PreProcessor.inputs` and `PreProcessor.outputs` accessors.

        Parameters
        ----------
        input_steps : int
            The number of time steps for each sample.
        output_steps : int
            Number of time steps for each sample.
        output_lookback : int
            The number of time steps to subtract from the end index of the inputs.
            This is default set to 1 but is set to zero for parallel series
        dim : str
            The name of the dimension to split the time series. Assumes the first dimension.
        """
        # Assert our data has 2-D, the result of splie_sequences will be 3-D
        assert len(self._data.shape) == 2

        # Alter output_lookback if doing parallel series
        if (
            set(self._inputs.coords[self._inputs.dims[-1]].to_numpy())
            == set(self._outputs.coords[self._outputs.dims[-1]].to_numpy())
            and output_lookback == 1
        ):
            output_lookback = 0

        # Use first dimension if not given
        if dim == None:
            dim = self._data.dims[0]

        # Iterate through 2D data to make 3D data for time series
        x, y = list(), list()
        for i in range(self._data.coords[dim].shape[0]):
            input_end_idx = i + input_steps
            output_end_idx = input_end_idx + output_steps
            if output_end_idx > self._data.coords[dim].shape[0]:
                break

            x.append(self._inputs.isel(**{dim: slice(i, input_end_idx)}).to_numpy())
            y.append(
                self._outputs.isel(
                    **{dim: slice(input_end_idx - output_lookback, output_end_idx)}
                ).to_numpy()
            )
        x = np.array(x)
        y = np.array(y)

        # Convert to xarray.DataArray(s)
        self._inputs = xr.DataArray(
            x,
            coords={
                "samples": range(x.shape[0]),
                "sequence_idx": range(x.shape[1]),
                self._inputs.dims[-1]: self._inputs.coords[
                    self._inputs.dims[-1]
                ].to_numpy(),
            },
        )
        self._outputs = xr.DataArray(
            y,
            coords={
                "samples": range(y.shape[0]),
                "sequence_idx": range(y.shape[1]),
                self._outputs.dims[-1]: self._outputs.coords[
                    self._outputs.dims[-1]
                ].to_numpy(),
            },
        )

    def train_test_split(self, scaler=None, scale_x: bool = True, scale_y: bool = True):
        """
        Split data into training and testing data sets.

        Parameters
        ----------
        scaler : sklearn scaler
            This an `sklearn.processing` such as `MinMaxScaler()` that scales the input
            and output data.
        scale_x : bool
            If `scaler` should be used on the inputs.
        scale_y : bool
            If `scaler` should be used on the outputs.
        """
        # Assuming the last dimension is the dimension for features
        # get the name of that dimension
        feature_dim_name = self._data.dims[-1]

        # Run sklearn.preprocessing.train_test_split on the coordinates
        # for the feature_dim_name dimension
        samples_idx = self._inputs.coords[self._inputs.dims[0]]
        train_idx, test_idx = train_test_split(
            samples_idx.to_numpy(),
            test_size=settings.values.test_size,
            random_state=settings.values.random_state,
        )

        # Split to xtrain, xtest, ytrain, ytest using indices and assuming the
        # first dimension in the array is the sample dimension
        self._split_data = (
            self._inputs.isel(
                **{self._inputs.dims[0]: train_idx.astype(int)}
            ).transpose(..., feature_dim_name),
            self._inputs.isel(**{self._inputs.dims[0]: test_idx.astype(int)}).transpose(
                ..., feature_dim_name
            ),
            self._outputs.isel(
                **{self._outputs.dims[0]: train_idx.astype(int)}
            ).transpose(..., feature_dim_name),
            self._outputs.isel(
                **{self._outputs.dims[0]: test_idx.astype(int)}
            ).transpose(..., feature_dim_name),
        )

        # Scale the data using the sklearn.preprocessing function given by the user
        if scaler:
            if scale_x:
                self._xscaler = copy.deepcopy(scaler)
                self._split_data[0].values = self._xscaler.fit_transform(
                    self._split_data[0].values.reshape(
                        -1, self._split_data[0].values.shape[-1]
                    )
                ).reshape(self._split_data[0].values.shape)
                self._split_data[1].values = self._xscaler.transform(
                    self._split_data[1].values.reshape(
                        -1, self._split_data[1].values.shape[-1]
                    )
                ).reshape(self._split_data[1].values.shape)

            if scale_y:
                self._yscaler = copy.deepcopy(scaler)
                self._split_data[2].values = self._yscaler.fit_transform(
                    self._split_data[2].values.reshape(
                        -1, self._split_data[2].values.shape[-1]
                    )
                ).reshape(self._split_data[2].values.shape)
                self._split_data[3].values = self._yscaler.transform(
                    self._split_data[3].values.reshape(
                        -1, self._split_data[3].values.shape[-1]
                    )
                ).reshape(self._split_data[3].values.shape)

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
        corr = (
            self._data.to_dataset(dim=self._data.dims[-1])
            .to_dataframe()
            .corr(method, min_periods)
        )

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
    def data(self) -> xr.DataArray:
        return self._data

    @property
    def inputs(self) -> xr.DataArray:
        return self._inputs

    @property
    def outputs(self) -> xr.DataArray:
        return self._outputs

    @property
    def split_data(self):
        return self._split_data

    @property
    def xscaler(self):
        return self._xscaler

    @property
    def yscaler(self):
        return self._yscaler

    # Setters
    @data.setter
    def data(self, data: xr.DataArray):
        self._data = data

    @inputs.setter
    def inputs(self, inputs: xr.DataArray):
        self._inputs = inputs

    @outputs.setter
    def outputs(self, outputs: xr.DataArray):
        self._outputs = outputs
