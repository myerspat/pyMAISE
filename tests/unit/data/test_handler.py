import pytest
import xarray as xr

import pyMAISE as mai


# Unit test for loading xs data from xs.csv
def test_load_xs():
    # Load settings
    mai.settings.init()

    # Load data through PreProcessor
    preprocessor = mai.load_xs()

    # Type assertion
    assert isinstance(preprocessor.data, xr.DataArray)
    assert isinstance(preprocessor.inputs, xr.DataArray)
    assert isinstance(preprocessor.outputs, xr.DataArray)

    # Shape assertions
    assert preprocessor.data.shape == (1000, 9)
    assert preprocessor.inputs.shape == (1000, 8)
    assert preprocessor.outputs.shape == (1000, 1)

    # Feature names assertions
    input_features = [
        "FissionFast",
        "CaptureFast",
        "FissionThermal",
        "CaptureThermal",
        "Scatter12",
        "Scatter11",
        "Scatter21",
        "Scatter22",
    ]
    output_features = ["k"]
    assert (
        list(preprocessor.data.coords["variable"].to_numpy())
        == input_features + output_features
    )
    assert list(preprocessor.inputs.coords["variable"].to_numpy()) == input_features
    assert list(preprocessor.outputs.coords["variable"].to_numpy()) == output_features

    # Element assertions
    assert preprocessor.data[0, 0] == 0.00644620
    assert preprocessor.data[0, -1] == 1.256376
    assert preprocessor.data[-1, 0] == 0.00627230
    assert preprocessor.data[-1, -1] == 1.240064
