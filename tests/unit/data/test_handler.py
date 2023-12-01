import sys

import numpy as np
import pytest
import xarray as xr

import pyMAISE as mai


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


def test_load_loca():
    # Load settings
    mai.settings.init()

    # Load data through PreProcessor
    preprocessor = mai.load_loca()

    # Shape assertions
    assert preprocessor.data.shape == (4001, 400, 44)
    assert preprocessor.inputs.shape == (4001, 400, 44)
    assert preprocessor.outputs.shape == (4001, 400, 4)

    # Assert begining and ending data
    for i in range(400):
        first_sample_feature_values = np.array(
            [
                1.06666667,
                1.2,
                50.0,
                0.8,
                1.2,
                1.06666667,
                0.8,
                1.2,
                0.93333333,
                0.93333333,
                1.06666667,
                0.93333333,
                0.93333333,
                0.93333333,
                0.93333333,
                0.93333333,
                1.2,
                0.8,
                1.15,
                0.85333333,
                6.53333333,
                1.00333333,
                0.99,
                0.97,
                1.06666667,
                0.93333333,
                1.02,
                1.00666667,
                1.01,
                0.97,
                1.00666667,
                1.00666667,
                1.15466667,
                1.818,
                0.9,
                1.0,
                0.9,
                1.4,
                1.03333333,
                1.1,
            ]
        )
        last_sample_feature_values = np.array(
            [
                1.0918164,
                0.98023356,
                35.3181022,
                0.95221506,
                0.9111622,
                0.98013086,
                0.98456057,
                0.93504432,
                1.05387288,
                1.12140451,
                0.82103804,
                1.02465197,
                0.93205184,
                0.96696788,
                0.98250758,
                0.86811134,
                0.95334107,
                1.09010678,
                1.02997878,
                1.2855111,
                7.92168275,
                1.00469921,
                1.00475051,
                0.99247073,
                0.83674474,
                0.82146543,
                1.00643982,
                0.99916991,
                1.00155855,
                1.00468544,
                1.00174953,
                1.00619003,
                1.23419344,
                1.78443828,
                1.05604381,
                3.1416632,
                0.96296582,
                1.46234724,
                0.95667249,
                1.01235371,
            ]
        )
        np.testing.assert_array_equal(
            np.round(preprocessor.data[1, i, :-4], decimals=8),
            first_sample_feature_values,
        )
        np.testing.assert_array_equal(
            np.round(preprocessor.data[-1, i, :-4], decimals=8),
            last_sample_feature_values,
        )
        np.testing.assert_array_equal(
            np.round(preprocessor.inputs[1, i, :-4], decimals=8),
            first_sample_feature_values,
        )
        np.testing.assert_array_equal(
            np.round(preprocessor.inputs[-1, i, :-4], decimals=8),
            last_sample_feature_values,
        )

    # Check first couple values in sequence data
    outputs_values = np.array(
        [
            [6.17969788e02, 1.55000000e07, 3.66000009e00, 0.00000000e00],
            [6.19476440e02, 1.56033000e07, 3.66000009e00, 0.00000000e00],
            [6.19972120e02, 1.57104000e07, 3.66000009e00, 0.00000000e00],
            [6.18127679e02, 1.58100000e07, 3.66000009e00, 0.00000000e00],
            [6.16912048e02, 1.54401000e07, 3.66000009e00, 0.00000000e00],
        ]
    )
    np.testing.assert_almost_equal(
        preprocessor.data[0:5, 0, 40:], outputs_values, decimal=6
    )
    np.testing.assert_almost_equal(
        preprocessor.inputs[0:5, 0, 40:], outputs_values, decimal=6
    )
    np.testing.assert_almost_equal(
        preprocessor.outputs[0:5, 0, :], outputs_values, decimal=6
    )
