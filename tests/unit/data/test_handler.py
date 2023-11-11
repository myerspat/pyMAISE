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
    assert preprocessor.data.shape == (4001, 400, 41)
    assert preprocessor.inputs.shape == (4001, 400, 41)
    assert preprocessor.outputs.shape == (4001, 400, 1)

    # Assert begining and ending data
    for i in range(400):
        first_sample_feature_values = np.array(
            [
                0.950390625,
                1.051171875,
                -26.26953125,
                1.1082031250000002,
                0.924609375,
                0.925390625,
                0.905859375,
                0.997265625,
                1.0878906250000002,
                0.990234375,
                0.921484375,
                0.964453125,
                0.901953125,
                0.816015625,
                0.8949218750000001,
                1.035546875,
                0.8691406250000001,
                0.800390625,
                0.906982421875,
                0.951015625,
                17.4541015625,
                0.99287109375,
                1.00337890625,
                1.01927734375,
                1.1457031250000005,
                1.000390625,
                1.0153515625,
                0.9954296875,
                1.02794921875,
                0.98857421875,
                1.0021484375,
                1.0095703125,
                1.3007119140625,
                1.203701171875,
                1.0916015625000002,
                2.0763671875,
                0.9634765625,
                1.0673828125,
                1.0052734375,
                0.9646484375,
            ]
        )
        last_sample_feature_values = np.array(
            [
                0.8753906250000001,
                1.164453125,
                -7.51953125,
                0.933203125,
                1.0996093750000002,
                0.8503906250000001,
                0.930859375,
                0.972265625,
                1.062890625,
                1.015234375,
                0.8464843750000001,
                0.839453125,
                1.1269531250000002,
                0.941015625,
                1.019921875,
                1.0105468750000002,
                1.000390625,
                0.8753906250000001,
                0.8788574218749999,
                0.566015625,
                16.416601562500002,
                1.00412109375,
                1.00962890625,
                1.01552734375,
                0.920703125,
                0.975390625,
                0.9828515625,
                1.0179296875,
                0.99419921875,
                1.00732421875,
                1.0046484375,
                0.9920703125,
                1.1883994140624998,
                1.152576171875,
                1.0541015625,
                2.4888671875000004,
                1.0509765625,
                1.1048828125,
                1.0427734375000002,
                0.9271484375,
            ]
        )
        np.testing.assert_array_equal(
            preprocessor.data[1, i, :-1],
            first_sample_feature_values,
        )
        np.testing.assert_array_equal(
            preprocessor.data[1, i, :-1],
            first_sample_feature_values,
        )
        np.testing.assert_array_equal(
            preprocessor.inputs[-1, i, :-1],
            last_sample_feature_values,
        )
        np.testing.assert_array_equal(
            preprocessor.inputs[-1, i, :-1],
            last_sample_feature_values,
        )

    # Check first couple values in sequence data
    outputs_values = np.array(
        [
            617.9697875976562,
            618.4946899414062,
            617.4552569347459,
            617.9203491210938,
            619.7262573242188,
        ]
    )
    np.testing.assert_array_equal(
        preprocessor.data[0:5, 0, -1],
        outputs_values
    )
    np.testing.assert_array_equal(
        preprocessor.inputs[0:5, 0, -1],
        outputs_values
    )
    np.testing.assert_array_equal(
        preprocessor.outputs[0:5, 0, -1],
        outputs_values
    )
