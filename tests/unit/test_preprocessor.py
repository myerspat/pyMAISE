import numpy as np
import pytest
import xarray as xr
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai


# ================================================================
def test_read_csv():
    # Init settings
    settings = {
        "verbosity": 1,
        "random_state": 42,
        "test_size": 0.3,
        "regression": True,
        "classification": False,
    }
    settings = mai.settings.init(settings_changes=settings)

    # Use load functions to test the functionality of both
    # ways of reading data
    # Start with XS data as this is given in one file
    preprocessor = mai.load_xs()

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

    # Check first two entries
    np.testing.assert_array_equal(
        preprocessor.data.to_numpy()[0, :],
        np.array(
            [
                0.00644620,
                0.00924782,
                0.13000700,
                0.08083560,
                0.01497270,
                0.48248300,
                0.00150595,
                1.12546,
                1.256376,
            ]
        ),
    )
    np.testing.assert_array_equal(
        preprocessor.data.to_numpy()[1, :],
        np.array(
            [
                0.00635893,
                0.00934703,
                0.12881100,
                0.08104840,
                0.01536280,
                0.49055800,
                0.00149675,
                1.12616,
                1.241534,
            ]
        ),
    )

    # Next check the fuel performance since it's multi-input
    # multi-output and is in two separate files
    preprocessor = mai.load_fp()

    # Shape assertions
    assert preprocessor.data.shape == (400, 17)
    assert preprocessor.inputs.shape == (400, 13)
    assert preprocessor.outputs.shape == (400, 4)

    # Feature names assertions
    input_features = [
        "fuel_dens",
        "porosity",
        "clad_thick",
        "pellet_OD",
        "pellet_h",
        "gap_thick",
        "inlet_T",
        "enrich",
        "rough_fuel",
        "rough_clad",
        "ax_pow",
        "clad_T",
        "pressure",
    ]
    output_features = [
        "fis_gas_produced",
        "max_fuel_centerline_temp",
        "max_fuel_surface_temp",
        "radial_clad_dia",
    ]

    # Place in alphabetical order because that's what xarray does
    alpha_idx = np.argsort(np.array(input_features + output_features))
    assert (
        list(preprocessor.data.coords["variable"].to_numpy())
        == np.array(input_features + output_features)[alpha_idx].tolist()
    )
    assert list(preprocessor.inputs.coords["variable"].to_numpy()) == input_features
    assert list(preprocessor.outputs.coords["variable"].to_numpy()) == output_features

    # Check first two entries
    np.testing.assert_array_equal(
        preprocessor.data.to_numpy()[0, :],
        np.array(
            [
                10466,
                0.040527,
                0.0005711,
                0.0041043,
                0.013077,
                1.92e-05,
                292.36,
                0.044852,
                1.84e-06,
                3.89e-07,
                0.99967,
                602.72,
                15504000,
                2.95e-05,
                1569.69931,
                699.6130331,
                1.88e-05,
            ]
        )[alpha_idx],
    )
    np.testing.assert_array_equal(
        preprocessor.data.to_numpy()[1, :],
        np.array(
            [
                10488,
                0.04178,
                0.00056984,
                0.004096,
                0.014227,
                1.90e-05,
                291.33,
                0.044942,
                1.90e-06,
                4.17e-07,
                0.98741,
                602.81,
                15591000,
                3.17e-05,
                1559.465162,
                699.9761907,
                1.87e-05,
            ]
        )[alpha_idx],
    )


# ================================================================
def test_split_sequances():
    # Initialize pyMAISE
    settings = {
        "verbosity": 1,
        "random_state": 42,
        "test_size": 0.3,
        "regression": True,
        "classification": False,
    }
    settings = mai.settings.init(settings_changes=settings)

    # Define raw sequence data
    num_sequences = 9
    raw_data = np.zeros((9, 3))
    raw_data[:, 0] = np.linspace(10, 90, num_sequences)
    raw_data[:, 1] = np.linspace(15, 95, num_sequences)
    raw_data[:, 2] = raw_data[:, 0] + raw_data[:, 1]

    univariate_xarray = (
        xr.Dataset(
            {
                "sec0": (["timesteps"], raw_data[:, 0]),
            },
            coords={
                "timesteps": np.linspace(0, 8, num_sequences),
            },
        )
        .to_array()
        .transpose(..., "variable")
    )
    multivariate_xarray = (
        xr.Dataset(
            {
                "sec0": (["timesteps"], raw_data[:, 0]),
                "sec1": (["timesteps"], raw_data[:, 1]),
                "sec2": (["timesteps"], raw_data[:, 2]),
            },
            coords={
                "timesteps": np.linspace(0, 8, num_sequences),
            },
        )
        .to_array()
        .transpose(..., "variable")
    )

    # ============================================================
    # Univariate
    # Input: (samples=6, timesteps=3, features=1)
    # Output: (samples=6, timesteps=1, features=1)
    # ============================================================
    # What is defined in preprocessor.data, preprocessor.input,
    # and preprocessor.output is already given in a univariate form
    # with "sec1" and "sec2" as inputs and "sec3" as output
    preprocessor = mai.PreProcessor()
    preprocessor.set_data(univariate_xarray, inputs=["sec0"])

    # Check contents prior to splitting
    np.testing.assert_array_equal(
        preprocessor.data.sel(variable="sec0").to_numpy(), raw_data[:, 0]
    )

    # Split data and confirm shapes and contents of first 2 samples
    preprocessor.split_sequences(
        input_steps=3,
        output_steps=1,
    )
    print(preprocessor.inputs.to_dataset(dim="variable").to_dataframe())
    print(preprocessor.outputs)
    assert preprocessor.inputs.shape == (6, 3, 1)
    assert preprocessor.outputs.shape == (6, 1, 1)

    np.testing.assert_array_equal(
        preprocessor.inputs.to_numpy()[0, :, :],
        np.array([[10], [20], [30]]),
    )
    np.testing.assert_array_equal(
        preprocessor.outputs.to_numpy()[0, :, :],
        np.array([[40]]),
    )
    np.testing.assert_array_equal(
        preprocessor.inputs.to_numpy()[1, :, :],
        np.array([[20], [30], [40]]),
    )
    np.testing.assert_array_equal(
        preprocessor.outputs.to_numpy()[1, :, :],
        np.array([[50]]),
    )

    # ============================================================
    # Multiple Input Series
    # Input: (samples=7, timesteps=3, features=2)
    # Output: (samples=7, timesteps=1, features=1)
    # ============================================================
    preprocessor.set_data(
        multivariate_xarray,
        inputs=["sec0", "sec1"],
    )

    # Check contents prior to splitting
    np.testing.assert_array_equal(preprocessor.data.values, raw_data)

    # Split data and confirm shapes and contents of first 2 samples
    preprocessor.split_sequences(
        input_steps=3,
        output_steps=0,
    )
    assert preprocessor.inputs.shape == (7, 3, 2)
    assert preprocessor.outputs.shape == (7, 1, 1)

    np.testing.assert_array_equal(
        preprocessor.inputs.to_numpy()[0, :, :],
        np.array([[10, 15], [20, 25], [30, 35]]),
    )
    np.testing.assert_array_equal(
        preprocessor.outputs.to_numpy()[0, :, :],
        np.array([[65]]),
    )
    np.testing.assert_array_equal(
        preprocessor.inputs.to_numpy()[1, :, :],
        np.array([[20, 25], [30, 35], [40, 45]]),
    )
    np.testing.assert_array_equal(
        preprocessor.outputs.to_numpy()[1, :, :],
        np.array([[85]]),
    )

    # ============================================================
    # Multiple Parallel Series
    # Input: (samples=6, timesteps=3, features=3)
    # Output: (samples=6, timesteps=1, features=3)
    # ============================================================
    preprocessor.set_data(
        multivariate_xarray,
        inputs=["sec0", "sec1", "sec2"],
    )

    # Split data and confirm shapes and contents of first 2 samples
    preprocessor.split_sequences(
        input_steps=3,
        output_steps=1,
    )
    assert preprocessor.inputs.shape == (6, 3, 3)
    assert preprocessor.outputs.shape == (6, 1, 3)

    np.testing.assert_array_equal(
        preprocessor.inputs.to_numpy()[0, :, :],
        np.array([[10, 15, 25], [20, 25, 45], [30, 35, 65]]),
    )
    np.testing.assert_array_equal(
        preprocessor.outputs.to_numpy()[0, :, :],
        np.array([[40, 45, 85]]),
    )
    np.testing.assert_array_equal(
        preprocessor.inputs.to_numpy()[1, :, :],
        np.array([[20, 25, 45], [30, 35, 65], [40, 45, 85]]),
    )
    np.testing.assert_array_equal(
        preprocessor.outputs.to_numpy()[1, :, :],
        np.array([[50, 55, 105]]),
    )

    # ============================================================
    # Univariate Multi-Step
    # Input: (samples=5, timesteps=3, features=1)
    # Output: (samples=5, timesteps=2, features=3)
    # ============================================================
    preprocessor.set_data(univariate_xarray, inputs=["sec0"])

    # Split data and confirm shapes and contents of first 2 samples
    preprocessor.split_sequences(
        input_steps=3,
        output_steps=2,
    )
    assert preprocessor.inputs.shape == (5, 3, 1)
    assert preprocessor.outputs.shape == (5, 2, 1)

    np.testing.assert_array_equal(
        preprocessor.inputs.to_numpy()[0, :, :],
        np.array([[10], [20], [30]]),
    )
    np.testing.assert_array_equal(
        preprocessor.outputs.to_numpy()[0, :, :],
        np.array([[40], [50]]),
    )
    np.testing.assert_array_equal(
        preprocessor.inputs.to_numpy()[1, :, :],
        np.array([[20], [30], [40]]),
    )
    np.testing.assert_array_equal(
        preprocessor.outputs.to_numpy()[1, :, :],
        np.array([[50], [60]]),
    )

    # ============================================================
    # Multiple Input Multi-Step Output
    # Input: (samples=6, timesteps=3, features=2)
    # Output: (samples=6, timesteps=2, features=1)
    # ============================================================
    preprocessor.set_data(
        multivariate_xarray,
        inputs=["sec0", "sec1"],
    )

    # Split data and confirm shapes and contents of first 2 samples
    preprocessor.split_sequences(
        input_steps=3,
        output_steps=1,
    )
    assert preprocessor.inputs.shape == (6, 3, 2)
    assert preprocessor.outputs.shape == (6, 2, 1)

    np.testing.assert_array_equal(
        preprocessor.inputs.to_numpy()[0, :, :],
        np.array([[10, 15], [20, 25], [30, 35]]),
    )
    np.testing.assert_array_equal(
        preprocessor.outputs.to_numpy()[0, :, :],
        np.array([[65], [85]]),
    )
    np.testing.assert_array_equal(
        preprocessor.inputs.to_numpy()[1, :, :],
        np.array([[20, 25], [30, 35], [40, 45]]),
    )
    np.testing.assert_array_equal(
        preprocessor.outputs.to_numpy()[1, :, :],
        np.array([[85], [105]]),
    )

    # ============================================================
    # Multiple Parallel Input and Multi-Step Output
    # Input: (samples=5, timesteps=3, features=3)
    # Output: (samples=5, timesteps=2, features=3)
    # ============================================================
    preprocessor.set_data(
        multivariate_xarray,
        inputs=["sec0", "sec1", "sec2"],
    )

    # Split data and confirm shapes and contents of first 2 samples
    preprocessor.split_sequences(
        input_steps=3,
        output_steps=2,
    )
    assert preprocessor.inputs.shape == (5, 3, 3)
    assert preprocessor.outputs.shape == (5, 2, 3)

    np.testing.assert_array_equal(
        preprocessor.inputs.to_numpy()[0, :, :],
        np.array([[10, 15, 25], [20, 25, 45], [30, 35, 65]]),
    )
    np.testing.assert_array_equal(
        preprocessor.outputs.to_numpy()[0, :, :],
        np.array([[40, 45, 85], [50, 55, 105]]),
    )
    np.testing.assert_array_equal(
        preprocessor.inputs.to_numpy()[1, :, :],
        np.array([[20, 25, 45], [30, 35, 65], [40, 45, 85]]),
    )
    np.testing.assert_array_equal(
        preprocessor.outputs.to_numpy()[1, :, :],
        np.array([[50, 55, 105], [60, 65, 125]]),
    )


# ================================================================
def test_train_test_split():
    # Init settings
    settings = {
        "verbosity": 1,
        "random_state": 42,
        "test_size": 0.3,
        "regression": True,
        "classification": False,
    }
    settings = mai.settings.init(settings_changes=settings)

    # Test train_test_split for 2D data without scaling
    # (multi-input, single output)
    preprocessor = mai.load_xs()
    preprocessor.train_test_split()
    xtrain, xtest, ytrain, ytest = preprocessor.split_data

    # Assert dimensions and size after split
    assert xtrain.shape == (700, 8)
    assert ytrain.shape == (700, 1)
    assert xtest.shape == (300, 8)
    assert ytest.shape == (300, 1)

    # Assert values did not change
    inputs = xtrain.combine_first(xtest)
    outputs = ytrain.combine_first(ytest)
    np.testing.assert_array_equal(
        inputs.sortby(variables=inputs.dims[0]).to_numpy(),
        preprocessor.inputs.to_numpy(),
    )
    np.testing.assert_array_equal(
        outputs.sortby(variables=inputs.dims[0]).to_numpy(),
        preprocessor.outputs.to_numpy(),
    )

    # Test train_test_split for 3D data with sklearn.preprocessor.MinMaxScaler()
    # (single-input, multi-output)
    multivariate_xarray = (
        xr.Dataset(
            {
                "sec0": (["timesteps"], np.linspace(10, 90, 9)),
                "sec1": (["timesteps"], np.linspace(15, 95, 9)),
                "sec2": (
                    ["timesteps"],
                    np.linspace(10, 90, 9) + np.linspace(15, 95, 9),
                ),
            },
            coords={
                "timesteps": np.linspace(0, 8, 9),
            },
        )
        .to_array()
        .transpose(..., "variable")
    )
    preprocessor.set_data(multivariate_xarray, inputs=["sec0"])

    # Split data and confirm shapes and contents of first 2 samples
    preprocessor.split_sequences(
        input_steps=3,
        output_steps=1,
    )
    preprocessor.train_test_split(scaler=MinMaxScaler())
    xtrain, xtest, ytrain, ytest = preprocessor.split_data

    # Assert dimensions and size after split
    assert xtrain.shape == (4, 3, 1)
    assert ytrain.shape == (4, 2, 2)
    assert xtest.shape == (2, 3, 1)
    assert ytest.shape == (2, 2, 2)

    # Assert train values are between 0 - 1 (min-max scaled)
    assert np.all((xtrain >= 0) & (xtrain <= 1))
    assert np.all((ytrain >= 0) & (ytrain <= 1))

    # Assert there exists a 0 and 1 in each feature
    for feature in range(xtrain.shape[-1]):
        assert (
            np.sum((xtrain.isel(**{xtrain.dims[-1]: [feature]})).to_numpy() == 1) == 1
        )
        assert (
            np.sum((xtrain.isel(**{xtrain.dims[-1]: [feature]})).to_numpy() == 0) == 1
        )
    for feature in range(ytrain.shape[-1]):
        assert (
            np.sum((ytrain.isel(**{ytrain.dims[-1]: [feature]})).to_numpy() == 1) == 1
        )
        assert (
            np.sum((ytrain.isel(**{ytrain.dims[-1]: [feature]})).to_numpy() == 0) == 1
        )

    # Assert values are the same once transformed back
    inputs = xtrain.combine_first(xtest)
    outputs = ytrain.combine_first(ytest)
    np.testing.assert_array_equal(
        preprocessor.xscaler.inverse_transform(
            inputs.sortby(variables=inputs.dims[0])
            .to_numpy()
            .reshape(-1, inputs.shape[-1])
        ),
        preprocessor.inputs.to_numpy().reshape(-1, preprocessor.inputs.shape[-1]),
    )
    np.testing.assert_array_equal(
        preprocessor.yscaler.inverse_transform(
            outputs.sortby(variables=inputs.dims[0])
            .to_numpy()
            .reshape(-1, outputs.shape[-1])
        ),
        preprocessor.outputs.to_numpy().reshape(-1, preprocessor.outputs.shape[-1]),
    )

# ================================================================
def test_correlation_matrix():
    # Init settings
    settings = {
        "verbosity": 1,
        "random_state": 42,
        "test_size": 0.3,
        "regression": True,
        "classification": False,
    }
    settings = mai.settings.init(settings_changes=settings)
    preprocessor = mai.load_xs()

    # Make sure correlation matrix runs
    preprocessor.correlation_matrix()
