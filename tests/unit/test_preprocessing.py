import numpy as np
import pytest
import xarray as xr
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai
from pyMAISE.datasets import load_fp, load_loca, load_xs
from pyMAISE.preprocessing import (
    SplitSequence,
    correlation_matrix,
    read_csv,
    scale_data,
    train_test_split,
)


# ================================================================
def test_read_csv():
    # Init settings
    settings = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        verbosity=1,
        random_state=42,
        cuda_visible_devices="-1",  # Use CPU only
    )

    # Start with XS data as this is given in one file
    data, inputs, outputs = load_xs()

    # Shape assertions
    assert data.shape == (1000, 9)
    assert inputs.shape == (1000, 8)
    assert outputs.shape == (1000, 1)

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
    assert list(data.coords["variable"].to_numpy()) == input_features + output_features
    assert list(inputs.coords["variable"].to_numpy()) == input_features
    assert list(outputs.coords["variable"].to_numpy()) == output_features

    # Check first two entries
    np.testing.assert_array_equal(
        data.values[0, :],
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
        data.values[1, :],
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
    data, inputs, outputs = load_fp()

    # Shape assertions
    assert data.shape == (400, 17)
    assert inputs.shape == (400, 13)
    assert outputs.shape == (400, 4)

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
        list(data.coords["variable"].to_numpy())
        == np.array(input_features + output_features)[alpha_idx].tolist()
    )
    assert list(inputs.coords["variable"].to_numpy()) == input_features
    assert list(outputs.coords["variable"].to_numpy()) == output_features

    # Check first two entries
    np.testing.assert_array_equal(
        data.values[0, :],
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
        data.values[1, :],
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
    settings = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        verbosity=1,
        random_state=42,
        cuda_visible_devices="-1",  # Use CPU only
    )

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

    # Split data and confirm shapes and contents of first 2 samples
    split_sequences = SplitSequence(
        input_steps=3,
        output_steps=1,
        output_position=1,
        sequence_inputs=["sec0"],
        sequence_outputs=["sec0"],
    )
    inputs, outputs = split_sequences.split(univariate_xarray)
    print(inputs.values)
    print(outputs.values)
    assert inputs.shape == (6, 3, 1)
    assert outputs.shape == (6, 1)

    np.testing.assert_array_equal(
        inputs.to_numpy()[0, :, :],
        np.array([[10], [20], [30]]),
    )
    np.testing.assert_array_equal(
        outputs.to_numpy()[0, :],
        np.array([40]),
    )
    np.testing.assert_array_equal(
        inputs.to_numpy()[1, :, :],
        np.array([[20], [30], [40]]),
    )
    np.testing.assert_array_equal(
        outputs.to_numpy()[1, :],
        np.array([50]),
    )

    # ============================================================
    # Multiple Input Series
    # Input: (samples=7, timesteps=3, features=2)
    # Output: (samples=7, timesteps=1, features=1)
    # ============================================================

    # Split data and confirm shapes and contents of first 2 samples
    split_sequences = SplitSequence(
        input_steps=3,
        output_steps=1,
        output_position=0,
        sequence_inputs=["sec0", "sec1"],
        sequence_outputs=["sec2"],
    )
    inputs, outputs = split_sequences.split(multivariate_xarray)
    assert inputs.shape == (7, 3, 2)
    assert outputs.shape == (7, 1)

    np.testing.assert_array_equal(
        inputs.to_numpy()[0, :, :],
        np.array([[10, 15], [20, 25], [30, 35]]),
    )
    np.testing.assert_array_equal(
        outputs.to_numpy()[0, :],
        np.array([65]),
    )
    np.testing.assert_array_equal(
        inputs.to_numpy()[1, :, :],
        np.array([[20, 25], [30, 35], [40, 45]]),
    )
    np.testing.assert_array_equal(
        outputs.to_numpy()[1, :],
        np.array([85]),
    )

    # ============================================================
    # Multiple Parallel Series
    # Input: (samples=6, timesteps=3, features=3)
    # Output: (samples=6, timesteps=1, features=3)
    # ============================================================

    # Split data and confirm shapes and contents of first 2 samples
    split_sequences = SplitSequence(
        input_steps=3,
        output_steps=1,
        output_position=1,
        sequence_inputs=["sec0", "sec1", "sec2"],
        sequence_outputs=["sec0", "sec1", "sec2"],
    )
    inputs, outputs = split_sequences.split(multivariate_xarray)
    assert inputs.shape == (6, 3, 3)
    assert outputs.shape == (6, 3)

    np.testing.assert_array_equal(
        inputs.to_numpy()[0, :, :],
        np.array([[10, 15, 25], [20, 25, 45], [30, 35, 65]]),
    )
    np.testing.assert_array_equal(
        outputs.to_numpy()[0, :],
        np.array([40, 45, 85]),
    )
    np.testing.assert_array_equal(
        inputs.to_numpy()[1, :, :],
        np.array([[20, 25, 45], [30, 35, 65], [40, 45, 85]]),
    )
    np.testing.assert_array_equal(
        outputs.to_numpy()[1, :],
        np.array([50, 55, 105]),
    )

    # ============================================================
    # Univariate Multi-Step
    # Input: (samples=5, timesteps=3, features=1)
    # Output: (samples=5, timesteps=2, features=3)
    # ============================================================

    # Split data and confirm shapes and contents of first 2 samples
    split_sequences = SplitSequence(
        input_steps=3,
        output_steps=2,
        output_position=1,
        sequence_inputs=["sec0"],
        sequence_outputs=["sec0"],
    )
    inputs, outputs = split_sequences.split(univariate_xarray)
    assert inputs.shape == (5, 3, 1)
    assert outputs.shape == (5, 2, 1)

    np.testing.assert_array_equal(
        inputs.to_numpy()[0, :, :],
        np.array([[10], [20], [30]]),
    )
    np.testing.assert_array_equal(
        outputs.to_numpy()[0, :, :],
        np.array([[40], [50]]),
    )
    np.testing.assert_array_equal(
        inputs.to_numpy()[1, :, :],
        np.array([[20], [30], [40]]),
    )
    np.testing.assert_array_equal(
        outputs.to_numpy()[1, :, :],
        np.array([[50], [60]]),
    )

    # ============================================================
    # Multiple Input Multi-Step Output
    # Input: (samples=6, timesteps=3, features=2)
    # Output: (samples=6, timesteps=2, features=1)
    # ============================================================

    # Split data and confirm shapes and contents of first 2 samples
    split_sequences = SplitSequence(
        input_steps=3,
        output_steps=2,
        output_position=0,
        sequence_inputs=["sec0", "sec1"],
        sequence_outputs=["sec2"],
    )
    inputs, outputs = split_sequences.split(multivariate_xarray)
    assert inputs.shape == (6, 3, 2)
    assert outputs.shape == (6, 2, 1)

    np.testing.assert_array_equal(
        inputs.to_numpy()[0, :, :],
        np.array([[10, 15], [20, 25], [30, 35]]),
    )
    np.testing.assert_array_equal(
        outputs.to_numpy()[0, :, :],
        np.array([[65], [85]]),
    )
    np.testing.assert_array_equal(
        inputs.to_numpy()[1, :, :],
        np.array([[20, 25], [30, 35], [40, 45]]),
    )
    np.testing.assert_array_equal(
        outputs.to_numpy()[1, :, :],
        np.array([[85], [105]]),
    )

    # ============================================================
    # Multiple Parallel Input and Multi-Step Output
    # Input: (samples=5, timesteps=3, features=3)
    # Output: (samples=5, timesteps=2, features=3)
    # ============================================================

    # Split data and confirm shapes and contents of first 2 samples
    split_sequences = SplitSequence(
        input_steps=3,
        output_steps=2,
        output_position=1,
        sequence_inputs=["sec0", "sec1", "sec2"],
        sequence_outputs=["sec0", "sec1", "sec2"],
    )
    inputs, outputs = split_sequences.split(multivariate_xarray)
    assert inputs.shape == (5, 3, 3)
    assert outputs.shape == (5, 2, 3)

    np.testing.assert_array_equal(
        inputs.to_numpy()[0, :, :],
        np.array([[10, 15, 25], [20, 25, 45], [30, 35, 65]]),
    )
    np.testing.assert_array_equal(
        outputs.to_numpy()[0, :, :],
        np.array([[40, 45, 85], [50, 55, 105]]),
    )
    np.testing.assert_array_equal(
        inputs.to_numpy()[1, :, :],
        np.array([[20, 25, 45], [30, 35, 65], [40, 45, 85]]),
    )
    np.testing.assert_array_equal(
        outputs.to_numpy()[1, :, :],
        np.array([[50, 55, 105], [60, 65, 125]]),
    )

    # ============================================================
    # LOCA Data Test (40 features, 1 sequential output)
    # Input: (samples=2001, timesteps=396, features=3)
    # Output: (samples=2001, timesteps=396, features=1)
    # ============================================================
    nominal_data, perturbed_data = load_loca(stack_series=False)

    # Split data
    split_sequences = SplitSequence(
        input_steps=4,
        output_steps=1,
        output_position=1,
        sequence_inputs=[
            "Pellet Cladding Temperature",
            "Core Pressure",
            "Water Level",
            "Break Flow Rate",
        ],
        sequence_outputs=[
            "Pellet Cladding Temperature",
            "Core Pressure",
            "Water Level",
            "Break Flow Rate",
        ],
        const_inputs=nominal_data.coords["features"].values[:-4],
    )
    nominal_inputs, nominal_outputs = split_sequences.split(nominal_data)
    assert nominal_data.shape == (1, 400, 44)
    assert nominal_inputs.shape == (1, 396, 56)
    assert nominal_outputs.shape == (1, 396, 4)

    perturbed_inputs, perturbed_outputs = split_sequences.split(perturbed_data)
    assert perturbed_data.shape == (2000, 400, 44)
    assert perturbed_inputs.shape == (2000, 396, 56)
    assert perturbed_outputs.shape == (2000, 396, 4)


# ================================================================
def test_train_test_split():
    # Init settings
    settings = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        verbosity=1,
        random_state=42,
        cuda_visible_devices="-1",  # Use CPU only
    )

    # Test train_test_split for 2D data without scaling
    # (multi-input, single output)
    _, inputs, outputs = load_xs()
    xtrain, xtest, ytrain, ytest = train_test_split(
        data=[inputs, outputs], test_size=0.3
    )

    # Assert dimensions and size after split
    assert xtrain.shape == (700, 8)
    assert ytrain.shape == (700, 1)
    assert xtest.shape == (300, 8)
    assert ytest.shape == (300, 1)

    # Assert values did not change
    recomb_inputs = xtrain.combine_first(xtest)
    recomb_outputs = ytrain.combine_first(ytest)
    np.testing.assert_array_equal(
        recomb_inputs.sortby(variables=recomb_inputs.dims[0]).to_numpy(),
        inputs.to_numpy(),
    )
    np.testing.assert_array_equal(
        recomb_outputs.sortby(variables=recomb_inputs.dims[0]).to_numpy(),
        outputs.to_numpy(),
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

    # Split data and confirm shapes and contents of first 2 samples
    split_sequences = SplitSequence(
        input_steps=3,
        output_steps=2,
        output_position=0,
        sequence_inputs=["sec0"],
        sequence_outputs=["sec1", "sec2"],
    )
    inputs, outputs = split_sequences.split(multivariate_xarray)

    # Train/test split
    xtrain, xtest, ytrain, ytest = train_test_split(
        data=[inputs, outputs], test_size=0.3
    )

    # Scale data
    xtrain, xtest, xscaler = scale_data(xtrain, xtest, MinMaxScaler())
    ytrain, ytest, yscaler = scale_data(ytrain, ytest, MinMaxScaler())

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
    recomb_inputs = xtrain.combine_first(xtest)
    recomb_outputs = ytrain.combine_first(ytest)
    np.testing.assert_array_equal(
        xscaler.inverse_transform(
            recomb_inputs.sortby(variables=recomb_inputs.dims[0])
            .to_numpy()
            .reshape(-1, recomb_inputs.shape[-1])
        ),
        inputs.to_numpy().reshape(-1, inputs.shape[-1]),
    )
    np.testing.assert_array_equal(
        yscaler.inverse_transform(
            recomb_outputs.sortby(variables=recomb_inputs.dims[0])
            .to_numpy()
            .reshape(-1, recomb_outputs.shape[-1])
        ),
        outputs.to_numpy().reshape(-1, outputs.shape[-1]),
    )


# ================================================================
def test_correlation_matrix():
    # Init settings
    settings = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        verbosity=1,
        random_state=42,
        cuda_visible_devices="-1",  # Use CPU only
    )
    data, _, _ = load_xs()

    # Make sure correlation matrix runs
    correlation_matrix(data)
