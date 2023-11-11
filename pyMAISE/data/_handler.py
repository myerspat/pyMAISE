import copy

import numpy as np
import pandas as pd
import xarray as xr
from pkg_resources import resource_filename

from pyMAISE.preprocessor import PreProcessor


# Get full pyMAISE data file path
def get_full_path(path: str):
    return resource_filename("pyMAISE", path)


# Load benchmark BWR cross section data
def load_xs():
    preprocessor = PreProcessor()
    preprocessor.read_csv(get_full_path("data/xs.csv"), slice(0, -1), slice(-1, None))
    return preprocessor


# Load benchmark MIT reactor data
def load_MITR():
    preprocessor = PreProcessor()
    preprocessor.read_csv(
        [get_full_path("data/crx.csv"), get_full_path("data/powery.csv")],
    )
    return preprocessor


# Load benchmark fuel perfromance data
def load_fp():
    preprocessor = PreProcessor()
    preprocessor.read_csv(
        [get_full_path("data/fp_inp.csv"), get_full_path("data/fp_out.csv")]
    )
    return preprocessor


# Load benchmark fuel centerline temperature data
def load_heat():
    preprocessor = PreProcessor()
    preprocessor.read_csv(get_full_path("data/heat.csv"), slice(0, -1), slice(-1, None))
    return preprocessor


# Rod ejection accident data
def load_rea():
    preprocessor = PreProcessor()
    preprocessor.read_csv(
        [get_full_path("data/rea_inputs.csv"), get_full_path("data/rea_outputs.csv")],
    )
    return preprocessor


# Load BWR micro-reactor data
def load_BWR():
    preprocessor = PreProcessor()
    preprocessor.read_csv(
        [get_full_path("data/bwr_input.csv"), get_full_path("data/bwr_output.csv")],
    )
    return preprocessor


# Load HTGR micro reactor quadrant power data before preprocessing
def load_qpower():
    preprocessor = PreProcessor()
    preprocessor.read_csv(
        get_full_path("data/microreactor.csv"), slice(29, 37), slice(4, 8)
    )
    return preprocessor


# Load HTGR micro-reactor quadrant power data after preprocessing using symmetry conditions
def load_pqpower():
    preprocessor = PreProcessor()
    preprocessor.read_csv(
        get_full_path("data/microreactor_preprocessed.csv"), slice(1, 9), slice(9, 14)
    )
    return preprocessor


# Load and prep LOCA data
def load_loca():
    # Paths
    input_path = get_full_path("data/loca_inp.csv")
    output_path = get_full_path("data/loca_out.csv")

    raw_inputs = pd.read_csv(input_path)
    raw_outputs = pd.read_csv(output_path, header=None)

    # Reshape data and add outputs onto inputs
    outputs = raw_outputs.values.T[:, :, np.newaxis]
    inputs = np.concatenate(
        (
            np.repeat(
                raw_inputs.values[:, np.newaxis, :], raw_outputs.shape[0], axis=1
            ),
            outputs,
        ),
        axis=2,
    )

    #
    preprocessor = PreProcessor()
    preprocessor.data = xr.DataArray(
        inputs,
        coords={
            "samples": np.linspace(0, inputs.shape[0], inputs.shape[0]).astype(int),
            "timesteps": np.linspace(0, inputs.shape[1], inputs.shape[1]).astype(int),
            "features": list(raw_inputs.columns) + ["PCT"],
        },
    )
    preprocessor.inputs = copy.deepcopy(preprocessor.data)
    preprocessor.outputs = xr.DataArray(
        outputs,
        coords={
            "samples": np.linspace(0, outputs.shape[0], outputs.shape[0]).astype(int),
            "timesteps": np.linspace(0, outputs.shape[1], outputs.shape[1]).astype(int),
            "features": "PCT",
        },
    )

    return preprocessor
