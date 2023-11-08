from pkg_resources import resource_filename

from pyMAISE.preprocessing import PreProcessor


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
