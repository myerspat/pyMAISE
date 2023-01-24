from pyMAISE.preprocessing import PreProcesser
from pkg_resources import resource_filename

# Get full pyMAISE data file path
def get_full_path(path: str):
    return resource_filename("pyMAISE", path)


# Load benchmark BWR cross section data
def load_xs():
    return PreProcesser(get_full_path("data/xs.csv"), slice(0, -1), slice(-1))


# Load benchmark BWR control rod positions
def load_rod_positions():
    return PreProcesser(
        get_full_path("data/rod_positions.csv"),
        slice(0, 6),
        slice(6, 29),
    )
