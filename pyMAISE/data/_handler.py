from pyMAISE.preprocessing import PreProcesser
from pkg_resources import resource_filename

# Load benchmark BWR cross section data
def load_xs():
    path = resource_filename("pyMAISE", "data/xs.csv")
    return PreProcesser(path, slice(0, -1), slice(-1))
