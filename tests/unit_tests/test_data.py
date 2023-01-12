from pandas import DataFrame
import pytest
import pyMAISE as mai

# Unit test for loading xs data from xs.csv
def test_load_xs():
    # Load settings
    mai.settings.init()

    # Load data through PreProcessor
    xs = mai.load_xs()._data

    # Type assertion
    assert isinstance(xs, DataFrame)

    # Size assertions
    assert xs.shape[0] == 1000
    assert xs.shape[1] == 9

    # Element assertions
    assert xs.iloc[0, 0] == 0.00644620
    assert xs.iloc[0, -1] == 1.256376
    assert xs.iloc[-1, 0] == 0.00627230
    assert xs.iloc[-1, -1] == 1.240064

    # Header assertions
    assert xs.columns.size == 9
    assert xs.columns[-1] == "k"
