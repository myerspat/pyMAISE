from setuptools import setup, find_packages

VERSION = "0.0.2"
DESCRIPTION = "Michigan Artificial Intelligance Standard Environment"
LONG_DESCRIPTION = "Machine learning and artificial intelligance benchmarking library for nuclear engineering applications."

setup(
    name="pyMAISE",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(include=["pyMAISE", "pyMAISE.*"]),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "scikit-optimize",
        "keras>=2.12.0",
        "tensorflow>=2.12.0",
        "keras-tuner",
        "scikeras",
        "matplotlib",
    ],
    extras_require={"test": ["pytest"], "stats": ["scipy"]},
    package_data={"pyMAISE.data": ["*.csv"]},
    author="Patrick Myers",
    author_email="myerspat@umich.edu",
)
