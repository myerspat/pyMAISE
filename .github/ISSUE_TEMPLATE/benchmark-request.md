---
name: Benchmark Request
about: Propose a benchmark
title: ''
labels: ''
assignees: ''

---

## Data
Please discuss the data you would like to incorporate into pyMAISE. If this data has already been published, then please link the paper.

## Tasks
- [ ] Add the data to `pyMAISE/datasets/`. Please respect the GitHub large file limit. The hosting of data locally is temporary until we find a permanent location.
- [ ] Add a load function to `pyMAISE/datasets/_handler.py` and include a data description in the docstring. The load function should return Xarray data structures.
- [ ] Create a Jupyter notebook for the benchmark in `docs/source/benchmarks/`.
- [ ] Add the relative path to the notebook to `docs/source/benchmarks.rst`.
- [ ] If a reference for the data set exists, then add the BibTeX citation to `docs/source/data_refs.bib`

## Definition of Done
All tasks above are completed, a pull request is opened, reviewers are satisfied, and the code is merged into develop.
