==============================================================
pyMAISE: Michigan Artificial Intelligence Standard Environment
==============================================================

pyMAISE is an artificial intelligence (AI) and machine learning (ML) benchmarking library for nuclear reactor applications. It offers to streamline the building, tuning, and comparison of various ML models for user-provided data sets. Additionally, pyMAISE offers benchmarked data sets with example Jupyter Notebooks for AI/ML comparison. Current ML algorithm support includes

- linear regression,
- lasso regression,
- logistic regression,
- decision tree regression and classification,
- support vector regression and classification,
- random forest regression and classification,
- k-nearest neighbors regression and classification,
- sequential neural networks.

These models are built using `scikit-learn <https://scikit-learn.org/stable/index.html>`_ and `Keras <https://keras.io>`_ :cite:`scikit-learn, chollet2015keras`. For more information such as installation, examples, and use, refer to the sections below.

--------
Contents
--------

.. toctree::
   :maxdepth: 2   
   
   installation
   user_guide
   dev_guide
   models
   pymaise_api
   examples/index
   license

.. _data_refs:
---------------
Data References
---------------
.. bibliography:: data_refs.bib
   :all:

-------------------
Software References
-------------------
.. bibliography:: software_refs.bib
   :all:
