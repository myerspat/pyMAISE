============
Installation
============

.. _prerequisites:

-------------
Prerequisites
-------------

.. admonition:: Required
   :class: error

   - `NumPy <https://numpy.org/>`_
   
   - `pandas <https://pandas.pydata.org/>`_

   - `scikit-learn <https://scikit-learn.org/stable/index.html>`_

   - `scikit-optimize <https://scikit-optimize.github.io/stable/>`_

   - `Keras <https://keras.io>`_

   - `TensorFlow <https://tensorflow.org>`_

   - `SciKeras <https://adriangb.com/scikeras/stable/>`_

   - `Matplotlib <https://matplotlib.org/stable/>`_

.. admonition:: Optional
   :class: note

   - `pytest <https://docs.pytest/en/7.2.x/contents.html>`_

   - `SciPy <https://scipy.org>`_

---
Pip
---

Install through Pip by running

.. code-block:: sh

   pip3 install pyMAISE

-----------
From Source
-----------

To install pyMAISE you can clone the repository. Make sure ``git`` is installed and run

.. code-block:: sh

   git clone https://github.com/myerspat/pyMAISE.git

To get a specific version, checkout the version's branch,

.. code-block:: sh

   git checkout <major>.<minor>.<patch>

In the ``pyMAISE`` directory you can run, 

.. code-block:: sh

   pip3 install -e .

to install the library.
