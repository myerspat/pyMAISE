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

   - `Xarray <https://docs.xarray.dev/en/stable/index.html>`_

   - `scikit-learn <https://scikit-learn.org/stable/index.html>`_

   - `scikit-optimize <https://scikit-optimize.github.io/stable/>`_

   - `Keras <https://keras.io>`_

   - `KerasTuner <https://keras.io/keras_tuner/>_`

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

This approach only supports released versions of pyMAISE. Install from source for latest updates.

-----------
From Source
-----------

To install pyMAISE you can clone the repository. Make sure ``git`` is installed and run

.. code-block:: sh

   git clone https://github.com/myerspat/pyMAISE.git

In the ``pyMAISE`` directory you can run 

.. code-block:: sh

   pip3 install .

to install the library. For developers, we recommend you add the ``-e`` option:

.. code-block:: sh

   pip3 install -e .
