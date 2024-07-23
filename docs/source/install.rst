:orphan:

Installation
============

Dependencies
------------

* ``mne`` (>=1.6)
* ``numpy`` (>=1.21)
* ``scipy`` (>=1.4.0)


We require that you use Python 3.9 or higher.
You may choose to install ``sleep-semantic-segmentation`` `via pip <#Installation via pip>`_,
or conda.

Installation via Conda
----------------------

To install Sleep-Semantic-Segmentation using conda in a virtual environment,
simply run the following at the root of the repository:

.. code-block:: bash

   # with python>=3.9 at least
   conda create -n sssm
   conda activate sssm
   conda install -c conda-forge sssm


Installation via Pip
--------------------

To install Sleep-Semantic-Segmentation including all dependencies required to use all features,
simply run the following at the root of the repository:

.. code-block:: bash

    python -m venv .venv
    pip install -U sssm

If you want to install a snapshot of the current development version, run:

.. code-block:: bash

   pip install --user -U https://api.github.com/repos/mne-tools/mne-connectivity/zipball/main

To check if everything worked fine, the following command should not give any
error messages:

.. code-block:: bash

   python -c 'import sssm'

sleep-semantic-segmentation works best with the latest stable release of SSSM. To ensure
SSSM is up-to-date, run:

.. code-block:: bash

   pip install --user -U sssm
