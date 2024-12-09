.. _install-page-label:

Installation guide
==================

Requirements
------------


The requirements are the following:

- ``python>=3.6``
- ``torch>=1.7.0``
- ``numpy>=1.18``
- ``opt_einsum`` (https://pypi.org/project/opt-einsum/)

The GPU (if available) version of pytorch is recommended to be installed. Read the `official installation guide <https://pytorch.org/get-started/locally/>`_  for further info.
 

Using pip
---------


You can install the package using the `pip` command: 

:: 

    pip install torchTT

The latest github version can be installed using:

::

    pip install git+https://github.com/ion-g-ion/torchTT

One can also clone the repository and manually install the package: 

::

    git clone https://github.com/ion-g-ion/torchTT
    cd torchtt
    pip install .


Tests 
-----

The directory `tests/ <tests/>`_ from the root folder contains all the `unittests`. To run them use the command:

::

    pytest tests/

