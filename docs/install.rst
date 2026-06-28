.. _install-page-label:

Installation guide
==================

Requirements
------------


The requirements are the following:

- ``python>=3.8``
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
    cd torchTT
    pip install .


Using uv
--------

You can install the package using `uv`:

::

    uv pip install torchTT

The latest github version can be installed using:

::

    uv pip install git+https://github.com/ion-g-ion/torchTT

One can also clone the repository and install the package using `uv`:

::

    git clone https://github.com/ion-g-ion/torchTT
    cd torchTT
    uv sync

Or install in editable mode:

::

    uv pip install -e .


Development Installation
------------------------

For development, you may want to install the package with additional development dependencies (pytest, sphinx, ipykernel, matplotlib):

**Using pip:**

::

    pip install -e ".[dev]"

**Using uv:**

::

    uv sync --extra dev

or

::

    uv pip install -e ".[dev]"

This will install the package in editable mode along with all development tools needed for testing, building documentation, and working with Jupyter notebooks.


Tests 
-----

The directory `tests/ <tests/>`_ from the root folder contains all the `unittests`. To run them use the command:

::

    pytest tests/


Building Documentation
----------------------

To build the documentation locally, you need:

1. **Install development dependencies** (see Development Installation above)

2. **Install pandoc** (required for rendering Jupyter notebooks in the documentation):

   - Ubuntu/Debian: ``sudo apt install pandoc``
   - macOS: ``brew install pandoc``
   - Windows: ``choco install pandoc`` or download from https://pandoc.org/installing.html

3. **Build the documentation:**

   ::

       make html

The generated documentation will be in ``_build/html/``.

