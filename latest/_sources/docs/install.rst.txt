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
- ``scipy>=0.16``

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


Optional C++ extension (Linux and macOS)
----------------------------------------

Installation attempts to compile the C++ extension and falls back to Python if
compilation fails. Windows uses the Python implementation; the C++ extension is
unsupported there.

Optional build dependencies:

- A C++ compiler compatible with your PyTorch version. PyTorch selects the required
  language standard (C++17 in older releases, C++20 in newer releases).
- Python development headers and the platform's C/C++ headers, linker, and runtime. On
  Ubuntu/Debian: ``sudo apt-get install build-essential python3-dev``. On macOS:
  ``xcode-select --install`` (Apple Clang and the macOS SDK). For other Linux
  distributions, install the equivalent packages; their names differ.
- PyTorch supplies the LibTorch headers and shared libraries. No separate LibTorch,
  BLAS/LAPACK, OpenMP development package, or CUDA toolkit is needed for this extension
  when using PyTorch wheels. The compiler and system headers must be installed through
  your OS, not a pip extra.

To build against the PyTorch already installed in your environment, run from the cloned
repository:

.. code-block:: bash

    python -m pip install torch  # Or choose a CPU/CUDA wheel using the PyTorch guide above.
    python -m pip install setuptools setuptools-scm wheel ninja numpy opt_einsum scipy
    python -m pip install --no-build-isolation -v .
    python -c "import torchtt; print(torchtt.cpp_enabled())"  # True if the extension loads.

With uv, use ``uv pip install`` for the installation commands above. Set ``CXX`` to
select another compiler. Rebuild after changing PyTorch versions;
``--no-build-isolation`` keeps the build and runtime PyTorch versions aligned. Use
``--no-cache-dir --force-reinstall --no-deps`` on the final installation command when
rebuilding an existing installation.

To skip compilation on Linux/macOS: ``TORCHTT_NO_CPP=1 python -m pip install .`` (or
``uv pip install .``). CI checks extension builds with GCC on Ubuntu and Apple Clang on
macOS, plus Python-only installs on Linux, macOS, and Windows. Other toolchains are not
covered by CI.


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

