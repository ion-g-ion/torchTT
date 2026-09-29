# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'torchtt'
copyright = '2023, Ion Gabriel Ion'
author = 'Ion Gabriel Ion'
try:  # keep the documented version in sync with the installed package
    from importlib.metadata import version as _pkg_version

    release = _pkg_version('torchTT')
except Exception:  # package not installed (e.g. plain `make html` in a checkout)
    release = '0.5.0'
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon', 'sphinx.ext.intersphinx', 'nbsphinx', 'sphinxcontrib.bibtex']

# sphinxcontrib-bibtex configuration: cite with :cite:p:`key` (listed in docs/references.rst)
bibtex_bibfiles = ['docs/references.bib']
bibtex_reference_style = 'author_year'

# nbsphinx configuration
nbsphinx_execute = 'never'  # Don't execute notebooks during build
# link every example notebook to Google Colab and to its source on GitHub
nbsphinx_prolog = r"""
.. note::

   Run this notebook in `Google Colab <https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/{{ env.docname }}.ipynb>`_ or see the source on `GitHub <https://github.com/ion-g-ion/torchTT/blob/main/{{ env.docname }}.ipynb>`_.
"""

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'https://github.com/ion-g-ion/torchTT/blob/main/logo_small.png?raw=true'
