# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CompSim-PINN'
copyright = '2025, Institute for Mathematics and Computer-Based Simulation (IMCS)'
author = 'Simon Völkl, Tarik Sahin, Daniel Wolff, Max von Danwitz, Alexander Popp'
release = '2025.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os, sys
sys.path.insert(0, os.path.abspath('../..'))

autodoc_mock_imports = ["torch", "deepxde"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",        # Google/NumPy style docstrings
    "sphinx.ext.autosummary",     # summary tables + stubs
    "sphinx.ext.viewcode",        # link to highlighted source
    "sphinx.ext.intersphinx",     # link to other documentations
    "myst_parser",                # Markdown parser
]

autosummary_generate = True             # auto-generate stub pages
autosummary_generate_overwrite = True   # overwrite auo-generated files on new run
autosummary_imported_members = True     # include objects re-exported from submodules
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", "https://docs.python.org/3/objects.inv"),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "https://github.com/GPflow/tensorflow-intersphinx/raw/master/tf2_py_objects.inv",
    ),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    'github_user': 'imcs-compsim',
    'github_repo': 'pinns_for_comp_mech',
    'github_banner': 'true',
}
