# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("/home/mrmn/moldovang/"))
sys.path.insert(1, os.path.abspath("/home/mrmn/moldovang/styleganpnria"))
sys.path.insert(2, os.path.abspath("/home/mrmn/moldovang/styleganpnria/gan"))

# sys.path.insert(0, os.path.abspath('/home/mrmn/moldovang/styleganpnria/gan'))
project = "styleGAN PNRIA"
copyright = "2024, Clement BROCHET, Gabriel MOLDOVAN, Julien RABAULT, Cyril REGAN"
author = "Clement BROCHET, Gabriel MOLDOVAN, Julien RABAULT, Cyril REGAN"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
