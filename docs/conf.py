# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

project = 'SimBA'
copyright = '2023, sronilsson'
author = 'sronilsson'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

myst_enable_extensions = ["html_image"]
myst_url_schemes = ["http", "https"]
extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx.ext.todo',
              'sphinx.ext.viewcode',
              #'sphinx_autodoc_typehints',
              'nbsphinx',
              'sphinx.ext.intersphinx']
intersphinx_mapping = {
  'python': ('https://docs.python.org/3', None),
  }
html_favicon = "../simba/assets/icons/SimBA_logo.png"
html_logo = "../simba/assets/icons/SimBA_logo.png"
latex_engine = 'xelatex'
latex_elements = {'papersize': 'a4paper')


# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.ipynb': 'nbsphinx',
# }


source_suffix = ['.rst']
nbsphinx_execute = 'never'
templates_path = ['_templates']
pygments_style = 'sphinx'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_style = 'css/simba_theme.css'
