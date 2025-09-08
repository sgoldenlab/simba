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
copyright = '2025, sronilsson'
author = 'sronilsson'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

myst_enable_extensions = ["html_image"]
myst_url_schemes = ["http", "https"]
extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.imgmath',
              'sphinx.ext.mathjax',
              'sphinx-mathjax-offline',
              'sphinx.ext.autodoc',
              'sphinx.ext.todo',
              'sphinx.ext.viewcode',
              'sphinxemoji.sphinxemoji',
              #'sphinx_autodoc_typehints',
              'sphinx_togglebutton',
              'nbsphinx',
              'sphinx.ext.intersphinx',
              'sphinxcontrib.video',
              'sphinx.ext.autosummary',
              'sphinxcontrib.youtube']

#mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

html_favicon = "../simba/assets/icons/readthedocs_logo.png"
html_logo = "../simba/assets/icons/readthedocs_logo.png"
latex_engine = 'xelatex'
latex_elements = {'papersize': 'letterpaper'}


source_suffix = ['.rst']
nbsphinx_execute = 'never'
templates_path = ['_templates']
pygments_style = 'sphinx'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['css/simba_theme.css',  # Include your existing CSS file
                  'custom.css']  # Include your additional CSS file


html_js_files = [
    "https://www.googletagmanager.com/gtag/js?id=G-PEKR9R5J47",
    "custom.js"
]


intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'shapely': ('https://shapely.readthedocs.io/en/stable/', None),
}