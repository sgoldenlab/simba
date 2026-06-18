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

myst_enable_extensions = ["html_image", "colon_fence", "deflist"]
myst_url_schemes = ["http", "https"]
myst_heading_anchors = 3  # generate anchors so in-page links (#step-1-...) resolve
extensions = ['sphinx.ext.napoleon',
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
              'sphinxcontrib.youtube',
              'myst_parser',
              'sphinx_reredirects']

# Math is rendered by a single renderer: MathJax (served locally via sphinx-mathjax-offline).
# Do not set mathjax_path here — it would override the offline bundle and re-introduce a CDN dependency.

html_favicon = "../simba/assets/icons/readthedocs_logo.png"
html_logo = "../simba/assets/icons/readthedocs_logo.png"
latex_engine = 'xelatex'
latex_elements = {'papersize': 'letterpaper'}


source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
nbsphinx_execute = 'never'
templates_path = ['_templates']
pygments_style = 'sphinx'
# Markdown migration is phased. Enabling the .md suffix would otherwise build every
# stray Markdown file in the tree (legacy copies, drafts, sandbox, GitHub-only docs).
# Only tutorials listed in MIGRATED_MD are published as .md (at repo-root depth);
# every other root-level .md stays excluded. Add to this set as each tutorial is
# migrated from tutorials_rst/*.rst.
import glob as _glob
MIGRATED_MD = {
    'Scenario1.md', 'Scenario2.md', 'Scenario3.md', 'Scenario4.md',
    'FSTTC.md', 'SHAP.md', 'anchored_rois.md', 'classifier_validation.md',
    'cue_light_tutorial.md', 'directionality_between_animals.md', 'feature_subsets.md',
    'kleinberg_filter.md', 'Multi_animal_pose.md', 'FAQ.md',
}
_unmigrated_md = [f for f in _glob.glob('*.md') if f not in MIGRATED_MD]

exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    'legacy', 'md', 'sandbox', 'nb/.ipynb_checkpoints', '_pilot*',
] + _unmigrated_md



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Markdown tutorials are published at the repo-root depth their relative paths assume
# (e.g. Scenario1.md -> /Scenario1.html). Redirect the legacy rST URLs so existing
# links (and search-engine results) keep working.  Target is relative to the old page.
redirects = {
    "tutorials_rst/scenario_1": "../Scenario1.html",
    "tutorials_rst/scenario_2": "../Scenario2.html",
    "tutorials_rst/scenario_3": "../Scenario3.html",
    "tutorials_rst/scenario_4": "../Scenario4.html",
    "tutorials_rst/FSTTC": "../FSTTC.html",
    "tutorials_rst/SHAP": "../SHAP.html",
    "tutorials_rst/anchored_rois": "../anchored_rois.html",
    "tutorials_rst/classifier_validation": "../classifier_validation.html",
    "tutorials_rst/cue_lights": "../cue_light_tutorial.html",
    "tutorials_rst/directionality_between_animals": "../directionality_between_animals.html",
    "tutorials_rst/feature_subsets": "../feature_subsets.html",
    "tutorials_rst/kleinberg_filter": "../kleinberg_filter.html",
    "tutorials_rst/multi_animal_pose": "../Multi_animal_pose.html",
    "tutorials_rst/FAQ": "../FAQ.html",
}
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

# Mock imports for modules that may not be available during documentation build
autodoc_mock_imports = ['torch', 'ultralytics', 'cupy']

# Hide type hints from signatures; the :param TYPE name: fields already document types.
autodoc_typehints = 'none'


def setup(app):
    """Copy the docs/images/ tree verbatim into <outdir>/images/ after an HTML build.

    Markdown tutorials embed media as raw ``<img>`` / ``<video src="images/...">``
    tags, which Sphinx does not track or copy. ``html_extra_path`` can't be used here
    because it copies a directory's *contents* to the output root (flattening the
    ``images/`` prefix the tags rely on). Copying explicitly preserves the structure
    and works identically on Read the Docs.
    """
    from sphinx.util.fileutil import copy_asset

    def _copy_images(app, exception):
        if exception is None and app.builder.name == 'html':
            src = os.path.join(app.srcdir, 'images')
            if os.path.isdir(src):
                copy_asset(src, os.path.join(app.outdir, 'images'))

    app.connect('build-finished', _copy_images)