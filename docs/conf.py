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
    'roi_tutorial_new_2025.md', 'Tools.md',
    'tutorial_process_videos.md', 'Pose_config.md', 'extractFeatures.md',
    'blob_track.md',
    'facemap_project.md', 'superanimal_topview_project.md', 'Visualizations.md',
    'yolo_train.md', 'yolo_inference.md', 'yolo_pose_plot.md',
    'spontaneous_alternation.md', 'light_dark_box.md', 'mutual_exclusivity_heuristic_rules.md',
    'validation_tutorial.md', 'cuml_simba.md',
    'third_party_annot_new.md', 'gpu_vs_cpu_video_processing_runtimes.md', 'blob_data_project_simba.md',
}
_unmigrated_md = [f for f in _glob.glob('*.md') if f not in MIGRATED_MD]

exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    'legacy', 'md', 'sandbox', 'nb/.ipynb_checkpoints', '_pilot*',
] + _unmigrated_md



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# The long Markdown tutorials have many emoji-prefixed Step headings; with the RTD
# defaults the sidebar expanded every one into a deep, hard-to-read wall. Constrain
# it: show the page + its top-level sections only, with [+] expanders for the rest.
html_theme_options = {
    # depth 2 = sidebar shows page TITLES only, never their internal sections. Higher
    # values flood the sidebar (and the tutorials landing page) with every heading,
    # producing an unreadable jumble. Sections are reachable via each page's own TOC.
    'navigation_depth': 2,
    'collapse_navigation': True,   # only the current path expands
    'sticky_navigation': True,
}
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
    "tutorials_rst/roi_tutorial_new_2025": "../roi_tutorial_new_2025.html",
    "tutorials_rst/tools": "../Tools.html",
    "tutorials_rst/process_videos": "../tutorial_process_videos.html",
    "tutorials_rst/create_user_defined_pose_config": "../Pose_config.html",
    "tutorials_rst/user_defined_feature_class": "../extractFeatures.html",
    "tutorials_rst/blob_tracking": "../blob_track.html",
    # stale duplicates removed from the toctree -> point at the real pages
    "tutorials_rst/roi": "../roi_tutorial_new_2025.html",
    "tutorials_rst/installation": "../installation.html",
    "tutorials_rst/third_party_annotations": "../third_party_annot_new.html",
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


# GitHub-style alerts (> [!NOTE], > [!WARNING], ...) are used throughout the
# Markdown tutorials. MyST 0.16.1 is too old to understand them (support landed in
# myst-parser 4.x, which needs a newer Python than the docs build pins), so it
# renders them as plain blockquotes with a literal "[!NOTE]" leaking through. This
# transform rewrites those blockquotes into proper admonition nodes at parse time,
# keeping the .md GitHub-native while rendering styled admonitions in Sphinx.
import re as _re
_GH_ALERT_RE = _re.compile(r'^\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*', _re.I)


def _convert_github_alerts(app, doctree):
    from docutils import nodes
    kinds = {'note': nodes.note, 'tip': nodes.tip, 'important': nodes.important,
             'warning': nodes.warning, 'caution': nodes.caution}
    for bq in list(doctree.findall(nodes.block_quote)):
        if not bq.children or not isinstance(bq.children[0], nodes.paragraph):
            continue
        para = bq.children[0]
        m = _GH_ALERT_RE.match(para.astext())
        if not m:
            continue
        # strip the leading "[!TYPE]" marker from the first text node of the paragraph
        if para.children and isinstance(para.children[0], nodes.Text):
            stripped = _GH_ALERT_RE.sub('', para.children[0].astext(), count=1)
            para.replace(para.children[0], nodes.Text(stripped))
        admonition = kinds[m.group(1).lower()]()
        admonition += bq.children
        bq.replace_self(admonition)


# Build-time lazy-loading of media (page-load speedup).
#
# These docs stack many large .webp/.png images and .webm/.mp4 demo clips per page;
# downloading the off-screen ones blocks first paint. We inject loading="lazy" +
# decoding="async" on every <img>, and preload="none" on every non-autoplay <video>,
# directly into the generated HTML. custom.js does the same client-side, but the
# browser's preload scanner can start fetching the first images before that script
# runs — doing it at build time guarantees the attributes are present in the served
# markup. Off-screen media then defers; in-viewport media (logo, hero) still loads
# immediately, so there is no LCP regression. Idempotent: the negative lookaheads
# skip tags that already carry loading=/preload=, so re-runs and the JS pass never
# double-apply.
_IMG_LAZY_RE = _re.compile(r'<img\b(?![^>]*\bloading=)([^>]*?)(\s*/?)>', _re.IGNORECASE)
_VIDEO_RE = _re.compile(r'<video\b([^>]*)>', _re.IGNORECASE)
_PRELOAD_TOKEN_RE = _re.compile(r'\bpreload\s*=\s*"?([\w-]+)"?', _re.IGNORECASE)


def _inject_lazy_media(app, exception):
    if exception is not None or getattr(app.builder, 'format', '') != 'html':
        return
    from pathlib import Path
    from sphinx.util import logging as _sphinx_logging
    logger = _sphinx_logging.getLogger(__name__)

    def _fix_img(m):
        attrs, close = m.group(1), m.group(2)
        add = ' loading="lazy"'
        if 'decoding=' not in attrs.lower():
            add += ' decoding="async"'
        return '<img' + attrs + add + close + '>'

    def _fix_video(m):
        # autoplay clips need their data, so never defer them. For the rest:
        # add preload="none" when absent, and downgrade the sphinxcontrib.video
        # default preload="auto" (full download on page load) to "none" so the
        # clip fetches nothing until the user presses play. An explicit
        # "metadata"/"none" the author chose is left untouched.
        attrs = m.group(1)
        if _re.search(r'\bautoplay\b', attrs, _re.IGNORECASE):
            return m.group(0)
        pm = _PRELOAD_TOKEN_RE.search(attrs)
        if pm is None:
            return '<video' + attrs + ' preload="none">'
        if pm.group(1).lower() == 'auto':
            return '<video' + attrs[:pm.start()] + 'preload="none"' + attrs[pm.end():] + '>'
        return m.group(0)

    changed = 0
    for html in Path(app.outdir).rglob('*.html'):
        text = html.read_text(encoding='utf-8')
        new = _VIDEO_RE.sub(_fix_video, _IMG_LAZY_RE.sub(_fix_img, text))
        if new != text:
            html.write_text(new, encoding='utf-8')
            changed += 1
    logger.info('lazy-media: injected loading/preload attributes into %d HTML files', changed)


# Auto-link glossary terms (see docs/glossary.rst) wherever they appear in prose,
# so jargon in docstrings/tutorials cross-references the glossary without authors
# hand-writing `:term:` everywhere. Curated to UNAMBIGUOUS terms only: multi-word
# phrases, acronyms/proper nouns, and distinctive single words. Deliberately
# excludes words that double as parameters/identifiers (feature, target, behavior,
# velocity, smoothing, geometry, p, F1, identity, precision, recall, validation,
# Observer, Solomon). Only the FIRST occurrence per page is linked, and matches
# inside code/literals/signatures/titles/existing refs are skipped — keeping the
# linking informative rather than noisy.
_GLOSSARY_TERMS = [
    "pose estimation", "convex hull", "bounding box", "random forest", "circular statistics",
    "Kleinberg smoothing", "burst detection", "egocentric alignment", "feature extraction",
    "outlier correction", "confusion matrix", "Gantt plot", "spontaneous alternation",
    "sliding window", "rolling window", "time bins", "machine results", "minimum bout length",
    "discrimination threshold", "probability threshold", "blob tracking", "contour tracking",
    "cross-validation", "feature importance", "pose confidence", "multi-animal tracking",
    "path plot", "aggregate statistics", "anchored ROI", "cue light", "severity scoring",
    "sequential analysis", "third-party annotation tool", "video info", "project config",
    "pixels per millimeter", "DeepLabCut", "SLEAP", "YOLO", "SHAP", "FSTTC", "CLAHE", "UMAP",
    "ROI", "FPS", "maDLC", "DLC", "BORIS", "Ethovision", "DeepEthogram", "BENTO",
    "SuperAnimal-TopView", "FaceMap", "AMBER", "px/mm", "ethogram", "occlusion", "centroid",
    "heatmap", "keypoint", "directionality", "interpolation", "classifier", "clustering",
]
_GLOSSARY_TERM_RE = _re.compile(
    r'(?<![\w-])(' + '|'.join(_re.escape(t) for t in sorted(set(_GLOSSARY_TERMS), key=len, reverse=True))
    + r')(s?)(?![\w-])', _re.IGNORECASE)


def _autolink_glossary_terms(app, doctree):
    from docutils import nodes
    from sphinx import addnodes
    if getattr(app.env, 'docname', None) == 'glossary':
        return  # the glossary defines the terms; don't self-link it
    SKIP = (nodes.literal, nodes.literal_block, nodes.reference, nodes.title,
            nodes.comment, addnodes.pending_xref, addnodes.desc_signature)
    try:
        doctree_block = nodes.doctest_block
        SKIP = SKIP + (doctree_block,)
    except AttributeError:
        pass
    linked = set()  # lowercased terms already linked on this page (first-occurrence only)
    for text_node in list(doctree.findall(nodes.Text)):
        anc = text_node.parent
        skip = False
        while anc is not None:
            if isinstance(anc, SKIP):
                skip = True
                break
            anc = anc.parent
        if skip:
            continue
        text = text_node.astext()
        match = None
        for m in _GLOSSARY_TERM_RE.finditer(text):
            if m.group(1).lower() not in linked:
                match = m
                break
        if match is None:
            continue
        term, plural = match.group(1), match.group(2)
        linked.add(term.lower())
        new_nodes = []
        if match.start():
            new_nodes.append(nodes.Text(text[:match.start()]))
        xref = addnodes.pending_xref('', refdomain='std', reftype='term',
                                     reftarget=term.lower(), refexplicit=False, refwarn=False)
        xref += nodes.inline(term + plural, term + plural, classes=['xref', 'std', 'std-term'])
        new_nodes.append(xref)
        if match.end() < len(text):
            new_nodes.append(nodes.Text(text[match.end():]))
        text_node.parent.replace(text_node, new_nodes)


def setup(app):
    """Build-time hooks for the Markdown tutorials.

    1. Copy the docs/images/ tree verbatim into <outdir>/images/ after an HTML build.
       Markdown tutorials embed media as raw ``<img>`` / ``<video src="images/...">``
       tags, which Sphinx does not track. ``html_extra_path`` can't be used because it
       copies a directory's *contents* to the output root (flattening the ``images/``
       prefix the tags rely on). Copying explicitly preserves the structure.
    2. Convert GitHub-style alerts to admonitions (see ``_convert_github_alerts``).
    3. Inject lazy-loading attributes into the generated HTML (see ``_inject_lazy_media``).
    """
    from sphinx.util.fileutil import copy_asset

    def _copy_images(app, exception):
        if exception is None and app.builder.name == 'html':
            src = os.path.join(app.srcdir, 'images')
            if os.path.isdir(src):
                copy_asset(src, os.path.join(app.outdir, 'images'))

    app.connect('build-finished', _copy_images)
    app.connect('build-finished', _inject_lazy_media)
    app.connect('doctree-read', _convert_github_alerts)
    app.connect('doctree-read', _autolink_glossary_terms)