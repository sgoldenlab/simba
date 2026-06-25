# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import datetime
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

project = 'SimBA'
copyright = f'{datetime.datetime.now().year}, sronilsson'
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
              'sphinx.ext.coverage',
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
html_show_sphinx = False          # drop the "Built with Sphinx using a theme provided by Read the Docs." footer credit

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
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
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


def _generate_github_contributors(app):
    """Fetch GitHub contributor commit counts and (re)generate the credits bar chart and
    avatar grid on every build.

    Runs at ``builder-inited`` (before sources are read) so the generated files exist when
    ``credits.rst`` is processed. On any failure (no network, API rate-limit, missing
    matplotlib) it prints a notice and keeps the committed snapshot, so the build never breaks.
    Set a ``GITHUB_TOKEN`` env var to raise the API rate limit on shared CI IPs.
    """
    import json
    import urllib.request

    here = os.path.dirname(os.path.abspath(__file__))
    gen_dir = os.path.join(here, '_generated')
    html_path = os.path.join(gen_dir, 'github_contributors.html')
    os.makedirs(gen_dir, exist_ok=True)

    try:
        headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'simba-docs'}
        token = os.environ.get('GITHUB_TOKEN')
        if token:
            headers['Authorization'] = f'Bearer {token}'
        req = urllib.request.Request(
            'https://api.github.com/repos/sgoldenlab/simba/contributors?per_page=100&anon=0',
            headers=headers)
        with urllib.request.urlopen(req, timeout=25) as resp:
            raw = json.load(resp)
        contribs = sorted(((c['login'], int(c['contributions'])) for c in raw if c.get('type') == 'User'),
                          key=lambda t: t[1], reverse=True)
        if not contribs:
            raise RuntimeError('no contributors returned')
    except Exception as e:  # noqa: BLE001 - never fail the build over this
        print(f'[credits] GitHub contributor fetch failed ({e!r}); keeping existing snapshot.')
        return

    stamp = datetime.datetime.now().strftime('%B %Y')

    # ---- per-contributor lines changed (additions/deletions), computed from local git ----
    # GitHub's stats/contributors REST endpoint is computed asynchronously and keeps returning
    # HTTP 202 (empty body) for minutes on a cold cache, so it never resolves within a CI build
    # window. The docs always build inside a clone of the repo, so we aggregate `git log
    # --numstat` locally instead: reliable, instant, no API/rate-limit. A single person commits
    # under several name/email identities, so identities are folded onto the API login set via
    # the noreply-email username, an exact name==login match, and a small alias map for the rest.
    import re
    import subprocess
    loc = {}
    try:
        repo_root = os.path.dirname(here)
        login_by_lower = {login.lower(): login for login, _ in contribs}
        # git author name (lowercased) -> GitHub login, for identities that are neither a
        # noreply email nor a literal login. Extend if a new contributor's bar shows no LOC.
        aliases = {
            'simon nilsson': 'sronilsson', 'simon': 'sronilsson',
            'jia jie choong': 'inoejj',
            'aasiya islam': 'aasiya-islam',
            'tzuk': 'tzukpolinsky', 'tzuk polinsky': 'tzukpolinsky',
            'jens schweihoff': 'JensBlack',
            'justin shenk': 'justinshenk',
            'nastacia l. goodwin': 'goodwinnastacia', 'nastacia goodwin': 'goodwinnastacia',
            "thomas o'shea-wheller": 'Toshea111',
        }
        noreply_re = re.compile(r'(?:\d+\+)?([^@]+)@users\.noreply\.github\.com$', re.I)

        def _resolve(email, name):
            m = noreply_re.match((email or '').strip())
            if m and m.group(1).lower() in login_by_lower:
                return login_by_lower[m.group(1).lower()]
            nl = (name or '').strip().lower()
            return login_by_lower.get(nl) or aliases.get(nl)

        out = subprocess.run(
            ['git', 'log', '--no-merges', '--numstat', '--format=COMMIT\t%ae\t%an'],
            cwd=repo_root, capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=60).stdout
        cur = None
        for line in out.splitlines():
            if line.startswith('COMMIT\t'):
                _, email, name = line.split('\t', 2)
                cur = _resolve(email, name)
                continue
            if cur is None or not line.strip():
                continue
            a, d, _rest = (line.split('\t', 2) + ['', '', ''])[:3]
            if a.isdigit() and d.isdigit():
                pa, pd = loc.get(cur, (0, 0))
                loc[cur] = (pa + int(a), pd + int(d))
        if not loc:
            print('[credits] no local git LOC resolved (shallow clone?); showing commits only.')
    except Exception as e:  # noqa: BLE001
        print(f'[credits] local git line-count failed ({e!r}); showing commits only.')

    # ---- responsive HTML bar chart with a GitHub card popover on hover (no image, no matplotlib) ----
    # Consumed by credits.rst via a `.. raw:: html :file:` include; regenerated every build.
    try:
        import html as _htmllib
        import math
        INK, MUT, TRACK, BLUE, LEAD = '#23272e', '#6b7280', '#eef2f7', '#2a7fb8', '#21567a'
        mx = max(c for _, c in contribs)

        def _w(v):  # log-scaled bar width (%), with a floor so 1-commit bars stay visible
            return 100.0 if mx <= 1 else round(max(5.0, math.log10(v) / math.log10(mx) * 100.0), 1)

        def _fmt(n):  # compact line count: 1.2M / 162K / 1.5K / 405
            if n >= 1_000_000:
                return f'{n / 1_000_000:.1f}M'
            if n >= 10_000:
                return f'{round(n / 1000)}K'
            if n >= 1000:
                return f'{n / 1000:.1f}K'
            return str(int(n))

        style = (
            "<style>\n"
            ".simba-cc{max-width:820px;margin:10px auto 4px;padding-top:74px;}\n"
            ".simba-cc-row{display:flex;align-items:center;gap:10px;margin:5px 0;position:relative;text-decoration:none!important;}\n"
            ".simba-cc-name{flex:0 0 140px;text-align:right;font-size:13px;color:#23272e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}\n"
            ".simba-cc-track{flex:1;background:#eef2f7;border-radius:5px;height:22px;display:block;}\n"
            ".simba-cc-bar{display:block;height:100%;border-radius:5px;transition:filter .12s ease;}\n"
            ".simba-cc-row:hover .simba-cc-bar{filter:brightness(1.08);}\n"
            ".simba-cc-count{flex:0 0 52px;font-size:12px;font-weight:700;color:#23272e;}\n"
            ".simba-cc-loc{flex:0 0 86px;font-size:11.5px;color:#6b7280;white-space:nowrap;}\n"
            ".simba-cc-card{position:absolute;left:150px;bottom:24px;z-index:40;display:none;align-items:center;gap:11px;"
            "background:#fff;border:1px solid #e2e6ec;border-radius:12px;box-shadow:0 8px 24px rgba(33,86,122,.20);"
            "padding:9px 16px 9px 9px;white-space:nowrap;pointer-events:none;}\n"
            ".simba-cc-card::after{content:'';position:absolute;left:26px;bottom:-7px;width:13px;height:13px;background:#fff;"
            "border-right:1px solid #e2e6ec;border-bottom:1px solid #e2e6ec;transform:rotate(45deg);}\n"
            ".simba-cc-card img{width:50px;height:50px;border-radius:50%;display:block;}\n"
            ".simba-cc-card .cci{display:flex;flex-direction:column;line-height:1.3;}\n"
            ".simba-cc-card .cci b{font-size:13.5px;color:#23272e;}\n"
            ".simba-cc-card .cci em{font-size:12px;color:#6b7280;font-style:normal;}\n"
            ".simba-cc-row:hover .simba-cc-card{display:flex;}\n"
            "</style>\n")

        rows = []
        for i, (login, c) in enumerate(contribs):
            le = _htmllib.escape(login)
            col = LEAD if i == 0 else BLUE
            commit_lbl = f"{c:,} commit" + ('' if c == 1 else 's')
            ad = loc.get(login)
            loc_lbl = (_fmt(ad[0] + ad[1]) + ' loc') if ad else ''
            loc_card = f' · +{_fmt(ad[0])} / −{_fmt(ad[1])} lines' if ad else ''
            rows.append(
                f'     <a class="simba-cc-row" href="https://github.com/{le}">'
                f'<span class="simba-cc-name">@{le}</span>'
                f'<span class="simba-cc-track"><span class="simba-cc-bar" style="width:{_w(c)}%;background:{col};"></span></span>'
                f'<span class="simba-cc-count">{c:,}</span>'
                f'<span class="simba-cc-loc">{loc_lbl}</span>'
                f'<span class="simba-cc-card"><img src="https://github.com/{le}.png?size=96" alt="@{le}" loading="lazy">'
                f'<span class="cci"><b>@{le}</b><em>{commit_lbl}{loc_card}</em></span></span>'
                f'</a>')
        chart = (style + '<div class="simba-cc">\n' + '\n'.join(rows) +
                 f'\n     <p style="text-align:right;font-size:11px;color:{MUT};font-style:italic;margin:10px 4px 0;">'
                 f'bar &amp; bold number = commits (log scale) · loc = lines changed (added + removed) · hover for details · {stamp}</p>\n   </div>')

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(chart + '\n')
    except Exception as e:  # noqa: BLE001
        print(f'[credits] contributor HTML render failed ({e!r}); keeping existing snapshot.')


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

    app.connect('builder-inited', _generate_github_contributors)
    app.connect('build-finished', _copy_images)
    app.connect('build-finished', _inject_lazy_media)
    app.connect('doctree-read', _convert_github_alerts)
    app.connect('doctree-read', _autolink_glossary_terms)