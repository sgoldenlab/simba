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


# ------------------------------------------------------------------ #
# Auto-build the landing-page demo list from every ".. video::"        #
# reference in the package docstrings and docs .rst, newest file       #
# first. The carousel and "explore all demos" wall (custom.js) read    #
# window.SIMBA_DEMOS_ALL and merge it after their curated openers, so   #
# every video embedded anywhere in the docs shows up in the showcase   #
# automatically -- no hand-maintained list to fall out of date.        #
# ------------------------------------------------------------------ #
import re as _re_vid, json as _json_vid, pathlib as _pl_vid


def _build_demo_manifest():
    here = _pl_vid.Path(__file__).resolve().parent
    img = here / "_static" / "img"
    if not img.is_dir():
        return
    existing = {p.name: p.stat().st_mtime for p in img.iterdir()
                if p.suffix.lower() in (".webm", ".mp4")}
    pat = _re_vid.compile(r'\.\.\s*video::\s*_static/img/([A-Za-z0-9_./-]+\.(?:webm|mp4))', _re_vid.I)
    refs = set()
    for root in (here.parent / "simba", here):          # package docstrings + docs .rst
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if f.suffix.lower() not in (".py", ".rst") or not f.is_file():
                continue
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for m in pat.finditer(txt):
                refs.add(m.group(1).split("/")[-1])
    names = [n for n in refs if n in existing]          # only videos whose file actually exists
    names.sort(key=lambda n: existing[n], reverse=True)  # newest file first
    (here / "_static" / "demo_manifest.js").write_text(
        "window.SIMBA_DEMOS_ALL = " + _json_vid.dumps(names) + ";\n", encoding="utf-8")


try:
    _build_demo_manifest()
except Exception as _e_vid:
    print("demo_manifest generation skipped:", _e_vid)


html_js_files = [
    "https://www.googletagmanager.com/gtag/js?id=G-PEKR9R5J47",
    "demo_manifest.js",   # generated above; defines window.SIMBA_DEMOS_ALL (must load before custom.js)
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
    SINCE = '2020-12-01'   # SimBA has been maintained by Simon since Dec 2020; only count contributions from then on

    # ---- per-contributor lines-of-code changed + commits SINCE 2020-12, computed from local git ----
    # LOC is the headline metric (linear bar). GitHub's stats/contributors REST endpoint returns
    # HTTP 202 for minutes on a cold cache, so we aggregate `git log --numstat` locally instead:
    # reliable, instant, no API/rate-limit. One person commits under several name/email identities,
    # so identities are folded onto the API login set via the noreply-email username, an exact
    # name==login match, and a small alias map for the rest.
    import re
    import subprocess
    loc = {}       # login -> (additions, deletions) since SINCE
    commits = {}   # login -> commit count since SINCE
    latest = {}    # login -> most recent commit date (YYYY-MM-DD) since SINCE
    monthly = {}   # 'YYYY-MM' -> {login -> commit count}
    def _mony(s):  # '2026-06-30' -> "Jun 2026"
        try:
            return datetime.datetime.strptime(s, '%Y-%m-%d').strftime('%b %Y')
        except Exception:  # noqa: BLE001
            return s
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
            ['git', 'log', '--no-merges', f'--since={SINCE}', '--numstat', '--format=COMMIT\t%ae\t%an\t%as'],
            cwd=repo_root, capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=60).stdout
        cur = None
        for line in out.splitlines():
            if line.startswith('COMMIT\t'):
                parts = line.split('\t', 3); email = parts[1] if len(parts) > 1 else ''
                name = parts[2] if len(parts) > 2 else ''; cdate = parts[3] if len(parts) > 3 else ''
                cur = _resolve(email, name)
                if cur is not None:
                    commits[cur] = commits.get(cur, 0) + 1
                    if cdate > latest.get(cur, ''):   # ISO dates compare lexically
                        latest[cur] = cdate
                    mo = cdate[:7]
                    if len(mo) == 7:
                        monthly.setdefault(mo, {})[cur] = monthly.setdefault(mo, {}).get(cur, 0) + 1
                continue
            if cur is None or not line.strip():
                continue
            a, d, _rest = (line.split('\t', 2) + ['', '', ''])[:3]
            if a.isdigit() and d.isdigit():
                pa, pd = loc.get(cur, (0, 0))
                loc[cur] = (pa + int(a), pd + int(d))
        if not loc:
            print('[credits] no local git LOC resolved (shallow clone?); keeping snapshot.')
    except Exception as e:  # noqa: BLE001
        print(f'[credits] local git line-count failed ({e!r}); keeping snapshot.')

    # Safety net: if LOC could not be computed here (e.g. a shallow CI clone) keep the committed
    # snapshot rather than overwriting it with an empty/partial chart.
    if not loc:
        return

    # Every GitHub contributor is listed (even those active only before Dec 2020); their since-Dec-2020
    # totals are simply 0. data: (login, commits, total_loc, additions, deletions) sorted by LOC desc.
    all_logins = list(dict.fromkeys([lg for lg, _ in contribs] + list(loc) + list(commits)))
    data = sorted(((lg, commits.get(lg, 0),
                    loc.get(lg, (0, 0))[0] + loc.get(lg, (0, 0))[1],
                    loc.get(lg, (0, 0))[0], loc.get(lg, (0, 0))[1]) for lg in all_logins),
                  key=lambda t: t[2], reverse=True)
    if not data:
        return

    # ---- authorship of the CURRENT release: git blame line ownership of the package source ----
    # Who wrote the lines that exist NOW (HEAD), not lifetime churn. `git blame` every tracked
    # simba/*.py, fold authors onto logins, express as a percentage of surviving lines.
    blame = {}
    try:
        files = subprocess.run(['git', 'ls-files', 'simba/*.py'], cwd=repo_root,
                               capture_output=True, text=True, timeout=30).stdout.split()
        for fp in files:
            b = subprocess.run(['git', 'blame', '--line-porcelain', '-w', 'HEAD', '--', fp],
                               cwd=repo_root, capture_output=True, text=True, encoding='utf-8',
                               errors='replace', timeout=30).stdout
            em = nm = None
            for line in b.splitlines():
                if line.startswith('author-mail '):
                    em = line[12:].strip('<> ')
                elif line.startswith('author '):
                    nm = line[7:]
                elif line.startswith('\t'):
                    lg = _resolve(em, nm)
                    if lg:
                        blame[lg] = blame.get(lg, 0) + 1
    except Exception as e:  # noqa: BLE001
        print(f'[credits] git blame authorship failed ({e!r}); skipping that chart.')
        blame = {}

    # ---- responsive HTML bar charts + monthly timeline (no image, no matplotlib) ----
    # Consumed by credits.rst via a `.. raw:: html :file:` include; regenerated every build.
    try:
        import html as _htmllib
        INK, MUT, TRACK, BLUE, LEAD = '#23272e', '#6b7280', '#eef2f7', '#2a7fb8', '#21567a'
        esc = _htmllib.escape

        def _fmt(n):  # compact count: 1.2M / 162K / 1.5K / 405
            if n >= 1_000_000:
                return f'{n / 1_000_000:.1f}M'
            if n >= 10_000:
                return f'{round(n / 1000)}K'
            if n >= 1000:
                return f'{n / 1000:.1f}K'
            return str(int(n))

        style = (
            "<style>\n"
            ".simba-cc-h{max-width:820px;margin:40px auto 10px;font-size:21px;font-weight:800;color:#21567a;text-align:center;line-height:1.25;}\n"
            ".simba-cc-h span{display:block;font-weight:400;color:#6b7280;font-size:13.5px;margin-top:2px;}\n"
            ".simba-cc-cap{text-align:right;font-size:11px;color:#6b7280;font-style:italic;margin:10px 4px 0;}\n"
            ".simba-cc{max-width:820px;margin:6px auto 4px;padding-top:38px;}\n"
            ".simba-cc-row{display:flex;align-items:center;gap:10px;margin:5px 0;position:relative;text-decoration:none!important;}\n"
            ".simba-cc-name{flex:0 0 140px;text-align:right;font-size:13px;color:#23272e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}\n"
            ".simba-cc-track{flex:1;background:#eef2f7;border-radius:5px;height:22px;display:block;}\n"
            ".simba-cc-bar{display:block;height:100%;border-radius:5px;transition:filter .12s ease;}\n"
            ".simba-cc-row:hover .simba-cc-bar{filter:brightness(1.08);}\n"
            ".simba-cc-count{flex:0 0 58px;font-size:12px;font-weight:700;color:#23272e;}\n"
            ".simba-cc-loc{flex:0 0 176px;font-size:11.5px;color:#6b7280;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}\n"
            "@media (max-width:600px){.simba-cc-name{flex-basis:96px;}.simba-cc-loc{display:none;}}\n"
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
            ".simba-tl{max-width:820px;margin:6px auto 4px;}\n"
            ".simba-tl-leg{display:flex;flex-wrap:wrap;gap:8px 16px;justify-content:center;font-size:12px;color:#23272e;margin:10px 0 0;}\n"
            ".simba-tl-leg span,.simba-tl-leg a{display:inline-flex;align-items:center;gap:5px;}\n"
            ".simba-tl-leg i{width:11px;height:11px;border-radius:3px;display:inline-block;}\n"
            ".simba-tl-li{text-decoration:none!important;color:#23272e;}\n"
            ".simba-tl-li:hover{color:#21567a;text-decoration:underline!important;}\n"
            "</style>\n")

        def bar_rows(items):
            out = []
            for i, it in enumerate(items):
                le = esc(it['login']); col = LEAD if i == 0 else BLUE
                out.append(
                    f'     <a class="simba-cc-row" href="https://github.com/{le}">'
                    f'<span class="simba-cc-name">@{le}</span>'
                    f'<span class="simba-cc-track"><span class="simba-cc-bar" style="width:{it["pct"]}%;background:{col};"></span></span>'
                    f'<span class="simba-cc-count">{it["main"]}</span>'
                    f'<span class="simba-cc-loc">{it["sub"]}</span>'
                    f'<span class="simba-cc-card"><img src="https://github.com/{le}.png?size=96" alt="@{le}" loading="lazy">'
                    f'<span class="cci"><b>@{le}</b><em>{it["card"]}</em></span></span></a>')
            return '\n'.join(out)

        # ---- Graph 1: lines changed (churn) since Dec 2020 ----
        mxloc = max(t[2] for t in data)
        g1 = []
        for (login, c, tot, add, dele) in data:
            last = latest.get(login, ''); cl = f"{c:,} commit" + ('' if c == 1 else 's')
            pct = 0 if tot == 0 else round(max(0.4, tot / mxloc * 100.0), 2)
            sub = (cl + (f" · latest {_mony(last)}" if last else '')) if c else 'none since Dec 2020'
            g1.append(dict(login=login, pct=pct, main=_fmt(tot), sub=sub,
                           card=f'{_fmt(tot)} lines changed · +{_fmt(add)} / −{_fmt(dele)} · {cl}' + (f", latest {last}" if last else '')))
        g1_html = ('<h4 class="simba-cc-h">Lines changed since Dec 2020 <span>(added + removed · linear)</span></h4>\n'
                   '<div class="simba-cc">\n' + bar_rows(g1) +
                   f'\n     <p class="simba-cc-cap">bar &amp; bold number = lines changed (added + removed) since Dec 2020 · '
                   f'linear scale · commits &amp; latest at right · {stamp}</p>\n   </div>')

        # ---- Graph 2: authorship of the current release (git blame %) ----
        g2_html = ''
        if blame:
            btot = sum(blame.values())
            g2 = []
            for login in sorted(all_logins, key=lambda lg: blame.get(lg, 0), reverse=True):
                n = blame.get(login, 0); pct = 100.0 * n / btot if btot else 0.0
                g2.append(dict(login=login, pct=0 if n == 0 else round(max(0.4, pct), 2), main=f'{pct:.1f}%',
                               sub=(f'{n:,} lines' if n else 'no surviving lines'),
                               card=f'{pct:.1f}% of current source · {n:,} lines in HEAD'))
            g2_html = ('\n<h4 class="simba-cc-h">Authorship of the current release <span>(git blame · % of lines in HEAD)</span></h4>\n'
                       '<div class="simba-cc">\n' + bar_rows(g2) +
                       f'\n     <p class="simba-cc-cap">bar &amp; bold number = share of the current <code>simba</code> Python source '
                       f'owned in git blame · {_fmt(btot)} lines total · {stamp}</p>\n   </div>')

        # ---- Graph 3: commits per month since Dec 2020, stacked by contributor ----
        months = sorted(monthly.keys())
        PALETTE = ['#21567a', '#2a7fb8', '#e08a1e', '#3a9d5d', '#b8433a', '#7b5bbd', '#c65b9c']
        OTHER = '#9aa3ad'
        color_logins = [lg for lg, _ in sorted(commits.items(), key=lambda t: t[1], reverse=True)][:len(PALETTE)]
        cmap = {lg: PALETTE[i] for i, lg in enumerate(color_logins)}
        maxc = max((sum(d.values()) for d in monthly.values()), default=1) or 1
        W, H, PL, PR, PT, PB = 820, 210, 30, 6, 10, 42
        plotw, ploth = W - PL - PR, H - PT - PB
        bw = plotw / max(1, len(months)); gap = min(2.0, bw * 0.18)
        parts = [f'<line x1="{PL}" y1="{PT+ploth}" x2="{W-PR}" y2="{PT+ploth}" stroke="#d5dbe2" stroke-width="1"/>',
                 f'<text x="{PL-4}" y="{PT+8}" text-anchor="end" font-size="10" fill="{MUT}">{maxc}</text>',
                 f'<text x="{PL-4}" y="{PT+ploth}" text-anchor="end" font-size="10" fill="{MUT}">0</text>']
        for i, mo in enumerate(months):
            x = PL + i * bw
            if mo.endswith('-01'):   # one tick per year (January); Dec-2020 start is unlabelled to avoid a collision
                xc = x + (bw - gap) / 2
                parts.append(f'<line x1="{xc:.1f}" y1="{PT+ploth}" x2="{xc:.1f}" y2="{PT+ploth+4}" stroke="#b9c1cb" stroke-width="1"/>')
                parts.append(f'<text x="{xc:.1f}" y="{PT+ploth+17}" text-anchor="middle" font-size="11" fill="{MUT}">{mo[:4]}</text>')
            d = monthly[mo]; y = PT + ploth
            order = [lg for lg in color_logins if lg in d] + [lg for lg in d if lg not in cmap]
            for lg in order:
                hh = d[lg] / maxc * ploth; y -= hh
                parts.append(f'<rect x="{x:.1f}" y="{y:.2f}" width="{max(0.6, bw-gap):.1f}" height="{hh:.2f}" '
                             f'fill="{cmap.get(lg, OTHER)}"><title>@{esc(lg)} · {mo}: {d[lg]} commits</title></rect>')
        svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;display:block;margin:6px auto 0;font-family:inherit;">'
               + ''.join(parts) + '</svg>')
        has_other = any(lg not in cmap for d in monthly.values() for lg in d)

        def _leg_item(lg):   # plain clickable link to the contributor's GitHub, no popup card
            le = esc(lg)
            return f'<a class="simba-tl-li" href="https://github.com/{le}"><i style="background:{cmap[lg]}"></i>@{le}</a>'
        leg = ''.join(_leg_item(lg) for lg in color_logins)
        if has_other:
            leg += f'<span style="cursor:default"><i style="background:{OTHER}"></i>others</span>'
        g3_html = ('\n<h4 class="simba-cc-h">Commits per month since Dec 2020 <span>(stacked by contributor)</span></h4>\n'
                   f'<div class="simba-tl">{svg}<div class="simba-tl-leg">{leg}</div>'
                   f'<p class="simba-cc-cap">each bar = one month · height = commits that month · colour = contributor · {stamp}</p></div>')

        chart = style + g1_html + '\n' + g2_html + '\n' + g3_html
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(chart + '\n')
    except Exception as e:  # noqa: BLE001
        print(f'[credits] contributor HTML render failed ({e!r}); keeping existing snapshot.')


def _generate_download_stats(app):
    """Fetch the PyPI download-stats CSV (BigQuery, last 30 days) from the ``download_stats``
    branch and render an interactive Chart.js dashboard for ``download_stats.rst``.

    The CSV is produced daily by the ``Get BigQuery Download Stats`` GitHub Action
    (``misc/bigquery_download_stats.py``) and force-pushed to the ``download_stats`` branch.
    On any failure (no network, parse error, missing pandas) a notice is printed and the
    committed snapshot is kept, so the build never breaks. Starts with a single panel:
    downloads by country.
    """
    import io
    import json
    import urllib.request

    here = os.path.dirname(os.path.abspath(__file__))
    gen_dir = os.path.join(here, '_generated')
    html_path = os.path.join(gen_dir, 'download_stats.html')
    os.makedirs(gen_dir, exist_ok=True)
    CSV_URL = 'https://raw.githubusercontent.com/sgoldenlab/simba/download_stats/misc/bigquery_download_stats.csv'

    try:
        import pandas as pd
        req = urllib.request.Request(CSV_URL, headers={'User-Agent': 'simba-docs'})
        with urllib.request.urlopen(req, timeout=25) as resp:
            df = pd.read_csv(io.StringIO(resp.read().decode('utf-8')))
        if df.empty or 'country' not in df.columns or 'download_count' not in df.columns:
            raise RuntimeError('unexpected CSV shape')
    except Exception as e:  # noqa: BLE001 - never fail the build over this
        print(f'[download_stats] CSV fetch/parse failed ({e!r}); keeping existing snapshot.')
        return

    try:
        import pycountry
    except Exception:  # noqa: BLE001 - fall back to raw codes if unavailable
        pycountry = None

    def _country_name(code):
        code = str(code)
        if pycountry is None:
            return code
        try:
            c = pycountry.countries.get(alpha_2=code)
            return (getattr(c, 'common_name', None) or c.name) if c is not None else code
        except Exception:  # noqa: BLE001
            return code

    TOP = 20
    by_country = df.groupby('country')['download_count'].sum().sort_values(ascending=False)
    top = by_country.head(TOP)
    codes = [str(c) for c in top.index]
    labels = [_country_name(c) for c in codes]
    vals = [int(v) for v in top.values]
    # per-bar colour: red (highest) -> blue (lowest) gradient across the sorted bars
    n = len(vals)
    (r0, g0, b0), (r1, g1, b1) = (214, 69, 59), (47, 127, 192)
    colors = ['#%02x%02x%02x' % (round(r0 + (r1 - r0) * t), round(g0 + (g1 - g0) * t), round(b0 + (b1 - b0) * t))
              for t in ([i / (n - 1) for i in range(n)] if n > 1 else [0.0])]
    data_json = json.dumps({'labels': labels, 'vals': vals, 'colors': colors})
    # full per-country totals (alpha-2 keyed) for the world map (used for the boolean fill + tooltip)
    map_json = json.dumps({str(c): int(v) for c, v in by_country.items()})

    # ---- continent rollup (ISO alpha-2 -> continent); unmapped codes fall into "Other" ----
    _CONT_CODES = {
        'Africa': 'DZ AO BJ BW BF BI CM CV CF TD KM CG CD CI DJ EG GQ ER ET GA GM GH GN GW KE LS LR LY MG MW ML MR MU MA MZ NA NE NG RW ST SN SC SL SO ZA SS SD SZ TZ TG TN UG ZM ZW',
        'Asia': 'AF AM AZ BH BD BT BN KH CN CY GE HK IN ID IR IQ IL JP JO KZ KW KG LA LB MO MY MV MN MM NP KP OM PK PS PH QA SA SG KR LK SY TW TJ TH TL TR TM AE UZ VN YE',
        'Europe': 'AL AD AT BY BE BA BG HR CZ DK EE FI FR DE GR HU IS IE IT XK LV LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH UA GB VA',
        'North America': 'AG BS BB BZ CA CR CU DM DO SV GD GT HT HN JM MX NI PA KN LC VC TT US VG AI AW BM KY GP MQ PR',
        'South America': 'AR BO BR CL CO EC FK GF GY PE PY SR UY VE',
        'Oceania': 'AU FJ KI MH FM NR NZ PW PG WS SB TO TV VU NC PF GU',
    }
    _code2cont = {cc: cont for cont, codes in _CONT_CODES.items() for cc in codes.split()}
    _cont_tot = {}
    for _cc, _v in by_country.items():
        _key = _code2cont.get(str(_cc), 'Other')
        _cont_tot[_key] = _cont_tot.get(_key, 0) + int(_v)
    _cont_order = sorted(_cont_tot, key=lambda k: _cont_tot[k], reverse=True)
    cont_json = json.dumps({'labels': _cont_order, 'vals': [_cont_tot[k] for k in _cont_order]})

    total = int(df['download_count'].sum())
    n_country = int(df['country'].nunique())
    n_version = int(df['package_version'].nunique())
    daily = df.groupby('download_date')['download_count'].sum()
    n_days = int(daily.shape[0]) or 1
    daily_avg = int(round(total / n_days))
    try:
        peak_val = int(daily.max())
        peak_date = pd.to_datetime(str(daily.idxmax())).strftime('%b %d')
    except Exception:  # noqa: BLE001
        peak_date, peak_val = '', 0
    try:
        dmin = str(df['download_date'].min())
        dmax = str(df['download_date'].max())
    except Exception:  # noqa: BLE001
        dmin = dmax = ''
    stamp = datetime.datetime.now().strftime('%B %d, %Y')

    # ---- time-series, version & day-of-week aggregations for the extra panels ----
    daily_sorted = daily.sort_index()
    date_labels = [pd.to_datetime(str(d)).strftime('%b %d') for d in daily_sorted.index]
    daily_vals = [int(v) for v in daily_sorted.values]
    cumulative_vals = [int(v) for v in daily_sorted.cumsum().values]
    ver = df.groupby('package_version')['download_count'].sum().sort_values(ascending=False)
    ver_top = ver.head(15)
    ver_labels = [str(v) for v in ver_top.index]
    ver_vals = [int(v) for v in ver_top.values]
    top6 = [str(v) for v in ver.head(6).index]
    piv = df.pivot_table(index='download_date', columns='package_version', values='download_count',
                         aggfunc='sum', fill_value=0).sort_index()
    piv.columns = [str(c) for c in piv.columns]
    adopt_dates = [pd.to_datetime(str(d)).strftime('%b %d') for d in piv.index]
    adopt = [{'label': v, 'data': [int(x) for x in piv[v].values]} for v in top6 if v in piv.columns]
    other_cols = [c for c in piv.columns if c not in top6]
    if other_cols:
        adopt.append({'label': 'other', 'data': [int(x) for x in piv[other_cols].sum(axis=1).values]})
    dow_series = pd.to_datetime(df['download_date']).dt.dayofweek
    dow_tot = df.groupby(dow_series)['download_count'].sum()
    dow_vals = [int(dow_tot.get(i, 0)) for i in range(7)]
    dash_json = json.dumps({'date_labels': date_labels, 'daily': daily_vals, 'cumulative': cumulative_vals,
                            'dow': dow_vals, 'ver_labels': ver_labels, 'ver': ver_vals,
                            'adopt_dates': adopt_dates, 'adopt': adopt})

    # ---- scholarly citations of the SimBA paper (OpenAlex, summed across journal + preprint DOIs) ----
    # OpenAlex is keyless, covers preprints + all venues, and uniquely returns a per-year breakdown.
    CITE_DOIS = ['10.1038/s41593-024-01649-9',   # Nature Neuroscience (2024)
                 '10.1101/2020.04.19.049452']    # bioRxiv preprint (2020)
    try:
        import collections
        _per_year = collections.defaultdict(int)
        _ctot = 0
        for _doi in CITE_DOIS:
            _url = f'https://api.openalex.org/works/doi:{_doi}?mailto=simon@netholabs.com'
            _req = urllib.request.Request(_url, headers={'User-Agent': 'simba-docs'})
            with urllib.request.urlopen(_req, timeout=25) as _resp:
                _w = json.loads(_resp.read().decode('utf-8'))
            _ctot += int(_w.get('cited_by_count') or 0)
            for _c in _w.get('counts_by_year', []):
                _per_year[int(_c['year'])] += int(_c['cited_by_count'])
        if not _per_year:
            raise RuntimeError('no counts_by_year returned')
        _yrs = sorted(_per_year)
        cite_years = [str(y) for y in _yrs]
        cite_counts = [_per_year[y] for y in _yrs]
        cite_total = _ctot
    except Exception as e:  # noqa: BLE001 - never fail the build over this; keep last-known values
        print(f'[download_stats] OpenAlex citation fetch failed ({e!r}); using fallback snapshot.')
        cite_years = ['2020', '2021', '2022', '2023', '2024', '2025', '2026']
        cite_counts = [21, 39, 56, 69, 89, 108, 50]
        cite_total = 433
    _cur_year = str(datetime.datetime.now().year)
    cite_partial = cite_years.index(_cur_year) if _cur_year in cite_years else -1
    cites_json = json.dumps({'years': cite_years, 'counts': cite_counts,
                             'total': cite_total, 'partial': cite_partial})

    # ---- NIH iCite Relative Citation Ratio (field- & time-normalised impact; 1.0 = median NIH paper) ----
    # Only the journal article is PubMed-indexed; the preprint has no PMID, so RCR is for the journal paper.
    CITE_PMID = '38778146'   # Nature Neuroscience article (resolved from its DOI)
    try:
        _ic_url = f'https://icite.od.nih.gov/api/pubs?pmids={CITE_PMID}'
        _ic_req = urllib.request.Request(_ic_url, headers={'User-Agent': 'simba-docs'})
        with urllib.request.urlopen(_ic_req, timeout=25) as _ic_resp:
            _ic = json.loads(_ic_resp.read().decode('utf-8'))['data'][0]
        rcr = round(float(_ic['relative_citation_ratio']), 1)
    except Exception as e:  # noqa: BLE001 - never fail the build over this; keep last-known value
        print(f'[download_stats] NIH iCite RCR fetch failed ({e!r}); using fallback snapshot.')
        rcr = 24.8

    try:
        MUT = '#8a9099'
        style = (
            "<style>\n"
            ".simba-dl{max-width:860px;margin:14px auto 6px;}\n"
            ".simba-dl-sub{font-size:13px;color:#6b7280;margin:0 0 14px;}\n"
            ".simba-dl-sub b{color:#23272e;}\n"
            ".simba-dl-cards{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 8px;}\n"
            ".simba-dl-card{flex:1 1 120px;min-width:104px;background:#fff;border:1px solid #e2e8f0;border-radius:12px;"
            "box-shadow:0 4px 14px rgba(33,86,122,.08);padding:13px 8px;text-align:center;overflow-wrap:anywhere;}\n"
            ".simba-dl-card .v{display:block;font-size:clamp(16px,5vw,22px);font-weight:800;color:#21567a;line-height:1.1;}\n"
            ".simba-dl-card .l{display:block;font-size:10.5px;color:#6b7280;margin-top:4px;}\n"
            ".simba-dl-h3{font-size:15px;color:#23272e;font-weight:700;margin:24px 0 10px;}\n"
            ".simba-dl-badge{display:inline-block;font-size:11px;font-weight:700;color:#8a5a00;background:#fdf1d6;"
            "border:1px solid #f0d68f;border-radius:999px;padding:2px 9px;margin-left:8px;vertical-align:middle;white-space:nowrap;cursor:help;}\n"
            ".simba-dl-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start;margin-top:8px;}\n"
            ".simba-dl-cell{min-width:0;}\n"
            ".simba-dl-cell .simba-dl-h3{margin:0 0 8px;min-height:38px;display:flex;align-items:center;flex-wrap:wrap;gap:0 4px;}\n"
            ".simba-dl-cell--wide{grid-column:1 / -1;}\n"
            "@media (max-width:680px){.simba-dl-grid{grid-template-columns:1fr;}}\n"
            ".simba-dl-panel{position:relative;height:340px;box-sizing:border-box;background:#fff;border:1px solid #e2e8f0;"
            "border-radius:14px;box-shadow:0 6px 20px rgba(33,86,122,.10);padding:14px 16px 10px;}\n"
            ".simba-dl-map{position:relative;height:680px;background:radial-gradient(120% 120% at 50% 0%,#f4f9fd,#e8f1f8);"
            "border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 6px 20px rgba(33,86,122,.10);padding:10px;box-sizing:border-box;}\n"
            ".simba-dl-legend{display:flex;align-items:center;gap:8px;justify-content:flex-end;font-size:11px;color:#6b7280;margin:8px 4px 0;}\n"
            ".simba-dl-legend i{width:140px;height:10px;border-radius:5px;display:inline-block;"
            "background:linear-gradient(90deg,#e8f4fb,#7fc1e3,#2a7fb8,#1a3f6b);}\n"
            ".simba-dl-legend .dot{width:11px;height:11px;border-radius:50%;background:#e0433a;display:inline-block;margin-left:14px;}\n"
            ".simba-dl-legend .sw{width:13px;height:13px;border-radius:3px;display:inline-block;border:1px solid rgba(0,0,0,.06);}\n"
            ".simba-dl-more{text-align:center;margin:26px 0 6px;}\n"
            ".simba-dl-more a{display:inline-flex;align-items:center;gap:8px;text-decoration:none !important;background:#21567a;"
            "color:#fff !important;font-weight:600;font-size:14px;padding:11px 20px;border-radius:24px;box-shadow:0 2px 10px rgba(33,86,122,.25);}\n"
            ".simba-dl-more a:hover{background:#19465f;}\n"
            ".simba-dl-chart{position:relative;height:600px;box-sizing:border-box;background:#fff;"
            "border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 6px 20px rgba(33,86,122,.10);padding:16px 18px 8px;}\n"
            ".simba-dl-foot{text-align:right;font-size:11px;color:#8a9099;font-style:italic;margin:12px 4px 0;}\n"
            "</style>\n")
        body = (
            '<div class="simba-dl">\n'
            f'  <p class="simba-dl-sub">PyPI downloads of <b>simba-uw-tf-dev</b> &middot; last 30 days &middot; {dmin} &ndash; {dmax}</p>\n'
            '  <div class="simba-dl-cards">\n'
            f'    <div class="simba-dl-card"><span class="v">{total:,}</span><span class="l">downloads &middot; 30 days</span></div>\n'
            f'    <div class="simba-dl-card"><span class="v">{daily_avg:,}</span><span class="l">avg / day</span></div>\n'
            f'    <div class="simba-dl-card"><span class="v">{peak_val:,}</span><span class="l">peak day ({peak_date})</span></div>\n'
            f'    <div class="simba-dl-card"><span class="v">{n_version:,}</span><span class="l">versions</span></div>\n'
            f'    <div class="simba-dl-card"><span class="v">{n_country}</span><span class="l">countries</span></div>\n'
            '  </div>\n'
            '  <div class="simba-dl-grid">\n'
            '    <div class="simba-dl-cell"><h3 class="simba-dl-h3">Downloads over time</h3>\n'
            '      <div class="simba-dl-panel"><canvas id="dlOverTime"></canvas></div></div>\n'
            '    <div class="simba-dl-cell"><h3 class="simba-dl-h3">Cumulative downloads</h3>\n'
            '      <div class="simba-dl-panel"><canvas id="dlCumulative"></canvas></div></div>\n'
            '    <div class="simba-dl-cell"><h3 class="simba-dl-h3">Downloads by day of week</h3>\n'
            '      <div class="simba-dl-panel"><canvas id="dlDow"></canvas></div></div>\n'
            '    <div class="simba-dl-cell"><h3 class="simba-dl-h3">Top 15 versions</h3>\n'
            '      <div class="simba-dl-panel"><canvas id="dlVersions"></canvas></div></div>\n'
            '    <div class="simba-dl-cell"><h3 class="simba-dl-h3">Version adoption over time</h3>\n'
            '      <div class="simba-dl-panel"><canvas id="dlAdoption"></canvas></div></div>\n'
            f'    <div class="simba-dl-cell"><h3 class="simba-dl-h3">Paper citations per year &middot; {cite_total:,} total'
            f'      <span class="simba-dl-badge" title="NIH iCite Relative Citation Ratio: field- and time-normalised citation impact, where 1.0 = the median NIH-funded paper.">NIH RCR {rcr:g} &middot; ~{round(rcr)}&times; median</span></h3>\n'
            '      <div class="simba-dl-panel"><canvas id="dlCitations"></canvas></div></div>\n'
            '    <div class="simba-dl-cell simba-dl-cell--wide"><h3 class="simba-dl-h3">Downloads by country &mdash; world map</h3>\n'
            '      <div class="simba-dl-map" id="dlMap"></div>\n'
            '      <div class="simba-dl-legend"><span>fewer</span><i></i><span>more</span>'
            '<span class="sw" style="background:#dce4ee;margin-left:14px"></span><span>none</span></div></div>\n'
            '    <div class="simba-dl-cell simba-dl-cell--wide"><h3 class="simba-dl-h3">Downloads by continent</h3>\n'
            '      <div class="simba-dl-panel" style="height:300px"><canvas id="dlContinents"></canvas></div></div>\n'
            f'    <div class="simba-dl-cell simba-dl-cell--wide"><h3 class="simba-dl-h3">Top {TOP} countries</h3>\n'
            '      <div class="simba-dl-chart"><canvas id="dlCountries"></canvas></div></div>\n'
            '  </div>\n'
            '  <p class="simba-dl-more"><a href="https://sronilsson.github.io/download_stats/" target="_blank" rel="noopener">'
            '\U0001F4CA  Full interactive download dashboard &rarr;</a></p>\n'
            f'  <p class="simba-dl-foot">Source: PyPI downloads via Google BigQuery &middot; updated {stamp}</p>\n'
            '  <p class="simba-dl-foot">Citation counts from <a href="https://openalex.org" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">OpenAlex</a>'
            ' &middot; SimBA journal (10.1038/s41593-024-01649-9) + preprint (10.1101/2020.04.19.049452) DOIs &middot; current year to date'
            ' &middot; RCR from <a href="https://icite.od.nih.gov" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">NIH iCite</a></p>\n'
            '</div>\n')
        script = (
            # RTD theme loads RequireJS; disable AMD while the UMD libs load so they set
            # window.Chart / window.jsVectorMap globals instead of registering as AMD modules
            '<script>window.__odef = window.define; try { window.define = undefined; } catch (e) {}</script>\n'
            '<link rel="stylesheet" href="_static/css/jsvectormap.min.css">\n'
            '<script src="_static/js/jsvectormap.min.js"></script>\n'
            '<script src="_static/js/jsvectormap-world.js"></script>\n'
            '<script>\n(function(){\n'
            f'  const MAP = {map_json};\n'
            '  const el = document.getElementById("dlMap");\n'
            '  if (!el || typeof jsVectorMap === "undefined") return;\n'
            '  // graduated choropleth. The jsVectorMap series scale is an ordinal lookup (value -> scale[value]),\n'
            '  // NOT a gradient, so we bucket each country into 1..N by log of its (highly skewed) count and\n'
            '  // index a colour ramp with that bucket. (Bucket 0 is skipped by the lib, so buckets start at 1.)\n'
            '  const SCALE = ["#e8f4fb", "#cfe6f5", "#9bcde9", "#5aa7d4", "#3a86c0", "#2a6299", "#143a5e"];\n'
            '  const NB = SCALE.length - 1;\n'
            '  const _counts = Object.keys(MAP).map((k) => MAP[k] || 0);\n'
            '  const _maxLog = Math.log10(Math.max.apply(null, _counts.concat([1])) + 1) || 1;\n'
            '  const BUCK = {};\n'
            '  for (const k in MAP) { let b = Math.ceil(Math.log10((MAP[k] || 0) + 1) / _maxLog * NB); BUCK[k] = b < 1 ? 1 : (b > NB ? NB : b); }\n'
            '  new jsVectorMap({\n'
            '    selector: "#dlMap", map: "world",\n'
            '    zoomButtons: true, zoomOnScroll: true, backgroundColor: "transparent",\n'
            '    regionStyle: {initial: {fill: "#dce4ee", stroke: "#ffffff", "stroke-width": 0.5},\n'
            '      hover: {fill: "#f0c44a", "fill-opacity": 1}},\n'
            '    series: {regions: [{attribute: "fill", values: BUCK, scale: SCALE}]},\n'
            '    onRegionTooltipShow(event, tooltip, code) {\n'
            '      const v = MAP[code] || 0;\n'
            '      tooltip.text(tooltip.text() + (v ? ": " + v.toLocaleString() + " downloads" : ": no downloads"), true);\n'
            '    }\n'
            '  });\n})();\n</script>\n'
            '<script src="_static/js/chart.umd.min.js"></script>\n'
            '<script>\n(function(){\n'
            '  if (!window.Chart) return;\n'
            f'  const DL = {data_json};\n'
            f'  const DASH = {dash_json};\n'
            f'  const CITES = {cites_json};\n'
            f'  const CONT = {cont_json};\n'
            '  const C = (id) => document.getElementById(id);\n'
            '  const GRID = "#eef2f7", INK = "#23272e", MUT = "#6b7280";\n'
            '  const PAL = ["#21567a","#2a7fb8","#38a8d4","#5cc6b3","#e0a33a","#e0653a","#b9c0c9"];\n'
            '  // light rounded track behind each horizontal bar (like the credits chart)\n'
            '  const track = {id: "track", beforeDatasetsDraw(chart) {\n'
            '    const ctx = chart.ctx, right = chart.chartArea.right, x0 = chart.scales.x.getPixelForValue(0);\n'
            '    ctx.save(); ctx.fillStyle = GRID;\n'
            '    chart.getDatasetMeta(0).data.forEach(function(b) {\n'
            '      const h = b.height, y = b.y - h / 2, w = right - x0, r = Math.min(6, h / 2);\n'
            '      if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x0, y, w, h, r); ctx.fill(); } else { ctx.fillRect(x0, y, w, h); }\n'
            '    }); ctx.restore();\n'
            '  }};\n'
            '  if (C("dlCountries")) new Chart(C("dlCountries"), {\n'
            '    type: "bar", plugins: [track],\n'
            '    data: {labels: DL.labels, datasets: [{label: "Downloads", data: DL.vals, backgroundColor: DL.colors, hoverBackgroundColor: DL.colors, borderRadius: 4, maxBarThickness: 24}]},\n'
            '    options: {indexAxis: "y", responsive: true, maintainAspectRatio: false, layout: {padding: {right: 6}},\n'
            '      plugins: {legend: {display: false}, tooltip: {callbacks: {label: (c) => " " + c.parsed.x.toLocaleString() + " downloads"}}},\n'
            '      scales: {x: {beginAtZero: true, grid: {display: false}, ticks: {color: MUT}}, y: {grid: {display: false}, ticks: {color: INK, font: {weight: "600"}}}}}\n'
            '  });\n'
            '  if (C("dlContinents")) new Chart(C("dlContinents"), {\n'
            '    type: "bar", plugins: [track],\n'
            '    data: {labels: CONT.labels, datasets: [{label: "Downloads", data: CONT.vals,\n'
            '      backgroundColor: PAL, hoverBackgroundColor: PAL, borderRadius: 4, maxBarThickness: 26}]},\n'
            '    options: {indexAxis: "y", responsive: true, maintainAspectRatio: false, layout: {padding: {right: 6}},\n'
            '      plugins: {legend: {display: false}, tooltip: {callbacks: {label: function(c) {\n'
            '        const t = CONT.vals.reduce((a, b) => a + b, 0), p = t ? Math.round(c.parsed.x / t * 100) : 0;\n'
            '        return " " + c.parsed.x.toLocaleString() + " downloads (" + p + "%)"; }}}},\n'
            '      scales: {x: {beginAtZero: true, grid: {display: false}, ticks: {color: MUT}}, y: {grid: {display: false}, ticks: {color: INK, font: {weight: "600"}}}}}\n'
            '  });\n'
            '  if (C("dlOverTime")) new Chart(C("dlOverTime"), {\n'
            '    type: "bar",\n'
            '    data: {labels: DASH.date_labels, datasets: [\n'
            '      {label: "Daily", data: DASH.daily, backgroundColor: "#7fc1e3", borderRadius: 3}]},\n'
            '    options: {responsive: true, maintainAspectRatio: false,\n'
            '      plugins: {legend: {display: false}},\n'
            '      scales: {y: {beginAtZero: true, grid: {color: GRID}, ticks: {color: MUT}, title: {display: true, text: "daily", color: MUT}},\n'
            '               x: {grid: {display: false}, ticks: {color: MUT, maxRotation: 45, minRotation: 45, autoSkip: true, maxTicksLimit: 10}}}}\n'
            '  });\n'
            '  if (C("dlCumulative")) new Chart(C("dlCumulative"), {\n'
            '    type: "line",\n'
            '    data: {labels: DASH.date_labels, datasets: [\n'
            '      {label: "Cumulative", data: DASH.cumulative, borderColor: "#21567a", backgroundColor: "rgba(33,86,122,.10)", fill: true, tension: .3, pointRadius: 0, pointHoverRadius: 5, pointHoverBackgroundColor: "#21567a", pointHoverBorderColor: "#fff", pointHoverBorderWidth: 2, borderWidth: 2.5}]},\n'
            '    options: {responsive: true, maintainAspectRatio: false, interaction: {intersect: false, mode: "index"},\n'
            '      plugins: {legend: {display: false}, tooltip: {displayColors: false,\n'
            '        callbacks: {title: (items) => "Through " + items[0].label,\n'
            '                    label: (c) => " " + c.parsed.y.toLocaleString() + " downloads total"}}},\n'
            '      scales: {y: {beginAtZero: true, grid: {color: GRID}, ticks: {color: MUT}, title: {display: true, text: "cumulative \\u00b7 30 days", color: MUT}},\n'
            '               x: {grid: {display: false}, ticks: {color: MUT, maxRotation: 45, minRotation: 45, autoSkip: true, maxTicksLimit: 10}}}}\n'
            '  });\n'
            '  if (C("dlDow")) new Chart(C("dlDow"), {\n'
            '    type: "bar",\n'
            '    data: {labels: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], datasets: [{label: "Downloads", data: DASH.dow,\n'
            '      backgroundColor: ["#2a7fb8","#2a7fb8","#2a7fb8","#2a7fb8","#2a7fb8","#e0a33a","#e0a33a"], borderRadius: 4, maxBarThickness: 52}]},\n'
            '    options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}},\n'
            '      scales: {y: {beginAtZero: true, grid: {color: GRID}, ticks: {color: MUT}}, x: {grid: {display: false}, ticks: {color: INK, font: {weight: "600"}}}}}\n'
            '  });\n'
            '  if (C("dlVersions")) new Chart(C("dlVersions"), {\n'
            '    type: "bar",\n'
            '    data: {labels: DASH.ver_labels, datasets: [{label: "Downloads", data: DASH.ver, backgroundColor: "#2a7fb8", borderRadius: 4, maxBarThickness: 30}]},\n'
            '    options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}, tooltip: {callbacks: {label: (c) => " " + c.parsed.y.toLocaleString() + " downloads"}}},\n'
            '      scales: {y: {beginAtZero: true, grid: {color: GRID}, ticks: {color: MUT}}, x: {grid: {display: false}, ticks: {color: INK, maxRotation: 60, minRotation: 45, font: {weight: "600", size: 10}}}}}\n'
            '  });\n'
            '  if (C("dlAdoption")) new Chart(C("dlAdoption"), {\n'
            '    type: "line",\n'
            '    data: {labels: DASH.adopt_dates, datasets: DASH.adopt.map(function(s, i){ return {label: s.label, data: s.data, backgroundColor: PAL[i % PAL.length], borderColor: PAL[i % PAL.length], fill: true, tension: .25, pointRadius: 0, borderWidth: 1}; })},\n'
            '    options: {responsive: true, maintainAspectRatio: false, interaction: {intersect: false, mode: "index"},\n'
            '      plugins: {legend: {display: true, position: "top", labels: {boxWidth: 10, font: {size: 10}}}},\n'
            '      scales: {y: {stacked: true, beginAtZero: true, grid: {color: GRID}, ticks: {color: MUT}}, x: {grid: {display: false}, ticks: {color: MUT, maxRotation: 45, minRotation: 45, autoSkip: true, maxTicksLimit: 10}}}}\n'
            '  });\n'
            '  if (C("dlCitations")) new Chart(C("dlCitations"), {\n'
            '    type: "bar",\n'
            '    data: {labels: CITES.years, datasets: [{label: "Citations", data: CITES.counts,\n'
            '      backgroundColor: CITES.years.map((y, i) => i === CITES.partial ? "#9ec9e6" : "#2a7fb8"), borderRadius: 4, maxBarThickness: 54}]},\n'
            '    options: {responsive: true, maintainAspectRatio: false,\n'
            '      plugins: {legend: {display: false}, tooltip: {displayColors: false, callbacks: {\n'
            '        title: (i) => i[0].label + (i[0].dataIndex === CITES.partial ? " (partial \\u00b7 to date)" : ""),\n'
            '        label: (c) => " " + c.parsed.y.toLocaleString() + " citing publications"}}},\n'
            '      scales: {y: {beginAtZero: true, grid: {color: GRID}, ticks: {color: MUT}, title: {display: true, text: "citing publications", color: MUT}},\n'
            '               x: {grid: {display: false}, ticks: {color: INK, font: {weight: "600"}}}}}\n'
            '  });\n'
            '})();\n</script>\n'
            '<script>try { window.define = window.__odef; } catch (e) {}</script>\n')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(style + body + script)
    except Exception as e:  # noqa: BLE001
        print(f'[download_stats] render failed ({e!r}); keeping existing snapshot.')


def _generate_commit_heatmap(app):
    """(Re)generate a GitHub-style calendar heatmap of commit activity on every build.

    Runs at ``builder-inited`` so the file exists when ``credits.rst`` is read. On any
    failure (shallow clone, no git) it keeps the committed snapshot so the build never breaks.
    """
    import subprocess
    import collections
    import datetime as _dt

    here = os.path.dirname(os.path.abspath(__file__))
    gen_dir = os.path.join(here, '_generated')
    html_path = os.path.join(gen_dir, 'commit_heatmap.html')
    os.makedirs(gen_dir, exist_ok=True)
    try:
        repo_root = os.path.dirname(here)
        out = subprocess.run(['git', 'log', '--no-merges', '--format=%as'],
                             cwd=repo_root, capture_output=True, text=True,
                             encoding='utf-8', errors='replace', timeout=60).stdout
        per_day = collections.Counter()
        for line in out.splitlines():
            s = line.strip()
            if len(s) == 10:
                per_day[s] += 1
        if not per_day:
            print('[heatmap] no commits resolved (shallow clone?); keeping snapshot.')
            return

        days = sorted(per_day)
        first_year, last_year = int(days[0][:4]), int(days[-1][:4])
        start_year = max(first_year, 2021)   # solo-maintenance era; skip the 2019-2020 lab period
        shown = {d: c for d, c in per_day.items() if int(d[:4]) >= start_year}
        total = sum(shown.values())
        active = len(shown)
        CELL, GAP = 10, 3
        STEP = CELL + GAP
        ML, MT = 30, 6
        COLORS = ["#ebedf0", "#cfe6f5", "#7fc1e3", "#2a7fb8", "#143a5e"]

        def bucket(n):
            if n <= 0:
                return 0
            if n <= 2:
                return 1
            if n <= 5:
                return 2
            if n <= 10:
                return 3
            return 4

        strip_h = 7 * STEP
        width = ML + 54 * STEP + 8
        today = _dt.date.today()
        blocks = []
        y_off = MT
        for Y in range(start_year, last_year + 1):
            start, end = _dt.date(Y, 1, 1), _dt.date(Y, 12, 31)
            if Y == today.year:
                end = today
            cells, col, d = [], 0, start
            while d <= end:
                sun = (d.weekday() + 1) % 7   # Sun=0 .. Sat=6
                if d != start and sun == 0:
                    col += 1
                n = per_day.get(d.isoformat(), 0)
                x, yy = ML + col * STEP, y_off + sun * STEP
                tip = f"{d.isoformat()}: {n} commit{'s' if n != 1 else ''}" if n else f"{d.isoformat()}: no commits"
                cells.append(f'<rect x="{x}" y="{yy}" width="{CELL}" height="{CELL}" rx="2" '
                             f'fill="{COLORS[bucket(n)]}"><title>{tip}</title></rect>')
                d += _dt.timedelta(days=1)
            blocks.append(f'<text x="0" y="{y_off + strip_h / 2}" font-size="11" fill="#6b7280" '
                          f'dominant-baseline="middle">{Y}</text>' + ''.join(cells))
            y_off += strip_h + 10
        height = y_off + 2
        svg = (f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;height:auto" '
               f'role="img" aria-label="SimBA commit activity, {first_year}-{last_year}">'
               + ''.join(blocks) + '</svg>')
        legend = ('<div class="simba-hm-legend">Less '
                  + ''.join(f'<span style="background:{c}"></span>' for c in COLORS) + ' More</div>')
        style = ("<style>.simba-hm{max-width:860px;margin:8px auto;}"
                 ".simba-hm-card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;"
                 "box-shadow:0 6px 20px rgba(33,86,122,.10);padding:16px 18px;overflow-x:auto;}"
                 ".simba-hm-cap{font-size:12.5px;color:#6b7280;margin:0 0 12px;}"
                 ".simba-hm-cap b{color:#21567a;}"
                 ".simba-hm-legend{display:flex;align-items:center;gap:3px;justify-content:flex-end;"
                 "font-size:11px;color:#6b7280;margin-top:10px;}"
                 ".simba-hm-legend span{width:10px;height:10px;border-radius:2px;display:inline-block;}"
                 "</style>")
        cap = (f'<p class="simba-hm-cap"><b>{total:,}</b> commits across <b>{active:,}</b> days, '
               f'{start_year}–{last_year} · a single maintainer</p>')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(style + f'<div class="simba-hm"><div class="simba-hm-card">{cap}{svg}{legend}</div></div>')
    except Exception as e:  # noqa: BLE001
        print(f'[heatmap] generation failed ({e!r}); keeping existing snapshot.')


def _generate_docs_authorship(app):
    """(Re)generate a small panel crediting authorship of the docs & tutorials on every build.

    Counts documentation source files and blames the prose (``.rst`` / ``.md``) to attribute
    authorship. Keeps the committed snapshot on any failure so the build never breaks.
    """
    import subprocess
    import re

    here = os.path.dirname(os.path.abspath(__file__))
    gen_dir = os.path.join(here, '_generated')
    html_path = os.path.join(gen_dir, 'docs_authorship.html')
    os.makedirs(gen_dir, exist_ok=True)
    SINCE = '2020-12-01'
    aliases = {
        'simon nilsson': 'sronilsson', 'simon': 'sronilsson', 'sronilsson': 'sronilsson',
        'jia jie choong': 'inoejj', 'inoejj': 'inoejj',
        'aasiya islam': 'aasiya-islam', 'aasiya-islam': 'aasiya-islam',
        'tzuk': 'tzukpolinsky', 'tzuk polinsky': 'tzukpolinsky',
        'jens schweihoff': 'JensBlack',
        'justin shenk': 'justinshenk',
        'nastacia l. goodwin': 'goodwinnastacia', 'nastacia goodwin': 'goodwinnastacia',
        "thomas o'shea-wheller": 'Toshea111',
        'sam golden lab': 'sgoldenlab', 'sgoldenlab': 'sgoldenlab',
        'florian duclot': 'florianduclot', 'florianduclot': 'florianduclot',
        'sophia hwang': 'sophihwang26', 'sophihwang26': 'sophihwang26',
    }
    noreply_re = re.compile(r'(?:\d+\+)?([^@]+)@users\.noreply\.github\.com$', re.I)

    def _login(email, name):
        mm = noreply_re.match((email or '').strip())
        if mm:
            return mm.group(1)
        nl = (name or '').strip().lower()
        if nl in aliases:
            return aliases[nl]
        return aliases.get((email or '').split('@')[0].strip().lower())

    def _h(n):
        if n >= 1_000_000:
            return f'{n / 1e6:.1f}M'
        if n >= 1000:
            return f'{n / 1e3:.1f}K'
        return str(int(n))

    def _mon(s):
        try:
            return datetime.datetime.strptime(s, '%Y-%m-%d').strftime('%b %Y')
        except Exception:  # noqa: BLE001
            return s

    try:
        repo_root = os.path.dirname(here)
        listed = subprocess.run(['git', 'ls-files', 'docs/*.rst', 'docs/*.md', 'docs/*.ipynb'],
                                cwd=repo_root, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=60).stdout
        allf = [f for f in listed.split('\n') if f.strip() and '/_build/' not in f]
        n_rst = sum(1 for f in allf if f.endswith('.rst') and '/_generated/' not in f)
        n_md = sum(1 for f in allf if f.endswith('.md'))
        n_nb = sum(1 for f in allf if f.endswith('.ipynb'))

        # Same metric as the code-contributors chart: lines changed (added + removed) since
        # Dec 2020, restricted to the documentation/tutorial prose sources via pathspec.
        out = subprocess.run(
            ['git', 'log', '--no-merges', f'--since={SINCE}', '--numstat',
             '--format=COMMIT\t%ae\t%an\t%as', '--', 'docs/*.rst', 'docs/*.md'],
            cwd=repo_root, capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=90).stdout
        loc, commits, latest, cur = {}, {}, {}, None
        for line in out.splitlines():
            if line.startswith('COMMIT\t'):
                parts = line.split('\t', 3)
                email = parts[1] if len(parts) > 1 else ''
                name = parts[2] if len(parts) > 2 else ''
                cdate = parts[3] if len(parts) > 3 else ''
                cur = _login(email, name)
                if cur is not None:
                    commits[cur] = commits.get(cur, 0) + 1
                    if cdate > latest.get(cur, ''):
                        latest[cur] = cdate
                continue
            if cur is None or not line.strip():
                continue
            a, d, _rest = (line.split('\t', 2) + ['', '', ''])[:3]
            if a.isdigit() and d.isdigit():
                pa, pd = loc.get(cur, (0, 0))
                loc[cur] = (pa + int(a), pd + int(d))
        if not loc:
            print('[docs_authorship] no doc contributions resolved (shallow clone?); keeping snapshot.')
            return

        data = sorted(((lg, loc[lg][0] + loc[lg][1], loc[lg][0], loc[lg][1],
                        commits.get(lg, 0), latest.get(lg, '')) for lg in loc),
                      key=lambda t: t[1], reverse=True)
        maxtot = data[0][1] or 1
        stamp = datetime.datetime.now().strftime('%B %Y')

        rows = []
        for i, (lg, tot, add, dele, ncom, ldate) in enumerate(data):
            wpct = 0 if tot == 0 else max(0.4, tot / maxtot * 100)
            color = '#21567a' if i == 0 else '#2a7fb8'
            locline = (f'{ncom:,} commit{"s" if ncom != 1 else ""} &middot; latest {_mon(ldate)}'
                       if ncom else 'none since Dec 2020')
            rows.append(
                f'<a class="simba-cc-row" href="https://github.com/{lg}">'
                f'<span class="simba-cc-name">@{lg}</span>'
                f'<span class="simba-cc-track"><span class="simba-cc-bar" '
                f'style="width:{wpct:.2f}%;background:{color};"></span></span>'
                f'<span class="simba-cc-count">{_h(tot)}</span>'
                f'<span class="simba-cc-loc">{locline}</span>'
                f'<span class="simba-cc-card"><img src="https://github.com/{lg}.png?size=96" '
                f'alt="@{lg}" loading="lazy"><span class="cci"><b>@{lg}</b>'
                f'<em>{_h(tot)} doc lines changed &middot; +{add:,} / −{dele:,} &middot; '
                f'{ncom:,} commits{f", latest {ldate}" if ldate else ""}</em></span></span></a>')
        cap = (f'bar &amp; bold number = documentation lines changed (added + removed) since Dec 2020 '
               f'&middot; linear scale &middot; {n_rst} pages &middot; {n_md} tutorials &middot; '
               f'{n_nb} notebooks &middot; {stamp}')
        html = ('<h4 class="simba-cc-h">Documentation lines changed since Dec 2020 '
                '<span>(added + removed &middot; linear)</span></h4>\n<div class="simba-cc">\n     '
                + '\n     '.join(rows)
                + f'\n     <p class="simba-cc-cap">{cap}</p>\n   </div>\n')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
    except Exception as e:  # noqa: BLE001
        print(f'[docs_authorship] generation failed ({e!r}); keeping existing snapshot.')


def _generate_code_growth(app):
    """(Re)generate a cumulative code-growth curve (net Python lines over time) on every build.

    Sums rename-aware ``git log --numstat`` additions minus deletions for ``*.py`` by month and
    plots the running total as a self-contained SVG. Keeps the committed snapshot on failure.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    gen_dir = os.path.join(here, '_generated')
    html_path = os.path.join(gen_dir, 'code_growth.html')
    os.makedirs(gen_dir, exist_ok=True)

    def _hn(v):
        if v >= 1_000_000:
            return f'{v / 1e6:.1f}M'
        if v >= 1000:
            return f'{v / 1e3:.0f}K'
        return str(int(v))

    try:
        repo_root = os.path.dirname(here)
        out = subprocess.run(
            ['git', 'log', '--no-merges', '--reverse', '-M', '--numstat', '--format=C\t%as'],
            cwd=repo_root, capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=180).stdout
        code_m, docs_m, curmon = {}, {}, None
        for line in out.splitlines():
            if line.startswith('C\t'):
                curmon = line.split('\t', 1)[1][:7]
                continue
            p = line.split('\t')
            if not (curmon and len(p) >= 3 and p[0].isdigit() and p[1].isdigit()):
                continue
            net = int(p[0]) - int(p[1])
            path = p[2].lower()
            if path.endswith('.py'):
                code_m[curmon] = code_m.get(curmon, 0) + net
            elif path.endswith('.rst') or path.endswith('.md'):
                docs_m[curmon] = docs_m.get(curmon, 0) + net
        if not code_m and not docs_m:
            print('[code_growth] no numstat resolved (shallow clone?); keeping snapshot.')
            return

        allm = sorted(set(code_m) | set(docs_m))
        yy, mm = int(allm[0][:4]), int(allm[0][5:7])
        ey, em = int(allm[-1][:4]), int(allm[-1][5:7])
        months = []
        while (yy, mm) <= (ey, em):
            months.append(f'{yy:04d}-{mm:02d}')
            mm += 1
            if mm > 12:
                mm, yy = 1, yy + 1
        cc = dc = 0
        code_cum, docs_cum = [], []
        for m in months:
            cc = max(0, cc + code_m.get(m, 0))
            dc = max(0, dc + docs_m.get(m, 0))
            code_cum.append(cc)
            docs_cum.append(dc)
        total = [c + d for c, d in zip(code_cum, docs_cum)]
        maxv = max(total) or 1
        code_final, docs_final = code_cum[-1], docs_cum[-1]
        stamp = datetime.datetime.now().strftime('%B %Y')

        W, H, L, R, T, B = 820, 210, 36, 812, 14, 164
        n = len(months)

        def X(i):
            return L + (R - L) * (i / (n - 1) if n > 1 else 0)

        def Y(v):
            return B - (B - T) * (v / maxv)

        codeY = [Y(v) for v in code_cum]
        totalY = [Y(v) for v in total]
        code_area = (f'M {X(0):.1f},{B:.1f}' + ''.join(f' L {X(i):.1f},{codeY[i]:.1f}' for i in range(n))
                     + f' L {X(n - 1):.1f},{B:.1f} Z')
        docs_band = (f'M {X(0):.1f},{codeY[0]:.1f}'
                     + ''.join(f' L {X(i):.1f},{codeY[i]:.1f}' for i in range(1, n))
                     + ''.join(f' L {X(i):.1f},{totalY[i]:.1f}' for i in range(n - 1, -1, -1)) + ' Z')
        total_line = ' '.join(f'{X(i):.1f},{totalY[i]:.1f}' for i in range(n))
        ticks = ''
        for i, m in enumerate(months):
            if m.endswith('-01'):
                x = X(i)
                ticks += (f'<line x1="{x:.1f}" y1="{B}" x2="{x:.1f}" y2="{B + 4}" stroke="#b9c1cb" '
                          f'stroke-width="1"/><text x="{x:.1f}" y="{B + 17}" text-anchor="middle" '
                          f'font-size="11" fill="#6b7280">{m[:4]}</text>')
        svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;display:block;'
               f'margin:6px auto 0;font-family:inherit;" role="img" '
               f'aria-label="Cumulative Python and documentation lines over time">'
               f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="#d5dbe2" stroke-width="1"/>'
               f'<text x="{L - 5}" y="{T + 4}" text-anchor="end" font-size="10" fill="#6b7280">{_hn(maxv)}</text>'
               f'<text x="{L - 5}" y="{B}" text-anchor="end" font-size="10" fill="#6b7280">0</text>'
               f'<path d="{docs_band}" fill="rgba(56,168,212,.22)"/>'
               f'<path d="{code_area}" fill="rgba(33,86,122,.16)"/>'
               f'<polyline points="{total_line}" fill="none" stroke="#21567a" stroke-width="2.2"/>'
               f'{ticks}</svg>')
        leg = ('<div class="simba-tl-leg"><span><i style="background:#21567a"></i>Python code</span>'
               '<span><i style="background:#38a8d4"></i>documentation</span></div>')
        cap = f'net lines accumulated over time · ≈{_hn(code_final)} Python + ≈{_hn(docs_final)} docs · {stamp}'
        html = ('<h4 class="simba-cc-h">Codebase growth '
                '<span>(cumulative Python + documentation lines)</span></h4>\n'
                f'<div class="simba-tl">{svg}{leg}<p class="simba-cc-cap">{cap}</p></div>\n')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
    except Exception as e:  # noqa: BLE001
        print(f'[code_growth] generation failed ({e!r}); keeping existing snapshot.')


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
    app.connect('builder-inited', _generate_download_stats)
    app.connect('builder-inited', _generate_commit_heatmap)
    app.connect('builder-inited', _generate_docs_authorship)
    app.connect('builder-inited', _generate_code_growth)
    app.connect('build-finished', _copy_images)
    app.connect('build-finished', _inject_lazy_media)
    app.connect('doctree-read', _convert_github_alerts)
    app.connect('doctree-read', _autolink_glossary_terms)