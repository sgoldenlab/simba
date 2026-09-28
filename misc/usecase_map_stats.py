# -*- coding: utf-8 -*-
"""
Generate docs/_generated/usecase_map.html — a "Global Reach" panel of published
SimBA use-cases, built from the public Google Sheet of citing studies.

Reuses the same vendored Chart.js as the download-stats page, so the styling
matches. The country and institution maps are the globe above it on the page
(misc/usecase_globe.py). Reads the sheet as public CSV — no credentials.

Run:  python misc/usecase_map_stats.py
"""
import urllib.request, csv, io, gzip, json, collections, os, re, sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from institution_coords import INSTITUTION_ALIASES, INSTITUTION_COORDS
from country_names import COUNTRY_CONTINENT, COUNTRY_NAMES

SHEET_ID = "169enc3Am2KQKifxj1F9KEKKLbftpMhBlw49zjl-egsY"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "_generated", "usecase_map.html")

# --- country name -> ISO-2 (covers sheet variants + misspellings) ---------
NAME2ISO = {
    "us": "US", "usa": "US", "united states": "US", "u.s.": "US", "u.s.a.": "US",
    "canada": "CA", "germany": "DE", "spain": "ES", "italy": "IT",
    "switzerland": "CH", "china": "CN", "poland": "PL", "netherlands": "NL",
    "the netherlands": "NL", "australia": "AU", "uk": "GB", "u.k.": "GB",
    "united kingdom": "GB", "england": "GB", "scotland": "GB", "israel": "IL",
    "sweden": "SE", "france": "FR", "ireland": "IE", "belgium": "BE",
    "mexico": "MX", "india": "IN", "morocco": "MA", "marocco": "MA",
    "austria": "AT", "japan": "JP", "thailand": "TH", "ecuador": "EC",
    "czech republic": "CZ", "czechia": "CZ", "hong kong": "HK",
    "hong kong sar": "HK", "hongkong": "HK", "brazil": "BR", "hungary": "HU",
    "portugal": "PT", "norway": "NO", "finland": "FI", "denmark": "DK",
    "singapore": "SG", "south korea": "KR", "korea": "KR",
    "republic of korea": "KR", "taiwan": "TW", "russia": "RU",
    "south africa": "ZA", "argentina": "AR", "chile": "CL", "new zealand": "NZ",
    "greece": "GR", "turkey": "TR", "iran": "IR", "slovenia": "SI",
    "romania": "RO", "ukraine": "UA", "indonesia": "ID", "philippines": "PH",
    "saudi arabia": "SA", "egypt": "EG", "colombia": "CO", "moldova": "MD",
    "montenegro": "ME", "lithuania": "LT", "estonia": "EE", "sri lanka": "LK",
    "nepal": "NP", "luxembourg": "LU",
}
# Spellings resolved automatically by misc/resolve_new_places.py (the daily job looks up any
# country string none of the lists above recognise); "spelling in lower case" -> ISO-2.
COUNTRY_AUTO_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "country_auto.json")
try:
    with open(COUNTRY_AUTO_JSON, encoding="utf-8") as _f:
        COUNTRY_AUTO = json.load(_f)
except (OSError, ValueError):
    COUNTRY_AUTO = {}
_NAME_ISO = {n.lower(): iso for iso, n in COUNTRY_NAMES.items()}


def split_countries(text):
    """Country strings from one COUNTRIES cell: split on comma/semicolon, or a period FOLLOWED
    BY a space (a typo separator, e.g. "Sweden. Israel") -- but not bare dots inside "U.S."."""
    return [c.strip() for c in re.split(r"[,;]|\.\s+", text or "") if c.strip()]


def country_iso(name):
    """ISO-2 for a country string, or None: the sheet variants above, then the standard names
    (misc/country_names.py, the sheet dropdown's list), then the auto-resolved spellings."""
    key = name.strip().lower()
    return NAME2ISO.get(key) or _NAME_ISO.get(key) or COUNTRY_AUTO.get(key)


SPECIES_NORM = {
    "mouse": "Mouse", "mice": "Mouse", "rat": "Rat", "rats": "Rat",
    "rodent": "Rodent", "rodents": "Rodent", "zebrafish": "Zebrafish",
    "fish": "Fish", "gerbil": "Gerbil", "gerbils": "Gerbil",
}

# Sequential orange ramp for institution study-count (validated ordinal ramp:
# one hue, monotone light->dark, visible steps, light end clears the map surface).
# Orange because the country choropleth already uses blue -- two sequential
# contexts on one view => second takes the next hue as its own one-hue ramp.
# Every bar on the page is one orange from the globe's ramp (misc/usecase_globe.py): bar length
# carries the value, so colour needs no second meaning. Checked on the white panels as an
# ordinal step: BAR 3.29:1, BAR_PARTIAL (the current, to-date year) 2.29:1.
BAR, BAR_PARTIAL, BAR_HOVER = "#e5692a", "#f59352", "#c2461a"


def inst_bucket(c):
    """Study count -> ramp index (0..4)."""
    if c <= 1:
        return 0
    if c == 2:
        return 1
    if c <= 4:
        return 2
    if c <= 7:
        return 3
    return 4


def esc(s):
    """Minimal HTML escaping for user text placed inside tooltip markup."""
    return ((s or "").replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def tooltip_html(name, count, studies, cap=6, total=None, unit=None):
    """Rich hover tooltip: name, count, and a capped study list. Used by the map pins
    and, via the bar charts' external tooltip, by the corpus panels -- so both look
    identical. total overrides the "+N more" base when studies is only a sample."""
    rows = []
    shown = sorted(studies, key=lambda t: t[0], reverse=True)[:cap]
    for yr, title, jrnl in shown:
        t = esc(title)
        if len(t) > 64:
            t = t[:63] + "…"
        yr_part = f'<b style="color:#f0a35a">{esc(yr)}</b> · ' if yr else ""
        jr_part = f' <span style="color:#b7c0cc">{esc(jrnl)}</span>' if jrnl else ""
        rows.append(f'<div style="margin-top:3px">{yr_part}{t}{jr_part}</div>')
    more = (total if total is not None else len(studies)) - len(shown)
    if more > 0:
        rows.append(f'<div style="margin-top:3px;color:#8b96a3">…+{more} more</div>')
    unit = unit or ("study" if count == 1 else "studies")
    return (f'<div style="text-align:left;max-width:280px;white-space:normal;'
            f'font-size:11px;line-height:1.35;color:#fff">'
            f'<b>{esc(name)}</b> — {count} {unit}{"".join(rows)}</div>')


def fetch_rows():
    req = urllib.request.Request(CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-16", "ignore")
    text = text.replace("\x00", "")
    return list(csv.DictReader(io.StringIO(text)))


def main():
    rows = fetch_rows()
    cols = {c.strip().upper(): c for c in rows[0].keys()}

    def cell(r, name):
        return (r.get(cols.get(name, "")) or "").strip()

    per_country = collections.Counter()
    per_continent = collections.Counter()
    per_year = collections.Counter()
    species = collections.Counter()
    journals = set()
    unmapped = collections.Counter()
    per_institution = collections.Counter()  # canonical name -> studies (deduped/study)
    unmapped_inst = collections.Counter()     # canonical names lacking coords

    for r in rows:
        seen_iso = set()
        for c in split_countries(cell(r, "COUNTRIES")):
            iso = country_iso(c)
            if not iso:
                unmapped[c] += 1
                continue
            seen_iso.add(iso)
        for iso in seen_iso:
            per_country[iso] += 1
            per_continent[COUNTRY_CONTINENT.get(iso, "Other")] += 1
        y = cell(r, "YEAR")
        if y.isdigit():
            per_year[int(y)] += 1
        sp = cell(r, "SPECIES").lower()
        if sp:
            species[SPECIES_NORM.get(sp, cell(r, "SPECIES"))] += 1
        j = cell(r, "JOURNAL")
        if j:
            journals.add(j.lower())
        seen_inst = set()  # count each institution once per study
        for tok in cell(r, "AUTHOR INSTITUTIONS").split(","):
            name = re.sub(r"\s+", " ", tok).strip()
            if not name:
                continue
            canon = INSTITUTION_ALIASES.get(name, name)
            if canon:
                seen_inst.add(canon)
        for canon in seen_inst:
            if canon in INSTITUTION_COORDS:
                per_institution[canon] += 1
            else:
                unmapped_inst[canon] += 1

    total = len(rows)
    n_countries = len(per_country)
    n_continents = len([k for k in per_continent if k != "Other"])
    n_species = len(species)
    n_journals = len(journals)
    years = sorted(per_year)
    yr_labels = [str(y) for y in years]
    yr_counts = [per_year[y] for y in years]
    top_country = per_country.most_common(20)
    ISO_NAME = COUNTRY_NAMES   # ISO-2 -> display name for tooltips/bars

    cont_labels = [c for c, _ in per_continent.most_common() if c != "Other"]
    cont_vals = [per_continent[c] for c in cont_labels]
    top_labels = [ISO_NAME.get(iso, iso) for iso, _ in top_country]
    top_vals = [n for _, n in top_country]
    sp_top = species.most_common(10)
    sp_labels = [s for s, _ in sp_top]
    sp_vals = [n for _, n in sp_top]

    # --- institutions: the full institution list ---
    n_institutions = len(per_institution)
    top_inst = per_institution.most_common()   # every institution, including single-study ones
    inst_labels = [n for n, _ in top_inst]
    inst_vals = [c for _, c in top_inst]
    inst_colors = [BAR] * len(inst_vals)

    if unmapped:
        print("Unmapped country strings (misc/resolve_new_places.py looks these up):", dict(unmapped))
    if unmapped_inst:
        print("Institutions without coords (add to institution_coords.py):", dict(unmapped_inst))

    # Corpus stats mined LOCALLY from the paper PDFs (misc/extract_corpus_stats.py) and
    # committed as misc/corpus_stats.json, so this CI/RTD render never touches the PDFs.
    corpus = None
    cpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_stats.json")
    try:
        with open(cpath, encoding="utf-8") as cf:
            corpus = json.load(cf)
    except Exception as e:
        print(f"[usecase] corpus_stats.json not loaded ({e!r}); corpus panels skipped.")
    if corpus:
        # Name the mined papers from the curated sheet entries, so the corpus tooltips
        # can list example studies the way the institution pins do.
        corpus["_paper_titles"] = _paper_titles(corpus, rows, cols)
        print(f"[usecase] named {len(corpus['_paper_titles'])} of "
              f"{len(corpus.get('papers', []))} mined papers from the sheet")

    pull_date = date.today().strftime("%B %d, %Y")
    html = _render(total, n_countries, n_continents, n_species, n_journals,
                   years, yr_labels, yr_counts, cont_labels,
                   cont_vals, top_labels, top_vals, ISO_NAME, pull_date,
                   n_institutions, inst_labels, inst_vals,
                   sp_labels, sp_vals, inst_colors, corpus)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {os.path.normpath(OUT)}: {total} studies, {n_countries} countries, "
          f"{n_continents} continents, {n_species} species, {n_journals} journals, "
          f"{n_institutions} institutions")


def _paper_titles(corpus, sheet_rows, cols):
    """Resolve each mined paper to its curated sheet entry -> {index: "2026 · Title (Journal)"}.

    corpus["papers"] holds normalised paper openings; a sheet title whose first 60
    normalised characters appear in one is that paper. 145/168 match outright; the
    rest fall back to word overlap, because a few PDFs carry a reworded or
    preprint-era version of the title."""
    heads = (corpus or {}).get("papers") or []
    if not heads:
        return {}

    def norm(s):
        return re.sub(r"[^a-z0-9]", "", (s or "").lower())

    entries = []
    for r in sheet_rows:
        title = (r.get(cols.get("TITLE", "")) or "").strip()
        if len(norm(title)) > 25:
            entries.append((norm(title), title,
                            (r.get(cols.get("YEAR", "")) or "").strip(),
                            (r.get(cols.get("JOURNAL", "")) or "").strip(),
                            set(re.findall(r"[a-z]{4,}", title.lower()))))
    out = {}
    for i, head in enumerate(heads):
        flat = norm(head)
        hit = next((e for e in entries if e[0][:60] in flat), None)
        if not hit:                     # reworded title: best word overlap, if decisive
            words = set(re.findall(r"[a-z]{4,}", head))
            best = max(entries, key=lambda e: len(e[4] & words) / max(len(e[4]), 1),
                       default=None)
            if best and len(best[4] & words) / max(len(best[4]), 1) >= 0.6:
                hit = best
        if hit:
            out[i] = (hit[2], hit[1], hit[3])       # (year, title, journal)
    return out


def _examples_for(corpus, titles, key, labels, counts, unit="papers"):
    """Per-bar tooltip markup, built with tooltip_html()."""
    ex = ((corpus or {}).get("examples") or {}).get(key) or {}
    out = []
    for lab, total in zip(labels, counts):
        studies = [titles[i] for i in ex.get(lab, []) if i in titles]
        out.append(tooltip_html(lab, total, studies, cap=4, total=total,
                                unit=unit if total != 1 else unit.rstrip("s")))
    return out


def _bar_helper():
    """Emitted once; every corpus-derived horizontal bar chart on the page calls it,
    so the Chart.js config exists in one place rather than per panel."""
    return """<script>
// One reused tooltip node, styled by the .jvm-tooltip rules, so every bar hover on
// the page looks the same. Chart.js draws its own
// tooltips on canvas and cannot render HTML, hence the external handler.
window.__ucTip = function (ctx, d) {
  let el = document.getElementById("ucBarTip");
  if (!el) {
    el = document.createElement("div");
    el.id = "ucBarTip";
    el.className = "jvm-tooltip";
    el.style.pointerEvents = "none";
    el.style.zIndex = "9999";
    document.body.appendChild(el);
  }
  const tt = ctx.tooltip;
  if (!tt.opacity) { el.classList.remove("active"); return; }
  const i = tt.dataPoints && tt.dataPoints.length ? tt.dataPoints[0].dataIndex : -1;
  const html = i >= 0 && d.html ? d.html[i] : "";
  if (!html) { el.classList.remove("active"); return; }
  el.innerHTML = html;
  el.classList.add("active");
  const r = ctx.chart.canvas.getBoundingClientRect();
  let left = window.scrollX + r.left + tt.caretX + 16;
  const top = window.scrollY + r.top + tt.caretY - 12;
  // flip to the left of the cursor rather than overflow the viewport
  if (left + el.offsetWidth > window.scrollX + document.documentElement.clientWidth - 8) {
    left = window.scrollX + r.left + tt.caretX - el.offsetWidth - 16;
  }
  el.style.left = Math.max(8, left) + "px";
  el.style.top = top + "px";
};
window.__ucBar = function (id, d) {
  if (!window.Chart) return;
  const el = document.getElementById(id);
  if (!el) return;
  new Chart(el, {
    type: "bar", plugins: window.__ucValues ? [window.__ucValues] : [],
    data: {labels: d.labels, datasets: [{data: d.vals, backgroundColor: "__BAR__",
      hoverBackgroundColor: "__BAR_HOVER__", borderRadius: 4, borderSkipped: "start", maxBarThickness: 18}]},
    options: {indexAxis: "y", responsive: true, maintainAspectRatio: false, layout: {padding: {right: 30}},
      plugins: {legend: {display: false}, tooltip: d.html
        ? {enabled: false, external: (ctx) => window.__ucTip(ctx, d)}
        : {displayColors: false, callbacks: {
            label: (c) => " " + c.parsed.x + " " + (c.parsed.x === 1 ? d.unit1 : d.unit)}}},
      scales: {x: {display: false, beginAtZero: true},
               y: {grid: {display: false}, border: {display: false}, ticks: {color: "#23272e",
                   font: {weight: "600"}, autoSkip: false}}}}
  });
};
</script>""".replace("__BAR_HOVER__", BAR_HOVER).replace("__BAR__", BAR)


def _bar_panel(pid, title, caption, rows, unit="papers", unit1="paper", height=None,
               examples=None):
    """(panel_html, script) for one horizontal bar panel built from
    [[label, count], ...] rows. examples adds named papers to each hover tooltip."""
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    h = height or (90 + 24 * len(rows))
    html = (f'<h3 class="simba-uc-h3">{title}</h3>'
            f'<p class="simba-uc-cap">{caption}</p>'
            f'<div class="simba-uc-panel" style="height:{h}px">'
            f'<canvas id="{pid}"></canvas></div>')
    j = json.dumps
    ex = f', html: {j(examples)}' if examples else ""
    js = (f'<script>window.__ucBar({j(pid)}, {{labels: {j(labels)}, vals: {j(vals)}, '
          f'unit: {j(unit)}, unit1: {j(unit1)}{ex}}});</script>')
    return html, js


def _build_behaviours(c):
    """Behaviours-automated panel: horizontal bars, one per behaviour, from
    corpus_stats.json['behaviours_automated'] (mined locally from the paper PDFs).
    Returns (panel_html, chart_js). Empty strings when the key is absent."""
    rows = (c or {}).get("behaviours_automated") or []
    n = (c or {}).get("n_behaviour_studies") or 0
    if not rows or not n:
        return "", ""
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    # One colour, not one per behaviour family: the families cannot be told apart under
    # colour-vision deficiency, and the family is kept in corpus_stats.json instead.
    height = 90 + 24 * len(rows)
    panel = (f'<h3 class="simba-uc-h3">Behaviours automated</h3>'
             f'<p class="simba-uc-cap"><b>INDICATIVE, NOT EXHAUSTIVE.</b> What SimBA was '
             f'used to score, read from the methods text of the collected papers: a behaviour '
             f'counts once per study when it is named near a SimBA mention <i>and</i> near a '
             f'classifier/scoring cue. Most studies score several. Keyword-based, so uncommon '
             f'behaviours may be undercounted.</p>'
             f'<div class="simba-uc-panel" style="height:{height}px"><canvas id="ucBehav"></canvas></div>')
    titles = (c or {}).get("_paper_titles") or {}
    ex = _examples_for(c, titles, "behaviours_automated", labels, vals, unit="studies")
    j = json.dumps
    script = (f'<script>window.__ucBar("ucBehav", {{labels: {j(labels)}, vals: {j(vals)}, '
              f'unit: "studies", unit1: "study", html: {j(ex)}}});</script>')
    return panel, script


def _build_corpus(c):
    """Corpus panels from corpus_stats.json: brain regions, methods and disease models
    as bar charts, then the concept word cloud as an impressionistic closer.
    Returns (panels_html, script). Empty strings if no corpus data."""
    # hide generic ML/plumbing terms -- the cloud is about what SimBA studies, not the algorithms
    EXCLUDE = {"machine learning", "pose estimation", "random forest", "cnn / resnet", "unsupervised",
               "svm", "xgboost", "transformer", "umap", "t-sne", "hdbscan", "bounding box", "keypoint tracking"}
    wc = [(w, n) for (w, n) in (c or {}).get("wordcloud", []) if w.lower() not in EXCLUDE]
    if not wc:
        return "", ""
    import hashlib
    # magma ramp (dark -> hot); skip the pale yellow top so words stay legible on white
    MAGMA = ["#160b39", "#3b0f70", "#641a80", "#8c2981", "#b73779",
             "#de4968", "#f7705c", "#fb8761"]
    fs = [n for _, n in wc]; lo, hi = min(fs), max(fs)
    def hsh(w):
        return int(hashlib.md5(w.encode()).hexdigest(), 16)
    def norm(n):
        return (((n - lo) / (hi - lo)) ** 0.6) if hi > lo else 0.5
    def sz(n):
        return round(11 + 26 * norm(n), 1)
    def col(n):                                    # colour by prevalence (magma heatmap)
        return MAGMA[min(len(MAGMA) - 1, int(norm(n) * len(MAGMA)))]
    # deterministic scramble (stable across rebuilds) so big/small words mix spatially,
    # with a small per-word vertical jitter for an organic cloud feel
    order = sorted(wc, key=lambda kv: hsh(kv[0]))
    spans = " ".join(
        f'<span style="font-size:{sz(n)}px;color:{col(n)};margin:1px 4px;'
        f'position:relative;top:{hsh(w) % 5 - 2}px;display:inline-block;font-weight:600;'
        f'opacity:.92;line-height:1" title="mentioned in {n} of {c.get("n_pdfs", "?")} papers">{esc(w)}</span>'
        for (w, n) in order)
    # --- categorised panels: regions / methods / disease models -----------------
    regions = (c or {}).get("regions") or []
    methods = (c or {}).get("methods") or []
    diseases = (c or {}).get("diseases") or []
    lead = "<b>INDICATIVE, NOT EXHAUSTIVE.</b> "
    # Full width rather than a two-column grid: the methods and model lists are long,
    # and their labels ("Chemogenetics (DREADD)") would eat half of a 421px column.
    cat_html, cat_js = "", ""
    titles = (c or {}).get("_paper_titles") or {}
    for pid, key, title, cap, rows in (
            ("ucRegions", "regions", "Brain regions",
             lead + "Papers naming each region anywhere in their full text.", regions),
            ("ucMethods", "methods", "Methods &amp; tools",
             lead + "What the studies pair SimBA with &mdash; recording methods, "
                    "molecular assays, and the other behaviour tools they cite.", methods),
            ("ucDiseases", "diseases", "Disease &amp; behavioural models",
             lead + "Models, conditions and paradigms the papers study.", diseases)):
        if rows:
            ex = _examples_for(c, titles, key, [r[0] for r in rows], [r[1] for r in rows])
            html, js = _bar_panel(pid, title, cap, rows, examples=ex)
            cat_html += html
            cat_js += js

    panels = (cat_html +
              f'<h3 class="simba-uc-h3">What SimBA has been used for</h3>'
              f'<p class="simba-uc-cap">Concepts appearing across the full text of {c.get("n_pdfs", "?")} '
              f'downloaded studies &mdash; methods, behaviours, brain regions, models and compounds. '
              f'Larger = mentioned in more papers. Keyword-based overview (indicative, not exhaustive). '
              f'Updated {c.get("generated", "")}.</p>'
              f'<div class="uc-cloud">{spans}</div>')
    return panels, cat_js


def _render(total, n_countries, n_continents, n_species, n_journals, years,
            yr_labels, yr_counts, cont_labels, cont_vals,
            top_labels, top_vals, iso_name, pull_date,
            n_institutions, inst_labels, inst_vals,
            sp_labels, sp_vals, inst_colors, corpus):
    yr_range = f"{years[0]}–{years[-1]}" if years else ""
    j = json.dumps
    corpus_panels, corpus_script = _build_corpus(corpus)
    behav_panel, behav_script = _build_behaviours(corpus)
    bar_helper = _bar_helper() if (behav_script or corpus_script) else ""
    # Tabler-style line icons (stroke=currentColor) for the stat cards
    _svg = ('<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{}</svg>')
    ic = {
        "studies": _svg.format('<path d="M14 3v4a1 1 0 0 0 1 1h4"/>'
            '<path d="M17 21H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2z"/>'
            '<path d="M9 13h6M9 17h6"/>'),
        "countries": _svg.format('<circle cx="12" cy="12" r="9"/>'
            '<path d="M3.6 9h16.8M3.6 15h16.8"/>'
            '<path d="M11.5 3a17 17 0 0 0 0 18M12.5 3a17 17 0 0 1 0 18"/>'),
        "species": _svg.format('<circle cx="6.5" cy="10" r="1.4"/><circle cx="10" cy="7" r="1.4"/>'
            '<circle cx="14" cy="7" r="1.4"/><circle cx="17.5" cy="10" r="1.4"/>'
            '<path d="M9 14.6c1-1.2 5-1.2 6 0 .8 1 1.9 1.4 1.9 2.9 0 1.2-1 1.9-2.1 1.6'
            '-1-.2-1.8-.6-2.8-.6s-1.8.4-2.8.6c-1.1.3-2.1-.4-2.1-1.6 0-1.5 1.1-1.9 1.9-2.9z"/>'),
        "journals": _svg.format('<path d="M3 19a9 9 0 0 1 9 0 9 9 0 0 1 9 0"/>'
            '<path d="M3 6a9 9 0 0 1 9 0 9 9 0 0 1 9 0"/><path d="M3 6v13M12 6v13M21 6v13"/>'),
        "institutions": _svg.format('<path d="M3 21h18M4 10h16M5 6l7-3 7 3M5 10v11M19 10v11'
            'M9 14v3M12 14v3M15 14v3"/>'),
    }
    # institutions as a multi-column list (name + mini-bar + count) -> readable names, several columns
    _maxv = max(inst_vals) if inst_vals else 1
    inst_rows = "".join(
        f'<div class="si-row"><b class="si-ct">{v}</b>'
        f'<span class="si-bar" style="width:{round(6 + 44 * v / _maxv)}px;background:{col}"></span>'
        f'<span class="si-nm">{esc(n)}</span></div>'
        for n, v, col in zip(inst_labels, inst_vals, inst_colors))
    inst_list_html = f'<div class="si-list">{inst_rows}</div>'
    return f"""<style>
.simba-uc{{max-width:860px;margin:14px auto 6px;}}
.simba-uc-sub{{font-size:13px;color:#6b7280;margin:0 0 14px;}}
.simba-uc-sub b{{color:#23272e;}}
.simba-uc-cards{{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 8px;}}
.simba-uc-card{{flex:1 1 120px;min-width:104px;background:#fff;border:1px solid #e2e8f0;border-radius:12px;box-shadow:0 4px 14px rgba(33,86,122,.08);padding:13px 8px;text-align:center;}}
.simba-uc-card .v{{display:block;font-size:clamp(16px,5vw,22px);font-weight:800;color:#21567a;line-height:1.1;}}
.simba-uc-card .l{{display:block;font-size:10.5px;color:#6b7280;margin-top:4px;}}
.simba-uc-card .ic{{display:block;width:22px;height:22px;margin:0 auto 7px;color:#21567a;opacity:.85;}}
.simba-uc-h3{{font-size:15px;color:#23272e;font-weight:700;margin:24px 0 10px;}}
.simba-uc-cap{{font-size:11.5px;color:#8b95a1;margin:-4px 0 10px;line-height:1.35;}}
.simba-uc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start;margin-top:18px;}}
.simba-uc-panel{{position:relative;height:320px;background:#fff;border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 6px 20px rgba(33,86,122,.10);padding:14px 16px 10px;box-sizing:border-box;}}
.simba-uc .si-list{{column-width:240px;column-gap:24px;}}
.simba-uc .si-row{{break-inside:avoid;display:flex !important;align-items:center;gap:8px;margin:4px 0;text-indent:0 !important;overflow:visible !important;}}
.simba-uc .si-ct{{flex:0 0 20px;text-align:right;font-size:11.5px;font-weight:800;color:#21567a;font-variant-numeric:tabular-nums;}}
.simba-uc .si-bar{{flex:0 0 auto;display:inline-block;height:9px;border-radius:2px;}}
.simba-uc .si-nm{{flex:1 1 auto;min-width:0;font-size:12.5px;color:#23272e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-indent:0 !important;}}
@media (max-width:680px){{.simba-uc-grid{{grid-template-columns:1fr;}}}}
.simba-uc-date{{font-size:12.5px;color:#6b7280;margin:0 0 14px;}}
.simba-uc-foot{{text-align:center;font-size:14.5px;color:#4b5563;margin:18px 4px 0;}}
.simba-uc-foot a{{font-weight:600;}}
.jvm-tooltip{{background-color:#1f2937 !important;border-radius:9px !important;padding:9px 11px !important;max-width:300px !important;box-shadow:0 8px 24px rgba(15,23,42,.32) !important;}}
.uc-cloud{{max-width:900px;margin:6px auto 4px;text-align:center;padding:18px 16px;background:#fff;border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 6px 20px rgba(33,86,122,.10);line-height:1.05;}}
</style>
<div class="simba-uc">
  <p class="simba-uc-date">Data pulled {pull_date}</p>
  <div class="simba-uc-cards">
    <div class="simba-uc-card">{ic['studies']}<span class="v">{total}</span><span class="l">published studies</span></div>
    <div class="simba-uc-card">{ic['countries']}<span class="v">{n_countries}</span><span class="l">countries</span></div>
    <div class="simba-uc-card">{ic['species']}<span class="v">{n_species}</span><span class="l">species</span></div>
    <div class="simba-uc-card">{ic['journals']}<span class="v">{n_journals}</span><span class="l">journals</span></div>
    <div class="simba-uc-card">{ic['institutions']}<span class="v">{n_institutions}</span><span class="l">institutions</span></div>
  </div>
  <div class="simba-uc-grid">
    <div><h3 class="simba-uc-h3">Studies per year</h3><p class="simba-uc-cap">Publications by calendar year; the current year is to-date.</p><div class="simba-uc-panel"><canvas id="ucYears"></canvas></div></div>
    <div><h3 class="simba-uc-h3">By continent</h3><p class="simba-uc-cap">Studies grouped by the continent of each contributing country.</p><div class="simba-uc-panel"><canvas id="ucCont"></canvas></div></div>
  </div>
  <h3 class="simba-uc-h3">Studies by species</h3>
  <p class="simba-uc-cap">Animal model studied (top 10 by study count).</p>
  <div class="simba-uc-panel" style="height:320px"><canvas id="ucSpecies"></canvas></div>
  <h3 class="simba-uc-h3">Top countries</h3>
  <p class="simba-uc-cap">Most studies by country (top 20).</p>
  <div class="simba-uc-panel" style="height:420px"><canvas id="ucCountries"></canvas></div>
  <h3 class="simba-uc-h3">All institutions</h3>
  <p class="simba-uc-cap">Every contributing institution by study count (number of studies shown at left); each counted once per study it contributed to.</p>
  <div class="simba-uc-panel" style="height:auto;padding:16px 18px">{inst_list_html}</div>
  {behav_panel}
  {corpus_panels}
  <p class="simba-uc-foot"><a href="https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Full list of studies (spreadsheet) &rarr;</a> &middot; one entry per study &middot; multi-country studies counted in each country</p>
</div>
<script>window.__odef2 = window.define; try {{ window.define = undefined; }} catch (e) {{}}</script>
<script src="_static/js/chart.umd.min.js"></script>
<script>
(function(){{
  if (!window.Chart) return;
  const YR = {{labels: {j(yr_labels)}, vals: {j(yr_counts)}}};
  const CONT = {{labels: {j(cont_labels)}, vals: {j(cont_vals)}}};
  const TOP = {{labels: {j(top_labels)}, vals: {j(top_vals)}}};
  const INST = {{labels: {j(inst_labels)}, vals: {j(inst_vals)}, col: {j(inst_colors)}}};
  const SP = {{labels: {j(sp_labels)}, vals: {j(sp_vals)}}};
  const C = (id) => document.getElementById(id);
  const INK = "#23272e";
  const BAR = {j(BAR)}, BAR_PARTIAL = {j(BAR_PARTIAL)}, BAR_HOVER = {j(BAR_HOVER)};
  // Value printed at the end of every bar, so the numbers read without hovering and the value
  // axis and gridlines can go. Shared on window so the corpus panels below reuse it.
  const values = {{id: "values", afterDatasetsDraw(chart) {{
    const ctx = chart.ctx, horiz = chart.options.indexAxis === "y";
    ctx.save();
    ctx.font = "600 11.5px " + Chart.defaults.font.family;
    ctx.fillStyle = INK;
    chart.getDatasetMeta(0).data.forEach(function(b, i) {{
      const t = String(chart.data.datasets[0].data[i]);
      if (horiz) {{ ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(t, b.x + 6, b.y); }}
      else {{ ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(t, b.x, b.y - 5); }}
    }});
    ctx.restore();
  }}}};
  window.__ucValues = values;
  const hbar = (labels, vals, thick) => ({{
    type: "bar", plugins: [values],
    data: {{labels: labels, datasets: [{{label: "Studies", data: vals, backgroundColor: BAR, hoverBackgroundColor: BAR_HOVER,
      borderRadius: 4, borderSkipped: "start", maxBarThickness: thick}}]}},
    options: {{indexAxis: "y", responsive: true, maintainAspectRatio: false, layout: {{padding: {{right: 30}}}},
      plugins: {{legend: {{display: false}}, tooltip: {{displayColors: false, callbacks: {{label: (c) => " " + c.parsed.x + (c.parsed.x === 1 ? " study" : " studies")}}}}}},
      scales: {{x: {{display: false, beginAtZero: true}},
               y: {{grid: {{display: false}}, border: {{display: false}}, ticks: {{color: INK, font: {{weight: "600"}}}}}}}}}}
  }});
  const cur = String(new Date().getFullYear());
  if (C("ucYears")) new Chart(C("ucYears"), {{
    type: "bar", plugins: [values],
    // the current, partial year is marked on the axis ("to date") and drawn in the lighter step
    data: {{labels: YR.labels.map((y) => y === cur ? [y, "to date"] : y), datasets: [{{label: "Studies", data: YR.vals,
      backgroundColor: YR.labels.map((y) => y === cur ? BAR_PARTIAL : BAR),
      hoverBackgroundColor: YR.labels.map((y) => y === cur ? BAR : BAR_HOVER),
      borderRadius: 4, borderSkipped: "start", maxBarThickness: 54}}]}},
    options: {{responsive: true, maintainAspectRatio: false, layout: {{padding: {{top: 20}}}},
      plugins: {{legend: {{display: false}},
        tooltip: {{displayColors: false, callbacks: {{
          title: (i) => YR.labels[i[0].dataIndex] + (YR.labels[i[0].dataIndex] === cur ? " (to date)" : ""),
          label: (c) => " " + c.parsed.y + " studies"}}}}}},
      scales: {{y: {{display: false, beginAtZero: true}},
               x: {{grid: {{display: false}}, border: {{display: false}}, ticks: {{color: INK, font: {{weight: "600"}}}}}}}}}}
  }});
  if (C("ucCont")) new Chart(C("ucCont"), hbar(CONT.labels, CONT.vals, 26));
  if (C("ucSpecies")) new Chart(C("ucSpecies"), hbar(SP.labels, SP.vals, 26));
  if (C("ucCountries")) new Chart(C("ucCountries"), hbar(TOP.labels, TOP.vals, 22));
  /* institutions are rendered as a static multi-column HTML list (see .ilist) */
}})();
</script>
{bar_helper}
{behav_script}
{corpus_script}
<script>try {{ window.define = window.__odef2; }} catch (e) {{}}</script>
"""


if __name__ == "__main__":
    main()
