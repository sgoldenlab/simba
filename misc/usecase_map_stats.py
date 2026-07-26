# -*- coding: utf-8 -*-
"""
Generate docs/_generated/usecase_map.html — a "Global Reach" panel of published
SimBA use-cases, built from the public Google Sheet of citing studies.

Reuses the same vendored libraries as the download-stats page (jsvectormap +
Chart.js), so the styling matches. Reads the sheet as public CSV — no credentials.

Run:  python misc/usecase_map_stats.py
"""
import urllib.request, csv, io, gzip, json, collections, os, re, sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from institution_coords import INSTITUTION_ALIASES, INSTITUTION_COORDS

SHEET_ID = "169enc3Am2KQKifxj1F9KEKKLbftpMhBlw49zjl-egsY"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0"
OUT = os.path.join("docs", "_generated", "usecase_map.html")

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
ISO2CONT = {
    "US": "North America", "CA": "North America", "MX": "North America",
    "DE": "Europe", "ES": "Europe", "IT": "Europe", "CH": "Europe", "PL": "Europe",
    "NL": "Europe", "GB": "Europe", "SE": "Europe", "FR": "Europe", "IE": "Europe",
    "BE": "Europe", "AT": "Europe", "CZ": "Europe", "HU": "Europe", "PT": "Europe",
    "NO": "Europe", "FI": "Europe", "DK": "Europe", "RU": "Europe", "SI": "Europe",
    "RO": "Europe", "UA": "Europe", "GR": "Europe", "MD": "Europe", "ME": "Europe",
    "LT": "Europe", "EE": "Europe", "LU": "Europe", "TR": "Europe",
    "CN": "Asia", "IL": "Asia", "IN": "Asia", "JP": "Asia", "TH": "Asia",
    "HK": "Asia", "SG": "Asia", "KR": "Asia", "TW": "Asia", "IR": "Asia",
    "ID": "Asia", "PH": "Asia", "SA": "Asia", "LK": "Asia", "NP": "Asia",
    "MA": "Africa", "ZA": "Africa", "EG": "Africa",
    "BR": "South America", "AR": "South America", "CL": "South America",
    "CO": "South America", "EC": "South America",
    "AU": "Oceania", "NZ": "Oceania",
}
SPECIES_NORM = {
    "mouse": "Mouse", "mice": "Mouse", "rat": "Rat", "rats": "Rat",
    "rodent": "Rodent", "rodents": "Rodent", "zebrafish": "Zebrafish",
    "fish": "Fish", "gerbil": "Gerbil", "gerbils": "Gerbil",
}

# Sequential orange ramp for institution study-count (validated ordinal ramp:
# one hue, monotone light->dark, visible steps, light end clears the map surface).
# Orange because the country choropleth already uses blue -- two sequential
# contexts on one view => second takes the next hue as its own one-hue ramp.
INST_RAMP = ["#ee8a4a", "#df6a2b", "#c14d1b", "#983412", "#712509"]
INST_BUCKETS = ["1", "2", "3–4", "5–7", "8+"]  # legend labels, index-aligned to ramp


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


def tooltip_html(name, count, studies, cap=6):
    """Rich hover tooltip: institution, study count, and a capped study list."""
    rows = []
    for yr, title, jrnl in sorted(studies, key=lambda t: t[0], reverse=True)[:cap]:
        t = esc(title)
        if len(t) > 64:
            t = t[:63] + "…"
        yr_part = f'<b style="color:#f0a35a">{esc(yr)}</b> · ' if yr else ""
        jr_part = f' <span style="color:#b7c0cc">{esc(jrnl)}</span>' if jrnl else ""
        rows.append(f'<div style="margin-top:3px">{yr_part}{t}{jr_part}</div>')
    more = len(studies) - cap
    if more > 0:
        rows.append(f'<div style="margin-top:3px;color:#8b96a3">…+{more} more</div>')
    unit = "study" if count == 1 else "studies"
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
    inst_studies = collections.defaultdict(list)  # canonical -> [(year, title, journal)]

    for r in rows:
        seen_iso = set()
        # split on comma/semicolon, or a period FOLLOWED BY space (a typo separator,
        # e.g. "Sweden. Israel") -- but not bare dots inside "U.S."/"U.K."
        for c in re.split(r"[,;]|\.\s+", cell(r, "COUNTRIES")):
            key = c.strip().lower()
            if not key:
                continue
            iso = NAME2ISO.get(key)
            if not iso:
                unmapped[c.strip()] += 1
                continue
            seen_iso.add(iso)
        for iso in seen_iso:
            per_country[iso] += 1
            per_continent[ISO2CONT.get(iso, "Other")] += 1
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
                inst_studies[canon].append((y, cell(r, "TITLE"), j))
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
    # ISO-3166 alpha-2 -> display names for tooltips/bars
    ISO_NAME = {v: k.title() for k, v in NAME2ISO.items()}
    ISO_NAME.update({"US": "United States", "GB": "United Kingdom", "KR": "South Korea",
                     "HK": "Hong Kong", "CZ": "Czech Republic", "AE": "UAE"})

    country_map = dict(per_country)
    cont_labels = [c for c, _ in per_continent.most_common() if c != "Other"]
    cont_vals = [per_continent[c] for c in cont_labels]
    top_labels = [ISO_NAME.get(iso, iso) for iso, _ in top_country]
    top_vals = [n for _, n in top_country]
    sp_top = species.most_common(10)
    sp_labels = [s for s, _ in sp_top]
    sp_vals = [n for _, n in sp_top]

    # --- institutions: map pins (sized by study count) + top-institutions bar ---
    n_institutions = len(per_institution)
    markers, mark_tips = [], []
    for name, c in sorted(per_institution.items(), key=lambda kv: (-kv[1], kv[0])):
        lat, lon = INSTITUTION_COORDS[name]
        markers.append({"name": name, "coords": [lat, lon],
                        "style": {"initial": {"r": round(4 + 2.2 * c ** 0.5, 1),
                                              "fill": INST_RAMP[inst_bucket(c)]}}})
        mark_tips.append(tooltip_html(name, c, inst_studies[name]))
    top_inst = per_institution.most_common(20)
    inst_labels = [n for n, _ in top_inst]
    inst_vals = [c for _, c in top_inst]
    inst_colors = [INST_RAMP[inst_bucket(c)] for c in inst_vals]
    # graduated color+size legend: [color, diameter_px, label] per bucket
    legend_dots = [[INST_RAMP[i], round(2 * (4 + 2.2 * rc ** 0.5), 1), INST_BUCKETS[i]]
                   for i, rc in enumerate((1, 2, 4, 7, 11))]

    if unmapped:
        print("Unmapped country strings (add to NAME2ISO):", dict(unmapped))
    if unmapped_inst:
        print("Institutions without coords (add to institution_coords.py):", dict(unmapped_inst))

    pull_date = date.today().strftime("%B %d, %Y")
    html = _render(total, n_countries, n_continents, n_species, n_journals,
                   years, country_map, yr_labels, yr_counts, cont_labels,
                   cont_vals, top_labels, top_vals, ISO_NAME, pull_date,
                   n_institutions, markers, mark_tips, inst_labels, inst_vals,
                   sp_labels, sp_vals, inst_colors, legend_dots)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUT}: {total} studies, {n_countries} countries, "
          f"{n_continents} continents, {n_species} species, {n_journals} journals, "
          f"{n_institutions} institutions pinned")


def _render(total, n_countries, n_continents, n_species, n_journals, years,
            country_map, yr_labels, yr_counts, cont_labels, cont_vals,
            top_labels, top_vals, iso_name, pull_date,
            n_institutions, markers, mark_tips, inst_labels, inst_vals,
            sp_labels, sp_vals, inst_colors, legend_dots):
    yr_range = f"{years[0]}–{years[-1]}" if years else ""
    j = json.dumps
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
    # graduated color+size key for the institution pins (built from legend_dots)
    dots_html = "".join(
        f'<span style="width:{d}px;height:{d}px;border-radius:50%;background:{col};'
        f'opacity:.85;display:inline-block;border:1px solid #fff"></span><span>{lab}</span>'
        for col, d, lab in legend_dots)
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
.simba-uc-map{{position:relative;height:520px;background:radial-gradient(120% 120% at 50% 0%,#f4f9fd,#e8f1f8);border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 6px 20px rgba(33,86,122,.10);padding:10px;box-sizing:border-box;}}
.simba-uc-legend{{display:flex;align-items:center;gap:8px;justify-content:flex-end;font-size:11px;color:#6b7280;margin:8px 4px 0;}}
.simba-uc-legend i{{width:140px;height:10px;border-radius:5px;display:inline-block;background:linear-gradient(90deg,#e8f4fb,#7fc1e3,#2a7fb8,#143a5e);}}
.simba-uc-legend .sw{{width:13px;height:13px;border-radius:3px;display:inline-block;border:1px solid rgba(0,0,0,.06);}}
.simba-uc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start;margin-top:18px;}}
.simba-uc-panel{{position:relative;height:320px;background:#fff;border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 6px 20px rgba(33,86,122,.10);padding:14px 16px 10px;box-sizing:border-box;}}
@media (max-width:680px){{.simba-uc-grid{{grid-template-columns:1fr;}}}}
.simba-uc-date{{font-size:12.5px;color:#6b7280;margin:0 0 14px;}}
.simba-uc-foot{{text-align:center;font-size:14.5px;color:#4b5563;margin:18px 4px 0;}}
.simba-uc-foot a{{font-weight:600;}}
.jvm-tooltip{{background-color:#1f2937 !important;border-radius:9px !important;padding:9px 11px !important;max-width:300px !important;box-shadow:0 8px 24px rgba(15,23,42,.32) !important;}}
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
  <h3 class="simba-uc-h3">Studies by country</h3>
  <div class="simba-uc-map" id="ucMapCountries"></div>
  <div class="simba-uc-legend"><span style="color:#9aa0a8">fewer</span><i></i><span>more</span><span class="sw" style="background:#dce4ee;margin-left:6px"></span><span>none</span></div>
  <h3 class="simba-uc-h3">Institutions &amp; locations</h3>
  <div class="simba-uc-map" id="ucMapPins"></div>
  <div class="simba-uc-legend"><span style="color:#9aa0a8">studies per institution:</span>{dots_html}</div>
  <div class="simba-uc-grid">
    <div><h3 class="simba-uc-h3">Studies per year</h3><div class="simba-uc-panel"><canvas id="ucYears"></canvas></div></div>
    <div><h3 class="simba-uc-h3">By continent</h3><div class="simba-uc-panel"><canvas id="ucCont"></canvas></div></div>
  </div>
  <h3 class="simba-uc-h3">Studies by species</h3>
  <div class="simba-uc-panel" style="height:320px"><canvas id="ucSpecies"></canvas></div>
  <h3 class="simba-uc-h3">Top countries</h3>
  <div class="simba-uc-panel" style="height:420px"><canvas id="ucCountries"></canvas></div>
  <h3 class="simba-uc-h3">Top institutions</h3>
  <div class="simba-uc-panel" style="height:560px"><canvas id="ucInst"></canvas></div>
  <p class="simba-uc-foot"><a href="https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Full list of studies (spreadsheet) &rarr;</a> &middot; one entry per study &middot; multi-country studies counted in each country</p>
</div>
<script>window.__odef2 = window.define; try {{ window.define = undefined; }} catch (e) {{}}</script>
<link rel="stylesheet" href="_static/css/jsvectormap.min.css">
<script src="_static/js/jsvectormap.min.js"></script>
<script src="_static/js/jsvectormap-world.js"></script>
<script>
(function(){{
  if (typeof jsVectorMap === "undefined") return;
  const MAP = {j(country_map)};
  const MARKERS = {j(markers)};
  const MTIP = {j(mark_tips)};
  const SCALE = ["#e8f4fb", "#cfe6f5", "#9bcde9", "#5aa7d4", "#3a86c0", "#2a6299", "#143a5e"];
  const NB = SCALE.length - 1;
  const counts = Object.keys(MAP).map((k) => MAP[k] || 0);
  const maxLog = Math.log10(Math.max.apply(null, counts.concat([1])) + 1) || 1;
  const BUCK = {{}};
  for (const k in MAP) {{ let b = Math.ceil(Math.log10((MAP[k] || 0) + 1) / maxLog * NB); BUCK[k] = b < 1 ? 1 : (b > NB ? NB : b); }}
  // Map 1 -- countries shaded (choropleth only)
  if (document.getElementById("ucMapCountries")) new jsVectorMap({{
    selector: "#ucMapCountries", map: "world",
    zoomButtons: true, zoomOnScroll: true, backgroundColor: "transparent",
    regionStyle: {{initial: {{fill: "#dce4ee", stroke: "#ffffff", "stroke-width": 0.5}},
      hover: {{fill: "#f0c44a", "fill-opacity": 1}}}},
    series: {{regions: [{{attribute: "fill", values: BUCK, scale: SCALE}}]}},
    onRegionTooltipShow(event, tooltip, code) {{
      const v = MAP[code] || 0;
      tooltip.text(tooltip.text() + (v ? ": " + v + (v === 1 ? " study" : " studies") : ": no studies yet"), true);
    }}
  }});
  // Map 2 -- institution / location pins (neutral land, no shading)
  if (document.getElementById("ucMapPins")) new jsVectorMap({{
    selector: "#ucMapPins", map: "world",
    zoomButtons: true, zoomOnScroll: true, backgroundColor: "transparent",
    regionStyle: {{initial: {{fill: "#dce4ee", stroke: "#ffffff", "stroke-width": 0.5}},
      hover: {{fill: "#dce4ee"}}}},
    markers: MARKERS,
    markerStyle: {{
      initial: {{fill: "#c14d1b", stroke: "#ffffff", "stroke-width": 1.3, fillOpacity: 0.85, r: 5}},
      hover: {{fill: "#f0c44a", cursor: "pointer", fillOpacity: 1}}
    }},
    onMarkerTooltipShow(event, tooltip, code) {{
      if (MTIP[code]) tooltip.text(MTIP[code], true);
    }}
  }});
}})();
</script>
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
  const GRID = "#eef2f7", INK = "#23272e", MUT = "#6b7280";
  const PAL = ["#21567a","#2a7fb8","#38a8d4","#5cc6b3","#e0a33a","#e0653a","#b9c0c9"];
  const track = {{id: "track", beforeDatasetsDraw(chart) {{
    const ctx = chart.ctx, right = chart.chartArea.right, x0 = chart.scales.x.getPixelForValue(0);
    ctx.save(); ctx.fillStyle = GRID;
    chart.getDatasetMeta(0).data.forEach(function(b) {{
      const h = b.height, y = b.y - h / 2, w = right - x0, r = Math.min(6, h / 2);
      if (ctx.roundRect) {{ ctx.beginPath(); ctx.roundRect(x0, y, w, h, r); ctx.fill(); }} else {{ ctx.fillRect(x0, y, w, h); }}
    }}); ctx.restore();
  }}}};
  const cur = String(new Date().getFullYear());
  if (C("ucYears")) new Chart(C("ucYears"), {{
    type: "bar",
    data: {{labels: YR.labels, datasets: [{{label: "Studies", data: YR.vals,
      backgroundColor: YR.labels.map((y) => y === cur ? "#9ec9e6" : "#2a7fb8"), borderRadius: 4, maxBarThickness: 54}}]}},
    options: {{responsive: true, maintainAspectRatio: false,
      plugins: {{legend: {{display: false}}, tooltip: {{displayColors: false, callbacks: {{
        title: (i) => i[0].label + (i[0].label === cur ? " (to date)" : ""),
        label: (c) => " " + c.parsed.y + " studies"}}}}}},
      scales: {{y: {{beginAtZero: true, grid: {{color: GRID}}, ticks: {{color: MUT}}, title: {{display: true, text: "studies", color: MUT}}}},
               x: {{grid: {{display: false}}, ticks: {{color: INK, font: {{weight: "600"}}}}}}}}}}
  }});
  if (C("ucCont")) new Chart(C("ucCont"), {{
    type: "bar", plugins: [track],
    data: {{labels: CONT.labels, datasets: [{{label: "Studies", data: CONT.vals,
      backgroundColor: PAL, hoverBackgroundColor: PAL, borderRadius: 4, maxBarThickness: 26}}]}},
    options: {{indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: {{legend: {{display: false}}, tooltip: {{callbacks: {{label: (c) => " " + c.parsed.x + " studies"}}}}}},
      scales: {{x: {{beginAtZero: true, grid: {{display: false}}, ticks: {{color: MUT}}}}, y: {{grid: {{display: false}}, ticks: {{color: INK, font: {{weight: "600"}}}}}}}}}}
  }});
  if (C("ucSpecies")) new Chart(C("ucSpecies"), {{
    type: "bar", plugins: [track],
    data: {{labels: SP.labels, datasets: [{{label: "Studies", data: SP.vals,
      backgroundColor: PAL, hoverBackgroundColor: PAL, borderRadius: 4, maxBarThickness: 26}}]}},
    options: {{indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: {{legend: {{display: false}}, tooltip: {{callbacks: {{label: (c) => " " + c.parsed.x + " studies"}}}}}},
      scales: {{x: {{beginAtZero: true, grid: {{display: false}}, ticks: {{color: MUT}}}}, y: {{grid: {{display: false}}, ticks: {{color: INK, font: {{weight: "600"}}}}}}}}}}
  }});
  if (C("ucCountries")) new Chart(C("ucCountries"), {{
    type: "bar", plugins: [track],
    data: {{labels: TOP.labels, datasets: [{{label: "Studies", data: TOP.vals,
      backgroundColor: "#2a7fb8", hoverBackgroundColor: "#2a7fb8", borderRadius: 4, maxBarThickness: 22}}]}},
    options: {{indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: {{legend: {{display: false}}, tooltip: {{callbacks: {{label: (c) => " " + c.parsed.x + " studies"}}}}}},
      scales: {{x: {{beginAtZero: true, grid: {{display: false}}, ticks: {{color: MUT}}}}, y: {{grid: {{display: false}}, ticks: {{color: INK, font: {{weight: "600"}}}}}}}}}}
  }});
  if (C("ucInst")) new Chart(C("ucInst"), {{
    type: "bar", plugins: [track],
    data: {{labels: INST.labels, datasets: [{{label: "Studies", data: INST.vals,
      backgroundColor: INST.col, hoverBackgroundColor: "#f0c44a", borderRadius: 4, maxBarThickness: 20}}]}},
    options: {{indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: {{legend: {{display: false}}, tooltip: {{callbacks: {{label: (c) => " " + c.parsed.x + " studies"}}}}}},
      scales: {{x: {{beginAtZero: true, grid: {{display: false}}, ticks: {{color: MUT}}}}, y: {{grid: {{display: false}}, ticks: {{color: INK, font: {{weight: "600"}}}}}}}}}}
  }});
}})();
</script>
<script>try {{ window.define = window.__odef2; }} catch (e) {{}}</script>
"""


if __name__ == "__main__":
    main()
