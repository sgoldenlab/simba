# -*- coding: utf-8 -*-
"""
Generate the Related Software page's tool table and its plain-text index:

* docs/_generated/related_tools_table.html -- one searchable, filterable table of every tool;
* docs/_generated/related_tools_index.rst  -- the same tools as a plain A-Z list, so the docs search
  (which indexes RST text, not the table's script) still finds tool names.

Rows come from two CSVs:

* misc/related_software.csv -- the curated list, grouped by section (edit this file by hand);
* misc/related_tools.csv    -- tools introduced by papers that cite SimBA (maintained by
  misc/related_tools_update.py).

A tool in both is shown once: the curated row's link, section and type, plus the citing paper's year,
authors and venue. Rows are embedded as JSON and rendered client-side (plain JS, no libraries).

Run:  python misc/related_tools_table.py
"""
import collections, csv, json, os, re
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
CURATED = os.path.join(HERE, "related_software.csv")
SRC = os.path.join(HERE, "related_tools.csv")
OUT = os.path.join(HERE, "..", "docs", "_generated", "related_tools_table.html")
INDEX = os.path.join(HERE, "..", "docs", "_generated", "related_tools_index.rst")

# Sections in display order; the citing-paper categories map onto the same names.
SECTIONS = ["Pose estimation", "3D pose estimation", "Tracking and identity", "Supervised behavior classification",
            "Unsupervised behavior discovery", "Kinematics and trajectory analysis", "Object detection",
            "Vocalisation and audio", "Neural data alignment", "Data standards and interoperability",
            "Analysis and real-time pipelines", "Manual annotation", "Home-cage monitoring", "Commercial platforms",
            "Hardware and acquisition", "Benchmarks and datasets", "Other"]
CATEGORIES = {"supervised-classification": "Supervised behavior classification",
              "unsupervised-behavior": "Unsupervised behavior discovery",
              "pose-estimation": "Pose estimation",
              "tracking": "Tracking and identity",
              "closed-loop/real-time": "Analysis and real-time pipelines",
              "gait/kinematics": "Kinematics and trajectory analysis",
              "hardware/acquisition": "Hardware and acquisition",
              "benchmark/dataset": "Benchmarks and datasets",
              "analysis-platform": "Analysis and real-time pipelines",
              "other-analysis": "Other"}
# Curated names that differ from the citing paper's tool name, so the two rows are shown as one.
ALIASES = {"AMBER-pipeline": "AMBER"}
CODE_HOSTS = re.compile(r"github\.com|gitlab\.|sourceforge\.net|bitbucket\.org|hackaday\.io|zenodo\.org")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def citing_fields(r):
    return {"y": int(r["year"]) if r["year"].isdigit() else 0,
            "lb": r.get("label", "").strip(),   # short descriptive label for tools the paper does not name
            "au": r.get("authors", "").strip(),
            "v": r.get("venue", "").strip(),
            "p": r.get("preprint") == "1",
            "cb": int(r.get("cited_by") or 0),
            "t": r["title"].strip(),
            "s": r["species"].strip(),
            "dl": r.get("details", "").strip(),   # longer description shown when a row is opened
            "alt": r["other_dois"].split()}


def gh_fields(r):
    return {"gs": int(r.get("gh_stars") or -1),      # GitHub stars, -1 = no GitHub repo
            "gp": r.get("gh_pushed", ""),            # last push, YYYY-MM
            "ga": r.get("gh_archived") == "1"}


def load():
    citing = {norm(r["tool_name"]): r for r in read(SRC) if r["tool_name"].strip()}
    used, tools = set(), []
    for r in read(CURATED):
        url, lic = r["url"].strip(), r["license"].strip()
        paper = "https://doi.org/" + r["paper"].strip() if r["paper"].strip() else ""
        if "doi.org/" in url:   # the tool's only link is its paper
            url, paper = "", paper or url
        t = {"n": r["name"].strip(), "lb": "", "c": r["section"].strip(), "sub": r["subsection"].strip(),
             "lic": lic, "mod": r["modality"].strip(), "si": r["simba_import"] == "1", "d": r["description"].strip(),
             "dl": r.get("details", "").strip(),
             "web": url, "wk": "code" if CODE_HOSTS.search(url) else "website",
             "code": url if (CODE_HOSTS.search(url) or (url and lic == "Open source")) else "", "url": paper,
             "y": 0, "au": "", "v": "", "p": False, "cb": 0, "t": "", "s": "", "alt": [], "src": "c"}
        t.update(gh_fields(r))
        keys = [norm(t["n"]), norm(t["n"].split("(")[0]), norm(t["n"].split(" / ")[0]), norm(ALIASES.get(t["n"], ""))]
        c = next((citing[k] for k in keys if k in citing), None)
        if c:   # also introduced by a paper citing SimBA: add the paper's metadata
            used.add(norm(c["tool_name"]))
            dl = t["dl"]
            t.update(citing_fields(c), src="cs")
            t["dl"] = dl or t["dl"]
            t["url"] = t["url"] or c["doi"].strip()
            if t["gs"] < 0:
                t.update(gh_fields(c))
        tools.append(t)
    for k, r in [(norm(r["tool_name"]), r) for r in read(SRC)]:
        if k and k in used:
            continue
        code = r["code_url"].strip()
        t = {"n": r["tool_name"].strip(), "c": CATEGORIES.get(r["category"], "Other"), "sub": "",
             "lic": "Open source" if code else "", "mod": "", "si": False, "d": r["one_line"].strip(),
             "web": code, "wk": "code", "code": code, "url": r["doi"].strip(), "src": "s"}
        t.update(citing_fields(r))
        t.update(gh_fields(r))
        tools.append(t)
    return tools


def write_index(tools):
    """Plain A-Z list in RST, inside a collapsed admonition, for the docs search and readers without JS."""
    names = sorted({(t["n"] or t["lb"] or t["t"]).strip() for t in tools}, key=str.lower)
    esc = lambda s: re.sub(r"([*`|_\\])", r"\\\1", s)
    lines = [".. admonition:: All tools, A-Z (plain list)", "   :class: toggle", ""] + \
            ["   * " + esc(n) for n in names] + [""]
    with open(INDEX, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main():
    tools = load()
    order = {s: i for i, s in enumerate(SECTIONS)}
    tools.sort(key=lambda t: (order.get(t["c"], len(SECTIONS)), (t["n"] or t["lb"] or t["t"]).lower()))
    present = [s for s in SECTIONS if any(t["c"] == s for t in tools)]
    years = sorted({t["y"] for t in tools if t["y"]}, reverse=True)
    j = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    n_cur = sum(t["src"] != "s" for t in tools)
    out = (TEMPLATE.replace("__DATE__", date.today().strftime("%B %d, %Y"))
           .replace("__TOTAL__", str(len(tools))).replace("__NCUR__", str(n_cur))
           .replace("__NCIT__", str(sum(t["src"] != "c" for t in tools)))
           .replace("__DATA__", j(tools)).replace("__CATS__", j(present)).replace("__YEARS__", j(years)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    write_index(tools)
    print(f"Wrote {os.path.normpath(OUT)}: {len(tools)} tools ({n_cur} curated) in {len(present)} sections")


# Raw string with __PLACEHOLDERS__ (no .format), so JS braces and backslashes are written as-is.
TEMPLATE = r"""<style>
.simba-rt{max-width:1000px;margin:14px auto 6px;font-size:13.5px;color:#23272e;}
.simba-rt-date{font-size:12.5px;color:#6b7280;margin:0 0 12px;}
.simba-rt-bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 8px;}
.simba-rt-bar input[type=search],.simba-rt-bar select{font:inherit;padding:7px 10px;border:1px solid #cbd5e1;border-radius:8px;background:#fff;color:#23272e;}
.simba-rt-bar input[type=search]{flex:1 1 170px;min-width:0;}
.simba-rt-bar select{flex:0 1 auto;min-width:0;max-width:190px;}
.simba-rt-bar label{display:flex;align-items:center;gap:5px;font-size:12.5px;color:#4b5563;white-space:nowrap;cursor:pointer;}
.simba-rt-bar button{font:inherit;padding:7px 12px;border:1px solid #cbd5e1;border-radius:8px;background:#f8fafc;color:#21567a;cursor:pointer;}
.simba-rt-checks{gap:14px !important;}
.simba-rt-count{font-size:12.5px;color:#6b7280;margin:0 0 8px;}
.simba-rt-wrap{background:#fff;border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 6px 20px rgba(33,86,122,.10);overflow:hidden;}
.simba-rt table{width:100%;table-layout:fixed;border-collapse:collapse;margin:0 !important;border:0 !important;}
.simba-rt th{text-align:left;font-size:12px;font-weight:700;color:#21567a;background:#f4f9fd;padding:9px 12px !important;border:0 !important;border-bottom:1px solid #e2e8f0 !important;cursor:pointer;user-select:none;white-space:nowrap;}
.simba-rt th[aria-sort="ascending"]::after{content:" \25B2";font-size:9px;}
.simba-rt th[aria-sort="descending"]::after{content:" \25BC";font-size:9px;}
.simba-rt td{padding:10px 12px !important;border:0 !important;border-top:1px solid #eef2f6 !important;vertical-align:top;white-space:normal !important;overflow-wrap:anywhere;background:#fff !important;}
.simba-rt tr.row{cursor:pointer;}
.simba-rt tr.row:hover td{background:#f8fbfe !important;}
.simba-rt .yr{font-variant-numeric:tabular-nums;font-weight:700;color:#21567a;white-space:nowrap !important;}
.simba-rt .nm{font-weight:700;color:#21567a;font-size:14px;margin-bottom:2px;}
.simba-rt .nm.lb{font-weight:600;font-style:italic;color:#4b6a85;}
.simba-rt .ttl{line-height:1.35;}
.simba-rt .ttl a{color:#1f4e79;}
.simba-rt .meta{font-size:12px;color:#6b7280;margin-top:3px;line-height:1.4;}
.simba-rt .meta a{color:#21567a;}
.simba-rt .sec{line-height:1.3;}
.simba-rt .sub{display:block;font-size:11.5px;color:#6b7280;margin-top:2px;}
.simba-rt .tag{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:9px;margin-left:6px;vertical-align:1px;background:#e8f4fb;color:#21567a;}
.simba-rt .tag.si{background:#fdf3e1;color:#8a5300;}
.simba-rt .tag.pre{background:#fdf0e6;color:#983412;}
.simba-rt .tag.old{background:#fff;color:#6b7280;border:1px solid #d1d5db;font-weight:600;}
.simba-rt .ty{display:inline-block;font-size:11px;font-weight:700;padding:1px 8px;border-radius:9px;white-space:nowrap;}
.simba-rt .ty.oss{background:#e7f6ec;color:#1d6b3a;}
.simba-rt .ty.comm{background:#eef2ff;color:#3b4a9c;}
.simba-rt .ty.free{background:#f3f4f6;color:#374151;}
.simba-rt .unk{color:#9ca3af;}
.simba-rt tr.dt td{border-top:0 !important;padding-top:0 !important;background:#fbfdff !important;}
.simba-rt .dtb{font-size:12.5px;line-height:1.55;color:#374151;border-left:3px solid #7fc1e3;padding:6px 10px;margin:0 0 4px;}
.simba-rt .dtb b{color:#21567a;}
.simba-rt .empty{text-align:center;color:#6b7280;padding:26px !important;}
.simba-rt-charts{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 12px;}
.simba-rt-chart{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:8px 12px 4px;position:relative;flex:1 1 300px;align-self:flex-start;min-width:0;box-sizing:border-box;}
.simba-rt-cats{display:grid;grid-template-columns:auto 1fr auto;gap:2px 8px;align-items:center;margin:4px 0 6px;font-size:10.5px;}
.simba-rt-cats .cl{color:#4b5563;max-width:200px;line-height:1.15;cursor:pointer;}
.simba-rt-cats .ct{height:9px;cursor:pointer;}
.simba-rt-cats .cb{height:9px;background:#2f7fb5;border-radius:0 4px 4px 0;min-width:2px;}
.simba-rt-cats .cn{font-weight:700;color:#23272e;font-variant-numeric:tabular-nums;text-align:right;cursor:pointer;}
.simba-rt-cats .row:hover .cb,.simba-rt-cats .row.sel .cb{background:#21567a;}
.simba-rt-cats .row{display:contents;}
.simba-rt-cats .row.sel .cl{color:#21567a;font-weight:700;}
.simba-rt-chart h4{margin:0;font-size:12.5px;color:#23272e;font-weight:700;}
.simba-rt-chart .csub{margin:0 0 2px;font-size:11px;color:#6b7280;}
.simba-rt-chart svg{display:block;width:100%;max-width:396px;height:auto;overflow:visible;margin:0 auto;}
.simba-rt-chart .bar{fill:#2f7fb5;}
.simba-rt-chart .bar.partial{fill:#2f7fb5;fill-opacity:.45;}
.simba-rt-chart .hit{fill:transparent;cursor:pointer;}
.simba-rt-chart .hit:hover + .bar,.simba-rt-chart .hit.sel + .bar{fill:#21567a;}
.simba-rt-chart .val{font-size:10.5px;font-weight:700;fill:#23272e;text-anchor:middle;font-variant-numeric:tabular-nums;}
.simba-rt-chart .yrl{font-size:10.5px;fill:#4b5563;text-anchor:middle;}
.simba-rt-chart .yrs{font-size:9px;fill:#6b7280;text-anchor:middle;}
.simba-rt-chart .base{stroke:#cbd5e1;stroke-width:1;}
.simba-rt-chart .bar,.simba-rt-chart text{pointer-events:none;}
.simba-rt-tip{position:absolute;pointer-events:none;background:#fff;border:1px solid #e2e8f0;border-radius:8px;box-shadow:0 4px 14px rgba(15,23,42,.12);padding:8px 10px;font-size:12px;line-height:1.45;color:#23272e;white-space:nowrap;display:none;z-index:2;}
.simba-rt-tip b{font-size:12.5px;}
.simba-rt-tip .muted{color:#6b7280;}
.simba-rt-foot{text-align:center;font-size:14px;color:#4b5563;margin:16px 4px 0;}
.simba-rt-foot a{color:#21567a;font-weight:700;cursor:pointer;}
.simba-rt-top{position:fixed;right:18px;bottom:18px;z-index:50;padding:7px 12px;border:1px solid #cbd5e1;border-radius:18px;background:#fff;color:#23272e;font-size:13px;text-decoration:none;box-shadow:0 2px 6px rgba(0,0,0,.12);opacity:0;pointer-events:none;transition:opacity .2s;cursor:pointer;}
.simba-rt-top.on{opacity:1;pointer-events:auto;}
.simba-rt-top:hover{border-color:#2f7fb5;color:#21567a;}
@media (max-width:680px){.simba-rt .c-ty,.simba-rt .c-yr{display:none;}.simba-rt-bar select{flex:1 1 45%;max-width:none;}.simba-rt th,.simba-rt td{padding-left:9px !important;padding-right:9px !important;}}
</style>
<div class="simba-rt" id="rtTop" data-total="__TOTAL__">
  <p class="simba-rt-date">Updated __DATE__ &middot; __TOTAL__ tools: __NCUR__ in the curated list, __NCIT__ introduced by papers citing SimBA</p>
  <div class="simba-rt-charts">
    <div class="simba-rt-chart">
      <h4>Tools by section</h4>
      <p class="csub">Click a section to filter</p>
      <div class="simba-rt-cats" id="rtCats" role="list"></div>
    </div>
    <div class="simba-rt-chart" id="rtChart">
      <h4>Tools from papers citing SimBA, per year</h4>
      <p class="csub">Hover for sections, click a bar to filter</p>
      <div class="simba-rt-tip" id="rtTip" role="status"></div>
    </div>
  </div>
  <div class="simba-rt-bar" id="rtBar">
    <input id="rtQ" type="search" placeholder="Search tools, species, sensors&hellip;" aria-label="Search tools">
    <select id="rtCat" aria-label="Filter by section"><option value="">All sections</option></select>
    <select id="rtLic" aria-label="Filter by type"><option value="">All types</option><option>Open source</option><option>Commercial</option><option>Freeware</option></select>
    <select id="rtSrc" aria-label="Filter by source"><option value="">All sources</option><option value="c">Curated list</option><option value="s">Papers citing SimBA</option></select>
    <select id="rtYr" aria-label="Filter by year"><option value="">All years</option></select>
  </div>
  <div class="simba-rt-bar simba-rt-checks">
    <label title="Produces data SimBA reads directly (pose formats or frame-wise annotations)"><input id="rtSi" type="checkbox"> SimBA import</label>
    <label title="Only tools with a public code or data link"><input id="rtCode" type="checkbox"> Has code</label>
    <label title="Only tools with a linked paper"><input id="rtPaper" type="checkbox"> Has paper</label>
    <label title="Hide tools whose GitHub repository is archived or has not been updated in two years"><input id="rtMaint" type="checkbox"> Hide unmaintained</label>
    <button id="rtReset" type="button">Reset</button>
  </div>
  <p class="simba-rt-count" id="rtCount" aria-live="polite"></p>
  <div class="simba-rt-wrap"><table>
    <thead><tr>
      <th data-k="key">Tool</th>
      <th data-k="ci" aria-sort="ascending" style="width:23%">Section</th>
      <th data-k="lic" class="c-ty" style="width:112px">Type</th>
      <th data-k="y" class="c-yr" style="width:62px">Year</th>
    </tr></thead>
    <tbody id="rtBody"></tbody>
  </table></div>
  <p class="simba-rt-foot">Click a row for details and links &middot; <a id="rtCsv" role="button" tabindex="0">Download the shown tools as CSV</a></p>
  <a class="simba-rt-top" id="rtUp" role="button" tabindex="0">&uarr; Back to filters</a>
</div>
<script>
(function(){
  const D = __DATA__, CAT = __CATS__, YR = __YEARS__;
  const $ = id => document.getElementById(id);
  const q = $("rtQ"), cat = $("rtCat"), lic = $("rtLic"), src = $("rtSrc"), yr = $("rtYr"), body = $("rtBody");
  const si = $("rtSi"), hasCode = $("rtCode"), hasPaper = $("rtPaper"), maint = $("rtMaint");
  const STALE_MONTHS = 24, now = new Date(), nowM = now.getFullYear() * 12 + now.getMonth();
  let shown = D;
  CAT.forEach(c => cat.add(new Option(c, c)));
  YR.forEach(y => yr.add(new Option(y, y)));
  D.forEach(d => {
    d.key = (d.n || d.lb || d.t).toLowerCase();
    d.ci = CAT.indexOf(d.c);
    d.hay = [d.n, d.lb, d.t, d.au, d.v, d.c, d.sub, d.s, d.d, d.dl, d.mod, d.lic, d.y].join(" ").toLowerCase();
    d.old = d.ga ? "Archived" : (d.gp && nowM - (+d.gp.slice(0, 4) * 12 + +d.gp.slice(5, 7) - 1) >= STALE_MONTHS ? "No updates since " + d.gp.slice(0, 4) : "");
  });
  let key = "ci", dir = 1;
  const open = new Set();

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function link(text, href) {
    const a = el("a", null, text);
    a.href = href; a.target = "_blank"; a.rel = "noopener";
    a.addEventListener("click", e => e.stopPropagation());
    return a;
  }
  // Spans must match the visible columns: on phones Type and Year are hidden, and a 4-column span
  // would keep their empty columns in the layout.
  const visibleCols = () => [...document.querySelectorAll(".simba-rt thead th")].filter(th => getComputedStyle(th).display !== "none").length;
  function sep(box) { if (box.childNodes.length) box.appendChild(document.createTextNode(" · ")); }

  const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const stars = n => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1).replace(/\.0$/, "") + "k" : String(n);
  const month = ym => ym ? MONTHS[+ym.slice(5, 7) - 1] + " " + ym.slice(0, 4) : "";
  const TY = {"Open source": "oss", "Commercial": "comm", "Freeware": "free"};

  function render() {
    const words = q.value.toLowerCase().split(/\s+/).filter(Boolean);
    const rows = D.filter(d =>
      (!cat.value || d.c === cat.value) &&
      (!lic.value || d.lic === lic.value) &&
      (!src.value || d.src.includes(src.value)) &&
      (!yr.value || String(d.y) === yr.value) &&
      (!si.checked || d.si) && (!hasCode.checked || d.code) && (!hasPaper.checked || d.url) &&
      (!maint.checked || !d.old) &&
      words.every(w => d.hay.includes(w)));
    rows.sort((a, b) => {
      const x = a[key], y = b[key];
      const c = typeof x === "number" ? x - y : String(x).localeCompare(String(y));
      return c * dir || a.ci - b.ci || a.key.localeCompare(b.key);
    });
    body.replaceChildren();
    rows.forEach(d => {
      const tr = el("tr", "row"), td = el("td");
      const nm = el("div", "nm" + (d.n ? "" : " lb"), d.n || d.lb);
      if (!d.n) nm.title = "The paper does not name its tool; this is a short description";
      if (d.si) { const t = el("span", "tag si", "SimBA import"); t.title = "Produces data SimBA reads directly"; nm.appendChild(t); }
      if (d.p) nm.appendChild(el("span", "tag pre", "Preprint"));
      if (d.old) { const t = el("span", "tag old", d.old); t.title = "Last code update on GitHub: " + d.gp; nm.appendChild(t); }
      td.appendChild(nm);
      const ttl = el("div", "ttl");
      if (d.src === "s") { if (d.url) ttl.appendChild(link(d.t, d.url)); else ttl.textContent = d.t; }
      else ttl.textContent = d.d;
      td.appendChild(ttl);
      const meta = el("div", "meta");
      if (d.mod) meta.appendChild(document.createTextNode(d.mod));
      [d.au, d.v, d.cb ? "cited by " + d.cb : ""].filter(Boolean).forEach(x => { sep(meta); meta.appendChild(document.createTextNode(x)); });
      if (d.gs >= 0) { sep(meta); const g = el("span", null, "★ " + stars(d.gs) + (d.gp ? " · updated " + month(d.gp) : "")); g.title = "GitHub stars and last code update"; meta.appendChild(g); }
      if (d.src !== "s") {
        if (d.web) { sep(meta); meta.appendChild(link(d.wk, d.web)); }
        if (d.url) { sep(meta); meta.appendChild(link("paper", d.url)); }
      } else if (!d.code) { sep(meta); meta.appendChild(el("span", "unk", "code: unknown")); }
      if (meta.childNodes.length) td.appendChild(meta);
      tr.appendChild(td);
      const sc = el("td", "sec", d.c);
      if (d.sub) sc.appendChild(el("span", "sub", d.sub));
      tr.appendChild(sc);
      const tc = el("td", "c-ty");
      if (d.lic) tc.appendChild(el("span", "ty " + TY[d.lic], d.lic));
      tr.appendChild(tc);
      tr.appendChild(el("td", "yr c-yr", d.y || ""));
      tr.addEventListener("click", () => { open.has(d) ? open.delete(d) : open.add(d); render(); });
      body.appendChild(tr);
      if (open.has(d)) {
        const dt = el("tr", "dt"), c = el("td");
        c.colSpan = visibleCols();
        const box = el("div", "dtb");
        const line = (label, node) => { if (box.childNodes.length) box.appendChild(el("br")); box.appendChild(el("b", null, label + ": ")); box.appendChild(typeof node === "string" ? document.createTextNode(node) : node); };
        line("What it does", d.dl || d.d || "—");
        if (d.src !== "s" && d.web) line(d.wk === "code" ? "Code" : "Website", link(d.web, d.web));
        if (d.src === "s") line("Code", d.code ? link(d.code, d.code) : el("span", "unk", "unknown (no public link found)"));
        if (d.url) line("Paper", link(d.t || d.url.replace(/^https?:\/\/(doi\.org\/)?/, ""), d.url));
        if (d.alt.length) {
          const s = el("span");
          d.alt.forEach((u, i) => { if (i) s.appendChild(document.createTextNode(", ")); s.appendChild(link(u.replace(/^https?:\/\/(doi\.org\/)?/, ""), u)); });
          line("Other versions", s);
        }
        if (d.s) line("Species", d.s);
        line("Listed", d.src === "c" ? "curated list" : d.src === "s" ? "introduced by a paper citing SimBA" : "curated list; also introduced by a paper citing SimBA");
        c.appendChild(box); dt.appendChild(c); body.appendChild(dt);
      }
    });
    if (!rows.length) {
      const tr = el("tr"), c = el("td", "empty", "No tools match these filters.");
      c.colSpan = visibleCols(); tr.appendChild(c); body.appendChild(tr);
    }
    shown = rows;
    $("rtCount").textContent = "Showing " + rows.length + " of " + D.length + " tools";
  }
  function update() { render(); markYear(); markCat(); }

  document.querySelectorAll(".simba-rt th[data-k]").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.k;
    dir = k === key ? -dir : (k === "y" ? -1 : 1);
    key = k;
    document.querySelectorAll(".simba-rt th[data-k]").forEach(h => h.removeAttribute("aria-sort"));
    th.setAttribute("aria-sort", dir > 0 ? "ascending" : "descending");
    render();
  }));
  [q, cat, lic, src, yr, si, hasCode, hasPaper, maint].forEach(e => e.addEventListener("input", update));
  $("rtReset").addEventListener("click", () => {
    q.value = cat.value = lic.value = src.value = yr.value = "";
    si.checked = hasCode.checked = hasPaper.checked = maint.checked = false;
    update();
  });

  // CSV of the rows currently shown (after search and filters), built in the browser.
  function downloadCsv() {
    const cols = [["Tool", "n"], ["Label", "lb"], ["Section", "c"], ["Subsection", "sub"], ["Type", "lic"], ["Modality", "mod"],
                  ["SimBA import", "si"], ["Description", "d"], ["Details", "dl"], ["Website or code", "web"], ["Paper", "url"], ["Paper title", "t"],
                  ["Year", "y"], ["Authors", "au"], ["Venue", "v"], ["Cited by", "cb"], ["Species", "s"], ["Last code update", "gp"]];
    const esc = v => { v = v == null || v === false || v === 0 ? "" : v === true ? "yes" : String(v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
    const lines = [cols.map(c => c[0]).join(",")].concat(shown.map(d => cols.map(c => esc(d[c[1]])).join(",")));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8" }));
    a.download = "simba_related_software.csv";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  $("rtCsv").addEventListener("click", downloadCsv);
  $("rtCsv").addEventListener("keydown", e => { if (e.key === "Enter") downloadCsv(); });

  // Tools per section: horizontal bars in one hue, in section order; a click sets the section filter.
  const catBox = $("rtCats"), catRows = {};
  const catMax = Math.max(...CAT.map(c => D.filter(d => d.c === c).length));
  CAT.forEach(c => {
    const n = D.filter(d => d.c === c).length, row = el("div", "row");
    row.setAttribute("role", "listitem");
    row.title = c + ": " + n + " tools";
    const track = el("div", "ct"), bar = el("div", "cb");
    bar.style.width = (100 * n / catMax) + "%";
    track.appendChild(bar);
    [el("span", "cl", c), track, el("span", "cn", n)].forEach(x => {
      x.addEventListener("click", () => { cat.value = cat.value === c ? "" : c; update(); });
      row.appendChild(x);
    });
    catBox.appendChild(row); catRows[c] = row;
  });
  function markCat() { Object.entries(catRows).forEach(([c, r]) => r.classList.toggle("sel", cat.value === c)); }

  // Tools from citing papers per year: one series, so one hue and no legend; the current year is drawn lighter.
  const NS = "http://www.w3.org/2000/svg", chart = $("rtChart"), tip = $("rtTip");
  const thisYear = now.getFullYear();
  const years = [...new Set(D.map(d => d.y).filter(Boolean))].sort((a, b) => a - b);
  const perYear = years.map(y => {
    const rows = D.filter(d => d.y === y), byCat = {};
    rows.forEach(d => byCat[d.c] = (byCat[d.c] || 0) + 1);
    return { y, n: rows.length, cats: Object.entries(byCat).sort((a, b) => b[1] - a[1]) };
  });
  const W = 396, H = 96, top = 14, bottom = 28, band = W / years.length, bw = Math.min(16, band * 0.5);
  const max = Math.max(...perYear.map(p => p.n)), sy = n => (H - top - bottom) * n / max;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Tools from papers citing SimBA per year: " + perYear.map(p => p.y + " " + p.n).join(", "));
  const mk = (tag, attrs, text) => {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    if (text != null) e.textContent = text;
    return e;
  };
  const baseY = H - bottom;
  svg.appendChild(mk("line", { class: "base", x1: 0, x2: W, y1: baseY + .5, y2: baseY + .5 }));
  const hits = {};
  perYear.forEach((p, i) => {
    const cx = band * i + band / 2, h = Math.max(sy(p.n), 2), x = cx - bw / 2, yTop = baseY - h, r = Math.min(4, h);
    const hit = mk("rect", { class: "hit", x: band * i, y: top - 12, width: band, height: H - top + 12 });
    // 4px rounded data end, square at the baseline
    const bar = mk("path", { class: "bar" + (p.y === thisYear ? " partial" : ""),
      d: `M${x},${baseY}V${yTop + r}Q${x},${yTop} ${x + r},${yTop}H${x + bw - r}Q${x + bw},${yTop} ${x + bw},${yTop + r}V${baseY}Z` });
    svg.appendChild(hit); svg.appendChild(bar);
    svg.appendChild(mk("text", { class: "val", x: cx, y: yTop - 4 }, p.n));
    svg.appendChild(mk("text", { class: "yrl", x: cx, y: baseY + 13 }, p.y));
    if (p.y === thisYear) svg.appendChild(mk("text", { class: "yrs", x: cx, y: baseY + 24 }, "so far"));
    hits[p.y] = hit;
    hit.addEventListener("mousemove", e => {
      tip.replaceChildren();
      tip.appendChild(el("b", null, p.y + (p.y === thisYear ? " (so far)" : "") + " · " + p.n + " tools"));
      p.cats.forEach(([c, n]) => { tip.appendChild(el("br")); tip.appendChild(el("span", "muted", n + "  " + c)); });
      tip.style.display = "block";
      const box = chart.getBoundingClientRect(), tw = tip.offsetWidth;
      tip.style.left = Math.min(Math.max(e.clientX - box.left + 12, 8), box.width - tw - 8) + "px";
      tip.style.top = (e.clientY - box.top + 12) + "px";
    });
    hit.addEventListener("mouseleave", () => tip.style.display = "none");
    hit.addEventListener("click", () => { yr.value = yr.value === String(p.y) ? "" : String(p.y); update(); });
  });
  function markYear() { Object.entries(hits).forEach(([y, h]) => h.classList.toggle("sel", yr.value === y)); }
  chart.appendChild(svg);

  // "Back to filters" once the filter bar has scrolled out of view.
  const bar = $("rtBar"), up = $("rtUp");
  const toTop = () => $("rtTop").scrollIntoView({ behavior: "smooth" });
  up.addEventListener("click", toTop);
  up.addEventListener("keydown", e => { if (e.key === "Enter") toTop(); });
  const toggleUp = () => up.classList.toggle("on", bar.getBoundingClientRect().bottom < 0);
  window.addEventListener("scroll", toggleUp, { passive: true });

  // Old section links (e.g. #pose-estimation, #tools-from-papers-citing-simba) open the table filtered.
  const slug = s => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const h = decodeURIComponent(location.hash.slice(1)).toLowerCase();
  if (h) {
    const hit = CAT.filter(c => h.endsWith(slug(c))).sort((a, b) => b.length - a.length)[0];
    if (hit) cat.value = hit;
    else if (h.endsWith("tools-from-papers-citing-simba")) src.value = "s";
    if (hit || src.value) setTimeout(toTop, 0);
  }
  update();
  toggleUp();
})();
</script>
"""

if __name__ == "__main__":
    main()
