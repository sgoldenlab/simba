# -*- coding: utf-8 -*-
"""
Generate docs/_generated/usecase_table.html — a searchable, filterable table of every
published SimBA use-case, built from the same public Google Sheet as the use-case map.

The rows are embedded as JSON and rendered client-side (plain JS, no libraries), so the
page works offline once built. Reads the sheet as public CSV — no credentials.

Run:  python misc/usecase_table.py
"""
import collections, json, os, re, sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from institution_coords import INSTITUTION_ALIASES
from usecase_map_stats import SHEET_ID, SPECIES_NORM, fetch_rows

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "_generated", "usecase_table.html")

# Journal-column values that are preprint servers / repositories rather than journals.
PREPRINT_SOURCES = {"biorxiv", "medrxiv", "arxiv", "researchsquare", "research square",
                    "ssrn", "researchgate"}


def institutions(raw):
    out = []
    for tok in raw.split(","):
        name = re.sub(r"\s+", " ", tok).strip()
        canon = INSTITUTION_ALIASES.get(name, name)
        if canon and canon not in out:
            out.append(canon)
    return out


def main():
    rows = fetch_rows()
    cols = {c.strip().upper(): c for c in rows[0].keys()}

    def cell(r, name):
        return (r.get(cols.get(name, "")) or "").strip()

    studies = []
    for r in rows:
        year, month = cell(r, "YEAR"), cell(r, "MONTH")
        journal, url = cell(r, "JOURNAL"), cell(r, "URL")
        sp = cell(r, "SPECIES")
        studies.append({
            "y": int(year) if year.isdigit() else 0,
            "m": int(month) if month.isdigit() else 0,
            "t": cell(r, "TITLE"),
            "u": cell(r, "USE CASE"),
            "j": journal,
            "p": journal.lower() in PREPRINT_SOURCES,
            "fa": cell(r, "FIRST AUTHOR"),
            "sa": cell(r, "SENIOR AUTHOR"),
            "i": institutions(cell(r, "AUTHOR INSTITUTIONS")),
            "c": cell(r, "COUNTRIES"),
            "s": SPECIES_NORM.get(sp.lower(), sp),
            "url": url if url.lower().startswith("http") else "",
        })
    studies.sort(key=lambda s: (-s["y"], -s["m"], s["t"].lower()))

    species = [s for s, _ in collections.Counter(s["s"] for s in studies if s["s"]).most_common()]
    years = sorted({s["y"] for s in studies if s["y"]}, reverse=True)
    n_pre = sum(s["p"] for s in studies)

    # "</" inside embedded JSON would close the <script> tag early.
    j = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.format(
        pull_date=date.today().strftime("%B %d, %Y"), total=len(studies),
        n_peer=len(studies) - n_pre, n_pre=n_pre, sheet_id=SHEET_ID,
        data=j(studies), species=j(species), years=j(years))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {os.path.normpath(OUT)}: {len(studies)} studies "
          f"({len(studies) - n_pre} peer-reviewed, {n_pre} preprints)")


TEMPLATE = """<style>
.simba-ut{{max-width:980px;margin:14px auto 6px;font-size:13.5px;color:#23272e;}}
.simba-ut-date{{font-size:12.5px;color:#6b7280;margin:0 0 12px;}}
.simba-ut-bar{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 10px;}}
.simba-ut-bar input,.simba-ut-bar select{{font:inherit;padding:7px 10px;border:1px solid #cbd5e1;border-radius:8px;background:#fff;color:#23272e;}}
.simba-ut-bar input{{flex:1 1 200px;min-width:0;}}
.simba-ut-bar select{{flex:0 1 auto;min-width:0;max-width:170px;}}
.simba-ut-bar button{{font:inherit;padding:7px 12px;border:1px solid #cbd5e1;border-radius:8px;background:#f8fafc;color:#21567a;cursor:pointer;}}
.simba-ut-count{{font-size:12.5px;color:#6b7280;margin:0 0 8px;}}
.simba-ut-wrap{{background:#fff;border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 6px 20px rgba(33,86,122,.10);overflow:hidden;}}
.simba-ut table{{width:100%;table-layout:fixed;border-collapse:collapse;margin:0 !important;border:0 !important;}}
.simba-ut th{{text-align:left;font-size:12px;font-weight:700;color:#21567a;background:#f4f9fd;padding:9px 12px !important;border:0 !important;border-bottom:1px solid #e2e8f0 !important;cursor:pointer;user-select:none;white-space:nowrap;}}
.simba-ut th[aria-sort="ascending"]::after{{content:" \\25B2";font-size:9px;}}
.simba-ut th[aria-sort="descending"]::after{{content:" \\25BC";font-size:9px;}}
.simba-ut td{{padding:10px 12px !important;border:0 !important;border-top:1px solid #eef2f6 !important;vertical-align:top;white-space:normal !important;overflow-wrap:anywhere;background:#fff !important;}}
.simba-ut tr.row{{cursor:pointer;}}
.simba-ut tr.row:hover td{{background:#f8fbfe !important;}}
.simba-ut .yr{{font-variant-numeric:tabular-nums;font-weight:700;color:#21567a;white-space:nowrap !important;}}
.simba-ut .ttl{{font-weight:600;line-height:1.35;}}
.simba-ut .ttl a{{color:#1f4e79;}}
.simba-ut .meta{{font-size:12px;color:#6b7280;margin-top:3px;line-height:1.4;}}
.simba-ut .tag{{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:9px;margin-left:6px;vertical-align:1px;}}
.simba-ut .tag.pre{{background:#fdf0e6;color:#983412;}}
.simba-ut .tag.peer{{background:#e8f4fb;color:#21567a;}}
.simba-ut tr.uc td{{border-top:0 !important;padding-top:0 !important;background:#fbfdff !important;}}
.simba-ut .ucb{{font-size:12.5px;line-height:1.5;color:#374151;border-left:3px solid #7fc1e3;padding:6px 10px;margin:0 0 4px;}}
.simba-ut .ucb b{{color:#21567a;}}
.simba-ut .empty{{text-align:center;color:#6b7280;padding:26px !important;}}
.simba-ut-foot{{text-align:center;font-size:14px;color:#4b5563;margin:16px 4px 0;}}
@media (max-width:680px){{.simba-ut .c-sp,.simba-ut .c-ct{{display:none;}}.simba-ut-bar select{{flex:1 1 40%;max-width:none;}}.simba-ut th,.simba-ut td{{padding-left:9px !important;padding-right:9px !important;}}}}
</style>
<div class="simba-ut">
  <p class="simba-ut-date">Data pulled {pull_date} &middot; {total} studies ({n_peer} peer-reviewed, {n_pre} preprints)</p>
  <div class="simba-ut-bar">
    <input id="utQ" type="search" placeholder="Search titles, authors, institutions, behaviours&hellip;" aria-label="Search studies">
    <select id="utSp" aria-label="Filter by species"><option value="">All species</option></select>
    <select id="utYr" aria-label="Filter by year"><option value="">All years</option></select>
    <select id="utTy" aria-label="Filter by publication type"><option value="">All types</option><option value="peer">Peer-reviewed</option><option value="pre">Preprints</option></select>
    <button id="utReset" type="button">Reset</button>
  </div>
  <p class="simba-ut-count" id="utCount" aria-live="polite"></p>
  <div class="simba-ut-wrap"><table>
    <thead><tr>
      <th data-k="date" aria-sort="descending" style="width:68px">Year</th>
      <th data-k="t">Study</th>
      <th data-k="s" class="c-sp" style="width:17%">Species</th>
      <th data-k="c" class="c-ct" style="width:15%">Countries</th>
    </tr></thead>
    <tbody id="utBody"></tbody>
  </table></div>
  <p class="simba-ut-foot">Click a row to see how SimBA was used &middot; <a href="https://docs.google.com/spreadsheets/d/{sheet_id}/edit" target="_blank" rel="noopener">Source spreadsheet &rarr;</a></p>
</div>
<script>
(function(){{
  const D = {data}, SP = {species}, YR = {years};
  const $ = id => document.getElementById(id);
  const q = $("utQ"), sp = $("utSp"), yr = $("utYr"), ty = $("utTy"), body = $("utBody");
  SP.forEach(s => sp.add(new Option(s, s)));
  YR.forEach(y => yr.add(new Option(y, y)));
  D.forEach(d => {{
    d.date = d.y * 100 + d.m;
    d.hay = [d.t, d.u, d.j, d.fa, d.sa, d.i.join(" "), d.c, d.s, d.y].join(" ").toLowerCase();
  }});
  let key = "date", dir = -1;
  const open = new Set();

  function el(tag, cls, text) {{
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }}

  function render() {{
    const words = q.value.toLowerCase().split(/\\s+/).filter(Boolean);
    const rows = D.filter(d =>
      (!sp.value || d.s === sp.value) &&
      (!yr.value || String(d.y) === yr.value) &&
      (!ty.value || (ty.value === "pre") === d.p) &&
      words.every(w => d.hay.includes(w)));
    rows.sort((a, b) => {{
      const x = a[key], y = b[key];
      const c = typeof x === "number" ? x - y : String(x).localeCompare(String(y));
      return c * dir || b.date - a.date;
    }});
    body.replaceChildren();
    rows.forEach(d => {{
      const tr = el("tr", "row");
      tr.appendChild(el("td", "yr", d.y || ""));
      const td = el("td");
      const ttl = el("div", "ttl");
      if (d.url) {{
        const a = el("a", null, d.t);
        a.href = d.url; a.target = "_blank"; a.rel = "noopener";
        a.addEventListener("click", e => e.stopPropagation());
        ttl.appendChild(a);
      }} else ttl.textContent = d.t;
      ttl.appendChild(el("span", "tag " + (d.p ? "pre" : "peer"), d.p ? "Preprint" : "Peer-reviewed"));
      td.appendChild(ttl);
      const authors = [d.fa, d.sa].filter((v, i, a) => v && a.indexOf(v) === i).join(" \\u2026 ");
      td.appendChild(el("div", "meta", [authors, d.j].filter(Boolean).join(" \\u00b7 ")));
      tr.appendChild(td);
      tr.appendChild(el("td", "c-sp", d.s));
      tr.appendChild(el("td", "c-ct", d.c));
      tr.addEventListener("click", () => {{ open.has(d) ? open.delete(d) : open.add(d); render(); }});
      body.appendChild(tr);
      if (open.has(d)) {{
        const uc = el("tr", "uc"), c = el("td");
        c.colSpan = 4;
        const box = el("div", "ucb");
        box.appendChild(el("b", null, "Use case: "));
        box.appendChild(document.createTextNode(d.u || "\\u2014"));
        if (d.i.length) {{
          box.appendChild(el("br"));
          box.appendChild(el("b", null, "Institutions: "));
          box.appendChild(document.createTextNode(d.i.join(", ")));
        }}
        c.appendChild(box); uc.appendChild(c); body.appendChild(uc);
      }}
    }});
    if (!rows.length) {{
      const tr = el("tr"), c = el("td", "empty", "No studies match these filters.");
      c.colSpan = 4; tr.appendChild(c); body.appendChild(tr);
    }}
    $("utCount").textContent = "Showing " + rows.length + " of " + D.length + " studies";
  }}

  document.querySelectorAll(".simba-ut th[data-k]").forEach(th => th.addEventListener("click", () => {{
    const k = th.dataset.k;
    dir = k === key ? -dir : (k === "date" ? -1 : 1);
    key = k;
    document.querySelectorAll(".simba-ut th[data-k]").forEach(h => h.removeAttribute("aria-sort"));
    th.setAttribute("aria-sort", dir > 0 ? "ascending" : "descending");
    render();
  }}));
  [q, sp, yr, ty].forEach(e => e.addEventListener("input", render));
  $("utReset").addEventListener("click", () => {{ q.value = sp.value = yr.value = ty.value = ""; render(); }});
  render();
}})();
</script>
"""

if __name__ == "__main__":
    main()
