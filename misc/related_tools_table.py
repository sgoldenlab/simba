# -*- coding: utf-8 -*-
"""
Generate docs/_generated/related_tools_table.html — a searchable, filterable table of
behavior-analysis tools introduced by papers that cite SimBA.

Rows come from misc/related_tools.csv (curated: one row per tool, deduplicated across
preprint/journal versions). They are embedded as JSON and rendered client-side (plain JS,
no libraries), in the same style as the use-case table.

Run:  python misc/related_tools_table.py
"""
import collections, csv, json, os, re
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "related_tools.csv")
OUT = os.path.join(HERE, "..", "docs", "_generated", "related_tools_table.html")
# Tools already in the curated tables of this page get an "Also listed above" tag.
PAGE = os.path.join(HERE, "..", "docs", "simba.related_software.rst")

CATEGORIES = {"supervised-classification": "Supervised classification",
              "unsupervised-behavior": "Unsupervised / self-supervised",
              "pose-estimation": "Pose estimation",
              "tracking": "Tracking",
              "closed-loop/real-time": "Real-time / closed-loop",
              "gait/kinematics": "Gait & kinematics",
              "hardware/acquisition": "Hardware & acquisition",
              "benchmark/dataset": "Benchmarks & datasets",
              "other-analysis": "Other analysis"}


def main():
    with open(SRC, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(PAGE, encoding="utf-8") as f:
        curated = "\n".join(l for l in f if l.lstrip().startswith("* - `"))
    tools = []
    for r in rows:
        name = re.sub(r"\s*\(.*?\)", "", r["tool_name"]).strip()
        tools.append({
            "listed": bool(name) and re.search(r"`" + re.escape(name) + r"[\s<]", curated) is not None,
            "y": int(r["year"]) if r["year"].isdigit() else 0,
            "n": r["tool_name"].strip(),
            "t": r["title"].strip(),
            "c": CATEGORIES.get(r["category"], "Other analysis"),
            "s": r["species"].strip(),
            "d": r["one_line"].strip(),
            "code": r["code_url"].strip(),
            "url": r["doi"].strip(),
            "alt": r["other_dois"].split(),
        })
    tools.sort(key=lambda t: (-t["y"], (t["n"] or t["t"]).lower()))

    cats = [c for c, _ in collections.Counter(t["c"] for t in tools).most_common()]
    years = sorted({t["y"] for t in tools if t["y"]}, reverse=True)
    j = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.format(pull_date=date.today().strftime("%B %d, %Y"), total=len(tools),
                           data=j(tools), cats=j(cats), years=j(years))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {os.path.normpath(OUT)}: {len(tools)} tools in {len(cats)} categories, "
          f"{sum(t['listed'] for t in tools)} also in the curated tables")


TEMPLATE = """<style>
.simba-rt{{max-width:980px;margin:14px auto 6px;font-size:13.5px;color:#23272e;}}
.simba-rt-date{{font-size:12.5px;color:#6b7280;margin:0 0 12px;}}
.simba-rt-bar{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 10px;}}
.simba-rt-bar input,.simba-rt-bar select{{font:inherit;padding:7px 10px;border:1px solid #cbd5e1;border-radius:8px;background:#fff;color:#23272e;}}
.simba-rt-bar input{{flex:1 1 200px;min-width:0;}}
.simba-rt-bar select{{flex:0 1 auto;min-width:0;max-width:210px;}}
.simba-rt-bar button{{font:inherit;padding:7px 12px;border:1px solid #cbd5e1;border-radius:8px;background:#f8fafc;color:#21567a;cursor:pointer;}}
.simba-rt-count{{font-size:12.5px;color:#6b7280;margin:0 0 8px;}}
.simba-rt-wrap{{background:#fff;border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 6px 20px rgba(33,86,122,.10);overflow:hidden;}}
.simba-rt table{{width:100%;table-layout:fixed;border-collapse:collapse;margin:0 !important;border:0 !important;}}
.simba-rt th{{text-align:left;font-size:12px;font-weight:700;color:#21567a;background:#f4f9fd;padding:9px 12px !important;border:0 !important;border-bottom:1px solid #e2e8f0 !important;cursor:pointer;user-select:none;white-space:nowrap;}}
.simba-rt th[aria-sort="ascending"]::after{{content:" \\25B2";font-size:9px;}}
.simba-rt th[aria-sort="descending"]::after{{content:" \\25BC";font-size:9px;}}
.simba-rt td{{padding:10px 12px !important;border:0 !important;border-top:1px solid #eef2f6 !important;vertical-align:top;white-space:normal !important;overflow-wrap:anywhere;background:#fff !important;}}
.simba-rt tr.row{{cursor:pointer;}}
.simba-rt tr.row:hover td{{background:#f8fbfe !important;}}
.simba-rt .yr{{font-variant-numeric:tabular-nums;font-weight:700;color:#21567a;white-space:nowrap !important;}}
.simba-rt .nm{{font-weight:700;color:#21567a;font-size:14px;margin-bottom:2px;}}
.simba-rt .ttl{{line-height:1.35;}}
.simba-rt .tag{{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:9px;margin-left:6px;vertical-align:1px;background:#e8f4fb;color:#21567a;}}
.simba-rt .ttl a{{color:#1f4e79;}}
.simba-rt tr.dt td{{border-top:0 !important;padding-top:0 !important;background:#fbfdff !important;}}
.simba-rt .dtb{{font-size:12.5px;line-height:1.5;color:#374151;border-left:3px solid #7fc1e3;padding:6px 10px;margin:0 0 4px;}}
.simba-rt .dtb b{{color:#21567a;}}
.simba-rt .empty{{text-align:center;color:#6b7280;padding:26px !important;}}
.simba-rt-foot{{text-align:center;font-size:14px;color:#4b5563;margin:16px 4px 0;}}
@media (max-width:680px){{.simba-rt .c-sp{{display:none;}}.simba-rt-bar select{{flex:1 1 40%;max-width:none;}}.simba-rt th,.simba-rt td{{padding-left:9px !important;padding-right:9px !important;}}}}
</style>
<div class="simba-rt">
  <p class="simba-rt-date">Updated {pull_date} &middot; {total} tools</p>
  <div class="simba-rt-bar">
    <input id="rtQ" type="search" placeholder="Search tools, titles, species&hellip;" aria-label="Search tools">
    <select id="rtCat" aria-label="Filter by category"><option value="">All categories</option></select>
    <select id="rtYr" aria-label="Filter by year"><option value="">All years</option></select>
    <button id="rtReset" type="button">Reset</button>
  </div>
  <p class="simba-rt-count" id="rtCount" aria-live="polite"></p>
  <div class="simba-rt-wrap"><table>
    <thead><tr>
      <th data-k="y" aria-sort="descending" style="width:68px">Year</th>
      <th data-k="key">Tool</th>
      <th data-k="c" style="width:24%">Category</th>
      <th data-k="s" class="c-sp" style="width:14%">Species</th>
    </tr></thead>
    <tbody id="rtBody"></tbody>
  </table></div>
  <p class="simba-rt-foot">Click a row for a short description and code link</p>
</div>
<script>
(function(){{
  const D = {data}, CAT = {cats}, YR = {years};
  const $ = id => document.getElementById(id);
  const q = $("rtQ"), cat = $("rtCat"), yr = $("rtYr"), body = $("rtBody");
  CAT.forEach(c => cat.add(new Option(c, c)));
  YR.forEach(y => yr.add(new Option(y, y)));
  D.forEach(d => {{
    d.key = (d.n || d.t).toLowerCase();
    d.hay = [d.n, d.t, d.c, d.s, d.d, d.y].join(" ").toLowerCase();
  }});
  let key = "y", dir = -1;
  const open = new Set();

  function el(tag, cls, text) {{
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }}
  function link(text, href) {{
    const a = el("a", null, text);
    a.href = href; a.target = "_blank"; a.rel = "noopener";
    a.addEventListener("click", e => e.stopPropagation());
    return a;
  }}

  function render() {{
    const words = q.value.toLowerCase().split(/\\s+/).filter(Boolean);
    const rows = D.filter(d =>
      (!cat.value || d.c === cat.value) &&
      (!yr.value || String(d.y) === yr.value) &&
      words.every(w => d.hay.includes(w)));
    rows.sort((a, b) => {{
      const x = a[key], y = b[key];
      const c = typeof x === "number" ? x - y : String(x).localeCompare(String(y));
      return c * dir || b.y - a.y || a.key.localeCompare(b.key);
    }});
    body.replaceChildren();
    rows.forEach(d => {{
      const tr = el("tr", "row");
      tr.appendChild(el("td", "yr", d.y || ""));
      const td = el("td");
      if (d.n) {{
        const nm = el("div", "nm", d.n);
        if (d.listed) nm.appendChild(el("span", "tag", "Also listed above"));
        td.appendChild(nm);
      }}
      const ttl = el("div", "ttl");
      if (d.url) ttl.appendChild(link(d.t, d.url)); else ttl.textContent = d.t;
      td.appendChild(ttl);
      tr.appendChild(td);
      tr.appendChild(el("td", null, d.c));
      tr.appendChild(el("td", "c-sp", d.s));
      tr.addEventListener("click", () => {{ open.has(d) ? open.delete(d) : open.add(d); render(); }});
      body.appendChild(tr);
      if (open.has(d)) {{
        const dt = el("tr", "dt"), c = el("td");
        c.colSpan = 4;
        const box = el("div", "dtb");
        box.appendChild(el("b", null, "What it does: "));
        box.appendChild(document.createTextNode(d.d || "\\u2014"));
        if (d.code) {{
          box.appendChild(el("br"));
          box.appendChild(el("b", null, "Code: "));
          box.appendChild(link(d.code, d.code));
        }}
        if (d.alt.length) {{
          box.appendChild(el("br"));
          box.appendChild(el("b", null, "Other versions: "));
          d.alt.forEach((u, i) => {{ if (i) box.appendChild(document.createTextNode(", ")); box.appendChild(link(u.replace(/^https?:\\/\\/(doi\\.org\\/)?/, ""), u)); }});
        }}
        c.appendChild(box); dt.appendChild(c); body.appendChild(dt);
      }}
    }});
    if (!rows.length) {{
      const tr = el("tr"), c = el("td", "empty", "No tools match these filters.");
      c.colSpan = 4; tr.appendChild(c); body.appendChild(tr);
    }}
    $("rtCount").textContent = "Showing " + rows.length + " of " + D.length + " tools";
  }}

  document.querySelectorAll(".simba-rt th[data-k]").forEach(th => th.addEventListener("click", () => {{
    const k = th.dataset.k;
    dir = k === key ? -dir : (k === "y" ? -1 : 1);
    key = k;
    document.querySelectorAll(".simba-rt th[data-k]").forEach(h => h.removeAttribute("aria-sort"));
    th.setAttribute("aria-sort", dir > 0 ? "ascending" : "descending");
    render();
  }}));
  [q, cat, yr].forEach(e => e.addEventListener("input", render));
  $("rtReset").addEventListener("click", () => {{ q.value = cat.value = yr.value = ""; render(); }});
  render();
}})();
</script>
"""

if __name__ == "__main__":
    main()
