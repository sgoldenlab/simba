# -*- coding: utf-8 -*-
"""
Generate docs/_generated/related_tools_table.html — a searchable, filterable table of
behavior-analysis tools introduced by papers that cite SimBA.

Rows come from misc/related_tools.csv (curated: one row per tool, deduplicated across
preprint/journal versions). They are embedded as JSON and rendered client-side (plain JS,
no libraries), in the same style as the use-case table.

Run:  python misc/related_tools_table.py
"""
import collections, csv, json, os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "related_tools.csv")
OUT = os.path.join(HERE, "..", "docs", "_generated", "related_tools_table.html")

# Display names match the section headings of the curated tables above this table on the page.
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


def main():
    with open(SRC, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tools = []
    for r in rows:
        tools.append({
            "y": int(r["year"]) if r["year"].isdigit() else 0,
            "n": r["tool_name"].strip(),
            "lb": r.get("label", "").strip(),   # short descriptive label for tools the paper does not name
            "au": r.get("authors", "").strip(),
            "v": r.get("venue", "").strip(),
            "p": r.get("preprint") == "1",
            "cb": int(r.get("cited_by") or 0),
            "gs": int(r.get("gh_stars") or -1),      # GitHub stars, -1 = no GitHub repo
            "gp": r.get("gh_pushed", ""),            # last push, YYYY-MM
            "ga": r.get("gh_archived") == "1",
            "t": r["title"].strip(),
            "c": CATEGORIES.get(r["category"], "Other"),
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
    print(f"Wrote {os.path.normpath(OUT)}: {len(tools)} tools in {len(cats)} categories")


TEMPLATE = """<style>
.simba-rt{{max-width:980px;margin:14px auto 6px;font-size:13.5px;color:#23272e;}}
.simba-rt-date{{font-size:12.5px;color:#6b7280;margin:0 0 12px;}}
.simba-rt-bar{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 10px;}}
.simba-rt-bar input,.simba-rt-bar select{{font:inherit;padding:7px 10px;border:1px solid #cbd5e1;border-radius:8px;background:#fff;color:#23272e;}}
.simba-rt-bar input{{flex:1 1 200px;min-width:0;}}
.simba-rt-bar select{{flex:0 1 auto;min-width:0;max-width:210px;}}
.simba-rt-bar label{{display:flex;align-items:center;gap:5px;font-size:12.5px;color:#4b5563;white-space:nowrap;cursor:pointer;}}
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
.simba-rt .nm.lb{{font-weight:600;font-style:italic;color:#4b6a85;}}
.simba-rt .meta{{font-size:12px;color:#6b7280;margin-top:3px;line-height:1.4;}}
.simba-rt .tag.pre{{background:#fdf0e6;color:#983412;}}
.simba-rt .tag.arc{{background:#f1f5f9;color:#475569;}}
.simba-rt .gh{{white-space:nowrap;}}
.simba-rt .tag{{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:9px;margin-left:6px;vertical-align:1px;background:#e8f4fb;color:#21567a;}}
.simba-rt .ttl a{{color:#1f4e79;}}
.simba-rt tr.dt td{{border-top:0 !important;padding-top:0 !important;background:#fbfdff !important;}}
.simba-rt .dtb{{font-size:12.5px;line-height:1.5;color:#374151;border-left:3px solid #7fc1e3;padding:6px 10px;margin:0 0 4px;}}
.simba-rt .dtb b{{color:#21567a;}}
.simba-rt .empty{{text-align:center;color:#6b7280;padding:26px !important;}}
.simba-rt-charts{{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 12px;}}
.simba-rt-chart{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:8px 12px 4px;position:relative;flex:1 1 300px;min-width:0;box-sizing:border-box;}}
.simba-rt-cats{{display:grid;grid-template-columns:auto 1fr auto;gap:2px 8px;align-items:center;margin:4px 0 6px;font-size:10.5px;}}
.simba-rt-cats .cl{{color:#4b5563;max-width:180px;line-height:1.15;cursor:pointer;}}
.simba-rt-cats .ct{{height:9px;cursor:pointer;}}
.simba-rt-cats .cb{{height:9px;background:#2f7fb5;border-radius:0 4px 4px 0;min-width:2px;}}
.simba-rt-cats .cn{{font-weight:700;color:#23272e;font-variant-numeric:tabular-nums;text-align:right;}}
.simba-rt-cats .row:hover .cb,.simba-rt-cats .row.sel .cb{{background:#21567a;}}
.simba-rt-cats .row{{display:contents;}}
.simba-rt-cats .row.sel .cl{{color:#21567a;font-weight:700;}}
.simba-rt-chart h4{{margin:0;font-size:12.5px;color:#23272e;font-weight:700;}}
.simba-rt-chart .sub{{margin:0 0 2px;font-size:11px;color:#6b7280;}}
.simba-rt-chart svg{{display:block;width:100%;max-width:396px;height:auto;overflow:visible;margin:0 auto;}}
.simba-rt-chart .bar{{fill:#2f7fb5;}}
.simba-rt-chart .bar.partial{{fill:#2f7fb5;fill-opacity:.45;}}
.simba-rt-chart .hit{{fill:transparent;cursor:pointer;}}
.simba-rt-chart .hit:hover + .bar,.simba-rt-chart .hit.sel + .bar{{fill:#21567a;}}
.simba-rt-chart .val{{font-size:10.5px;font-weight:700;fill:#23272e;text-anchor:middle;font-variant-numeric:tabular-nums;}}
.simba-rt-chart .yrl{{font-size:10.5px;fill:#4b5563;text-anchor:middle;}}
.simba-rt-chart .yrs{{font-size:9px;fill:#6b7280;text-anchor:middle;}}
.simba-rt-chart .base{{stroke:#cbd5e1;stroke-width:1;}}
.simba-rt-chart .bar,.simba-rt-chart text{{pointer-events:none;}}
.simba-rt-tip{{position:absolute;pointer-events:none;background:#fff;border:1px solid #e2e8f0;border-radius:8px;box-shadow:0 4px 14px rgba(15,23,42,.12);padding:8px 10px;font-size:12px;line-height:1.45;color:#23272e;white-space:nowrap;display:none;z-index:2;}}
.simba-rt-tip b{{font-size:12.5px;}}
.simba-rt-tip .muted{{color:#6b7280;}}
.simba-rt-foot{{text-align:center;font-size:14px;color:#4b5563;margin:16px 4px 0;}}
.simba-rt-foot a{{color:#21567a;font-weight:700;cursor:pointer;}}
.simba-rt .unk{{color:#9ca3af;}}
@media (max-width:680px){{.simba-rt .c-sp{{display:none;}}.simba-rt-bar select{{flex:1 1 40%;max-width:none;}}.simba-rt th,.simba-rt td{{padding-left:9px !important;padding-right:9px !important;}}}}
</style>
<div class="simba-rt" data-total="{total}">
  <p class="simba-rt-date">Updated {pull_date} &middot; {total} tools</p>
  <div class="simba-rt-charts">
    <div class="simba-rt-chart" id="rtChart">
      <h4>Tools introduced per year</h4>
      <p class="sub">Hover for categories, click a bar to filter</p>
      <div class="simba-rt-tip" id="rtTip" role="status"></div>
    </div>
    <div class="simba-rt-chart">
      <h4>Tools by category</h4>
      <p class="sub">Click a bar to filter</p>
      <div class="simba-rt-cats" id="rtCats" role="list"></div>
    </div>
  </div>
  <div class="simba-rt-bar">
    <input id="rtQ" type="search" placeholder="Search tools, titles, species&hellip;" aria-label="Search tools">
    <select id="rtCat" aria-label="Filter by category"><option value="">All categories</option></select>
    <select id="rtYr" aria-label="Filter by year"><option value="">All years</option></select>
    <label title="Only tools with a public code or data link"><input id="rtCode" type="checkbox"> Has code</label>
    <button id="rtReset" type="button">Reset</button>
  </div>
  <p class="simba-rt-count" id="rtCount" aria-live="polite"></p>
  <div class="simba-rt-wrap"><table>
    <thead><tr>
      <th data-k="y" aria-sort="descending" style="width:68px">Year</th>
      <th data-k="key">Tool</th>
      <th data-k="c" style="width:21%">Category</th>
      <th data-k="s" class="c-sp" style="width:12%">Species</th>
    </tr></thead>
    <tbody id="rtBody"></tbody>
  </table></div>
  <p class="simba-rt-foot">Click a row for a short description and code link &middot; <a id="rtCsv" role="button" tabindex="0">Download the shown tools as CSV</a></p>
</div>
<script>
(function(){{
  const D = {data}, CAT = {cats}, YR = {years};
  const $ = id => document.getElementById(id);
  const q = $("rtQ"), cat = $("rtCat"), yr = $("rtYr"), hasCode = $("rtCode"), body = $("rtBody");
  let shown = D;
  CAT.forEach(c => cat.add(new Option(c, c)));
  YR.forEach(y => yr.add(new Option(y, y)));
  D.forEach(d => {{
    d.key = (d.n || d.lb || d.t).toLowerCase();
    d.hay = [d.n, d.lb, d.t, d.au, d.v, d.c, d.s, d.d, d.y].join(" ").toLowerCase();
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

  const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const stars = n => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1).replace(/\.0$/, "") + "k" : String(n);
  const month = ym => ym ? MONTHS[+ym.slice(5, 7) - 1] + " " + ym.slice(0, 4) : "";

  function render() {{
    const words = q.value.toLowerCase().split(/\\s+/).filter(Boolean);
    const rows = D.filter(d =>
      (!cat.value || d.c === cat.value) &&
      (!yr.value || String(d.y) === yr.value) &&
      (!hasCode.checked || d.code) &&
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
      if (d.n) td.appendChild(el("div", "nm", d.n));
      else if (d.lb) {{
        const nm = el("div", "nm lb", d.lb);
        nm.title = "The paper does not name its tool; this is a short description";
        td.appendChild(nm);
      }}
      const ttl = el("div", "ttl");
      if (d.url) ttl.appendChild(link(d.t, d.url)); else ttl.textContent = d.t;
      td.appendChild(ttl);
      const meta = el("div", "meta", [d.au, d.v, d.cb ? "cited by " + d.cb : ""].filter(Boolean).join(" · "));
      if (d.gs >= 0) {{
        const gh = el("span", "gh", (meta.textContent ? " · " : "") + "★ " + stars(d.gs) + (d.gp ? " · updated " + month(d.gp) : ""));
        gh.title = "GitHub stars and last code update";
        meta.appendChild(gh);
      }}
      if (!d.code) meta.appendChild(el("span", "unk", (meta.textContent ? " · " : "") + "code: unknown"));
      if (d.ga) meta.appendChild(el("span", "tag arc", "Archived"));
      if (d.p) meta.appendChild(el("span", "tag pre", "Preprint"));
      if (meta.childNodes.length) td.appendChild(meta);
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
        box.appendChild(el("br"));
        box.appendChild(el("b", null, "Code: "));
        box.appendChild(d.code ? link(d.code, d.code) : el("span", "unk", "unknown (no public link found)"));
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
    shown = rows;
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
  [q, cat, yr, hasCode].forEach(e => e.addEventListener("input", () => {{ render(); markYear(); markCat(); }}));
  $("rtReset").addEventListener("click", () => {{
    q.value = cat.value = yr.value = ""; hasCode.checked = false; render(); markYear(); markCat();
  }});

  // CSV of the rows currently shown (after search and filters), built in the browser.
  function downloadCsv() {{
    const cols = [["Year", "y"], ["Tool", "n"], ["Label", "lb"], ["Title", "t"], ["Category", "c"], ["Species", "s"],
                  ["Description", "d"], ["Code", "code"], ["Paper", "url"], ["Authors", "au"], ["Venue", "v"],
                  ["Cited by", "cb"]];
    const esc = v => {{ v = v == null ? "" : String(v); return /[",\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }};
    const lines = [cols.map(c => c[0]).join(",")].concat(shown.map(d => cols.map(c => esc(d[c[1]])).join(",")));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob(["\\ufeff" + lines.join("\\n")], {{ type: "text/csv;charset=utf-8" }}));
    a.download = "simba_related_tools.csv";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }}
  $("rtCsv").addEventListener("click", downloadCsv);
  $("rtCsv").addEventListener("keydown", e => {{ if (e.key === "Enter") downloadCsv(); }});

  // Tools per category: horizontal bars in the same hue, largest first; a click sets the category filter.
  const catBox = $("rtCats"), catRows = {{}};
  const catMax = Math.max(...CAT.map(c => D.filter(d => d.c === c).length));
  CAT.forEach(c => {{
    const n = D.filter(d => d.c === c).length, row = el("div", "row");
    row.setAttribute("role", "listitem");
    row.title = c + ": " + n + " tools";
    const track = el("div", "ct"), bar = el("div", "cb");
    bar.style.width = (100 * n / catMax) + "%";
    track.appendChild(bar);
    [el("span", "cl", c), track, el("span", "cn", n)].forEach(x => {{
      x.addEventListener("click", () => {{ cat.value = cat.value === c ? "" : c; render(); markCat(); }});
      row.appendChild(x);
    }});
    catBox.appendChild(row); catRows[c] = row;
  }});
  function markCat() {{ Object.entries(catRows).forEach(([c, r]) => r.classList.toggle("sel", cat.value === c)); }}

  // Tools per year: one series, so one hue and no legend; the current year is drawn lighter as partial.
  const NS = "http://www.w3.org/2000/svg", chart = $("rtChart"), tip = $("rtTip");
  const thisYear = new Date().getFullYear();
  const years = [...new Set(D.map(d => d.y).filter(Boolean))].sort((a, b) => a - b);
  const perYear = years.map(y => {{
    const rows = D.filter(d => d.y === y), byCat = {{}};
    rows.forEach(d => byCat[d.c] = (byCat[d.c] || 0) + 1);
    return {{ y, n: rows.length, cats: Object.entries(byCat).sort((a, b) => b[1] - a[1]) }};
  }});
  const W = 396, H = 96, top = 14, bottom = 28, band = W / years.length, bw = Math.min(16, band * 0.5);
  const max = Math.max(...perYear.map(p => p.n)), sy = n => (H - top - bottom) * n / max;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Tools introduced per year: " + perYear.map(p => p.y + " " + p.n).join(", "));
  const mk = (tag, attrs, text) => {{
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    if (text != null) e.textContent = text;
    return e;
  }};
  const baseY = H - bottom;
  svg.appendChild(mk("line", {{ class: "base", x1: 0, x2: W, y1: baseY + .5, y2: baseY + .5 }}));
  const hits = {{}};
  perYear.forEach((p, i) => {{
    const cx = band * i + band / 2, h = Math.max(sy(p.n), 2), x = cx - bw / 2, yTop = baseY - h, r = Math.min(4, h);
    const hit = mk("rect", {{ class: "hit", x: band * i, y: top - 12, width: band, height: H - top + 12 }});
    // 4px rounded data end, square at the baseline
    const bar = mk("path", {{ class: "bar" + (p.y === thisYear ? " partial" : ""),
      d: `M${{x}},${{baseY}}V${{yTop + r}}Q${{x}},${{yTop}} ${{x + r}},${{yTop}}H${{x + bw - r}}Q${{x + bw}},${{yTop}} ${{x + bw}},${{yTop + r}}V${{baseY}}Z` }});
    svg.appendChild(hit); svg.appendChild(bar);
    svg.appendChild(mk("text", {{ class: "val", x: cx, y: yTop - 4 }}, p.n));
    svg.appendChild(mk("text", {{ class: "yrl", x: cx, y: baseY + 13 }}, p.y));
    if (p.y === thisYear) svg.appendChild(mk("text", {{ class: "yrs", x: cx, y: baseY + 24 }}, "so far"));
    hits[p.y] = hit;
    hit.addEventListener("mousemove", e => {{
      tip.replaceChildren();
      tip.appendChild(el("b", null, p.y + (p.y === thisYear ? " (so far)" : "") + " · " + p.n + " tools"));
      p.cats.forEach(([c, n]) => {{ tip.appendChild(el("br")); tip.appendChild(el("span", "muted", n + "  " + c)); }});
      tip.style.display = "block";
      const box = chart.getBoundingClientRect(), tw = tip.offsetWidth;
      tip.style.left = Math.min(Math.max(e.clientX - box.left + 12, 8), box.width - tw - 8) + "px";
      tip.style.top = (e.clientY - box.top + 12) + "px";
    }});
    hit.addEventListener("mouseleave", () => tip.style.display = "none");
    hit.addEventListener("click", () => {{
      yr.value = yr.value === String(p.y) ? "" : String(p.y);
      render(); markYear();
    }});
  }});
  function markYear() {{ Object.entries(hits).forEach(([y, h]) => h.classList.toggle("sel", yr.value === y)); }}
  chart.appendChild(svg);
  render();
}})();
</script>
"""

if __name__ == "__main__":
    main()
