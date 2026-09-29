# -*- coding: utf-8 -*-
"""
Generate docs/_generated/usecase_globe.html, an interactive globe.gl view of the published SimBA
studies (top of the published-studies page) with three views:

* Countries      - countries raised & coloured by number of studies (log scale). Default.
* Institutions   - one bar per institution (height/colour = studies).
* Collaborations - an arc between every pair of institutions that published a study together.

Clicking a country or institution opens a side panel with its studies and a link to the studies
table filtered to it. The globe library only downloads when the panel scrolls into view, and a
message replaces the globe if the browser has no WebGL or the CDN is unreachable.

Reads the same public sheet and institution coordinates as misc/usecase_map_stats.py.

globe_html() is the reusable part: docs/conf.py calls it for the countries-only globe of PyPI
downloads on the download-statistics page.

Run:  python misc/usecase_globe.py
"""
import collections, json, os, re, sys
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from country_names import COUNTRY_NAMES, COUNTRY_NUMERIC
from institution_coords import INSTITUTION_ALIASES, INSTITUTION_COORDS
from usecase_map_stats import country_iso, fetch_rows, inst_bucket, split_countries

OUT = os.path.join(HERE, "..", "docs", "_generated", "usecase_globe.html")
# One-hue ramp shared by all three views: pale peach = few, deep red-orange = many.
# Validated as an ordinal ramp (--mode dark): darkest step clears the land 2.52:1 and the ocean 3.83:1.
RAMP = ["#fee3c8", "#fbbd8c", "#f59352", "#e5692a", "#c2461a"]


def studies_globe(tiles=True):
    """(html, summary) for the published-studies globe, built from the sheet."""
    rows = fetch_rows()
    cols = {c.strip().upper(): c for c in rows[0].keys()}

    def cell(r, name):
        return (r.get(cols.get(name, "")) or "").strip()

    studies = collections.defaultdict(list)          # institution -> ["2024 · Title"]
    pairs = collections.Counter()                     # (inst_a, inst_b) -> shared studies
    country_studies = collections.defaultdict(list)   # ISO-2 -> ["2024 · Title"], each study once per country
    for r in rows:
        insts = set()
        for tok in cell(r, "AUTHOR INSTITUTIONS").split(","):
            name = re.sub(r"\s+", " ", tok).strip()
            canon = INSTITUTION_ALIASES.get(name, name)
            if canon in INSTITUTION_COORDS:
                insts.add(canon)
        label = f"{cell(r, 'YEAR')} · {cell(r, 'TITLE')}"
        for i in insts:
            studies[i].append(label)
        for iso in {country_iso(c) for c in split_countries(cell(r, "COUNTRIES"))} - {None}:
            country_studies[iso].append(label)
        for a, b in combinations(sorted(insts), 2):
            if INSTITUTION_COORDS[a] != INSTITUTION_COORDS[b]:
                pairs[(a, b)] += 1

    points = [{"name": n, "lat": INSTITUTION_COORDS[n][0], "lng": INSTITUTION_COORDS[n][1],
               "n": len(s), "col": RAMP[inst_bucket(len(s))], "studies": sorted(s, reverse=True)}
              for n, s in studies.items()]
    arcs = [{"a": a, "b": b, "n": n,
             "sLat": INSTITUTION_COORDS[a][0], "sLng": INSTITUTION_COORDS[a][1],
             "eLat": INSTITUTION_COORDS[b][0], "eLng": INSTITUTION_COORDS[b][1]}
            for (a, b), n in pairs.items()]
    countries = {iso: {"n": len(v), "items": sorted(v, reverse=True)} for iso, v in country_studies.items()}

    html = globe_html(
        countries, points, arcs, unit=("study", "studies"),
        headline=f"{len(rows)} studies",
        stats=f"{len(countries)} countries &middot; {len(points)} institutions &middot; {len(arcs)} collaboration links",
        country_sub="Raised countries: height &amp; colour show the number of studies &middot; "
                    "multi-country studies count in each",
        none_text="no studies yet", table_links=True, tiles=tiles)
    return html, f"{len(countries)} countries, {len(points)} institutions, {len(arcs)} collaboration arcs"


def main():
    html, summary = studies_globe()   # with the street map / satellite basemap
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {os.path.normpath(OUT)}: {summary}")


def globe_html(countries, points=(), arcs=(), *, unit, headline, stats, country_sub, none_text,
               modes=None, table_links=False, share_total=None, tiles=False):
    """HTML for one interactive globe.

    countries   {ISO-2: {"n": count, "items": [text, ...]}}; items (optional) are listed when a
                country is clicked.
    points/arcs institutions and collaboration pairs (as built in main()); with none of them the
                globe shows only its Countries view and no view buttons.
    unit        ("study", "studies") -- singular/plural of what is counted.
    headline, stats  the two lines at the top left; country_sub the Countries-view caption;
                none_text the tooltip for a country with nothing.
    table_links link panels to the published-studies table.
    share_total if given, a clicked country also shows its share of this total.
    tiles       draw a street-level basemap (Esri World Dark Gray tiles) that shows through when zoomed in.
    """
    if modes is None:
        modes = ("country", "inst", "collab") if points else ("country",)
    names = {"country": "Countries", "inst": "Institutions", "collab": "Collaborations"}
    buttons = "" if len(modes) < 2 else (
        '<div class="sg-modes" role="group" aria-label="Globe view">' + "".join(
            f'<button type="button" data-mode="{m}" aria-pressed="{str(i == 0).lower()}">{names[m]}</button>'
            for i, m in enumerate(modes)) + "</div>")
    j = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    subs = {"__POINTS__": j(list(points)), "__ARCS__": j(list(arcs)), "__COUNTRIES__": j(countries),
            "__HEADLINE__": headline, "__STATS__": stats, "__MODE_BUTTONS__": buttons,
            "__BOX_CLASS__": "has-modes" if buttons else "",
            "__UNIT__": j(list(unit)), "__UNITS__": unit[1], "__COUNTRY_SUB__": j(country_sub),
            "__NONE_TEXT__": j(none_text), "__LINKS__": j(bool(table_links)), "__SHARE_TOTAL__": j(share_total), "__TILES__": j(bool(tiles)),
            "__C_MAX__": f"{max(c['n'] for c in countries.values()):,}" if countries else "0",
            "__NUMERIC__": j(COUNTRY_NUMERIC), "__NAMES__": j({k: COUNTRY_NAMES[k] for k in countries if k in COUNTRY_NAMES}),
            "__RAMP__": j(RAMP), "__RAMP_CSS__": ",".join(RAMP)}
    subs.update({f"__R{i}__": c for i, c in enumerate(RAMP)})
    html = TEMPLATE
    for k, v in subs.items():
        html = html.replace(k, v)
    return html


TEMPLATE = r"""<style>
.simba-globe{position:relative;max-width:980px;margin:14px auto 10px;height:560px;border-radius:16px;overflow:hidden;
  /* backlight: a bright glow centred behind the globe, over the dark panel gradient */
  background:radial-gradient(circle at 50% 50%,rgba(190,225,255,.75) 0,rgba(120,180,250,.45) 28%,rgba(60,120,210,.2) 44%,rgba(0,0,0,0) 64%),
             radial-gradient(120% 90% at 50% 40%,#16344f 0%,#0b1a2a 60%,#07111c 100%);box-shadow:0 10px 30px rgba(10,30,50,.35);}
.simba-globe #simbaGlobe{position:absolute;inset:0;}
.simba-globe .sg-hud{position:absolute;left:18px;top:14px;z-index:2;color:#dbe8f3;pointer-events:none;font-size:12.5px;line-height:1.45;}
.simba-globe .sg-hud b{display:block;font-size:19px;color:#fff;letter-spacing:.01em;margin-bottom:2px;}
.simba-globe .sg-hud span{color:#8fb3cf;}
.simba-globe.pinned .sg-hud span{display:none;}
/* view switch: a vertical stack of large buttons so the three views are easy to find */
.simba-globe .sg-modes{position:absolute;left:18px;top:96px;z-index:3;display:flex;flex-direction:column;gap:6px;
  background:rgba(12,24,38,.82);border:1px solid #2b4a66;border-radius:14px;padding:6px;}
.simba-globe .sg-modes button{font:600 14px system-ui,sans-serif;border:1px solid #2b4a66;border-radius:10px;padding:10px 18px;
  min-width:160px;text-align:left;cursor:pointer;background:rgba(255,255,255,.06);color:#dbe8f3;transition:background .15s ease,color .15s ease;}
.simba-globe .sg-modes button:hover{background:rgba(255,255,255,.14);color:#fff;}
.simba-globe .sg-modes button[aria-pressed="true"]{background:#e5692a;border-color:#e5692a;color:#fff;}
.simba-globe .sg-legend{position:absolute;right:16px;bottom:12px;z-index:2;gap:10px;align-items:center;
  font-size:11px;color:#a9c3d8;pointer-events:none;flex-wrap:wrap;justify-content:flex-end;display:none;}
.simba-globe.m-country .sg-legend.country,.simba-globe.m-inst .sg-legend.inst,.simba-globe.m-collab .sg-legend.collab{display:flex;}
.simba-globe .sg-legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px;}
.simba-globe .sg-legend .ramp{width:110px;height:9px;border-radius:5px;background:linear-gradient(90deg,__RAMP_CSS__);}
.simba-globe .sg-legend .arc{width:22px;height:2px;border-radius:1px;background:__R2__;}
.simba-globe .sg-hint{position:absolute;left:18px;bottom:12px;z-index:2;font-size:11px;color:#7f9bb3;pointer-events:none;}
.simba-globe .sg-credit{position:absolute;left:18px;bottom:28px;z-index:2;font-size:10px;color:#6f8aa3;}
.simba-globe .sg-credit a{color:inherit !important;}
.simba-globe .sg-credit[hidden]{display:none;}
/* Map / Satellite switch (street-map globes only): under the view buttons, or top-left without them */
.simba-globe .sg-base{position:absolute;left:18px;top:96px;z-index:3;display:flex;background:rgba(12,24,38,.82);
  border:1px solid #2b4a66;border-radius:14px;padding:4px;gap:4px;}
.simba-globe.has-modes .sg-base{top:270px;}
.simba-globe .sg-base[hidden]{display:none;}
.simba-globe .sg-base button{font:600 12.5px system-ui,sans-serif;border:0;border-radius:10px;padding:7px 14px;cursor:pointer;
  background:none;color:#a9c3d8;}
.simba-globe .sg-base button:hover{color:#fff;}
.simba-globe .sg-base button[aria-pressed="true"]{background:#2f8f9d;color:#fff;}
.simba-globe .sg-base .sep{width:1px;margin:3px 2px;background:#2b4a66;}
.simba-globe .sg-msg{position:absolute;inset:0;z-index:4;display:flex;align-items:center;justify-content:center;text-align:center;
  padding:0 40px;color:#c3d3e2;font:14px/1.5 system-ui,sans-serif;}
.simba-globe .sg-msg[hidden]{display:none;}
.simba-globe.failed .sg-base,.simba-globe.failed .sg-modes,.simba-globe.failed .sg-legend,.simba-globe.failed .sg-hint,.simba-globe.failed .sg-hud span{display:none !important;}
.simba-globe .sg-panel{position:absolute;right:14px;top:14px;z-index:3;width:290px;max-height:calc(100% - 60px);overflow:auto;
  background:rgba(12,24,38,.94);border:1px solid #2b4a66;border-radius:12px;padding:12px 14px;color:#e6eef5;
  font:12.5px/1.45 system-ui,sans-serif;box-shadow:0 10px 30px rgba(0,0,0,.45);}
.simba-globe .sg-panel[hidden]{display:none;}
.simba-globe .sg-panel h4{margin:0 22px 2px 0 !important;font-size:14px;color:#fff !important;line-height:1.3;}
.simba-globe .sg-panel .c{color:#f0a36b;font-weight:700;font-size:12px;}
.simba-globe .sg-panel ul{margin:8px 0 10px !important;padding-left:16px;}
.simba-globe .sg-panel li{margin:0 0 4px;color:#c3d3e2;list-style:disc;}
.simba-globe .sg-panel li b{color:__R1__;font-weight:600;}
.simba-globe .sg-panel .x{position:absolute;right:8px;top:6px;background:none;border:0;color:#8fb3cf;font-size:18px;cursor:pointer;line-height:1;}
.simba-globe .sg-panel a.go{display:inline-block;text-decoration:none !important;font-weight:600;font-size:12.5px;padding:6px 12px;
  border-radius:16px;background:#2f8f9d;color:#fff !important;}
.simba-globe .sg-panel a.go:hover{background:#277a86;}
.simba-globe-tip{background:rgba(15,28,43,.96);color:#e6eef5;border:1px solid #2b4a66;border-radius:9px;padding:9px 11px;
  max-width:300px;font:12px/1.4 system-ui,sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.4);}
.simba-globe-tip b{color:#fff;font-size:12.5px;}
.simba-globe-tip .c{color:#f0a36b;font-weight:700;}
.simba-globe-tip ul{margin:5px 0 0;padding-left:14px;}
.simba-globe-tip li{margin:0 0 2px;color:#b9cadb;}
@media (max-width:680px){.simba-globe{height:440px;}
  .simba-globe .sg-modes{flex-direction:row;top:auto;bottom:34px;left:10px;right:10px;justify-content:center;}
  .simba-globe .sg-modes button{min-width:0;flex:1;text-align:center;padding:8px 6px;font-size:12.5px;}
  .simba-globe .sg-hint,.simba-globe .sg-hud span{display:none;}
  .simba-globe .sg-base,.simba-globe.has-modes .sg-base{top:auto;bottom:80px;left:auto;right:10px;}.simba-globe .sg-legend{display:none !important;}
  .simba-globe .sg-panel{left:10px;right:10px;top:auto;bottom:10px;width:auto;max-height:55%;}}
</style>
<div class="simba-globe m-country __BOX_CLASS__" id="simbaGlobeBox">
  <div class="sg-hud"><b>__HEADLINE__</b>__STATS__<br><span id="simbaGlobeSub"></span></div>
  __MODE_BUTTONS__
  <div class="sg-legend country"><span><i style="background:#2a3440;border:1px solid #5a6b7d"></i>none</span><span style="margin-left:6px">1</span><i class="ramp"></i><span>__C_MAX__ __UNITS__ (log scale)</span></div>
  <div class="sg-legend inst"><span style="color:#8fb3cf">studies:</span><span><i style="background:__R0__"></i>1</span><span><i style="background:__R1__"></i>2</span><span><i style="background:__R2__"></i>3&ndash;4</span><span><i style="background:__R3__"></i>5&ndash;7</span><span><i style="background:__R4__"></i>8+</span></div>
  <div class="sg-legend collab"><span>partners: few</span><i class="ramp"></i><span>many (bigger dot = more)</span><span style="margin-left:8px"><i class="arc"></i>co-authored study</span></div>
  <div class="sg-hint" id="simbaGlobeHint"></div>
  <div class="sg-credit" id="simbaGlobeCredit" hidden></div>
  <div class="sg-base" id="simbaGlobeBase" role="group" aria-label="Basemap" hidden><button type="button" data-base="map" aria-pressed="true">Map</button><button type="button" data-base="sat" aria-pressed="false">Satellite</button><span class="sep"></span><button type="button" data-tilt aria-pressed="false" title="View the ground at an angle, like Google Earth">Tilt</button></div>
  <div class="sg-panel" id="simbaGlobePanel" hidden></div>
  <div class="sg-msg" id="simbaGlobeMsg" hidden></div>
  <div id="simbaGlobe"></div>
</div>
<script>
(function () {
  var LIB = "https://cdn.jsdelivr.net/npm/globe.gl@2.46.2/dist/globe.gl.min.js";
  // Natural Earth outlines (world-atlas): the coarse 1:110m set first, then the 1:50m set -- prepared in idle
  // time a few seconds later (or at once on a close zoom) -- sharp borders up close, and tiny countries appear
  var GEO_LO = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json";
  var GEO_HI = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-50m.json";
  var TOPO = "https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js";
  var HI_ALT = 1.1;   // camera altitude (in globe radii) below which the detailed outlines load
  var POINTS = __POINTS__, ARCS = __ARCS__, COUNTRIES = __COUNTRIES__, RAMP = __RAMP__;
  var UNIT = __UNIT__, NONE_TEXT = __NONE_TEXT__, LINKS = __LINKS__, SHARE_TOTAL = __SHARE_TOTAL__;
  var NUMERIC = __NUMERIC__, NAMES = __NAMES__, TILES = __TILES__;   // outline id (ISO numeric) -> ISO-2; ISO-2 -> display name
  // with the street map on: below TILE_ALT the country fills fade to see-through; below CITY_ALT they (and the
  // coarse country borders, jagged at city scale) disappear, leaving the map and the institution dots
  var TILE_ALT = 0.9, CITY_ALT = 0.06;
  var rampAt = function (t) { return RAMP[Math.min(RAMP.length - 1, Math.floor(RAMP.length * t))]; };
  var box = document.getElementById("simbaGlobeBox"), el = document.getElementById("simbaGlobe");
  var panel = document.getElementById("simbaGlobePanel"), msg = document.getElementById("simbaGlobeMsg");
  if (!box) return;
  var TEXT = {
    country: [__COUNTRY_SUB__, "a country"],
    inst: ["Bars: studies per institution &middot; taller &amp; darker = more studies", "a bar"],
    collab: ["Arcs: institutions that published a study together &middot; hover or tap a dot to see its partners", "a dot"]
  };

  function fail(text) {
    msg.innerHTML = "<div>" + text + '<br><span style="color:#8fb3cf;font-size:12.5px">The charts below show the same data.</span></div>';
    msg.hidden = false; box.classList.add("failed");
  }
  function webgl() {
    try { var c = document.createElement("canvas"); return !!(window.WebGLRenderingContext && (c.getContext("webgl") || c.getContext("experimental-webgl"))); }
    catch (e) { return false; }
  }
  function start() {
    if (!webgl()) return fail("Your browser can&rsquo;t display the 3D globe (WebGL is unavailable).");
    msg.innerHTML = "Loading globe&hellip;"; msg.hidden = false;
    // The docs pages load require.js: hide its `define` while the UMD bundle runs, or globe.gl
    // registers as an AMD module instead of window.Globe (same trick as the Chart.js panels).
    var amd = window.define, restore = function () { try { window.define = amd; } catch (e) {} };
    try { window.define = undefined; } catch (e) {}
    (function load(urls) {
      if (!urls.length) { restore(); msg.hidden = true; try { init(); } catch (e) { fail("The 3D globe couldn&rsquo;t start."); } return; }
      var s = document.createElement("script");
      s.src = urls[0]; s.async = true;
      s.onload = function () { load(urls.slice(1)); };
      s.onerror = function () { restore(); fail("The 3D globe couldn&rsquo;t load (the globe library is unreachable)."); };
      document.head.appendChild(s);
    })([LIB, TOPO]);
  }
  // Wait until the page's scripts have all run before touching `define`: other panels toggle it
  // while the page parses, and a restore landing mid-download would capture globe.gl into
  // require.js. DOMContentLoaded (not "load") so images, videos and analytics don't delay it.
  function whenLoaded() {
    if (document.readyState !== "loading") start();
    else document.addEventListener("DOMContentLoaded", start, { once: true });
  }
  // Lazy-load: only fetch the ~1.9 MB library once the globe is close to the viewport.
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (es) {
      if (es.some(function (e) { return e.isIntersecting; })) { io.disconnect(); whenLoaded(); }
    }, { rootMargin: "300px" });
    io.observe(box);
  } else { whenLoaded(); }

  function init() {
    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); };
    var plural = function (n, one, many) { return n.toLocaleString() + " " + (n === 1 ? one : many); };
    var mode = "country", hoverInst = null, hoverArc = null, hoverC = null, pinned = null, geo = null, ctl;

    // ---------- institutions & collaborations ----------
    var maxN = Math.max.apply(null, POINTS.map(function (p) { return p.n; }));
    var partners = {};
    ARCS.forEach(function (a) {
      (partners[a.a] = partners[a.a] || []).push({ name: a.b, n: a.n });
      (partners[a.b] = partners[a.b] || []).push({ name: a.a, n: a.n });
    });
    var COLLAB_POINTS = POINTS.filter(function (p) { return partners[p.name]; });
    var deg = function (p) { return partners[p.name] ? partners[p.name].length : 0; };
    var maxDeg = Math.max.apply(null, COLLAB_POINTS.map(deg));
    var touches = function (a, name) { return a.a === name || a.b === name; };
    var focus = function () { return hoverInst || pinned; };
    var lit = function (a) { var f = focus(); return a === hoverArc || (f && touches(a, f)); };
    var arcColor = function (a) {
      if (lit(a)) return [RAMP[0], RAMP[1]];
      return (focus() || hoverArc) ? "rgba(245,147,82,0.06)" : "rgba(245,147,82,0.4)";
    };
    var refreshArcs = function () { g.arcColor(arcColor).arcStroke(g.arcStroke()); refreshPoints(); };
    // bars and dots start just above the flat country layer (0.002 in these views) so clicks reach them,
    // and their height scales with zoom like their size
    var pAlt = function (p) { return 0.003 + zs * (mode === "inst" ? 0.02 + 0.28 * Math.sqrt(p.n / maxN) : 0.006); };
    var zs = 1;   // zoom scale: 1 at the default view, smaller as the camera closes in
    var pRad = function (p) {
      if (mode === "inst") return zs * (0.35 + 0.25 * Math.sqrt(p.n / maxN));
      return zs * (0.45 + 1.3 * Math.sqrt(deg(p) / maxDeg) + (focus() === p.name ? 0.2 : 0));   // flat disc sized by partners
    };
    var pCol = function (p) {
      if (mode === "inst") return p.col;
      var f = focus();
      if (f === p.name) return "#ffffff";
      if (f && !partners[f].some(function (q) { return q.name === p.name; })) return "rgba(254,227,200,0.18)";
      return rampAt(maxDeg > 1 ? Math.log(deg(p)) / Math.log(maxDeg) : 1);   // colour = number of partners
    };
    var refreshPoints = function () { g.pointAltitude(pAlt).pointRadius(pRad).pointColor(pCol); };

    // ---------- countries ----------
    var CRAMP = RAMP;   // shared ramp (see RAMP in the Python above)
    var LAND = "#2a3440";
    var cMax = Math.max.apply(null, Object.keys(COUNTRIES).map(function (k) { return COUNTRIES[k].n; }));
    var iso = function (f) { return NUMERIC[f.id]; };
    var cname = function (f) { return NAMES[iso(f)] || f.properties.name; };
    var cn = function (f) { var v = COUNTRIES[iso(f)]; return v ? v.n : 0; };
    // log scale: the US would otherwise flatten every other country into the bottom colour
    var cT = function (n) { return cMax > 1 ? Math.log(n) / Math.log(cMax) : 1; };
    var cCol = function (n) { return CRAMP[Math.min(CRAMP.length - 1, Math.floor(CRAMP.length * cT(n)))]; };
    var close = 0;   // 0 normal; 1 past TILE_ALT: flat, see-through fills; 2 past CITY_ALT: no fills or borders
    var alpha = function (hex, a) { var v = parseInt(hex.slice(1), 16); return "rgba(" + (v >> 16) + "," + ((v >> 8) & 255) + "," + (v & 255) + "," + a + ")"; };
    var polyAlt = function (f) {
      var n = cn(f);
      if (close) return 0.001;
      return mode === "country" && n ? 0.012 + 0.06 * cT(n) + (f === hoverC ? 0.03 : 0) : mode === "country" ? 0.004 : 0.002;
    };
    var polyCap = function (f) {
      var n = cn(f), on = mode === "country" && n;
      if (close === 2) return "rgba(0,0,0,0)";
      if (close) return on ? alpha(cCol(n), f === hoverC ? 0.55 : 0.35) : "rgba(0,0,0,0)";
      return on ? cCol(n) : LAND;
    };
    var polySide = function (f) { return !close && mode === "country" && cn(f) ? "rgba(0,0,0,0.25)" : "rgba(0,0,0,0)"; };
    // hover keeps the country's own colour (so it still reads correctly on the scale): it lifts and gets a white outline
    var polyStroke = function (f) { return close === 2 ? "rgba(0,0,0,0)" : f === hoverC ? "#ffffff" : "rgba(140,170,200,0.35)"; };
    var refreshPolys = function () { g.polygonAltitude(polyAlt).polygonCapColor(polyCap).polygonSideColor(polySide).polygonStrokeColor(polyStroke); };
    // camera target: mean vertex of the country's largest ring (this dataset has no label points)
    function centre(f) {
      var gm = f.geometry, polys = gm.type === "Polygon" ? [gm.coordinates] : gm.coordinates, best = polys[0][0];
      polys.forEach(function (p) { if (p[0].length > best.length) best = p[0]; });
      var x = 0, y = 0, x0 = 180, x1 = -180, y0 = 90, y1 = -90;
      best.forEach(function (pt) { x += pt[0]; y += pt[1]; x0 = Math.min(x0, pt[0]); x1 = Math.max(x1, pt[0]); y0 = Math.min(y0, pt[1]); y1 = Math.max(y1, pt[1]); });
      return [x / best.length, y / best.length, Math.max(x1 - x0, y1 - y0)];   // lng, lat, extent in degrees
    }

    // ---------- side panel ----------
    function showPanel(html, lat, lng, alt) {
      ctl.autoRotate = false;
      // pan to it, but never zoom back out: keep the current zoom if it's already closer than the target
      g.pointOfView({ lat: lat, lng: lng, altitude: Math.min(g.pointOfView().altitude, alt || 1.5) }, 900);
      panel.innerHTML = '<button class="x" type="button" aria-label="Close">&times;</button>' + html;
      panel.querySelector(".x").addEventListener("click", closePanel);
      panel.hidden = false; box.classList.add("pinned");
    }
    function closePanel() { pinned = null; panel.hidden = true; box.classList.remove("pinned"); refreshArcs(); }
    var list = function (items) { return "<ul>" + items.map(function (s) { return "<li>" + s + "</li>"; }).join("") + "</ul>"; };
    function pinInst(p) {
      pinned = p.name; refreshArcs();
      var body = mode === "collab"
        ? '<span class="c">' + plural(partners[p.name].length, "collaborating institution", "collaborating institutions") + '</span>' +
          list(partners[p.name].slice().sort(function (a, b) { return b.n - a.n; }).map(function (q) { return esc(q.name) + " <b>&times;" + q.n + "</b>"; }))
        : '<span class="c">' + plural(p.n, "study", "studies") + '</span>' + list(p.studies.map(esc));
      showPanel("<h4>" + esc(p.name) + "</h4>" + body + (LINKS ?
        '<a class="go" href="published_studies.html#studies=' + encodeURIComponent(p.name) + '">See its studies in the table &rarr;</a>' : ""), p.lat, p.lng);
    }
    function pinCountry(f) {
      var ctr = centre(f), c = COUNTRIES[iso(f)] || {n: 0}, items = c.items || [];
      // fly closer to small countries (and fetch the detailed outlines for them)
      var alt = Math.max(0.25, Math.min(1.5, ctr[2] / 35));
      if (alt < HI_ALT) loadHi();
      var share = SHARE_TOTAL ? '<br><span style="color:#c3d3e2">' + (100 * c.n / SHARE_TOTAL).toFixed(1) + "% of all " + UNIT[1] + "</span>" : "";
      showPanel("<h4>" + esc(cname(f)) + '</h4><span class="c">' + plural(c.n, UNIT[0], UNIT[1]) + "</span>" + share +
        (items.length ? list(items.map(esc)) : "<br><br>") + (LINKS ?
        '<a class="go" href="published_studies.html#country=' + encodeURIComponent(iso(f)) + "&label=" + encodeURIComponent(cname(f)) +
        '">See these studies in the table &rarr;</a>' : ""), ctr[1], ctr[0], alt);
    }

    // ---------- globe ----------
    var g = Globe()(el)
      .backgroundColor("rgba(0,0,0,0)")
      .showAtmosphere(true).atmosphereColor("#c4e4ff").atmosphereAltitude(0.38)
      .pointLat("lat").pointLng("lng").pointsMerge(false).pointsTransitionDuration(0).arcsTransitionDuration(0)
      // lighter geometry than globe.gl's defaults (64-segment arcs, 12-sided markers, 5-degree country tops):
      // hardly visible at these sizes, but far fewer triangles to draw every frame
      .arcCurveResolution(32).arcCircularResolution(4).pointResolution(8)
      .pointAltitude(pAlt).pointRadius(pRad).pointColor(pCol)
      .pointLabel(function (p) {
        if (mode === "collab") return '<div class="simba-globe-tip"><b>' + esc(p.name) + '</b><br><span class="c">' + plural(partners[p.name].length, "collaborating institution", "collaborating institutions") + "</span></div>";
        return '<div class="simba-globe-tip"><b>' + esc(p.name) + '</b><br><span class="c">' + plural(p.n, "study", "studies") + "</span>" +
          list(p.studies.slice(0, 6).map(function (s) { return esc(s.length > 90 ? s.slice(0, 88) + "…" : s); }).concat(p.n > 6 ? ["…and " + (p.n - 6) + " more (click for all)"] : [])) + "</div>";
      })
      .onPointHover(function (p) { hoverInst = p ? p.name : null; if (mode === "collab") refreshArcs(); })
      .onPointClick(pinInst)
      .arcStartLat("sLat").arcStartLng("sLng").arcEndLat("eLat").arcEndLng("eLng")
      .arcColor(arcColor).arcAltitudeAutoScale(0.45)
      .arcStroke(function (a) { return zs * ((lit(a) ? 0.45 : 0.15) + 0.15 * Math.min(a.n, 4)); })
      .arcLabel(function (a) { return '<div class="simba-globe-tip"><b>' + esc(a.a) + "</b> &harr; <b>" + esc(a.b) + '</b><br><span class="c">' + plural(a.n, "shared study", "shared studies") + "</span></div>"; })
      .onArcHover(function (a) { hoverArc = a || null; refreshArcs(); })
      .onGlobeClick(closePanel);
    var mat = g.globeMaterial(); mat.color.set("#070f19"); mat.shininess = 4;   // choropleth-style dark ocean
    g.renderer().setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));   // retina at 2-3x costs 2-4x the drawing

    var toFeatures = function (topo) {
      return topojson.feature(topo, topo.objects.countries).features.filter(function (f) {
        return f.id !== "010";   // no Antarctica
      });
    };
    var hiState = 0;   // 0 = coarse only, 1 = loading detailed, 2 = detailed shown
    // Which outlines to draw. Until the street map has shown it can load, all land is drawn (grey where there
    // is no data), so a slow or unreachable tile server never leaves countries floating on an empty sphere.
    // Once tiles load, the map draws the land itself: only countries with data are kept (and none in the
    // Institutions / Collaborations views) -- far fewer shapes to render and to hit-test.
    var tilesReady = false;
    var polyData = function () {
      if (!TILES || !tilesReady) return geo || [];
      return mode === "country" ? (geo || []).filter(function (f) { return cn(f) > 0; }) : [];
    };
    function loadHi() {
      if (hiState) return;
      hiState = 1;
      fetch(GEO_HI).then(function (r) { return r.json(); }).then(function (topo) {
        geo = toFeatures(topo); hoverC = null; hiState = 2;
        g.polygonsTransitionDuration(0).polygonsData(polyData());   // swap in place: no rise animation
        setTimeout(function () { g.polygonsTransitionDuration(600); }, 50);
      }).catch(function () { hiState = 0; });   // keep the coarse outlines; retry on the next zoom
    }
    // keyless Esri basemaps (CARTO's now need an API key); addressed {z}/{y}/{x}
    var ESRI = "/ArcGIS/rest/services/", HOSTS = ["https://server.arcgisonline.com", "https://services.arcgisonline.com"];
    var BASEMAPS = {
      map: { url: ESRI + "Canvas/World_Dark_Gray_Base/MapServer/tile/",
             credit: 'Basemap: <a href="https://www.esri.com" target="_blank" rel="noopener">Esri</a>, HERE, Garmin, &copy; ' +
                     '<a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors' },
      sat: { url: ESRI + "World_Imagery/MapServer/tile/",
             credit: 'Imagery: <a href="https://www.esri.com" target="_blank" rel="noopener">Esri</a>, Maxar, Earthstar Geographics' }
    };
    var credit = document.getElementById("simbaGlobeCredit"), baseSwitch = document.getElementById("simbaGlobeBase");
    function setBasemap(b, switching) {
      if (switching && g.globeTileEngineClearCache) g.globeTileEngineClearCache();   // drop the other style's tiles
      g.globeTileEngineUrl(function (x, y, l) {   // alternate the two hosts so the browser fetches more tiles at once
        return HOSTS[(x + y) % 2] + BASEMAPS[b].url + l + "/" + y + "/" + x;
      }).globeTileEngineMaxLevel(16);
      credit.innerHTML = BASEMAPS[b].credit;
      baseSwitch.querySelectorAll("button[data-base]").forEach(function (x) { x.setAttribute("aria-pressed", x.dataset.base === b ? "true" : "false"); });
    }
    // Fallback: if the tile server fails (3 tries across both hosts) or nothing loads within 6 s, this visitor
    // gets the plain globe instead --
    // all land drawn, no Map / Satellite / Tilt, the normal zoom limit.
    function noTiles() {
      if (!TILES) return;
      TILES = false; tilesReady = false;
      if (tilted) setTilt(false);
      g.globeTileEngineUrl(null);
      credit.hidden = baseSwitch.hidden = true;
      close = 0; ctl.minDistance = 112;
      if (geo) { g.polygonsTransitionDuration(0).polygonsData(polyData()); refreshPolys(); setTimeout(function () { g.polygonsTransitionDuration(600); }, 50); }
    }
    if (TILES) {
      var timer = setTimeout(noTiles, 6000);
      (function probe(n) {   // one brief network error must not switch a visitor to the plain globe
        var im = new Image();
        im.crossOrigin = "anonymous";
        im.onload = function () {
          clearTimeout(timer);
          if (!TILES) return;   // already gave up
          tilesReady = true;
          if (geo) g.polygonsTransitionDuration(0).polygonsData(polyData());
          setTimeout(function () { g.polygonsTransitionDuration(600); }, 50);
        };
        im.onerror = function () { if (n < 2) setTimeout(function () { probe(n + 1); }, 400); else { clearTimeout(timer); noTiles(); } };
        im.src = HOSTS[n % 2] + ESRI + "Canvas/World_Dark_Gray_Base/MapServer/tile/1/0/0";
      })(0);
      setBasemap("map");
      credit.hidden = baseSwitch.hidden = false;
      baseSwitch.querySelectorAll("button[data-base]").forEach(function (x) {
        x.addEventListener("click", function () { if (x.getAttribute("aria-pressed") !== "true") setBasemap(x.dataset.base, true); });
      });
      var tiltBtn = baseSwitch.querySelector("button[data-tilt]");
      tiltBtn.addEventListener("click", function () { setTilt(!tilted); tiltBtn.setAttribute("aria-pressed", tilted ? "true" : "false"); });
    }
    // ---- tilt: pitch the camera up towards the horizon, like Google Earth. globe.gl's controls are left alone
    // (they re-aim the camera at the globe's centre on every update), so dragging and zooming work as normal:
    // the pitch is added after each controls update, just before the frame is drawn ----
    var tilted = false, tiltAmt = 0, TILT = 50 * Math.PI / 180;
    var lastView = "";
    function applyTilt() {
      if (!tiltAmt) return;
      var k = Math.max(0, Math.min(1, (1.2 - g.pointOfView().altitude) / 0.9));   // full below ~1,900 km up, none above ~7,600 km
      if (!k) return;
      var cam = g.camera(); cam.rotateX(TILT * tiltAmt * k);
      // globe.gl told the tile engine about the camera *before* this pitch, so it would load tiles for the view
      // straight down; announce the pitched view (only when it changed) so the tiles on screen load instead
      var view = [cam.position.x, cam.position.y, cam.position.z, tiltAmt * k].map(function (v) { return v.toFixed(2); }).join();
      if (view !== lastView) { lastView = view; ctl.dispatchEvent({ type: "change" }); }
    }
    function setTilt(on) {
      tilted = on; ctl.autoRotate = false;
      var pov = g.pointOfView();
      if (on && pov.altitude > 0.6) g.pointOfView({ lat: pov.lat, lng: pov.lng, altitude: 0.35 }, 900);   // come in close enough to see it
      var from = tiltAmt, to = on ? 1 : 0, start = performance.now();
      (function step(now) {   // ease the pitch in/out
        var k = Math.min(1, (now - start) / 600);
        tiltAmt = from + (to - from) * (k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2);
        if (k < 1) requestAnimationFrame(step);
      })(start);
    }

    // Smooth zooming: marks resize once the zoom pauses (not on every frame), and zoom-stage changes restyle
    // the countries instantly -- an animated transition rebuilds every country's geometry on every frame.
    var zsTimer = null;
    function instantPolys() {
      g.polygonsTransitionDuration(0); refreshPolys();
      setTimeout(function () { g.polygonsTransitionDuration(600); }, 50);
    }
    g.onZoom(function (pov) {
      if (pov.altitude < HI_ALT) loadHi();
      clearTimeout(zsTimer);
      zsTimer = setTimeout(function () {
        var z = Math.max(TILES ? 0.005 : 0.08, Math.min(1, Math.pow(pov.altitude / 2.3, 0.75)));   // slower than the zoom: stays clickable
        if (Math.abs(z - zs) / zs > 0.15) { zs = z; if (mode !== "country") refreshArcs(); }
      }, 150);
      var c = !TILES ? 0 : pov.altitude < CITY_ALT ? 2 : pov.altitude < TILE_ALT ? 1 : 0;
      if (c !== close) { close = c; if (geo) instantPolys(); }
    });
    fetch(GEO_LO).then(function (r) { return r.json(); }).then(function (topo) {
      if (hiState !== 2) geo = toFeatures(topo);   // the detailed set may already have arrived
      g.polygonsData(polyData()).polygonAltitude(polyAlt).polygonCapColor(polyCap)
        .polygonSideColor(polySide)
        .polygonStrokeColor(polyStroke)   // light borders keep coastlines readable on the dark land
        .polygonsTransitionDuration(600).polygonCapCurvatureResolution(8)
        .polygonLabel(function (f) {
          if (mode !== "country") return "";
          var n = cn(f); return '<div class="simba-globe-tip"><b>' + esc(cname(f)) + '</b><br><span class="c">' + (n ? plural(n, UNIT[0], UNIT[1]) : NONE_TEXT) + "</span></div>";
        })
        .onPolygonHover(function (f) { if (mode === "country") { hoverC = f; instantPolys(); } })
        .onPolygonClick(function (f) { if (mode === "country" && cn(f)) pinCountry(f); else closePanel(); });
      // the globe redraws every frame, so "idle" may never come: give the idle callback a deadline
      var idle = window.requestIdleCallback || function (f) { return setTimeout(f, 1); };
      setTimeout(function () { idle(loadHi, { timeout: 2000 }); }, 2500);
    }).catch(function () { if (mode === "country") fail("The country outlines couldn&rsquo;t load."); });

    function setMode(m) {
      mode = m; hoverC = hoverInst = hoverArc = null;
      ["country", "inst", "collab"].forEach(function (k) { box.classList.toggle("m-" + k, k === m); });
      box.querySelectorAll(".sg-modes button").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.mode === m ? "true" : "false"); });
      closePanel();
      document.getElementById("simbaGlobeSub").innerHTML = TEXT[m][0];
      document.getElementById("simbaGlobeHint").innerHTML = "Drag to rotate &middot; scroll to zoom &middot; click " + TEXT[m][1] + " for details";
      g.pointsData(m === "inst" ? POINTS : m === "collab" ? COLLAB_POINTS : []).arcsData(m === "collab" ? ARCS : []);
      if (geo) g.polygonsData(polyData());
      refreshPoints(); if (geo) refreshPolys();
    }
    box.querySelectorAll(".sg-modes button").forEach(function (b) {
      b.addEventListener("click", function () { if (b.dataset.mode !== mode) setMode(b.dataset.mode); });
    });
    setMode("country");

    g.pointOfView({ lat: 30, lng: -40, altitude: 2.3 }, 0);
    ctl = g.controls();
    var ctlUpdate = ctl.update.bind(ctl);   // globe.gl calls this every frame: add the tilt pitch after it
    ctl.update = function () { var r = ctlUpdate.apply(null, arguments); applyTilt(); return r; };
    ctl.autoRotate = !reduce; ctl.autoRotateSpeed = 1.1; ctl.enableZoom = true; ctl.minDistance = TILES ? 100.3 : 112; ctl.maxDistance = 520;   // radius 100 = 6,371 km: 100.3 is ~20 km up
    el.addEventListener("mouseenter", function () { ctl.autoRotate = false; });
    el.addEventListener("mouseleave", function () { ctl.autoRotate = !reduce && !pinned && panel.hidden && !tilted; });
    function size() { g.width(el.clientWidth).height(el.clientHeight); }
    size(); window.addEventListener("resize", size);
    // pause rendering while off-screen (saves battery on long docs pages)
    new IntersectionObserver(function (es) { es.forEach(function (e) { e.isIntersecting ? g.resumeAnimation() : g.pauseAnimation(); }); }).observe(el);
  }
})();
</script>
"""

if __name__ == "__main__":
    main()
