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

Run:  python misc/usecase_globe.py
"""
import collections, json, os, re, sys
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from institution_coords import INSTITUTION_ALIASES, INSTITUTION_COORDS
from usecase_map_stats import country_iso, fetch_rows, inst_bucket, split_countries

OUT = os.path.join(HERE, "..", "docs", "_generated", "usecase_globe.html")
# One-hue ramp shared by all three views: pale peach = few, deep red-orange = many.
# Validated as an ordinal ramp (--mode dark): darkest step clears the land 2.52:1 and the ocean 3.83:1.
RAMP = ["#fee3c8", "#fbbd8c", "#f59352", "#e5692a", "#c2461a"]


def main():
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
    countries = {iso: sorted(v, reverse=True) for iso, v in country_studies.items()}

    j = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    subs = {"__POINTS__": j(points), "__ARCS__": j(arcs), "__COUNTRIES__": j(countries),
            "__N_STUDIES__": str(len(rows)), "__N_INST__": str(len(points)), "__N_LINKS__": str(len(arcs)),
            "__N_COUNTRIES__": str(len(countries)), "__C_MAX__": str(max(map(len, countries.values()))),
            "__RAMP__": j(RAMP), "__RAMP_CSS__": ",".join(RAMP)}
    subs.update({f"__R{i}__": c for i, c in enumerate(RAMP)})
    html = TEMPLATE
    for k, v in subs.items():
        html = html.replace(k, v)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {os.path.normpath(OUT)}: {len(countries)} countries, {len(points)} institutions, "
          f"{len(arcs)} collaboration arcs")


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
.simba-globe .sg-msg{position:absolute;inset:0;z-index:4;display:flex;align-items:center;justify-content:center;text-align:center;
  padding:0 40px;color:#c3d3e2;font:14px/1.5 system-ui,sans-serif;}
.simba-globe .sg-msg[hidden]{display:none;}
.simba-globe.failed .sg-modes,.simba-globe.failed .sg-legend,.simba-globe.failed .sg-hint,.simba-globe.failed .sg-hud span{display:none !important;}
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
  .simba-globe .sg-hint,.simba-globe .sg-hud span{display:none;}.simba-globe .sg-legend{display:none !important;}
  .simba-globe .sg-panel{left:10px;right:10px;top:auto;bottom:10px;width:auto;max-height:55%;}}
</style>
<div class="simba-globe m-country" id="simbaGlobeBox">
  <div class="sg-hud"><b>__N_STUDIES__ studies</b>__N_COUNTRIES__ countries &middot; __N_INST__ institutions &middot; __N_LINKS__ collaboration links<br><span id="simbaGlobeSub"></span></div>
  <div class="sg-modes" role="group" aria-label="Globe view">
    <button type="button" data-mode="country" aria-pressed="true">Countries</button><button type="button" data-mode="inst" aria-pressed="false">Institutions</button><button type="button" data-mode="collab" aria-pressed="false">Collaborations</button>
  </div>
  <div class="sg-legend country"><span><i style="background:#2a3440;border:1px solid #5a6b7d"></i>none</span><span style="margin-left:6px">1</span><i class="ramp"></i><span>__C_MAX__ studies (log scale)</span></div>
  <div class="sg-legend inst"><span style="color:#8fb3cf">studies:</span><span><i style="background:__R0__"></i>1</span><span><i style="background:__R1__"></i>2</span><span><i style="background:__R2__"></i>3&ndash;4</span><span><i style="background:__R3__"></i>5&ndash;7</span><span><i style="background:__R4__"></i>8+</span></div>
  <div class="sg-legend collab"><span>partners: few</span><i class="ramp"></i><span>many (bigger dot = more)</span><span style="margin-left:8px"><i class="arc"></i>co-authored study</span></div>
  <div class="sg-hint" id="simbaGlobeHint"></div>
  <div class="sg-panel" id="simbaGlobePanel" hidden></div>
  <div class="sg-msg" id="simbaGlobeMsg" hidden></div>
  <div id="simbaGlobe"></div>
</div>
<script>
(function () {
  var LIB = "https://cdn.jsdelivr.net/npm/globe.gl@2.46.2/dist/globe.gl.min.js";
  var GEO = "https://cdn.jsdelivr.net/npm/globe.gl@2.46.2/example/datasets/ne_110m_admin_0_countries.geojson";
  var POINTS = __POINTS__, ARCS = __ARCS__, COUNTRIES = __COUNTRIES__, RAMP = __RAMP__;
  var rampAt = function (t) { return RAMP[Math.min(RAMP.length - 1, Math.floor(RAMP.length * t))]; };
  var box = document.getElementById("simbaGlobeBox"), el = document.getElementById("simbaGlobe");
  var panel = document.getElementById("simbaGlobePanel"), msg = document.getElementById("simbaGlobeMsg");
  if (!box) return;
  var TEXT = {
    country: ["Raised countries: height &amp; colour show the number of studies &middot; multi-country studies count in each", "a country"],
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
    var s = document.createElement("script");
    s.src = LIB; s.async = true;
    s.onload = function () { restore(); msg.hidden = true; try { init(); } catch (e) { fail("The 3D globe couldn&rsquo;t start."); } };
    s.onerror = function () { restore(); fail("The 3D globe couldn&rsquo;t load (the globe library is unreachable)."); };
    document.head.appendChild(s);
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
    var plural = function (n, one, many) { return n + " " + (n === 1 ? one : many); };
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
    var pAlt = function (p) { return mode === "inst" ? 0.02 + 0.28 * Math.sqrt(p.n / maxN) : 0.006; };
    var pRad = function (p) {
      if (mode === "inst") return 0.35 + 0.25 * Math.sqrt(p.n / maxN);
      return 0.45 + 1.3 * Math.sqrt(deg(p) / maxDeg) + (focus() === p.name ? 0.2 : 0);   // flat disc sized by partners
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
    var cMax = Math.max.apply(null, Object.keys(COUNTRIES).map(function (k) { return COUNTRIES[k].length; }));
    var NE_FIX = { France: "FR", Norway: "NO" };   // Natural Earth 110m tags these ISO_A2 = -99
    var iso = function (f) { var c = f.properties.ISO_A2; return c === "-99" ? NE_FIX[f.properties.NAME] : c; };
    var cn = function (f) { var v = COUNTRIES[iso(f)]; return v ? v.length : 0; };
    // log scale: the US (80+) would otherwise flatten every other country into the bottom colour
    var cT = function (n) { return cMax > 1 ? Math.log(n) / Math.log(cMax) : 1; };
    var cCol = function (n) { return CRAMP[Math.min(CRAMP.length - 1, Math.floor(CRAMP.length * cT(n)))]; };
    var polyAlt = function (f) { var n = cn(f); return mode === "country" && n ? 0.012 + 0.06 * cT(n) + (f === hoverC ? 0.03 : 0) : 0.004; };
    var polyCap = function (f) { var n = cn(f); return mode === "country" && n ? cCol(n) : LAND; };
    // hover keeps the country's own colour (so it still reads correctly on the scale): it lifts and gets a white outline
    var polyStroke = function (f) { return f === hoverC ? "#ffffff" : "rgba(140,170,200,0.35)"; };
    var refreshPolys = function () { g.polygonAltitude(polyAlt).polygonCapColor(polyCap).polygonStrokeColor(polyStroke); };
    // camera target: mean vertex of the country's largest ring (this dataset has no label points)
    function centre(f) {
      var gm = f.geometry, polys = gm.type === "Polygon" ? [gm.coordinates] : gm.coordinates, best = polys[0][0];
      polys.forEach(function (p) { if (p[0].length > best.length) best = p[0]; });
      var x = 0, y = 0; best.forEach(function (pt) { x += pt[0]; y += pt[1]; });
      return [x / best.length, y / best.length];
    }

    // ---------- side panel ----------
    function showPanel(html, lat, lng) {
      ctl.autoRotate = false;
      g.pointOfView({ lat: lat, lng: lng, altitude: 1.5 }, 900);
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
      showPanel("<h4>" + esc(p.name) + "</h4>" + body +
        '<a class="go" href="published_studies.html#studies=' + encodeURIComponent(p.name) + '">See its studies in the table &rarr;</a>', p.lat, p.lng);
    }
    function pinCountry(f) {
      var ctr = centre(f), studies = COUNTRIES[iso(f)] || [];
      showPanel("<h4>" + esc(f.properties.NAME) + '</h4><span class="c">' + plural(studies.length, "study", "studies") + "</span>" + list(studies.map(esc)) +
        '<a class="go" href="published_studies.html#country=' + encodeURIComponent(iso(f)) + "&label=" + encodeURIComponent(f.properties.NAME) +
        '">See these studies in the table &rarr;</a>', ctr[1], ctr[0]);
    }

    // ---------- globe ----------
    var g = Globe()(el)
      .backgroundColor("rgba(0,0,0,0)")
      .showAtmosphere(true).atmosphereColor("#c4e4ff").atmosphereAltitude(0.38)
      .pointLat("lat").pointLng("lng").pointsMerge(false)
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
      .arcStroke(function (a) { return (lit(a) ? 0.45 : 0.15) + 0.15 * Math.min(a.n, 4); })
      .arcLabel(function (a) { return '<div class="simba-globe-tip"><b>' + esc(a.a) + "</b> &harr; <b>" + esc(a.b) + '</b><br><span class="c">' + plural(a.n, "shared study", "shared studies") + "</span></div>"; })
      .onArcHover(function (a) { hoverArc = a || null; refreshArcs(); })
      .onGlobeClick(closePanel);
    var mat = g.globeMaterial(); mat.color.set("#070f19"); mat.shininess = 4;   // choropleth-style dark ocean

    fetch(GEO).then(function (r) { return r.json(); }).then(function (d) {
      geo = d.features.filter(function (f) { return f.properties.ISO_A2 !== "AQ"; });
      g.polygonsData(geo).polygonAltitude(polyAlt).polygonCapColor(polyCap)
        .polygonSideColor(function (f) { return mode === "country" && cn(f) ? "rgba(0,0,0,0.25)" : "rgba(0,0,0,0)"; })
        .polygonStrokeColor(polyStroke)   // light borders keep coastlines readable on the dark land
        .polygonsTransitionDuration(600)
        .polygonLabel(function (f) {
          if (mode !== "country") return "";
          var n = cn(f); return '<div class="simba-globe-tip"><b>' + esc(f.properties.NAME) + '</b><br><span class="c">' + (n ? plural(n, "study", "studies") : "no studies yet") + "</span></div>";
        })
        .onPolygonHover(function (f) { if (mode === "country") { hoverC = f; refreshPolys(); } })
        .onPolygonClick(function (f) { if (mode === "country" && cn(f)) pinCountry(f); else closePanel(); });
    }).catch(function () { if (mode === "country") fail("The country outlines couldn&rsquo;t load."); });

    function setMode(m) {
      mode = m; hoverC = hoverInst = hoverArc = null;
      ["country", "inst", "collab"].forEach(function (k) { box.classList.toggle("m-" + k, k === m); });
      box.querySelectorAll(".sg-modes button").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.mode === m ? "true" : "false"); });
      closePanel();
      document.getElementById("simbaGlobeSub").innerHTML = TEXT[m][0];
      document.getElementById("simbaGlobeHint").innerHTML = "Drag to rotate &middot; scroll to zoom &middot; click " + TEXT[m][1] + " for details";
      g.pointsData(m === "inst" ? POINTS : m === "collab" ? COLLAB_POINTS : []).arcsData(m === "collab" ? ARCS : []);
      refreshPoints(); if (geo) refreshPolys();
    }
    box.querySelectorAll(".sg-modes button").forEach(function (b) {
      b.addEventListener("click", function () { if (b.dataset.mode !== mode) setMode(b.dataset.mode); });
    });
    setMode("country");

    g.pointOfView({ lat: 30, lng: -40, altitude: 2.3 }, 0);
    ctl = g.controls();
    ctl.autoRotate = !reduce; ctl.autoRotateSpeed = 1.1; ctl.enableZoom = true; ctl.minDistance = 180; ctl.maxDistance = 520;
    el.addEventListener("mouseenter", function () { ctl.autoRotate = false; });
    el.addEventListener("mouseleave", function () { ctl.autoRotate = !reduce && !pinned && panel.hidden; });
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
