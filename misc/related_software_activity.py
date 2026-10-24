# -*- coding: utf-8 -*-
"""
Generate docs/_generated/related_software_extras.html — page helpers for the Related Software page:

* a jump bar of the page's sections, each with its tool count (built in the browser from the
  page itself, so it never goes stale when the curated tables are edited);
* a "Back to top" button;
* a "no updates since YYYY" / "archived" tag next to curated tools whose GitHub repository has
  gone quiet. Last-push dates come from the GitHub API and are embedded here as JSON.

Repositories are taken from the tool links in docs/simba.related_software.rst; tools whose link
is a homepage rather than a GitHub repo are mapped in HOMEPAGE_REPOS. Uses GITHUB_TOKEN when set
(the weekly Action provides one); without it GitHub allows 60 requests an hour, so some repos keep
their previous values.

Run:  python misc/related_software_activity.py
"""
import json, os, re, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RST = os.path.join(HERE, "..", "docs", "simba.related_software.rst")
OUT = os.path.join(HERE, "..", "docs", "_generated", "related_software_extras.html")

GITHUB_REPO_RE = re.compile(r"^https?://github\.com/([\w.-]+)/([\w.-]+)/?$")
LINK_RE = re.compile(r"^   \* - `.+? <(.+?)>`__")
HOMEPAGE_REPOS = {"https://www.deeplabcut.org/": "DeepLabCut/DeepLabCut",
                  "https://sleap.ai/": "talmolab/sleap",
                  "https://kristinbranson.github.io/APT/": "kristinbranson/APT",
                  "https://trex.run/": "mooch443/trex",
                  "https://imagej.net/plugins/trackmate/": "trackmate-sc/TrackMate",
                  "https://zebrazoom.org/": "oliviermirat/ZebraZoom",
                  "https://www.fasttrack.sh/": "FastTrackOrg/FastTrack",
                  "https://argos.readthedocs.io/en/latest/": "subhacom/argos",
                  "https://www.janelia.org/open-science/jaaba": "kristinbranson/JAABA",
                  "https://animovement.dev/": "animovement/animovement",
                  "https://rupertoverall.net/Rtrack/index.html": "rupertoverall/Rtrack",
                  "https://rupertoverall.net/ColonyTrack": "rupertoverall/ColonyTrack",
                  "https://swarm-lab.github.io/swaRm/": "swarm-lab/swaRm",
                  "https://nwb.org/": "NeurodataWithoutBorders/pynwb",
                  "https://bonsai-rx.org/": "bonsai-rx/bonsai",
                  "https://www.boris.unito.it/": "olivierfriard/BORIS"}


def repos():
    """{link URL on the page: owner/repo} for every curated tool with a GitHub repository."""
    out = {}
    with open(RST, encoding="utf-8") as f:
        for line in f:
            m = LINK_RE.match(line)
            if not m:
                continue
            url = m.group(1)
            g = GITHUB_REPO_RE.match(url)
            if g:
                out[url] = f"{g.group(1)}/{re.sub(r'[.]git$', '', g.group(2))}"
            elif url in HOMEPAGE_REPOS:
                out[url] = HOMEPAGE_REPOS[url]
    return out


def previous():
    """Activity embedded by the last run, kept when a lookup fails."""
    if not os.path.isfile(OUT):
        return {}
    with open(OUT, encoding="utf-8") as f:
        m = re.search(r"const ACT = (\{.*?\});", f.read())
    return json.loads(m.group(1)) if m else {}


def main():
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "simba-related-software"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    act, old, failed = {}, previous(), []
    for url, repo in repos().items():
        try:
            req = urllib.request.Request(f"https://api.github.com/repos/{repo}", headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                g = json.loads(r.read())
            act[url] = {"p": (g.get("pushed_at") or "")[:7], "a": bool(g.get("archived"))}
        except Exception:
            failed.append(repo)
            if url in old:
                act[url] = old[url]
    data = json.dumps(act, sort_keys=True).replace("</", "<\\/")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("__ACT__", data))
    print(f"Wrote {os.path.normpath(OUT)}: activity for {len(act)} repositories"
          + (f"; lookup failed for {', '.join(failed)}" if failed else ""))


# Plain string (no .format), so JS braces and backslashes are written as-is.
TEMPLATE = """<style>
.simba-rs-jump{display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:6px;margin:14px 0 22px;}
.simba-rs-jump a{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:6px 10px;border:1px solid #e5e7eb;border-radius:8px;color:#23272e;text-decoration:none;font-size:13px;line-height:1.25;background:#fff;}
.simba-rs-jump a:hover{border-color:#2f7fb5;background:#f3f8fc;}
.simba-rs-jump a.main{grid-column:1/-1;border-color:#2f7fb5;background:#f3f8fc;font-weight:600;}
.simba-rs-jump .n{color:#6b7280;font-weight:400;font-variant-numeric:tabular-nums;white-space:nowrap;}
.simba-rs-top{position:fixed;right:18px;bottom:18px;z-index:50;padding:7px 12px;border:1px solid #cbd5e1;border-radius:18px;background:#fff;color:#23272e;font-size:13px;text-decoration:none;box-shadow:0 2px 6px rgba(0,0,0,.12);opacity:0;pointer-events:none;transition:opacity .2s;}
.simba-rs-top.on{opacity:1;pointer-events:auto;}
.simba-rs-top:hover{border-color:#2f7fb5;color:#21567a;}
.rst-content .wy-table-responsive table.docutils td{white-space:normal;}
.simba-rs-stale{display:inline-block;margin-left:6px;padding:0 6px;border:1px solid #d1d5db;border-radius:9px;font-size:11px;line-height:16px;color:#6b7280;white-space:nowrap;vertical-align:1px;}
</style>
<nav class="simba-rs-jump" id="simba-rs-jump" aria-label="Sections on this page"></nav>
<a class="simba-rs-top" id="simba-rs-top" href="#simba-rs-jump">&uarr; Back to top</a>
<script>
(function () {
  const ACT = __ACT__;
  const STALE_MONTHS = 24;

  function ready(fn) {
    if (document.readyState !== "loading") fn(); else document.addEventListener("DOMContentLoaded", fn);
  }

  ready(function () {
    const nav = document.getElementById("simba-rs-jump");
    const page = nav.closest("section, .section");

    // Jump bar: one link per section, with its tool count; the searchable table goes first.
    const links = [];
    page.querySelectorAll(":scope > section > h2, :scope > .section > h2").forEach(function (h) {
      const sec = h.parentElement;
      const hc = h.cloneNode(true);
      hc.querySelectorAll(".headerlink").forEach(function (x) { x.remove(); });   // the hidden link icon
      const name = hc.textContent.trim();
      const rt = sec.querySelector(".simba-rt");
      const n = rt ? +rt.dataset.total : sec.querySelectorAll("table.docutils tbody tr").length;
      const a = document.createElement("a");
      a.href = "#" + sec.id;
      a.innerHTML = "<span></span><span class=n></span>";
      a.firstChild.textContent = rt ? name + " (searchable)" : name;
      a.lastChild.textContent = n + " tools";
      if (rt) { a.className = "main"; links.unshift(a); } else links.push(a);
    });
    links.forEach(function (a) { nav.appendChild(a); });

    // Back to top: shown once the jump bar has scrolled out of view.
    const top = document.getElementById("simba-rs-top");
    function toggle() { top.classList.toggle("on", nav.getBoundingClientRect().bottom < 0); }
    window.addEventListener("scroll", toggle, {passive: true});
    toggle();

    // Maintenance tags on curated tools whose GitHub repository has gone quiet.
    const now = new Date(), nowM = now.getFullYear() * 12 + now.getMonth();
    page.querySelectorAll("table.docutils td:first-child a.reference.external").forEach(function (a) {
      const g = ACT[a.getAttribute("href")];
      if (!g || !g.p) return;
      const y = +g.p.slice(0, 4), m = +g.p.slice(5, 7) - 1;
      let txt = "";
      if (g.a) txt = "archived";
      else if (nowM - (y * 12 + m) >= STALE_MONTHS) txt = "no updates since " + y;
      if (!txt) return;
      const s = document.createElement("span");
      s.className = "simba-rs-stale";
      s.textContent = txt;
      s.title = "Last code update on GitHub: " + g.p + (g.a ? " (repository archived)" : "");
      a.after(s);
    });
  });
})();
</script>
"""

if __name__ == "__main__":
    main()
