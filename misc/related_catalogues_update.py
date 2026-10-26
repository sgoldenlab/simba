# -*- coding: utf-8 -*-
"""
Watch three external tool catalogues for entries that are not on the Related Software page yet:

* OpenBehavior (https://edspace.american.edu/openbehavior/) -- projects in its video-analysis and
  home-cage categories, read from the site's public WordPress API;
* the Home Cage Monitoring Technologies Catalogue (https://www.thebehaviourdatabase.org/catalogue.php),
  read from the JSON the catalogue page itself loads;
* Open Neuroscience (https://open-neuroscience.com/) -- projects tagged Behaviour, read from the site's
  source on GitHub (one markdown file per project).

Every entry seen is recorded in misc/related_catalogues_seen.csv, so each one is reported once. Entries
that are new since the last run and not already in the table (matched by name or link
against misc/related_software.csv and misc/related_tools.csv) are written to
docs/_generated/related_catalogues_new.html, which the page shows under its list of other catalogues
until the tool is added to misc/related_software.csv (or marked ``skip`` in the seen CSV).

Run:  python misc/related_catalogues_update.py           (weekly, by .github/workflows/related_tools.yml)
      python misc/related_catalogues_update.py --seed    (record everything as seen without listing it)
"""
import csv, html, io, json, os, re, sys, tarfile, urllib.request
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
CURATED = os.path.join(HERE, "related_software.csv")
CITING = os.path.join(HERE, "related_tools.csv")
SEEN = os.path.join(HERE, "related_catalogues_seen.csv")
OUT = os.path.join(HERE, "..", "docs", "_generated", "related_catalogues_new.html")

OB_API = "https://edspace.american.edu/openbehavior/wp-json/wp/v2/"
OB_CATEGORIES = {"1Video Analysis", "Behavior Video Analysis", "Homecage"}
HCM_DATA = "https://thebehaviourdatabase.org/get_data.php"
HCM_PAGE = "https://www.thebehaviourdatabase.org/catalogue.php"
ON_SOURCE = "https://codeload.github.com/open-neuroscience/open-neuroscience.github.io/tar.gz/HEAD"
ON_POST = "https://open-neuroscience.com/en/post/"
FIELDS = ["source", "id", "name", "url", "first_seen", "status"]   # status: listed | new | skip


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "simba-related-catalogues"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def openbehavior():
    cats, page = {}, 1
    while True:
        batch = get_json(f"{OB_API}project_category?per_page=100&page={page}&_fields=id,name")
        cats.update({c["id"]: c["name"] for c in batch})
        if len(batch) < 100:
            break
        page += 1
    out, page = [], 1
    while True:
        batch = get_json(f"{OB_API}project?per_page=100&page={page}&_fields=id,title,link,project_category")
        for p in batch:
            names = {cats.get(i, "") for i in p["project_category"]}
            if names & OB_CATEGORIES:
                out.append({"source": "OpenBehavior", "id": str(p["id"]),
                            "name": html.unescape(p["title"]["rendered"]).strip(), "url": p["link"]})
        if len(batch) < 100:
            break
        page += 1
    return out


def home_cage_catalogue():
    return [{"source": "Home Cage Monitoring Catalogue", "id": str(s["system_id"]), "name": s["name"].strip(),
             "url": s["url"] if (s.get("url") or "").startswith("http") else HCM_PAGE}
            for s in get_json(HCM_DATA)["systems"]]


def open_neuroscience():
    req = urllib.request.Request(ON_SOURCE, headers={"User-Agent": "simba-related-catalogues"})
    with urllib.request.urlopen(req, timeout=120) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    out = []
    for m in tar.getmembers():
        parts = m.name.split("/")
        if not (m.isfile() and parts[-1] == "index.md" and parts[-3:-2] == ["post"]):
            continue
        text = tar.extractfile(m).read().decode("utf-8", "replace")
        tags = re.search(r"^tags:\s*\[(.*)\]", text, re.M)
        title = re.search(r"^title:\s*(.+)$", text, re.M)
        if not (tags and title and "behaviour" in tags.group(1).lower()):
            continue
        link = re.search(r"## Project Links\s+(\S+)", text)
        url = link.group(1) if link else ""
        if not url.startswith("http"):
            url = ON_POST + parts[-2].lower() + "/"
        out.append({"source": "Open Neuroscience", "id": parts[-2], "name": title.group(1).strip().strip("'\""), "url": url})
    return out


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def on_page(entry, page_text, page_norm):
    """True when the entry's short name or tool link already appears on the page."""
    short = re.split(r"\s*[:–—]\s+|\s+-\s+|\s*\(", entry["name"])[0]
    key = norm(short)
    if len(key) >= 4 and key in page_norm:
        return True
    url = re.sub(r"^https?://(www\.)?", "", entry["url"]).rstrip("/").lower()
    return entry["source"] != "OpenBehavior" and len(url) > 8 and url in page_text


def main():
    seed = "--seed" in sys.argv
    with open(CURATED, encoding="utf-8") as f:
        curated = list(csv.DictReader(f))
    with open(CITING, encoding="utf-8") as f:
        citing = [r["tool_name"] for r in csv.DictReader(f) if r["tool_name"]]
    page_text = " ".join(r["url"] for r in curated).lower()
    page_norm = " ".join(norm(n) for n in [r["name"] for r in curated] + citing)
    seen = {}
    if os.path.isfile(SEEN):
        with open(SEEN, encoding="utf-8", newline="") as f:
            seen = {(r["source"], r["id"]): r for r in csv.DictReader(f)}

    entries, failed = [], []
    for name, fetch in (("OpenBehavior", openbehavior), ("Home Cage Monitoring Catalogue", home_cage_catalogue),
                        ("Open Neuroscience", open_neuroscience)):
        try:
            entries += fetch()
        except Exception as e:
            failed.append(f"{name} ({e})")

    today, added = date.today().isoformat(), []
    for e in entries:
        k = (e["source"], e["id"])
        if k not in seen:
            seen[k] = dict(e, first_seen=today, status="listed" if seed else "new")
            if not seed:
                added.append(e)
    for r in seen.values():   # a new entry stops being listed once it is on the page
        if r["status"] == "new" and on_page(r, page_text, page_norm):
            r["status"] = "listed"

    with open(SEEN, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(seen.values(), key=lambda r: (r["source"], int(r["id"]) if r["id"].isdigit() else 0)))

    pending = sorted((r for r in seen.values() if r["status"] == "new"), key=lambda r: r["first_seen"], reverse=True)
    if pending:
        items = "".join(f'<li><a href="{html.escape(r["url"])}">{html.escape(r["name"])}</a> '
                        f'<span class="src">({html.escape(r["source"])}, {r["first_seen"]})</span></li>' for r in pending)
        body = (f'<div class="simba-rc-new"><p>New in these catalogues and not yet reviewed for this page:</p>'
                f'<ul>{items}</ul></div>')
    else:
        body = "<!-- no new catalogue entries -->"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(STYLE + body + "\n")

    print(f"{len(entries)} catalogue entries checked; {len(added)} new this run; {len(pending)} awaiting review")
    for e in added:
        print(f"  NEW {e['source']}: {e['name']} | {e['url']}")
    for msg in failed:
        print(f"WARNING: {msg} skipped")


STYLE = """<style>
.simba-rc-new{font-size:13px;margin:6px 0 0;}
.simba-rc-new p{margin:0 0 2px;}
.simba-rc-new ul{margin:0 0 0 18px;}
.simba-rc-new .src{color:#6b7280;}
</style>
"""

if __name__ == "__main__":
    main()
