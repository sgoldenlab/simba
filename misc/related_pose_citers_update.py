# -*- coding: utf-8 -*-
"""
Watch new papers citing SLEAP or DeepLabCut for ones that look like they introduce a reusable tool.

Most papers citing a pose estimator are studies that used it, so nothing here is added to the
Related Software page. Hits are printed to the run log (shown in the workflow's job summary) for a
person to review and add to misc/related_software.csv by hand.

Each run reads OpenAlex works citing the papers in WATCHED that were published in the last
WINDOW_DAYS, skips any already in misc/related_pose_citers_seen.csv or already on the page, and keeps
those whose title and abstract say they present a tool or software (or name it in the title) and
mention code being available. Every paper checked is recorded in the seen CSV, so each one is
reported once. Papers indexed by OpenAlex more than WINDOW_DAYS after publication are missed.

Seen CSV status: flagged (reported for review) | no (did not look like a tool) |
                 backlog (looked like a tool when the list was seeded; never reported) | seeded

Run:  python misc/related_pose_citers_update.py           (weekly, by .github/workflows/related_tools.yml)
      python misc/related_pose_citers_update.py --seed    (record the current window as seen without reporting)
"""
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
SEEN = os.path.join(HERE, "related_pose_citers_seen.csv")
ON_PAGE = [os.path.join(HERE, f) for f in ("related_software.csv", "related_tools.csv", "related_tools_seen.csv")]
FIELDS = ["doi", "cites", "year", "title", "first_seen", "status"]
WINDOW_DAYS = 365
WATCHED = {   # DOI of the cited paper -> label
    "10.1038/s41592-022-01426-1": "SLEAP",
    "10.1038/s41593-018-0209-y": "DeepLabCut",
    "10.1038/s41596-019-0176-0": "DeepLabCut",    # protocol paper
    "10.1038/s41592-022-01443-0": "DeepLabCut",   # multi-animal DeepLabCut
}

NOT_PAPER = re.compile(r"^(a )?(review|survey)\b|\breview\b|perspective|primer|commentary|^correction|^erratum|^reply|"
                       r"^response|^elife assessment|^author response|^decision letter|^reviewer #", re.I)
OFF_TOPIC = re.compile(r"surg|endoscop|robot(ic)? (arm|manip)|patient|clinical trial|tumou?r|histolog", re.I)
ANIMAL = re.compile(r"\b(mouse|mice|murine|rats?|rodents?|zebrafish|fish|drosophila|fly|flies|insects?|ants?|bees?|honeybees?|"
                    r"worms?|elegans|primates?|monkeys?|macaques?|marmosets?|birds?|pigeons?|songbirds?|animals?|livestock|"
                    r"pigs?|cattle|cows?|sheep|dogs?|cats?|octopus|squid|bats?|voles?|gerbils?)\b", re.I)
NOUN = (r"(tool|toolbox|toolkit|software|package|pipeline|framework|platform|system|app|application|gui|interface|"
        r"library|method|algorithm|model|tracker|dataset|benchmark|apparatus|device|rig|workflow)")
INTRO = re.compile(r"\b(we|here we|this paper|this work|this study)\b[^.]{0,60}\b(present|presents|introduce|introduces|"
                   r"develop|developed|describe|describes|propose|proposes|built|created|designed|release|report|provide|"
                   r"offer)\b[^.]{0,90}\b" + NOUN + r"\b", re.I)
OPEN = re.compile(r"open[- ]source|freely available|publicly available|github\.com|gitlab|zenodo|pip install|python package|"
                  r"r package|user[- ]friendly|graphical user interface|\bGUI\b|code (is|are) available|available at", re.I)
NAMED = re.compile(r"^[^:]{2,40}:\s")
TITLE_TOOL = re.compile(r"\b(toolbox|toolkit|software|open[- ]source|package|pipeline|framework|platform|tracker|"
                        r"benchmark|plugin|library)\b", re.I)
LINK = re.compile(r"https?://(?:github\.com|gitlab\.com|zenodo\.org|osf\.io)/[^\s,;)]+", re.I)


def get(url):
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        url += "&api_key=" + key
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "simba-related-pose-citers"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5)


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:60]


def abstract(inv):
    pos = {p: w for w, ps in (inv or {}).items() for p in ps}
    return " ".join(pos[i] for i in sorted(pos))


def citers(doi, since):
    """Works citing `doi` published on or after `since` (page paging: cursor paging needs an API key)."""
    wid = get("https://api.openalex.org/works/doi:" + doi + "?select=id")["id"].rsplit("/", 1)[-1]
    sel = "id,doi,title,publication_year,type,abstract_inverted_index"
    out, page = [], 1
    while page <= 50:   # OpenAlex stops page paging at 10,000 results
        r = get(f"https://api.openalex.org/works?filter=cites:{wid},from_publication_date:{since}"
                f"&per-page=200&page={page}&select={sel}")
        out += r["results"]
        if len(r["results"]) < 200:
            break
        page += 1
        time.sleep(0.3)
    return out


def looks_like_tool(title, abs_text):
    """(True, why) when the paper reads as introducing a reusable tool."""
    if NOT_PAPER.search(title) or OFF_TOPIC.search(title):
        return False, ""
    if not abs_text:   # OpenAlex has no abstract for many journals: judge the title alone
        hit = TITLE_TOOL.search(title) or NAMED.match(title)
        return bool(hit), "title only, no abstract"
    text = title + " " + abs_text
    if not ANIMAL.search(text):
        return False, ""
    intro = INTRO.search(abs_text)
    score = 2 * bool(intro) + 2 * bool(OPEN.search(text)) + bool(TITLE_TOOL.search(title)) + bool(NAMED.match(title))
    return score >= 3, (intro.group(0) if intro else "tool-like title, code available")[:160]


def main():
    seed = "--seed" in sys.argv
    seen = {}
    if os.path.isfile(SEEN):
        with open(SEEN, encoding="utf-8", newline="") as f:
            seen = {r["doi"]: r for r in csv.DictReader(f)}
    seen_titles = {norm_title(r["title"]) for r in seen.values()}
    known = ""
    for path in ON_PAGE:
        with open(path, encoding="utf-8") as f:
            known += " ".join(" ".join(r.values()) for r in csv.DictReader(f)).lower()

    since = (date.today() - timedelta(days=WINDOW_DAYS)).isoformat()
    works, failed = {}, []
    for doi, label in WATCHED.items():
        try:
            for w in citers(doi, since):
                labels = works.setdefault(w["id"], (w, set()))[1]
                labels.add(label)
        except Exception as e:
            failed.append(f"{label} {doi} ({e})")

    today, checked, flagged = date.today().isoformat(), 0, []
    for w, labels in works.values():
        doi = (w.get("doi") or w["id"]).replace("https://doi.org/", "").lower()
        title = (w.get("title") or "").strip()
        if not title or doi in seen or norm_title(title) in seen_titles:
            continue
        checked += 1
        seen_titles.add(norm_title(title))
        links = []
        if w.get("type") in ("review", "editorial", "erratum", "retraction", "paratext", "letter", "peer-review"):
            tool, why = False, ""
        elif doi in known or title.lower()[:60] in known:
            tool, why = False, ""   # already on the page or handled by the SimBA citer pipeline
        else:
            abs_text = abstract(w.get("abstract_inverted_index"))
            tool, why = looks_like_tool(title, abs_text)
            links = LINK.findall(abs_text)
        status = ("backlog" if tool else "seeded") if seed else ("flagged" if tool else "no")
        seen[doi] = {"doi": doi, "cites": "; ".join(sorted(labels)), "year": w.get("publication_year") or "",
                     "title": title, "first_seen": today, "status": status}
        if tool and not seed:
            flagged.append((seen[doi], why, links))

    with open(SEEN, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(seen.values(), key=lambda r: (r["first_seen"], r["doi"])))

    print(f"Papers citing SLEAP / DeepLabCut since {since}: {len(works)} found, {checked} new this run, "
          f"{len(flagged)} look like tools" + (" (seed run: none reported)" if seed else ""))
    for r, why, links in flagged:
        print(f"  CHECK [{r['cites']}] {r['year']} | {r['title'][:140]}\n"
              f"        https://doi.org/{r['doi']}" + (f" | {links[0]}" if links else "") + f"\n        > {why}")
    if flagged:
        print("  Not added to the page. Add a tool to misc/related_software.csv by hand if it fits.")
    for msg in failed:
        print(f"WARNING: {msg} skipped")


if __name__ == "__main__":
    main()
