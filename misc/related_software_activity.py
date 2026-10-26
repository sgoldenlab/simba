# -*- coding: utf-8 -*-
"""
Refresh the automatically collected fields of the Related Software table:

* curated tools (misc/related_software.csv): GitHub stars, last push (YYYY-MM) and archived flag, and
  the year, authors, venue and citation count of the tool's paper (OpenAlex);
* curated and citing tools (misc/related_tools.csv): the license and main language GitHub reports for
  the repository, and the PyPI package with its latest release (YYYY-MM).

The Related Software table tags tools that are archived or have not been updated in two years, and
its "Hide unmaintained" filter uses the same columns. A PyPI package is recorded only when its own
project links point back to the tool's GitHub repository, so a different package that happens to share
the name is never listed.

Tools whose link is a homepage rather than a GitHub repo are mapped in HOMEPAGE_REPOS. Uses
GITHUB_TOKEN and OPENALEX_API_KEY when set (the weekly Action provides them); without a token GitHub
allows 60 requests an hour, so some rows keep their previous values.

Run:  python misc/related_software_activity.py      (then python misc/related_tools_table.py)
"""
import csv, json, os, re, time, urllib.error, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "related_software.csv")
CITING = os.path.join(HERE, "related_tools.csv")

GITHUB_REPO_RE = re.compile(r"^https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+)")
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
                  "https://www.boris.unito.it/": "olivierfriard/BORIS",
                  "https://jarvis-mocap.github.io/jarvis-docs/": "JARVIS-MoCap/JARVIS-HybridNet"}
PREPRINT_VENUES = ("biorxiv", "medrxiv", "arxiv", "research square", "ssrn", "preprints.org")
GH_HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "simba-related-software"}
if os.environ.get("GITHUB_TOKEN"):
    GH_HEADERS["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]


def repo_of(url):
    g = GITHUB_REPO_RE.match(url.strip())
    if g:
        return f"{g.group(1)}/{re.sub(r'[.]git$', '', g.group(2))}"
    return HOMEPAGE_REPOS.get(url)


def get_json(url, headers=None):
    if "api.openalex.org" in url and os.environ.get("OPENALEX_API_KEY"):
        url += ("&" if "?" in url else "?") + "api_key=" + os.environ["OPENALEX_API_KEY"]
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "simba-related-software"})
    for attempt in range(3):   # retry dropped connections; an HTTP error such as 404 is final
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError:
            raise
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def license_name(g):
    lic = g.get("license") or {}
    if not lic:
        return "Not detected"
    spdx = lic.get("spdx_id") or ""
    return "Other (see repository)" if spdx in ("", "NOASSERTION") else spdx


def pypi_package(repo, names):
    """(name, latest release YYYY-MM) of the PyPI package whose project links point to this repo;
    ("", "") when there is none, and None when PyPI could not be reached (keep the previous values)."""
    want = "github.com/" + repo.lower()
    for name in dict.fromkeys(n for n in names if n):
        try:
            p = get_json(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
        except urllib.error.HTTPError:
            continue   # no package of that name
        except Exception:
            return None
        info = p.get("info") or {}
        links = [info.get("home_page") or "", info.get("download_url") or ""] + list((info.get("project_urls") or {}).values())
        if any(re.sub(r"(\.git)?/*$", "", (u or "").lower()).split("#")[0].endswith(want) or want + "/" in (u or "").lower() for u in links):
            uploads = [f.get("upload_time", "") for f in p.get("urls") or []]
            return info.get("name") or name, (max(uploads) if uploads else "")[:7]
    return "", ""


def github(rows, url_col, name_col, activity):
    done, failed = 0, []
    for r in rows:
        repo = repo_of(r[url_col].strip())
        if not repo:
            r["gh_license"] = r["gh_language"] = r["pypi"] = r["pypi_release"] = ""
            if activity:
                r["gh_stars"] = r["gh_pushed"] = r["gh_archived"] = ""
            continue
        try:
            g = get_json(f"https://api.github.com/repos/{repo}", GH_HEADERS)
        except Exception:
            failed.append(repo)   # keep the previous values
            continue
        if activity:
            r["gh_stars"] = str(g.get("stargazers_count", ""))
            r["gh_pushed"] = (g.get("pushed_at") or "")[:7]
            r["gh_archived"] = "1" if g.get("archived") else "0"
        r["gh_license"] = license_name(g)
        r["gh_language"] = g.get("language") or ""
        name = re.split(r"\s*[(/]", r[name_col])[0].strip()
        pkg = pypi_package(repo, [r.get("pypi"), repo.split("/")[1], name.replace(" ", "-")])
        if pkg is not None:
            r["pypi"], r["pypi_release"] = pkg
        done += 1
    return done, failed


def paper_metadata(rows):
    """Year, first author, venue and citation count of each curated tool's paper."""
    done = 0
    for r in rows:
        doi = r["paper"].strip() or (r["url"].split("doi.org/", 1)[1] if "doi.org/" in r["url"] else "")
        if not doi:
            r["year"] = r["authors"] = r["venue"] = r["cited_by"] = ""
            continue
        try:
            w = get_json("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi.lower()))
        except Exception:
            continue   # keep the previous values
        names = [a["author"]["display_name"].split()[-1] for a in w.get("authorships", []) if a.get("author")]
        r["authors"] = names[0] + (" et al." if len(names) > 2 else " & " + names[1] if len(names) == 2 else "") if names else ""
        venue = re.sub(r"\s*\(.*\)$", "", ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "")
        r["venue"] = "" if venue in ("PubMed", "PubMed Central", "Europe PMC") else venue
        r["year"] = str(w.get("publication_year") or "")
        r["cited_by"] = str(w.get("cited_by_count") or 0)
        done += 1
    return done


def main():
    fields, rows = read(CSV)
    done, failed = github(rows, "url", "name", activity=True)
    papers = paper_metadata(rows)
    write(CSV, fields, rows)
    print(f"Updated GitHub activity for {done} curated tools and paper details for {papers}"
          + (f"; lookup failed for {', '.join(failed)}" if failed else ""))
    fields, rows = read(CITING)
    done, failed = github(rows, "code_url", "tool_name", activity=False)
    write(CITING, fields, rows)
    print(f"Updated GitHub license/language and PyPI for {done} citing tools"
          + (f"; lookup failed for {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
