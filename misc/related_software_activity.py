# -*- coding: utf-8 -*-
"""
Refresh GitHub stars, last push (YYYY-MM) and archived flag for the curated tools in
misc/related_software.csv. The Related Software table tags tools that are archived or have not
been updated in two years, and its "Hide unmaintained" filter uses the same columns.

Tools whose link is a homepage rather than a GitHub repo are mapped in HOMEPAGE_REPOS. Uses
GITHUB_TOKEN when set (the weekly Action provides one); without it GitHub allows 60 requests an
hour, so some rows keep their previous values.

Run:  python misc/related_software_activity.py      (then python misc/related_tools_table.py)
"""
import csv, json, os, re, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "related_software.csv")

GITHUB_REPO_RE = re.compile(r"^https?://github\.com/([\w.-]+)/([\w.-]+)/?$")
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


def repo_of(url):
    g = GITHUB_REPO_RE.match(url)
    if g:
        return f"{g.group(1)}/{re.sub(r'[.]git$', '', g.group(2))}"
    return HOMEPAGE_REPOS.get(url)


def main():
    with open(CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "simba-related-software"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    done, failed = 0, []
    for r in rows:
        repo = repo_of(r["url"].strip())
        if not repo:
            r["gh_stars"] = r["gh_pushed"] = r["gh_archived"] = ""
            continue
        try:
            req = urllib.request.Request(f"https://api.github.com/repos/{repo}", headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                g = json.loads(resp.read())
        except Exception:
            failed.append(repo)   # keep the previous values
            continue
        r["gh_stars"] = str(g.get("stargazers_count", ""))
        r["gh_pushed"] = (g.get("pushed_at") or "")[:7]
        r["gh_archived"] = "1" if g.get("archived") else "0"
        done += 1
    with open(CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Updated GitHub activity for {done} curated tools" + (f"; lookup failed for {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
