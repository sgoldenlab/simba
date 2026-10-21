# -*- coding: utf-8 -*-
"""
Weekly update of the "Tools from papers citing SimBA" table (docs/simba.related_software.rst).

Two steps, with a human or LLM classification in between:

    python misc/related_tools_update.py fetch
        Finds papers citing SimBA that are not yet in misc/related_tools_seen.csv:
          * OpenAlex works citing any SimBA paper (set OPENALEX_API_KEY for cursor paging),
          * the last DAYS of bioRxiv preprints likely to be behavior tools. bioRxiv DOIs get their
            reference lists weeks late, so these wait as "pending" until OpenAlex shows whether
            they cite SimBA. Pending rows with no references after PENDING_DAYS are dropped.
        Writes misc/_related_tools_candidates.json (title + abstract per paper, not committed).

    python misc/related_tools_update.py apply decisions.json
        decisions.json is a list of {"doi", "verdict": "yes"|"no", "tool_name", "category",
        "species", "one_line", "code_url"}. Every paper is recorded in related_tools_seen.csv,
        "yes" rows are appended to related_tools.csv, and the HTML table is regenerated.

    python misc/related_tools_update.py auto
        fetch, then classify the candidates with the keyword rules in rule_decision(), then apply.
        Used by .github/workflows/related_tools.yml. On the hand-sorted citing papers the rules
        were right ~80% of the time when they said "tool" and found ~65% of the tools.
"""
import csv, json, os, re, sys, urllib.parse, urllib.request
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
SEEN = os.path.join(HERE, "related_tools_seen.csv")
TOOLS = os.path.join(HERE, "related_tools.csv")
CANDIDATES = os.path.join(HERE, "_related_tools_candidates.json")

SIMBA_WORKS = {"W4398209670": "10.1038/s41593-024-01649-9",   # Nature Neuroscience (2024)
               "W3016904080": "10.1101/2020.04.19.049452",    # bioRxiv preprint (2020)
               "W4401250068": "10.1038/s44277-024-00014-9"}   # primer (2024)
DAYS = 8
PENDING_DAYS = 60
BIORXIV_CATEGORIES = {"neuroscience", "animal behavior and cognition"}
BEHAVIOR_RE = re.compile(r"\b(behavio(u)?r(al)?|pose|keypoint|tracking|video)\b", re.I)
METHOD_RE = re.compile(r"\b(classif\w*|machine learning|deep learning|pose estimation|keypoint|automated|tracking)\b", re.I)
# Keyword rules for the unattended weekly run. A paper counts as a tool when its title starts
# with a tool name ("B-REAL: ..."), names an acronym next to a method word ("... classifier (R2C2)"),
# or its abstract says "we present/introduce/develop ... <tool/pipeline/dataset/...>".
NAME_RE = re.compile(r"^\s*([A-Za-z][\w\-.+]{1,24}(?:\s[\w\-.+]{1,20}){0,2})\s*[:–—]\s")
ACRO_RE = re.compile(r"\(([A-Z][A-Za-z0-9\-]{2,15})\)")
ACRO_CONTEXT_RE = re.compile(r"(classifier|system|pipeline|tool|toolbox|platform|software|model|dataset)", re.I)
INTRO_RE = re.compile(r"\b(we|here,? we|this (work|paper|study))\s+(present|introduce|develop(ed)?|describe|propose|release|built|created?)\b"
                      r"[^.]{0,160}\b(tool|toolbox|toolkit|software|package|pipeline|platform|framework|system|dataset|benchmark"
                      r"|app|application|library|model|method|device|apparatus|setup)\b", re.I)
EXCLUDE_RE = re.compile(r"\b(review|perspective|primer on|meta-analysis|thesis|commentary)\b", re.I)
ANIMAL_RE = re.compile(r"\b(behavio(u)?r\w*|pose|keypoint|tracking|ethogram|animal|rodent|mice|mouse|rats?)\b", re.I)
CATEGORY_RULES = [("benchmark/dataset", r"\b(dataset|benchmark|database)\b"),
                  ("closed-loop/real-time", r"\b(real-time|closed-loop|online)\b"),
                  ("pose-estimation", r"\b(pose estimation|keypoint)"),
                  ("tracking", r"\btrack(ing|er)\b"),
                  ("unsupervised-behavior", r"\b(unsupervised|self-supervised|clustering|embedding|motif)"),
                  ("gait/kinematics", r"\b(gait|kinematic|locomotion|walking)"),
                  ("hardware/acquisition", r"\b(camera|device|hardware|apparatus|microscope|implant|raspberry)"),
                  ("supervised-classification", r"\b(classif\w*|recogni\w*|detect\w*|scor(e|ing))\b")]
SPECIES_RULES = [("Mouse", r"\b(mouse|mice)\b"), ("Rat", r"\brats?\b"), ("Zebrafish", r"\bzebrafish\b"),
                 ("Drosophila", r"\b(drosophila|flies)\b"), ("Macaque", r"\b(macaque|monkey)s?\b"),
                 ("Marmoset", r"\bmarmosets?\b")]
SEEN_COLS = ["doi", "verdict", "year", "source", "checked", "title"]
TOOL_COLS = ["tool_name", "title", "year", "category", "species", "one_line", "code_url", "doi", "other_dois"]


def get(url):
    key = os.environ.get("OPENALEX_API_KEY")
    if key and "api.openalex.org" in url:
        url += ("&" if "?" in url else "?") + "api_key=" + key
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def norm_doi(d):
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", (d or "").strip()).lower()


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:60]


def abstract(inv):
    if not inv:
        return ""
    pos = {p: w for w, ps in inv.items() for p in ps}
    return " ".join(pos[i] for i in sorted(pos))


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        # "\n", not csv's default "\r\n": the Linux runner would otherwise rewrite every line each week
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def openalex_citers():
    works = {}
    use_cursor = bool(os.environ.get("OPENALEX_API_KEY"))
    for wid in SIMBA_WORKS:
        cursor, page = "*", 1
        while True:
            paging = f"cursor={cursor}" if use_cursor else f"page={page}"
            r = get(f"https://api.openalex.org/works?filter=cites:{wid}&per-page=200&{paging}")
            for w in r["results"]:
                works[w["id"]] = w
            if not r["results"] or len(r["results"]) < 200:
                break
            cursor, page = r["meta"].get("next_cursor"), page + 1
            if use_cursor and not cursor:
                break
    return works.values()


def cites_simba(doi):
    """True / False once OpenAlex has the reference list, None while it is still empty."""
    try:
        w = get("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi))
    except Exception:
        return None
    refs = w.get("referenced_works") or []
    if not refs:
        return None
    return any(r.rsplit("/", 1)[-1] in SIMBA_WORKS for r in refs)


def fetch():
    seen = read_csv(SEEN)
    seen_doi = {r["doi"] for r in seen}
    seen_title = {norm_title(r["title"]) for r in seen}
    today = date.today()
    candidates = []

    def is_new(doi, title):
        return doi not in seen_doi and norm_title(title) not in seen_title

    for w in openalex_citers():
        doi, title = norm_doi(w.get("doi")), w.get("title") or ""
        if title and is_new(doi, title):
            candidates.append({"doi": doi, "title": title, "year": w.get("publication_year"), "source": "openalex",
                               "abstract": abstract(w.get("abstract_inverted_index"))})
            seen_doi.add(doi)
            seen_title.add(norm_title(title))

    # Pending preprints: classify once they are confirmed to cite SimBA, drop when they do not.
    for r in seen:
        if r["verdict"] != "pending":
            continue
        c = cites_simba(r["doi"])
        if c is True:
            candidates.append({"doi": r["doi"], "title": r["title"], "year": r["year"], "source": "biorxiv",
                               "abstract": ""})
        elif c is False or (today - date.fromisoformat(r["checked"])).days > PENDING_DAYS:
            r["verdict"] = "no"

    # This week's bioRxiv preprints that look like behavior-analysis methods.
    start, cursor = (today - timedelta(days=DAYS)).isoformat(), 0
    while True:
        r = get(f"https://api.biorxiv.org/details/biorxiv/{start}/{today.isoformat()}/{cursor}")
        batch = r.get("collection", [])
        for p in batch:
            doi = p["doi"].lower()
            if (p["category"] in BIORXIV_CATEGORIES and BEHAVIOR_RE.search(p["abstract"])
                    and METHOD_RE.search(p["abstract"]) and is_new(doi, p["title"])):
                seen.append({"doi": doi, "verdict": "pending", "year": p["date"][:4], "source": "biorxiv",
                             "checked": today.isoformat(), "title": p["title"]})
                seen_doi.add(doi)
                seen_title.add(norm_title(p["title"]))
        cursor += len(batch)
        if not batch or cursor >= int(r["messages"][0].get("total", 0)):
            break

    write_csv(SEEN, seen, SEEN_COLS)
    with open(CANDIDATES, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=1, ensure_ascii=False)
    n_pending = sum(r["verdict"] == "pending" for r in seen)
    print(f"{len(candidates)} new papers to classify -> {os.path.normpath(CANDIDATES)}; {n_pending} preprints pending")


def apply(path):
    with open(path, encoding="utf-8") as f:
        decisions = json.load(f)
    seen, tools = read_csv(SEEN), read_csv(TOOLS)
    by_doi = {r["doi"]: r for r in seen}
    known = {(t["tool_name"].lower() or norm_title(t["title"])) for t in tools}
    today = date.today().isoformat()
    added = []
    for d in decisions:
        doi = norm_doi(d["doi"])
        row = by_doi.get(doi)
        if row is None:
            row = {"doi": doi, "source": d.get("source", "openalex")}
            seen.append(row)
        row.update(verdict=d["verdict"], year=d.get("year", row.get("year", "")),
                   title=d.get("title", row.get("title", "")), checked=today)
        key = d.get("tool_name", "").strip().lower() or norm_title(row["title"])
        if d["verdict"] == "yes" and key not in known:
            known.add(key)
            tools.append({"tool_name": d.get("tool_name", "").strip(), "title": row["title"], "year": row["year"],
                          "category": d.get("category", ""), "species": d.get("species", ""),
                          "one_line": d.get("one_line", ""), "code_url": d.get("code_url", ""),
                          "doi": "https://doi.org/" + doi if doi.startswith("10.") else doi, "other_dois": ""})
            added.append(d.get("tool_name") or row["title"])
    write_csv(SEEN, seen, SEEN_COLS)
    tools.sort(key=lambda t: (-int(t["year"] or 0), (t["tool_name"] or t["title"]).lower()))
    write_csv(TOOLS, tools, TOOL_COLS)
    sys.path.insert(0, HERE)
    import related_tools_table
    related_tools_table.main()
    print(f"Added {len(added)} tools: {', '.join(added) or '-'}")


def rule_decision(c):
    title, text = c["title"], c["title"] + ". " + c["abstract"]
    decision = {"doi": c["doi"], "title": title, "year": c["year"], "source": c["source"], "verdict": "no"}
    if EXCLUDE_RE.search(title) or not ANIMAL_RE.search(text):
        return decision
    m, acro = NAME_RE.match(title), ACRO_RE.search(title)
    named = bool(m) and not m.group(1).lower().startswith(("the ", "a ", "an ")) and \
        (" " not in m.group(1) or any(ch.isupper() or ch.isdigit() for ch in m.group(1)[1:]))
    intro = INTRO_RE.search(text)
    if not (named or (acro and ACRO_CONTEXT_RE.search(title)) or intro):
        return decision
    species = [s for s, pat in SPECIES_RULES if re.search(pat, text, re.I)]
    sentence = intro.group(0) if intro else title.split(":", 1)[-1]
    decision.update(verdict="yes", tool_name=m.group(1) if named else (acro.group(1) if acro else ""),
                    category=next((k for k, pat in CATEGORY_RULES if re.search(pat, text, re.I)), "other-analysis"),
                    species=species[0] if len(species) == 1 else "Multiple" if species else "",
                    one_line=re.sub(r"\s+", " ", sentence).strip()[:140], code_url="")
    return decision


def auto():
    fetch()
    with open(CANDIDATES, encoding="utf-8") as f:
        candidates = json.load(f)
    for c in candidates:
        if not c["abstract"] and c["doi"].startswith(("10.1101/", "10.64898/")):
            try:
                c["abstract"] = get("https://api.biorxiv.org/details/biorxiv/" + c["doi"])["collection"][-1]["abstract"]
            except Exception:
                pass
    with open(CANDIDATES, "w", encoding="utf-8") as f:
        json.dump([rule_decision(c) for c in candidates], f, indent=1, ensure_ascii=False)
    apply(CANDIDATES)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "auto":
        auto()
    elif len(sys.argv) >= 2 and sys.argv[1] == "fetch":
        fetch()
    elif len(sys.argv) >= 3 and sys.argv[1] == "apply":
        apply(sys.argv[2])
    else:
        print(__doc__)
