# -*- coding: utf-8 -*-
"""
One-time geocoder for SimBA use-case institutions.

Reads the AUTHOR INSTITUTIONS column from the public studies Google Sheet,
cleans/normalises the free-text names (typos, abbreviations, duplicates),
geocodes the survivors via OpenStreetMap Nominatim (rate-limited + cached),
and writes:

  institution_coords.py   ->  INSTITUTION_ALIASES + INSTITUTION_COORDS
  geocode_cache.json       ->  raw Nominatim responses (so re-runs are instant)

Prints a report of anything that could not be placed, for manual review.

Usage:
  python misc/geocode_institutions.py                  # full regeneration
  python misc/geocode_institutions.py --fill-missing   # keep existing coords,
      only geocode institutions newly added to the sheet, and list what was
      added (eyeball those) and what still needs a manual coord in MANUAL.

Nominatim usage policy: <=1 req/sec, real User-Agent. This is not run at
docs-build time.
"""
import urllib.request, urllib.parse, csv, io, json, time, collections, os, re, sys

SHEET_ID = "169enc3Am2KQKifxj1F9KEKKLbftpMhBlw49zjl-egsY"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "geocode_cache.json")
OUT = os.path.join(HERE, "institution_coords.py")
UA = "SimBA-docs-geocoder/1.0 (https://github.com/sgoldenlab/simba; simon@netholabs.com)"

# --- 1. raw token -> canonical name (fix typos, merge dupes). ""  = drop. ----
ALIASES = {
    # OCR / spacing typos
    "University ofWashington": "University of Washington",
    "niversity of Pennsylvania": "University of Pennsylvania",
    "etherlands Institute for Neuroscience": "Netherlands Institute for Neuroscience",
    "Pontificia UniversidadCatólica del Ecuador": "Pontificia Universidad Católica del Ecuador",
    "University of California San Diego.": "University of California San Diego",
    "Institute of Physiology of the Czech Academy of\nSciences": "Institute of Physiology of the Czech Academy of Sciences",
    # duplicate / partial names -> canonical
    "The University of Hong Kong": "University of Hong Kong",
    "Washington University": "Washington University in St. Louis",
    "St. Louis": "Washington University in St. Louis",
    "Stanford": "Stanford University",
    "Cornell": "Cornell University",
    "McGill": "McGill University",
    "Concordia": "Concordia University",
    "Texas A&M": "Texas A&M University",
    "University of Leuven": "KU Leuven",
    "Jaume I University": "Universitat Jaume I",
    "Polish Academy of Science": "Polish Academy of Sciences",
    "University of Bordeaux": "Université de Bordeaux",
    "Technical University Darmstadt": "Technische Universität Darmstadt",
    "University Darmstadt": "Technische Universität Darmstadt",
    "University of New South Wales": "UNSW Sydney",
    "Children's Hospital of Philadelphia": "Children’s Hospital of Philadelphia",
    "The Children's Hospital of Philadelphia Research Institute": "Children’s Hospital of Philadelphia",
    "National Autonomous University of Mexico": "UNAM",
    "École Polytechnique Fédérale de Lausanne": "EPFL",
    "Brain Mind Institute": "EPFL",
    "Laboratory of Synaptic Mechanisms": "EPFL",
    "University of California": "University of California San Diego",
    # abbreviations -> canonical (given a manual coord below)
    "NIDA": "National Institute on Drug Abuse",
    "NIH": "National Institutes of Health",
    "NIHM": "National Institute of Mental Health",
    "National Institute of Mental Health": "National Institute of Mental Health",
    "NYU": "New York University",
    "KU": "KU Leuven",
    "Max Planck": "Max Planck Institute of Psychiatry",
    "RCSI": "Royal College of Surgeons in Ireland",
    "Salk": "Salk Institute for Biological Studies",
    "UNAM": "UNAM",
    "EPFL": "EPFL",
    "Donders Institute for Brain": "Donders Institute",
    # non-geographic fragments / noise -> drop
    "Inc": "",
    "University": "",
    "Zhengzhou": "",
    "Department of Earth and Environmental Sciences": "",
}

# --- 2. canonical -> hand-verified [lat, lon]. Wins over Nominatim. ----------
# Used for abbreviations and anything Nominatim is likely to place wrongly.
MANUAL = {
    "National Institute on Drug Abuse": [39.2946, -76.5836],   # Baltimore, MD
    "National Institutes of Health": [39.0003, -77.1029],      # Bethesda, MD
    "National Institute of Mental Health": [39.0003, -77.1029],
    "National Institute of Diabetes and Digestive and Kidney Diseases": [39.0003, -77.1029],
    "National Institute on Deafness and other Communication Disorders": [39.0003, -77.1029],
    "New York University": [40.7295, -73.9965],
    "KU Leuven": [50.8779, 4.7005],
    "Salk Institute for Biological Studies": [32.8872, -117.2454],
    "Max Planck Institute of Psychiatry": [48.1497, 11.5670],  # Munich
    "Max Planck": [48.1497, 11.5670],
    "EPFL": [46.5191, 6.5668],                                 # Lausanne
    "UNAM": [19.3320, -99.1870],                               # Mexico City
    "UNSW Sydney": [-33.9173, 151.2313],
    "Royal College of Surgeons in Ireland": [53.3392, -6.2626],# Dublin
    "Broad Institute": [42.3626, -71.0866],                    # Cambridge, MA
    "BROAD Institute": [42.3626, -71.0866],
    "Washington University in St. Louis": [38.6488, -90.3108],
    "University of Washington": [47.6553, -122.3035],          # Seattle
    "Donders Institute": [51.8199, 5.8637],                    # Nijmegen
    "Netherlands Institute for Neuroscience": [52.3593, 4.9531],
    "The International Brain Laboratory": [51.5246, -0.1340],   # UCL hub, London
    "Platea Biosciences": [42.2626, -71.8023],                 # Worcester, MA
    # --- hand-verified fixes for Nominatim "no result" failures ---
    "Aquatic Technology Promotion Station of Xiangshan County": [29.4776, 121.8695],  # Xiangshan, Ningbo
    "Biobizkaia Health Research Institute": [43.2983, -2.9903],   # Barakaldo, Spain
    "Boston University Medical Center": [42.3360, -71.0723],
    "CIBERsam-ISCiii": [40.5088, -3.6903],                        # ISCIII, Madrid
    "CaaMTech": [47.5301, -122.0326],                             # Issaquah, WA
    "Canadian Institute for Advanced Research (CIFAR)": [43.6532, -79.3832],  # Toronto
    "Federal University of Paraíba": [-7.1386, -34.8452],         # João Pessoa
    "Georgetown University Medical Center": [38.9118, -77.0752],
    "HUN-REN Biological Research Centre": [46.2472, 20.1447],     # Szeged
    "Harvard Stem Cell Institute": [42.3770, -71.1167],           # Cambridge, MA
    "Hospital General Dr. Manuel Gea González": [19.3007, -99.1880],  # Mexico City
    "Hungarian Centre of Excellence for Molecular Medicine": [46.2530, 20.1414],  # Szeged
    "IRCCS Humanitas Research Hospital": [45.3592, 9.1725],       # Rozzano, Milan
    "Inscopix": [37.4419, -122.1430],                             # Palo Alto, CA
    "Institute of Physiology of the Czech Academy of Sciences": [50.0407, 14.4740],  # Prague
    "MCCI Corporation": [42.4430, -76.5019],                      # Ithaca, NY
    "Medical Research Council Harwell": [51.5745, -1.3110],       # Harwell, UK
    "National Research Council of Italy": [41.9033, 12.5147],     # CNR, Rome
    "Nencki Institute of Experimental Biology of the Polish Academy of Sciences": [52.2560, 21.0300],  # Warsaw
    "Ningbo Institute of Oceanography": [29.8683, 121.5440],
    "Peking University School of Life Sciences": [39.9925, 116.3058],  # Beijing
    "SMART Biomedical Microsystems Laborator Université de Sherbrooke": [45.3785, -71.9245],
    "The Second Affiliated Hospital of Nanjing Medical University": [32.0616, 118.7788],
    "University of Cagliari": [39.2238, 9.1217],
    "University of South Florida Health": [28.0625, -82.4088],    # Tampa
    "Roche": [47.5580, 7.6020],                                   # HQ Basel, CH (Nominatim mis-placed to France)
    # --- corrections for Nominatim continent-level mismatches (verified by hand) ---
    "University of Hong Kong": [22.2830, 114.1371],   # was Germany
    "University of Geneva": [46.1952, 6.1408],         # was Ontario, Canada
    "University of North Carolina": [35.9049, -79.0469],  # Chapel Hill; was San Diego
    "University of Oxford": [51.7548, -1.2544],        # was California
    "University of Texas": [30.2849, -97.7341],        # Austin; was San Diego
    "DLH Corporation": [33.7490, -84.3880],            # Atlanta HQ; was Kashmir
    "University of Nebraska": [40.8206, -96.7056],     # Lincoln; was Washington DC
    "Florey Institute": [-37.7963, 144.9614],          # Melbourne; was Sheffield
    "VA Boston": [42.2977, -71.1449],                  # was Virginia
    "Royal Hospital for Women": [-33.9169, 151.2380],  # Randwick, Sydney; was London
    "McGill University": [45.5048, -73.5772],          # downtown campus (was Macdonald campus)
    "Centros de Integración Juvenil": [19.3600, -99.1800],  # Mexico City HQ; was Acapulco
    "Netholabs": [51.5074, -0.1278],                   # London (per Simon)
    "Boehringer Ingelheim Pharma GmbH & Co.": [49.9764, 8.0917],  # Ingelheim am Rhein, DE
    "Humboldt University": [52.5178, 13.3936],          # Berlin
}

DROP = {k for k, v in ALIASES.items() if v == ""}


def fetch_rows():
    req = urllib.request.Request(CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-16", "ignore")
    text = text.replace("\x00", "")
    return list(csv.DictReader(io.StringIO(text)))


def canonical(token):
    t = re.sub(r"\s+", " ", token).strip()
    if t in ALIASES:
        return ALIASES[t]
    return t


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_existing():
    """Read the coords already baked into institution_coords.py (for --fill-missing)."""
    if not os.path.exists(OUT):
        return {}
    ns = {}
    with open(OUT, encoding="utf-8") as f:
        exec(compile(f.read(), OUT, "exec"), ns)
    return ns.get("INSTITUTION_COORDS", {})


def nominatim(name, cache):
    if name in cache:
        return cache[name]
    q = urllib.parse.urlencode({"q": name, "format": "json", "limit": 1})
    url = f"https://nominatim.openstreetmap.org/search?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8"))
    except Exception as e:
        data = {"__error__": str(e)}
    cache[name] = data
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    time.sleep(1.1)  # policy: <=1 req/sec
    return data


def main():
    # --fill-missing: reuse coords already in institution_coords.py and only
    # geocode canonical names not yet present (fast incremental top-up when the
    # sheet gains a study). Default (no flag): full regeneration from scratch.
    fill = "--fill-missing" in sys.argv
    existing = load_existing() if fill else {}

    rows = fetch_rows()
    col = [c for c in rows[0] if c.strip().upper() == "AUTHOR INSTITUTIONS"][0]

    counts = collections.Counter()          # canonical -> study count
    alias_used = {}                          # raw -> canonical (non-identity only)
    for r in rows:
        for tok in (r[col] or "").split(","):
            tok = tok.strip()
            if not tok:
                continue
            canon = canonical(tok)
            norm = re.sub(r"\s+", " ", tok).strip()
            if canon == "":
                alias_used[norm] = ""  # explicit drop of a non-institution fragment
                continue
            counts[canon] += 1
            if norm != canon:
                alias_used[norm] = canon

    cache = load_cache()
    coords, failed, low_conf, added = {}, [], [], []
    for name in sorted(counts):
        if name in MANUAL:
            coords[name] = MANUAL[name]
            continue
        if fill and name in existing:
            coords[name] = existing[name]  # keep as-is, no network call
            continue
        res = nominatim(name, cache)
        if isinstance(res, dict) and "__error__" in res:
            failed.append((name, res["__error__"]))
            continue
        if not res:
            failed.append((name, "no result"))
            continue
        top = res[0]
        coords[name] = [round(float(top["lat"]), 4), round(float(top["lon"]), 4)]
        if fill:
            added.append(name)  # newly geocoded this run -> review for wrong-continent hits
        # crude confidence: flag very generic OSM class matches for review
        if top.get("class") in ("boundary", "place") and top.get("type") in ("administrative", "city", "town"):
            low_conf.append((name, top.get("display_name", "")[:70]))

    # --- write the baked module ---
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# -*- coding: utf-8 -*-\n")
        f.write('"""Auto-generated by geocode_institutions.py. INSTITUTION study coords.\n')
        f.write('Resolve a raw sheet token with: COORDS.get(ALIASES.get(tok, tok)).\n"""\n\n')
        f.write("INSTITUTION_ALIASES = {\n")
        for k in sorted(alias_used):
            f.write(f"    {k!r}: {alias_used[k]!r},\n")
        f.write("}\n\n")
        f.write("INSTITUTION_COORDS = {\n")
        for k in sorted(coords):
            lat, lon = coords[k]
            f.write(f"    {k!r}: [{lat}, {lon}],  # {counts[k]}\n")
        f.write("}\n")

    print(f"mode:                   {'fill-missing' if fill else 'full regen'}")
    print(f"canonical institutions: {len(counts)}")
    print(f"geocoded OK:            {len(coords)}")
    if fill:
        print(f"NEWLY ADDED this run ({len(added)}) -- eyeball for wrong-continent hits:")
        for n in added:
            print(f"   + {n}  ->  {coords[n]}")
    print(f"FAILED ({len(failed)}):")
    for n, why in failed:
        print(f"   - {n}  [{why}]")
    print(f"LOW-CONFIDENCE / city-level match ({len(low_conf)}) -- review:")
    for n, dn in low_conf:
        print(f"   ? {n}  ->  {dn}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
