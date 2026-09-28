# -*- coding: utf-8 -*-
"""
Daily top-up for the use-case pages, run by CI before the generators:

* institutions new to the sheet are geocoded (geocode_institutions.main(fill=True)) and written
  into institution_coords.py, so they appear on the globe the same day;
* country strings none of the lists recognise are looked up once on OpenStreetMap Nominatim
  (country-level search) and saved to country_auto.json, so they are counted everywhere;
* everything placed or left unresolved is written as a Markdown report (--report PATH) for the
  review issue -- only items not reported before (places_reported.json), so the issue is not
  commented on every day about the same thing.

Run:  python misc/resolve_new_places.py [--report report.md]
"""
import json, os, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geocode_institutions
from country_names import COUNTRY_NAMES
from usecase_map_stats import COUNTRY_AUTO_JSON, country_iso, fetch_rows, split_countries

REPORTED = os.path.join(HERE, "places_reported.json")


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")


def lookup_country(name):
    """(ISO-2, place name) for a country string, or (None, reason). One Nominatim request."""
    q = urllib.parse.urlencode({"q": name, "format": "json", "limit": 1,
                                "featuretype": "country", "addressdetails": 1})
    req = urllib.request.Request(f"https://nominatim.openstreetmap.org/search?{q}",
                                 headers={"User-Agent": geocode_institutions.UA})
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8"))
    except Exception as e:
        return None, f"lookup failed ({e})"
    finally:
        time.sleep(1.1)  # Nominatim policy: <=1 req/sec
    if not data:
        return None, "no match"
    code = ((data[0].get("address") or {}).get("country_code") or "").upper()
    place = data[0].get("display_name", "")
    return (code, place) if code in COUNTRY_NAMES else (None, f"no country found ({place})")


def osm(lat, lon):
    return f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=7/{lat}/{lon}"


def main():
    report_path = sys.argv[sys.argv.index("--report") + 1] if "--report" in sys.argv else None
    rows = fetch_rows()
    col = next(c for c in rows[0] if c.strip().upper() == "COUNTRIES")

    # ---- countries ----
    unknown = sorted({c for r in rows for c in split_countries(r.get(col)) if country_iso(c) is None})
    auto = load_json(COUNTRY_AUTO_JSON, {})
    new_countries, bad_countries = [], []
    for c in unknown:
        iso, info = lookup_country(c)
        if iso:
            auto[c.lower()] = iso
            new_countries.append((c, iso))
        else:
            bad_countries.append((c, info))
    if new_countries:
        write_json(COUNTRY_AUTO_JSON, auto)

    # ---- institutions ----
    inst = geocode_institutions.main(fill=True)

    # ---- report: only what has not been reported before ----
    seen = set(load_json(REPORTED, []))
    sections = [
        ("Institutions added to the globe", "inst",
         [(n, f"**{n}** → {ll[0]}, {ll[1]} ([map]({osm(*ll)}))") for n, ll in inst["added"]]),
        ("Institutions placed only at city/region level — check the location", "lowconf",
         [(n, f"**{n}** → {place}") for n, place in inst["low_conf"]]),
        ("Institutions that couldn't be located (not on the globe; add to `MANUAL` in "
         "`misc/geocode_institutions.py`)", "instfail",
         [(n, f"**{n}** — {why}") for n, why in inst["failed"]]),
        ("Country spellings recognised automatically", "country",
         [(c, f"`{c}` → **{COUNTRY_NAMES[iso]}** ({iso})") for c, iso in new_countries]),
        ("Country spellings that couldn't be recognised (not counted; fix the sheet)", "countryfail",
         [(c, f"`{c}` — {why}") for c, why in bad_countries]),
    ]
    parts = []
    for title, kind, items in sections:
        fresh = [(k, line) for k, line in items if f"{kind}:{k}" not in seen]
        if fresh:
            parts.append(f"### {title}\n" + "\n".join(f"- {line}" for _, line in fresh))
            seen.update(f"{kind}:{k}" for k, _ in fresh)
    if parts:
        report = ("The daily use-case job updated the places below from the studies sheet. "
                  "Please check each one; wrong institution locations are fixed in `MANUAL` / "
                  "`ALIASES` in `misc/geocode_institutions.py`, wrong countries in the sheet "
                  "(its COUNTRIES dropdown should use the names in `misc/country_names.py`).\n\n"
                  + "\n\n".join(parts) + "\n")
        write_json(REPORTED, sorted(seen))
        print(report)
        if report_path:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
    else:
        print("[places] nothing new to report")


if __name__ == "__main__":
    main()
