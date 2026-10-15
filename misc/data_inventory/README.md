# SimBA data inventory

Generates the docs page [`docs/data_inventory.rst`](../../docs/data_inventory.rst)
(<https://simba-uw-tf-dev.readthedocs.io/en/latest/data_inventory.html>, linked from `example_datasets.rst`) and its downloads in
`docs/_static/data_inventory/`: every public SimBA file on OSF (all 28 projects on the sgoldenlab account), the mouse
resident-intruder Google Drive folder and the data files tracked in this repo, with archive contents, body parts,
classifier settings, feature columns, checksums and maintainer notes.

**Both outputs are generated. Do not edit them by hand**; change `build_data_inventory.py` / `page_layout.py` and rebuild.
The crawlers only read (OSF API, public Drive pages, HTTP range requests into archives); nothing on OSF or Drive is changed.

## When to refresh

After uploading, moving or deleting data on OSF or the Drive folder, or after changing which docs link to it.
The page shows its snapshot date (`SNAPSHOT_DATE` in `build_data_inventory.py`); update it when you refresh.

## Refresh

Run from this folder. Crawl results go to `data/` (gitignored, ~13 MB); some crawlers write next to themselves, move
those JSON files into `data/`.

```
python crawlers/crawl_osf.py tmu6y      # repeat for d69jt uhjzf 5t4y9 dg385 fvbuh; rename each osf_inventory.json -> osf_<root>.json
                                        # then merge into data/osf_all.json (list of nodes, each with "root": <root id>)
python crawlers/crawl_gdrive.py         # -> gdrive_inventory.json
python crawlers/zip_contents.py data/gdrive_inventory.json gdrive   # -> gdrive_zip_contents.json
python crawlers/zip_contents.py data/osf_all.json osf              # -> osf_zip_contents.json (single zips)
python crawlers/split_zip_contents.py                               # -> osf_split_zip_contents.json (.zip.001 ...)
python crawlers/fetch_hashes.py                                     # -> data/osf_hashes.json (sha256, used for duplicates)
python crawlers/repo_refs.py                                        # -> data/repo_refs.json (OSF/Drive links in the repo)
python crawlers/bodyparts_from_archives.py                          # -> data/archive_bodyparts.json (body parts from CollectedData CSVs)
python crawlers/feature_columns.py                                  # -> data/feature_columns.json (header of one training CSV per dataset)
# classifier settings: download the OSF 'SimBA meta files' CSVs + dg385 model_settings.csv into data/meta_files/<node>__<path with __>
python crawlers/current_feature_columns.py                           # -> data/current_feature_columns.json (columns this SimBA version computes; needs a SimBA env)
python crawlers/repo_data_files.py                                  # -> data/repo_data_files.json (data files tracked in git)
python build_data_inventory.py                                      # -> docs/data_inventory.rst + docs/_static/data_inventory/
```

The build is deterministic: unchanged crawl data gives byte-identical outputs, so `git diff` shows only real changes.
It deletes files in `docs/_static/data_inventory/` that the page no longer links.

## Files

- `build_data_inventory.py`: loads the crawl data, builds rows and checks (duplicates, empty folders, settings mismatches ...), CSS, downloads.
- `page_layout.py`: the page itself (jump menu, start here, searchable table + filter JS, browse by location, repo files, maintainer notes).
- `crawlers/`: one script per source, see above. The Drive crawler scrapes public folder pages (no API), so it is the one most likely to break.

## Verified by hand (2026-10-15), not re-checked by a rebuild

- `.sav` loading: `3mc7g/lateral_threat.sav` (490 features) and Drive `Allogroom_vigorous.sav` (498 features) load in py3.6.13/sklearn 0.22.2, fail in py3.10/sklearn 1.4.2.
- Drive annotation CSVs: 548 cols = index + 48 pose (16 bp) + 498 features + 1 behavior.
- No config.yaml/pose_cfg.yaml anywhere on OSF, including inside archives.
- Training CSV columns: Drive mouse 16 bp/498 features; rat + CRIM13 16 bp/490 features + `video_no`, `frames` bookkeeping columns, behavior cols named *_prediction; example project + Tutorial Data 14 bp/454 features.
- vs. current SimBA (`crawlers/current_feature_columns.py`, compared automatically on the page): all sets use old names for the 8 hull features
  (same positions); rat + CRIM13 lack the 8 nose/centroid-to-lateral distances, hence the 490 vs 498 classifier split.
  Inference (`clf_predict_proba`) passes features as an array and checks only the count, so renamed columns still predict.
- 8 of 29 classifier settings files name a different classifier inside; the 11 CRIM13 ones have shifted columns.
