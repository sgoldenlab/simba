import glob
import os

import pandas as pd

from simba.misc_tools import detect_bouts, get_fn_ext

DATA_FOLDER = "/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/csv/machine_results"
FPS = 25
SAVE_PATH = "/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/csv/my_timestamps.csv"
CLASSIFIERS = [
    "Freezing",
    "Normal Swimming",
    "Fast Swimming",
    "Erratic Turning",
    "Normal Turning",
    "Wall Bumping",
    "Top",
    "Bottom",
    "Floor Skimming",
]


def time_stamp_extractor(df: pd.DataFrame):
    results = pd.DataFrame()
    bouts = detect_bouts(data_df=df, target_lst=CLASSIFIERS, fps=FPS)[
        ["Event", "Start_time", "End Time"]
    ]
    for clf in CLASSIFIERS:
        clf_df = bouts[bouts["Event"] == clf]
        clf_df["Start_time"] = pd.to_timedelta(
            clf_df["Start_time"].astype(float), unit="s"
        )
        clf_df["Start_time"] = clf_df["Start_time"].astype(str).map(lambda x: x[7:])
        clf_df["End Time"] = pd.to_timedelta(clf_df["End Time"].astype(float), unit="s")
        clf_df["End Time"] = clf_df["End Time"].astype(str).map(lambda x: x[7:])
        results = pd.concat([results, clf_df])
    return results


data_files = glob.glob(DATA_FOLDER + "/*.csv")
out = []
for file_path in data_files:
    _, video_name, _ = get_fn_ext(file_path)
    df = pd.read_csv(file_path, usecols=CLASSIFIERS)
    results = time_stamp_extractor(df=df)
    results.insert(0, "VIDEO NAME", video_name)
    out.append(results)
out = pd.concat(out, axis=0).reset_index(drop=True)
out.to_csv(SAVE_PATH)
print(f"DATA SAVED AT {SAVE_PATH}")
