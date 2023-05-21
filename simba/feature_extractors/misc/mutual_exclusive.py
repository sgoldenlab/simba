import glob
import os
from copy import deepcopy

import pandas as pd

DATA_DIR = "/Users/simon/Desktop/envs/simba_dev/tests/test_data/visualization_tests/project_folder/csv/machine_results"
SAVE_DIR = "/Users/simon/Desktop/envs/simba_dev/tests/test_data/visualization_tests/project_folder/csv/output"
FIRST_CLF = "Swimming_normal"
SECOND_CLF = "Swimming_fast"


data_paths = glob.glob(DATA_DIR + "/*.csv")
for file_path in data_paths:
    df = pd.read_csv(file_path, index_col=0)
    output_df = deepcopy(df)
    missing_cols = list(set(["FIRST_CLF", "SECOND_CLF"]) - set(list(df.columns)))
    if missing_cols:
        print(f"ERROR: {missing_cols} column(s) missing in file {file_path}")
    df.loc[df.FIRST_CLF == 1, SECOND_CLF] = 0
