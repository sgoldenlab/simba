import pandas as pd
import os, glob
from pathlib import Path

FEATURES_DIR = '/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/csv/features_extracted'
TARGETS_DIR = '/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/csv/targets_inserted'

CLF_NAMES = ['Locomote_Off_Wheel',
             'Move_Under_Wheel',
             'Rear',
             'Run_On_Wheel',
             'Stumble_On_Wheel',
             'Thrown_Off_Wheel']

files = glob.glob(FEATURES_DIR + '/*.csv')
for file_path in files:
    features_df = pd.read_csv(file_path, index_col=0)
    file_extension = Path(file_path).suffix
    file_name = os.path.basename(file_path.rsplit(file_extension, 1)[0])
    target_path = os.path.join(TARGETS_DIR, file_name + '.csv')
    if os.path.isfile(target_path):
        target_df = pd.read_csv(target_path, index_col=0)[CLF_NAMES]
        out_df = pd.concat([features_df, target_df], axis=1)
        out_df.to_csv(target_path, index=True)
        print(target_path + ' saved...')