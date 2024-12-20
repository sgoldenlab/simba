import os.path

import pandas as pd

from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext)

ADDITIONAL_FEATURES_LIST_PATH = r"C:\troubleshooting\mitra\additional_features_TAIL.csv"
FEATURES_DIR = r'C:\troubleshooting\mitra\project_folder\csv\features_extracted'
NEW_FEATURES_DIR = r'C:\troubleshooting\mitra\project_folder\videos\additional\bg_removed\rotated\tail_features_additional'
SAVE_DIR = r"C:\troubleshooting\mitra\project_folder\videos\additional\bg_removed\rotated\tail_features_additional\APPENDED"
#CLF_NAMES = ['rearing', 'grooming', 'immobility', 'lay-on-belly', 'straub_tail', 'circling', 'shaking']

additional_feature_names = list(pd.read_csv(ADDITIONAL_FEATURES_LIST_PATH, index_col=None)['ADDITIONAL_FEATURES'])
new_features_files = find_files_of_filetypes_in_directory(directory=NEW_FEATURES_DIR, extensions=['.csv'])

for file_path in new_features_files:
    print(file_path)
    df = pd.read_csv(file_path, index_col=0)
    #df_clf = df[CLF_NAMES]
    #df = df.drop(CLF_NAMES, axis=1)
    video_name = get_fn_ext(filepath=file_path)[1]
    features_path = os.path.join(FEATURES_DIR, video_name + '.csv')
    features_df = pd.read_csv(features_path, index_col=0)[additional_feature_names]
    df = pd.concat([df, features_df], axis=1)
    save_path = os.path.join(SAVE_DIR, video_name + '.csv')
    df.to_csv(save_path)

    #additional_features = df[additional_feature_names]