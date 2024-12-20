import pandas as pd
import glob
import os
from simba.utils.read_write import get_fn_ext
import shutil

# ANNOTATED_VIDEOS_PATH = '/Users/simon/Desktop/envs/simba/troubleshooting/mitra/annotated_videos.csv'
# DATA_PATH = '/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/input_csv/originals'
# OUT_DIR = '/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/input_csv/'
# NOT_ANNOT_DIR = '/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/input_csv/not_annotated'
#
#
# file_paths = glob.glob(DATA_PATH + '/*.csv')
# annotated_lst = list(pd.read_csv(ANNOTATED_VIDEOS_PATH)['VIDE_NAME'])
#
# for file_path in file_paths:
#     dir, file_name, ext = get_fn_ext(filepath=file_path)
#     file_name = file_name.split('.', 1)[0]
#     if file_name in annotated_lst:
#         save_path = os.path.join(OUT_DIR, file_name + '.csv')
#     else:
#         save_path = os.path.join(NOT_ANNOT_DIR, file_name + '.csv')
#     shutil.copy(src=file_path, dst=save_path)

annot_df = pd.read_csv('/Users/simon/Desktop/envs/simba/troubleshooting/mitra/annotations.csv')
annot_df[['START','STOP']] = annot_df['START-STOP'].str.split('-',expand=True)[[0, 1]]
file_paths = glob.glob('/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/features_extracted' + '/*.csv')
for file_path in file_paths:
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    video_annot = annot_df[annot_df['VIDEO'] == file_name]
    if len(video_annot) > 0:
        data_df = pd.read_csv(file_path)
        for clf in annot_df['BEHAVIOR'].unique():
            data_df[clf] = 0
            video_clf_annot = video_annot[video_annot['BEHAVIOR'] == clf]

            annotations_idx = list(video_clf_annot.apply(lambda x: list(range(int(x["START"]), int(x["STOP"]) + 1)),1))
            print(annotations_idx)


