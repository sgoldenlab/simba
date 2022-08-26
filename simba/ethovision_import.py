import pandas as pd
import os, glob
import numpy as np
from configparser import ConfigParser, NoOptionError
import pandas as pd
from simba.rw_dfs import read_df, save_df
from simba.features_scripts.unit_tests import read_video_info, read_video_info_csv
from simba.drop_bp_cords import get_fn_ext
from simba.drop_bp_cords import get_workflow_file_format


class ImportEthovision(object):
    def __init__(self, config_path=None, folder_path=None):
        print('Appending Ethovision annotations...')
        self.config = ConfigParser()
        self.config.read(str(config_path))
        self.files_found = glob.glob(folder_path + '/*.xlsx') + glob.glob(folder_path + '/*.xls')
        self.files_found = [x for x in self.files_found if '~$' not in x]
        model_nos = self.config.getint('SML settings', 'No_targets')
        self.classifier_names = []
        for i in range(model_nos):
            self.classifier_names.append(self.config.get('SML settings', 'target_name_' + str(i + 1)))
        self.project_path = self.config.get('General settings', 'project_path')
        self.vid_config_path = os.path.join(self.project_path, 'logs', 'video_info.csv')
        self.features_folder = os.path.join(self.project_path, 'csv', 'features_extracted')
        self.targets_insert_folder_path = os.path.join(self.project_path, 'csv', 'targets_inserted')
        self.video_config_df = read_video_info_csv(self.vid_config_path)
        self.workflow_file_format = get_workflow_file_format(self.config)
        self.read_files()
        print('All Ethovision annotations added. Files with annotation are located in the project_folder/csv/targets_inserted directory.')

    def read_files(self):
        for file_path in self.files_found:
            ethovision_df = pd.read_excel(file_path, sheet_name=None)
            manual_scoring_sheet_name = list(ethovision_df.keys())[-1]
            ethovision_df = pd.read_excel(file_path, sheet_name=manual_scoring_sheet_name, index_col=0, header=None)
            video_path = ethovision_df.loc['Video file'].values[0]
            dir_name, self.video_name, ext = get_fn_ext(video_path)
            self.features_file_path = os.path.join(self.features_folder, self.video_name + '.' + self.workflow_file_format)
            print('Processing annotations for video ' + str(self.video_name) + '...')
            _, _, fps = read_video_info(self.video_config_df, str(self.video_name))
            header_lines_n = int(ethovision_df.loc['Number of header lines:'].values[0]) - 2
            ethovision_df = ethovision_df.iloc[header_lines_n:].reset_index(drop=True)
            ethovision_df.columns = list(ethovision_df.iloc[0])
            ethovision_df = ethovision_df.iloc[2:].reset_index(drop=True)
            self.clf_dict = {}
            for clf in self.classifier_names:
                self.clf_dict[clf] = {}
                clf_data = ethovision_df[ethovision_df['Behavior'] == clf]
                starts = list(clf_data['Recording time'][clf_data['Event'] == 'state start'])
                ends = list(clf_data['Recording time'][clf_data['Event'] == 'state stop'])
                self.clf_dict[clf]['start_frames'] = [int(x * fps) for x in starts]
                self.clf_dict[clf]['end_frames'] = [int(x * fps) for x in ends]
                frame_list = []
                for cnt, start in enumerate(self.clf_dict[clf]['start_frames']):
                    frame_list.extend(list(range(start, self.clf_dict[clf]['end_frames'][cnt])))
                self.clf_dict[clf]['frames'] = frame_list
            self.insert_annotations()

    def insert_annotations(self):
        self.features_df = read_df(self.features_file_path, self.workflow_file_format)
        for clf in self.classifier_names:
            self.features_df[clf] = 0
            self.features_df[clf] = np.where(self.features_df.index.isin(self.clf_dict[clf]['frames']), 1, 0)
        self.save_data()

    def save_data(self):
        save_file_name = os.path.join(self.targets_insert_folder_path, self.video_name + '.' + self.workflow_file_format)
        save_df(self.features_df, self.workflow_file_format, save_file_name)
        print('Added Ethovision annotations for video {} ... '.format(self.video_name))

#test = ImportEthovision(config_path= r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini", folder_path=r'Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\ethovision_import')