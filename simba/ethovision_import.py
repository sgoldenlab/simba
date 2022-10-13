__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
import os, glob
import numpy as np
from configparser import ConfigParser, NoOptionError
import pandas as pd
from simba.rw_dfs import read_df, save_df
from simba.read_config_unit_tests import read_config_entry, read_config_file
from simba.features_scripts.unit_tests import read_video_info, read_video_info_csv
from simba.drop_bp_cords import get_fn_ext
from simba.drop_bp_cords import get_workflow_file_format


class ImportEthovision(object):
    """
    Class for appending ETHOVISION human annotations onto featurized pose-estimation data.
    Results (parquets' or CSVs) are saved within the the project_folder/csv/targets_inserted directory of
    the SimBA project

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    folder_path: str
        path to folder holding ETHOVISION data files is XLSX or XLS format

    Notes
    -----
    `Third-party import GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md>`__.

    Examples
    -----
    >>> ImportEthovision(config_path="MyConfigPath", folder_path="MyEthovisionFolderPath")
    """

    def __init__(self, config_path: str,
                 folder_path: str):

        print('Appending Ethovision annotations...')
        self.config = read_config_file(config_path)
        self.files_found = glob.glob(folder_path + '/*.xlsx') + glob.glob(folder_path + '/*.xls')
        self.files_found = [x for x in self.files_found if '~$' not in x]
        if len(self.files_found) == 0:
            print('SIMBA ERROR: No ethovision xlsx or xls files found in {}'.format(str(folder_path)))
            raise ValueError('SIMBA ERROR: No ethovision xlsx or xls files found in {}'.format(str(folder_path)))
        model_nos = self.config.getint('SML settings', 'no_targets')
        self.classifier_names = []
        for i in range(model_nos):
            self.classifier_names.append(self.config.get('SML settings', 'target_name_' + str(i + 1)))
        self.project_path = self.config.get('General settings', 'project_path')
        self.vid_config_path = os.path.join(self.project_path, 'logs', 'video_info.csv')
        self.features_folder = os.path.join(self.project_path, 'csv', 'features_extracted')
        self.targets_insert_folder_path = os.path.join(self.project_path, 'csv', 'targets_inserted')
        self.video_config_df = read_video_info_csv(self.vid_config_path)
        self.workflow_file_format = get_workflow_file_format(self.config)
        self.__read_files()
        print('All Ethovision annotations added. Files with annotation are located in the project_folder/csv/targets_inserted directory.')

    def __read_files(self):
        for file_path in self.files_found:
            ethovision_df = pd.read_excel(file_path, sheet_name=None)
            manual_scoring_sheet_name = list(ethovision_df.keys())[-1]
            ethovision_df = pd.read_excel(file_path, sheet_name=manual_scoring_sheet_name, index_col=0, header=None)
            try:
                video_path = ethovision_df.loc['Video file'].values[0]
            except KeyError:
                print('SIMBA ERROR: "Video file" does not exist in the sheet named {} in file {}'.format(manual_scoring_sheet_name,file_path))
                raise ValueError('SIMBA ERROR: "Video file" does not exist in the sheet named {} in file {}'.format(manual_scoring_sheet_name,file_path))
            try:
                if np.isnan(video_path):
                    print('SIMBA ERROR: "Video file" does not have a value in the sheet named {} in file {}'.format(manual_scoring_sheet_name,file_path))
                    raise ValueError('SIMBA ERROR: "Video file" does not have a value in the sheet named {} in file {}'.format(manual_scoring_sheet_name,file_path))
            except:
                pass
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
            self.__insert_annotations()

    def __insert_annotations(self):
        self.features_df = read_df(self.features_file_path, self.workflow_file_format)
        for clf in self.classifier_names:
            self.features_df[clf] = 0
            self.features_df[clf] = np.where(self.features_df.index.isin(self.clf_dict[clf]['frames']), 1, 0)
        self.__save_data()

    def __save_data(self):
        save_file_name = os.path.join(self.targets_insert_folder_path, self.video_name + '.' + self.workflow_file_format)
        save_df(self.features_df, self.workflow_file_format, save_file_name)
        print('Added Ethovision annotations for video {} ... '.format(self.video_name))

#test = ImportEthovision(config_path= r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini", folder_path=r'Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\ethovision_import')
# test = ImportEthovision(config_path= r"/Users/simon/Desktop/simbapypi_dev/tests/test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini", folder_path=r'/Users/simon/Desktop/simbapypi_dev/tests/test_data/multi_animal_dlc_two_c57/ethovision_import')
#

# @pytest.mark.parametrize("config_path, folder_path", [('test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini', 'test_data/multi_animal_dlc_two_c57/ethovision_import')])
# def test_ethovision_import_use_case(config_path, folder_path):
#     _ = ImportEthovision(config_path=config_path, folder_path=folder_path)