from simba.read_config_unit_tests import (check_int,
                                          check_str,
                                          check_float,
                                          read_config_entry,
                                          check_file_exist_and_readable,
                                          read_config_file)
import os, glob
import pandas as pd
from simba.drop_bp_cords import getBpNames, get_fn_ext
from simba.rw_dfs import read_df, save_df
from simba.train_model_functions import insert_column_headers_for_outlier_correction

class OutlierCorrectionSkipper(object):
    def __init__(self,
                 config_path: str = None):
        self.config = read_config_file(config_path)
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.bp_file_path = os.path.join(self.project_path, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
        self.in_dir = os.path.join(self.project_path, 'csv', 'input_csv')
        self.out_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)
        check_file_exist_and_readable(self.bp_file_path)
        self.x_cols, self.y_cols, self.p_cols = getBpNames(config_path)
        self.header_line = []
        for cols in zip(self.x_cols, self.y_cols, self.p_cols):
            self.header_line.extend((list(cols)))
        self.files_found = glob.glob(self.in_dir + '/*.' + self.file_type)
        print('Processing {} file(s)...'.format(str(len(self.files_found))))

    def skip_outlier_correction(self):
        self.file_cnt = 0
        for file_cnt, file_path in enumerate(self.files_found):
            _, video_name, ext = get_fn_ext(file_path)
            data_df = read_df(file_path, self.file_type)
            print(data_df)
            if self.file_type == 'csv':
                data_df = data_df.drop(data_df.index[[0, 1]]).apply(pd.to_numeric)
            if 'scorer' in data_df.columns:
                data_df = data_df.set_index('scorer')
            data_df = insert_column_headers_for_outlier_correction(data_df=data_df, new_headers=self.header_line, filepath=file_path)
            data_df.index.name = None
            save_path = os.path.join(self.out_dir, video_name + '.' + self.file_type)
            save_df(data_df, self.file_type, save_path)
            self.file_cnt += 1
            print('Skipped outlier correction for video {} ...'.format(video_name))

        print('SIMBA COMPLETE: Skipped outlier correction for {} files.'.format(str(self.file_cnt)))

# test = OutlierCorrectionSkipper(config_path='/Users/simon/Desktop/troubleshooting/Zebrafish/project_folder/project_config.ini')
# test.skip_outlier_correction()

# test = OutlierCorrectionSkipper(config_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Parquet_test2\project_folder\project_config.ini")
# test.skip_outlier_correction()

# config_ini = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Parquet_test2\project_folder\project_config.ini"
# skip_outlier_c(config_ini)