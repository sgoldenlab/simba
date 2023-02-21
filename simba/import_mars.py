from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry,
                                          read_project_path_and_file_type)
import os, glob
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from simba.rw_dfs import save_df
from simba.misc_tools import (get_fn_ext,
                              smooth_data_gaussian,
                              smooth_data_savitzky_golay)
from simba.enums import Paths, Methods, Dtypes
from simba.interpolate_pose import Interpolate
import pyarrow.parquet as pq
import pyarrow as pa


class MarsImporter(object):
    """
    Class for importing two animal BENTO pose-estimation data (in JSON format) into a SimBA project in
    parquet or CSV format.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    data_folder: str
        Path to file or folder with data in `.json` format.
    interpolation_settings: str
        String defining the pose-estimation interpolation method. OPTIONS: 'None', 'Animal(s): Nearest',
        'Animal(s): Linear', 'Animal(s): Quadratic','Body-parts: Nearest', 'Body-parts: Linear',
        'Body-parts: Quadratic'.
    smoothing_settings: dict
        Dictionary defining the pose estimation smoothing method. EXAMPLE: {'Method': 'Savitzky Golay',
        'Parameters': {'Time_window': '200'}})

    Notes
    -----
    `Multi-animal import tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Multi_animal_pose.md>`__.

    Examples
    -----
    >>> mars_importer = MarsImporter(config_path=r'MyConfigPath', data_folder=r'MyMarsDataFolder', interpolation_settings='None', smoothing_settings={'Method': 'None', 'Parameters': {'Time_window': '200'}})
    >>> mars_importer.import_data()

    References
    ----------
    .. [1] Segalin et al., The Mouse Action Recognition System (MARS) software pipeline for automated analysis of social behaviors in mice, `eLife`, 2021.

    """


    def __init__(self,
                 config_path: str,
                 data_path: str,
                 interpolation_method:  str,
                 smoothing_method: dict):

        self.config, self._config_path = read_config_file(ini_path=config_path), config_path
        self.data_path = data_path
        self.interpolation_method, self.smoothing_method = interpolation_method, smoothing_method
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.save_dir = os.path.join(self.project_path, Paths.INPUT_CSV.value)
        if os.path.isdir(data_path):
            self.files_found = glob.glob(data_path + '/*.json')
        else:
            self.files_found = [data_path]
        if len(self.files_found) == 0:
            print('SIMBA ERROR: Zero .json files found in {} directory'.format(data_path))
            raise ValueError()
        body_part_names = ['Nose', 'Ear_left', 'Ear_right', 'Neck', 'Hip_left', 'Hip_right', 'Tail']
        self.keypoint_headers, self.scores_headers = [], []
        for animal in ['1', '2']:
            for body_part in body_part_names:
                for coordinate in ['x', 'y']:
                    self.keypoint_headers.append(body_part + '_' + animal + '_' + coordinate)
        for animal in ['1', '2']:
            for body_part in body_part_names:
                self.scores_headers.append(body_part + '_' + animal + '_p')
        self.headers = deepcopy(self.keypoint_headers)
        index = 3 - 1
        for elem in self.scores_headers:
            self.headers.insert(index, elem)
            index += 3

    def __merge_dfs(self, df_1: pd.DataFrame, df_2: pd.DataFrame):
        df = pd.DataFrame()
        for cnt, c in enumerate(df_1.columns):
            df[len(df.columns)] = df_1[c]
            df[len(df.columns)] = df_2[c]
        return df

    def __create_multi_index_headers(self, df: pd.DataFrame):
        multi_index_tuples = []
        for column in range(len(df.columns)):
            multi_index_tuples.append(tuple(('MARS', 'MARS', df.columns[column])))
        df.columns = pd.MultiIndex.from_tuples(multi_index_tuples, names=['scorer', 'bodypart', 'coords'])
        return df

    def __perform_interpolation(self,
                                file_path: str,
                                workflow_file_type: str,
                                config_path: str,
                                interpolation_method: str):
        if workflow_file_type == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, index_col=0)
        interpolate_body_parts = Interpolate(config_path, df)
        interpolate_body_parts.detect_headers()
        interpolate_body_parts.fix_missing_values(interpolation_method)
        interpolate_body_parts.reorganize_headers()
        if workflow_file_type == 'parquet':
            table = pa.Table.from_pandas(interpolate_body_parts.new_df)
            pq.write_table(table, file_path)
        if workflow_file_type == 'csv':
            interpolate_body_parts.new_df.to_csv(file_path)

    def __run_smoothing(self):
        if self.smoothing_method['Method'] == Methods.GAUSSIAN.value:
            print('Performing Gaussian smoothing on video {}...'.format(self.file_name))
            time_window = self.smoothing_method['Parameters']['Time_window']
            smooth_data_gaussian(config=self.config, file_path=self.save_path, time_window_parameter=time_window)

        if self.smoothing_method['Method'] == Methods.SAVITZKY_GOLAY.value:
            print('Performing Savitzky Golay smoothing on video {}...'.format(self.file_name))
            time_window = self.smoothing_method['Parameters']['Time_window']
            smooth_data_savitzky_golay(config=self.config, file_path=self.save_path, time_window_parameter=time_window)

    def import_data(self):
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.file_name, _ = get_fn_ext(file_path)
            print('Importing data for video {}...'.format(self.file_name))
            self.save_path = os.path.join(self.save_dir, self.file_name + '.' + self.file_type)
            with open(file_path, 'r') as j:
                data = json.loads(j.read())
            key_points, scores = np.array(data['keypoints']).astype(int), np.array(data['scores'])
            animal_1_scores, animal_2_scores = pd.DataFrame(scores[:, 0]), pd.DataFrame(scores[:, 1])
            data_df = []
            for a in [key_points[:, 0], key_points[:, 1]]:
                m, n, r = a.shape
                arr = np.column_stack((np.repeat(np.arange(m),n),a.reshape(m*n,-1)))
                df = pd.DataFrame(arr)
                df_x, df_y = df[df.index % 2 != 0].set_index(0), df[df.index % 2 != 1].set_index(0)
                data_df.append(self.__merge_dfs(df_x, df_y))
            data_df = pd.concat(data_df, axis=1)
            data_df.columns = self.keypoint_headers
            scores_df = pd.concat([animal_1_scores, animal_2_scores], axis=1)
            scores_df.columns = self.scores_headers
            data_df = pd.concat([data_df, scores_df], axis=1)[self.headers]
            data_df = self.__create_multi_index_headers(df=data_df)
            save_df(data_df, self.file_type, self.save_path)

            if self.interpolation_method != Dtypes.NONE.value:
                print('Performing interpolation...')
                self.__perform_interpolation(self.save_path, self.file_type, self._config_path, self.interpolation_method)
            if self.smoothing_method['Method'] != Dtypes.NONE.value:
                self.__run_smoothing()
            print('Video imported {}.'.format(self.file_name))
        print('SIMBA COMPLETE: {} data files imported to SimBA project.'.format(str(len(self.files_found))))

