__author__ = "Simon Nilsson", "JJ Choong"

from datetime import datetime
from simba.read_config_unit_tests import read_config_entry, check_that_column_exist, read_config_file
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.misc_tools import check_multi_animal_status, line_length_numba
from simba.drop_bp_cords import create_body_part_dictionary, getBpNames, get_fn_ext, checkDirectionalityCords
import pandas as pd
from simba.rw_dfs import read_df
import itertools
import numpy as np
import os, glob

class DirectingOtherAnimalsAnalyzer(object):
    """
    Class for calculating when animals are directing towards body-parts of other animals. Results are stored in
    the `project_folder/logs/directionality_dataframes` directory of the SimBA project

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    -----
    Requires the pose-estimation data for the ``left ear``, ``right ear`` and ``nose`` of
    each individual animals.

    Examples
    -----
    >>> directing_analyzer = DirectingOtherAnimalsAnalyzer(config_path='MyProjectConfig')
    >>> directing_analyzer.process_directionality()
    >>> directing_analyzer.create_directionality_dfs()
    >>> directing_analyzer.save_directionality_dfs()
    >>> directing_analyzer.summary_statistics()
    """



    def __init__(self,
                 config_path: str):

        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.config = read_config_file(config_path)
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.animal_cnt = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        if self.animal_cnt < 2:
            print('SIMBA ERROR: Cannot analyze directionality between animals in a 1 animal project.')
            raise ValueError('SIMBA ERROR: Cannot analyze directionality between animals in a 1 animal project.')
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.logs_dir = os.path.join(self.project_path, 'logs')
        self.data_in_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.data_out_dir = os.path.join(self.project_path , 'logs', 'directionality_dataframes')
        if not os.path.exists(self.data_out_dir): os.makedirs(self.data_out_dir)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.animal_cnt)
        self.x_cols, self.y_cols, self.p_cols = getBpNames(config_path)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.animal_cnt, self.x_cols, self.y_cols, [], [])
        self.animal_permutations = list(itertools.permutations(self.animal_bp_dict, 2))
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        print('Processing {} video(s)...'.format(str(len(self.files_found))))

    def process_directionality(self):
        """
        Method to compute when animals are directing towards body-parts belonging to other animals.

        Returns
        -------
        Attribute: dict
            results_dict
        """

        self.results_dict = {}
        for file_cnt, file_path in enumerate(self.files_found):
            _, video_name, _ = get_fn_ext(file_path)
            self.results_dict[video_name] = {}
            data_df = read_df(file_path, self.file_type)
            direct_bp_dict = checkDirectionalityCords(self.animal_bp_dict)
            for animal_permutation in self.animal_permutations:
                self.results_dict[video_name]['{} {} {}'.format(animal_permutation[0], 'directing towards',animal_permutation[1])] = {}
                first_animal_bps, second_animal_bps = direct_bp_dict[animal_permutation[0]], self.animal_bp_dict[animal_permutation[1]]
                first_ear_left_arr = data_df[[first_animal_bps['Ear_left']['X_bps'], first_animal_bps['Ear_left']['Y_bps']]].to_numpy()
                first_ear_right_arr = data_df[[first_animal_bps['Ear_right']['X_bps'], first_animal_bps['Ear_right']['Y_bps']]].to_numpy()
                first_nose_arr = data_df[[first_animal_bps['Nose']['X_bps'], first_animal_bps['Nose']['Y_bps']]].to_numpy()
                other_animal_x_bps, other_animal_y_bps = second_animal_bps['X_bps'], second_animal_bps['Y_bps']
                for x_bp, y_bp in zip(other_animal_x_bps, other_animal_y_bps):
                    target_cord_arr = data_df[[x_bp, y_bp]].to_numpy()
                    direction_data = line_length_numba(left_ear_array=first_ear_left_arr, right_ear_array=first_ear_right_arr, nose_array=first_nose_arr, target_array=target_cord_arr)
                    x_min = np.minimum(direction_data[:, 1], first_nose_arr[:, 0])
                    y_min = np.minimum(direction_data[:, 2], first_nose_arr[:, 1])
                    delta_x = abs((direction_data[:, 1] - first_nose_arr[:, 0]) / 2)
                    delta_y = abs((direction_data[:, 2] - first_nose_arr[:, 1]) / 2)
                    x_middle, y_middle = np.add(x_min, delta_x), np.add(y_min, delta_y)
                    direction_data = np.concatenate((y_middle.reshape(-1, 1), direction_data), axis=1)
                    direction_data = np.concatenate((x_middle.reshape(-1, 1), direction_data), axis=1)
                    direction_data = np.delete(direction_data, [2, 3, 4], 1)
                    direction_data = np.hstack((direction_data,target_cord_arr))
                    bp_data = pd.DataFrame(direction_data, columns=['Eye_x', 'Eye_y', 'Directing_BOOL', x_bp, y_bp])
                    bp_data = bp_data[['Eye_x', 'Eye_y', x_bp, y_bp, 'Directing_BOOL']]
                    bp_data.insert(loc=0, column='Animal_2_body_part', value=x_bp[:-2])
                    bp_data.insert(loc=0, column='Animal_2', value=animal_permutation[1])
                    bp_data.insert(loc=0, column='Animal_1', value=animal_permutation[0])
                    self.results_dict[video_name]['{} {} {}'.format(animal_permutation[0], 'directing towards', animal_permutation[1])][x_bp[:-2]] = bp_data
            print('Direction analysis complete for video {} ({}/{})...'.format(video_name, str(file_cnt + 1), str(len(self.files_found))))

    def create_directionality_dfs(self):
        """
        Method to transpose results created by :meth:`~simba.DirectingOtherAnimalsAnalyzer.process_directionality`.
        into dict of dataframes

        Returns
        -------
        Attribute: dict
            directionality_df_dict
        """



        print('Transposing directionality data...')
        self.directionality_df_dict = {}
        for video_name, video_data in self.results_dict.items():
            out_df_lst = []
            for animal_permutation, permutation_data in video_data.items():
                for bp_name, bp_data in permutation_data.items():
                    directing_df = bp_data[bp_data['Directing_BOOL'] == 1].reset_index().rename(columns={'index': 'Frame_#', bp_name + '_x':'Animal_2_bodypart_x', bp_name + '_y':'Animal_2_bodypart_y'})
                    directing_df.insert(loc=0, column='Video', value=video_name)
                    out_df_lst.append(directing_df)
            self.directionality_df_dict[video_name] = pd.concat(out_df_lst, axis=0).drop('Directing_BOOL', axis=1)

    def save_directionality_dfs(self):

        """
        Method to save result created by :meth:`~simba.DirectingOtherAnimalsAnalyzer.create_directionality_dfs`.
        into CSV files on disk. Results are stored in `project_folder/logs` directory of the SimBA project.

        Returns
        -------
        None
        """


        for video_name, video_data in self.directionality_df_dict.items():
            save_name = os.path.join(self.data_out_dir, video_name + '.csv')
            video_data.to_csv(save_name)
            print('Detailed directional data saved for video {} ...'.format(str(video_name)))
        print('All detailed directional data saved in the {} directory'.format(str(self.data_out_dir)))

    def summary_statistics(self):

        """
        Method to save aggregate statistics of data created by :meth:`~simba.DirectingOtherAnimalsAnalyzer.create_directionality_dfs`.
        into CSV files on disk. Results are stored in `project_folder/logs` directory of the SimBA project.

        Returns
        -------
        None
        """

        print('Computing summary statistics...')
        out_df_lst = []
        for video_name, video_data in self.results_dict.items():
            _, _, fps = read_video_info(self.vid_info_df, video_name)
            for animal_permutation, permutation_data in video_data.items():
                idx_directing = set()
                for bp_name, bp_data in permutation_data.items():
                    idx_directing.update(list(bp_data.index[bp_data['Directing_BOOL'] == 1]))
                value = round(len(idx_directing) / fps, 3)
                out_df_lst.append(pd.DataFrame([[video_name, animal_permutation, value]], columns=['Video', 'Animal permutation', 'Value (s)']))
        self.summary_df = pd.concat(out_df_lst, axis=0).sort_values(by=['Video', 'Animal permutation']).set_index('Video')
        self.summary_df.to_csv(os.path.join(self.logs_dir, 'Direction_data_{}{}'.format(str(self.datetime), '.csv')))
        print('Summary directional statistics saved at ' + os.path.join(self.logs_dir, 'Direction_data_{}{}'.format(str(self.datetime), '.csv')))
        print('SIMBA COMPLETE: ALL DIRECTIONAL DATA ANALYZED AND SAVED IN PROJECT.')

# test = DirectingOtherAnimalsAnalyzer(config_path='/Users/simon/Desktop/train_model_project/project_folder/project_config.ini')
# test.process_directionality()
# test.create_directionality_dfs()
# # test.save_directionality_dfs()
# # test.summary_statistics()


