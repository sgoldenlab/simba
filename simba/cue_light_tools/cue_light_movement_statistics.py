import os, glob
from simba.read_config_unit_tests import (read_config_entry, read_config_file, check_file_exist_and_readable)
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.rw_dfs import read_df
from simba.misc_tools import get_fn_ext, check_multi_animal_status
from simba.drop_bp_cords import getBpNames, create_body_part_dictionary
from collections import defaultdict
import pandas as pd
import numpy as np
from statistics import mean
from simba.ROI_analyzer import ROIAnalyzer
from datetime import datetime
from simba.cue_light_tools.cue_light_tools import find_frames_when_cue_light_on


class CueLightMovementAnalyzer(object):
    """
    Class for computing aggregate statistics of animal movement in relation to the cue light
    ON and OFF states.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    pre_window: int
        Time period (in millisecond) before the onset of each cue light to compute aggregate classification
        statistics for.
    post_window: int
        Time period (in millisecond) after the offset of each cue light to compute aggregate classification
        statistics for.
    cue_light_names: list
        Names of cue lights, as defined in the SimBA ROI interface.
    threshold: float
        The body-part post-estimation probability threshold. SimBA omits movement calculations for frames where the
        body-part probability threshold is lower than the user-specified threshold.
    roi_setting: bool
        If True, SimBA calculates movement statistics within non-light cue ROIs.

    Notes
    ----------
    `Cue light tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`__.


    Examples
    ----------
    >>> cue_light_movement_analyzer = CueLightMovementAnalyzer(config_path='MyProjectConfig', cue_light_names=['Cue_light'], pre_window=1000, post_window=1000, threshold=0.0, roi_setting=True)
    >>> cue_light_movement_analyzer.calculate_whole_session_movement()
    >>> cue_light_movement_analyzer.organize_results()
    >>> cue_light_movement_analyzer.save_results()
    """

    def __init__(self,
                 config_path: str,
                 pre_window: int,
                 post_window: int,
                 cue_light_names: list,
                 threshold: float,
                 roi_setting: bool):

        self.config_path, self.cue_light_names = config_path, cue_light_names
        self.config, self.roi_setting = read_config_file(config_path), roi_setting
        self.p_threshold = threshold
        self.pre_window, self.post_window = pre_window, post_window
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.in_dir = os.path.join(self.project_path, 'csv', 'cue_lights')
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.logs_path = os.path.join(self.project_path, 'logs')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.files_found = glob.glob(self.in_dir + '/*.' + self.file_type)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, self.x_cols, self.y_cols, self.pcols,[])
        self.bp_dict, self.bp_columns = defaultdict(list), []
        for cnt, animal in enumerate(self.multi_animal_id_lst):
            bp_name = self.config.get('ROI settings', 'animal_' + str(cnt + 1) + '_bp')
            if bp_name == 'None':
                print('SIMBA ERROR: Please analyze ROI data and set body-parts first.')
                raise ValueError
            for c in ['_x', '_y', '_p']:
                self.bp_dict[animal].append(bp_name + c)
                self.bp_columns.append(bp_name + c)
        print('Analyzing {} files...'.format(str(len(self.files_found))))

        if roi_setting:
            self.roi_analyzer = ROIAnalyzer(ini_path=self.config_path,
                                            data_path=os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location'),
                                            calculate_distances=False)
            self.roi_analyzer.read_roi_dfs()
            self.roi_analyzer.analyze_ROIs()
            self.entries_exits_df = pd.concat(self.roi_analyzer.entry_exit_df_lst, axis=0)
            self.entries_exits_df = self.entries_exits_df[~self.entries_exits_df['Shape'].isin(self.cue_light_names)]
            self.entries_exits_df['inside_lst'] = self.entries_exits_df.apply(lambda x: list(range(int(x['Entry_times']), int(x['Exit_times'] + 1))), axis=1)

    def __euclidean_distance(self, bp_1_x_vals, bp_2_x_vals, bp_1_y_vals, bp_2_y_vals, px_per_mm):
        series = (np.sqrt((bp_1_x_vals - bp_2_x_vals) ** 2 + (bp_1_y_vals - bp_2_y_vals) ** 2)) / px_per_mm
        return series

    def calculate_whole_session_movement(self):
        """
        Method to calculate movement in relation to cue-light(s) onsets.

        Returns
        -------
        Attribute: dict
            roi_results
        """

        self.results = {}
        if self.roi_setting:
            self.roi_results = {}
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            self.results[self.video_name] = {}
            self.data_df = read_df(file_path, self.file_type).reset_index(drop=True)
            self.video_info_settings, self.px_per_mm, self.fps = read_video_info(self.vid_info_df, self.video_name)
            self.prior_window_frames_cnt = int(self.pre_window / (1000 / self.fps))
            self.post_window_frames_cnt = int(self.post_window / (1000 / self.fps))
            self.light_on_dict = find_frames_when_cue_light_on(data_df=self.data_df,
                                          cue_light_names=self.cue_light_names,
                                          fps=self.fps,
                                          prior_window_frames_cnt=self.prior_window_frames_cnt,
                                          post_window_frames_cnt=self.post_window_frames_cnt)




            for animal_name, animal_bps in self.bp_dict.items():
                self.results[self.video_name][animal_name] = {}
                animal_df = self.data_df[animal_bps]
                if self.p_threshold > 0.00:
                    animal_df = animal_df[animal_df[animal_bps[2]] >= self.p_threshold]
                animal_df = animal_df.iloc[:, 0:2].reset_index(drop=True)
                df_shifted = animal_df.shift(1)
                df_shifted = df_shifted.combine_first(animal_df).add_suffix('_shifted')
                animal_df = pd.concat([animal_df, df_shifted], axis=1)
                self.movement = self.__euclidean_distance(animal_df[animal_bps[0]], animal_df[animal_bps[0] + '_shifted'], animal_df[animal_bps[1]], animal_df[animal_bps[1] + '_shifted'], self.px_per_mm) / 10
                self.light_movement = {}
                for cue_light in self.cue_light_names:
                    self.light_movement[cue_light] = {}
                    self.light_movement[cue_light]['pre_window_movement'] = self.movement[self.movement.index.isin(self.light_on_dict[cue_light]['pre_window_frames'])]
                    self.light_movement[cue_light]['light_movement'] = self.movement[self.movement.index.isin(self.light_on_dict[cue_light]['light_on_frames'])]
                    self.light_movement[cue_light]['post_window_movement'] = self.movement[self.movement.index.isin(self.light_on_dict[cue_light]['post_window_frames'])]
                for cue_light in self.cue_light_names:
                    self.results[self.video_name][animal_name][cue_light] = {}
                    for state_name, df in zip(['pre-cue', 'cue', 'post-cue'], [self.light_movement[cue_light]['pre_window_movement'], self.light_movement[cue_light]['light_movement'], self.light_movement[cue_light]['post_window_movement']]):
                       self.results[self.video_name][animal_name][cue_light][state_name] = {}
                       self.results[self.video_name][animal_name][cue_light][state_name]['Distance (cm)'] = round((df.sum() / 10), 4)
                       velocity_lst = []
                       for sliced_df in np.array_split(df, self.fps): velocity_lst.append(sliced_df.sum())
                       self.results[self.video_name][animal_name][cue_light][state_name]['Velocity (cm/s)'] = round((mean(velocity_lst) / 10), 4)
                if self.roi_setting:
                    self.roi_results[self.video_name] = {}
                    self.roi_results[self.video_name][animal_name] = {}
                    for roi_name in self.entries_exits_df['Shape'].unique():
                        self.roi_results[self.video_name][animal_name][roi_name] = {}
                        inside_roi_frms = self.entries_exits_df['inside_lst'][self.entries_exits_df['Shape'] == roi_name]
                        inside_roi_frms = [i for s in inside_roi_frms for i in s]
                        for cue_light in self.cue_light_names:
                            self.roi_results[self.video_name][animal_name][roi_name][cue_light] = {}
                            self.overlap_light = list(set(inside_roi_frms).intersection(self.light_on_dict[cue_light]['light_on_frames']))
                            self.overlap_pre_window_frames = list(set(inside_roi_frms).intersection(self.light_on_dict[cue_light]['pre_window_frames']))
                            self.overlap_post_window_frames = list(set(inside_roi_frms).intersection(self.light_on_dict[cue_light]['post_window_frames']))
                            for state_name, lst in zip(['pre-cue', 'cue', 'post-cue'],
                                                      [self.overlap_pre_window_frames,
                                                      self.overlap_light,
                                                      self.overlap_post_window_frames]):
                                self.roi_results[self.video_name][animal_name][roi_name][cue_light][state_name] = round(((len(lst) * (1000 / self.fps)) / 1000), 4)

    def organize_results(self):
        """
        Method to organize movement results into dataframe

        Returns
        -------
        Attribute: pd.DataFrame
            results_roi_df
        """

        self.results_df = pd.DataFrame(columns=['Video', 'Animal', 'Cue light', 'Time period', 'Measure', 'Value'])
        for video_name, video_data in self.results.items():
            for animal_name, animal_data in video_data.items():
                for light_name, light_data in animal_data.items():
                    for period_name, period_data in light_data.items():
                        for measure_name, measure_data in period_data.items():
                            self.results_df.loc[len(self.results_df)] = [video_name, animal_name, light_name, period_name, measure_name, measure_data]

        if self.roi_setting:
            self.results_roi_df = pd.DataFrame(columns=['Video', 'Animal', 'ROI Name', 'Cue light', 'Time period', 'Time in ROI (s)'])
            for video_name, video_data in self.roi_results.items():
                for animal_name, animal_data in video_data.items():
                    for roi_name, roi_data in animal_data.items():
                        for light_name, light_data in roi_data.items():
                            for period_name, period_data in light_data.items():
                                self.results_roi_df.loc[len(self.results_roi_df)] = [video_name, animal_name, roi_name, light_name, period_name, period_data]

    def save_results(self):
        """
        Method to save movement cue light results into the SimBA project folder.
        Results are stored in the `project_folder/logs` directory of the SimBA project.

        Returns
        -------
        None
        """

        save_results_path = os.path.join(self.logs_path, 'Cue_lights_movement_statistics_{}.csv'.format(self.datetime))
        self.results_df = self.results_df.sort_values('Video').reset_index(drop=True)
        self.results_df.to_csv(save_results_path)
        print('SIMBA COMPLETE: Cue light movement statistics saved in project_folder/logs directory.')
        if self.roi_setting:
            save_roi_results_path = os.path.join(self.logs_path, 'Cue_lights_roi_statistics_{}.csv'.format(self.datetime))
            self.results_roi_df = self.results_roi_df.sort_values('Video').reset_index(drop=True)
            self.results_roi_df.to_csv(save_roi_results_path )
            print('SIMBA COMPLETE: Cue light ROI statistics saved in project_folder/logs directory.')