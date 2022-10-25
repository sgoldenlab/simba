import pandas as pd
from simba.read_config_unit_tests import (read_config_entry, read_config_file, check_file_exist_and_readable)
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
import os, glob
from simba.drop_bp_cords import getBpNames, create_body_part_dictionary
from simba.misc_tools import check_multi_animal_status, get_fn_ext
from simba.rw_dfs import read_df
from simba.read_config_unit_tests import check_that_column_exist
from datetime import datetime
from simba.cue_light_tools.cue_light_tools import find_frames_when_cue_light_on

class CueLightClfAnalyzer(object):
    """
    Class for computing aggregate statistics when classified behaviors are occurring in relation to the cue light
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
    clf_list: list
        Names of the classifiers we want to compute aggregate statistics for.

    Notes
    ----------
    `Cue light tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`__.

    Examples
    ----------
    >>> cue_light_clf_analyzer = CueLightClfAnalyzer(config_path='MyProjectConfig', pre_window=1000, post_window=1000, cue_light_names=['Cue_light'], clf_list=['Attack'])
    >>> cue_light_clf_analyzer.analyze_clf()
    >>> cue_light_clf_analyzer.organize_results()
    >>> cue_light_clf_analyzer.save_data()
    """

    def __init__(self,
                 config_path: str,
                 pre_window: int,
                 post_window: int,
                 cue_light_names: list,
                 clf_list: list):



        self.pre_window, self.post_window = pre_window, post_window
        self.config_path, self.cue_light_names = config_path, cue_light_names
        self.config = read_config_file(config_path)
        self.clf_list = clf_list
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.cue_light_data_dir = os.path.join(self.project_path, 'csv', 'cue_lights')
        self.machine_results_dir = os.path.join(self.project_path, 'csv', 'machine_results')
        self.logs_path = os.path.join(self.project_path, 'logs')
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.files_found_cue_light = glob.glob(self.cue_light_data_dir + '/*.' + self.file_type)
        if len(self.files_found_cue_light) == 0:
            print('SIMBA ERROR: No cue light data found. Please analyze cue light data before analyzing classifications based on cue light data')
            raise ValueError('SIMBA ERROR: No cue light data found. Please analyze cue light data before analyzing classifications based on cue light data')
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, self.x_cols, self.y_cols, self.pcols,[])

    def analyze_clf(self):
        """
        Method to calculate classifier data during cue lights periods

        Returns
        -------
        Attribute: dict
            results
        """

        self.results = {}
        self.clf_frms = {}
        for file_cnt, file_path in enumerate(self.files_found_cue_light):
            _, self.video_name, _ = get_fn_ext(file_path)
            machine_results_path = os.path.join(self.machine_results_dir, self.video_name + '.' + self.file_type)
            if not os.path.isfile(machine_results_path):
                print('SIMBA ERROR: No machine classifications exist for {}.'
                      ' Skipping cue light classifier analysis for video {}'.format(self.video_name, self.video_name))
                continue
            else:
                self.results[self.video_name] = {}
                cue_light_df = read_df(file_path, self.file_type)
                self.video_info_settings, self.px_per_mm, self.fps = read_video_info(self.vid_info_df, self.video_name)
                self.prior_window_frames_cnt = int(self.pre_window / (1000 / self.fps))
                self.post_window_frames_cnt = int(self.post_window / (1000 / self.fps))
                machine_results_df = read_df(machine_results_path, self.file_type)
                data_df = pd.concat([machine_results_df, cue_light_df[self.cue_light_names]], axis=1)
                del cue_light_df, machine_results_df
                cue_light_frames_dict = find_frames_when_cue_light_on(data_df=data_df,
                                              cue_light_names=self.cue_light_names,
                                              fps=self.fps,
                                              prior_window_frames_cnt=self.prior_window_frames_cnt,
                                              post_window_frames_cnt=self.post_window_frames_cnt)
                for clf in self.clf_list:
                    check_that_column_exist(df=data_df, column_name=clf, file_name=file_path)
                    self.results[self.video_name][clf] = {}
                    self.clf_frms[clf] = list(data_df.index[data_df[clf] == 1])
                    for cue_light_name, cue_light_data in cue_light_frames_dict.items():
                        self.results[self.video_name][clf][cue_light_name] = {}
                        for period_name, period in zip(['pre-cue', 'cue', 'post-cue'], ['pre_window_frames', 'light_on_frames', 'post_window_frames']):
                            clf_in_period_frms = list(set(self.clf_frms[clf]).intersection(cue_light_data[period]))
                            self.results[self.video_name][clf][cue_light_name][period_name] = round(((len(clf_in_period_frms) * (1000 / self.fps)) / 1000), 4)


    def organize_results(self):
        """
        Method to organize classifier data into a summary dataframe.

        Returns
        -------
        Attribute: pd.DataFrame
            results_df
        """

        self.results_df = pd.DataFrame(columns=['Video', 'Classifier', 'Cue light', 'Time period', 'Time (s)'])
        for video_name, video_data in self.results.items():
            for clf_name, clf_data in video_data.items():
                for light_name, light_data in clf_data.items():
                    for period_name, period_data in light_data.items():
                        self.results_df.loc[len(self.results_df)] = [video_name, clf_name, light_name, period_name, period_data]
            clf_outside_windows = (((len(self.clf_frms[clf_name]) * (1000 / self.fps)) / 1000) - self.results_df['Time (s)'][self.results_df['Video'] == video_name].sum())
            if clf_outside_windows < 0.00: clf_outside_windows = 0.00
            self.results_df.loc[len(self.results_df)] = [video_name, clf_name, None, 'Outside light cue time periods', clf_outside_windows]

    def save_data(self):
        """
        Method to save cue light classification data to disk. Results are stored in the `project_folder/logs`
        directory of the SimBA project.

        Returns
        -------
        None
        """

        save_results_path = os.path.join(self.logs_path, 'Cue_lights_clf_statistics_{}.csv'.format(self.datetime))
        self.results_df.to_csv(save_results_path)
        print('SIMBA COMPLETE: Cue light classifier statistics saved in project_folder/logs directory.')


# test = CueLightClfAnalyzer(config_path='/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/project_config.ini',
#                            pre_window=1000,
#                            post_window=1000,
#                            cue_light_names=['Cue_light'],
#                            clf_list=['Attack'])
# test.analyze_clf()
# test.organize_results()
# test.save_data()