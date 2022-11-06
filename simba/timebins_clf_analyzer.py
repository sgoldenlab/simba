__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
from simba.read_config_unit_tests import read_config_entry, check_that_column_exist ,read_config_file, check_int
from simba.misc_tools import detect_bouts
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.train_model_functions import get_all_clf_names
from datetime import datetime
import os, glob
from simba.rw_dfs import read_df
from simba.drop_bp_cords import get_fn_ext
from collections import defaultdict

class TimeBinsClf(object):

    """
    Class for aggregating classification results into time-bins. Results are stored in
    the ``project_folder/logs`` directory of the SimBA project`

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    bin_length: int
        Integer representing the time bin size in seconds
    measurements: list
        Aggregate statistic measures to calculate for each time bin. OPTIONS: ['First occurance (s)', 'Event count',
        Total event duration (s)', 'Mean event duration (s)', 'Median event duration (s)', 'Mean event interval (s)',
        'Median event interval (s)']
    classifiers: list
        Names of classifiers to calculate aggregate statistics in time-bins for. EXAMPLE: ['Attack', 'Sniffing']

    Example
    ----------

    >>> timebin_clf_analyzer = TimeBinsClf(config_path='MyConfigPath', bin_length=15, measurements=['Event count', 'Total event duration (s)'])
    >>> timebin_clf_analyzer.analyze_timebins_clf()

    """

    def __init__(self,
                 config_path: str,
                 bin_length: int,
                 measurements: list,
                 classifiers: list):

        if len(measurements) == 0:
            print('SIMBA ERROR: Please select at least ONE aggregate statistic measurement.')
            raise ValueError('SIMBA ERROR: Please select at least ONE aggregate statistic measurement.')
        check_int(name='Bin length', value=bin_length)
        self.bin_length, self.measurements = int(bin_length), measurements
        self.classifiers, self.config = classifiers, read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.data_in_dir = os.path.join(self.project_path, 'csv', 'machine_results')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.target_cnt = read_config_entry(self.config, 'SML settings', 'no_targets', data_type='int')
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.target_cnt)
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        print('Processing {} video(s)...'.format(str(len(self.files_found))))
        self.out_df_lst = []

    def analyze_timebins_clf(self):
        """
        Method for running the classifier time-bin analysis. Results are stored in the
        ``project_folder/logs`` directory of the SimBA project.

        Returns
        ----------
        None
        """


        video_dict = {}
        for file_cnt, file_path in enumerate(self.files_found):
            dir_name, file_name, extension = get_fn_ext(file_path)
            data_df = read_df(file_path, self.file_type)
            video_settings, px_per_mm, fps = read_video_info(self.vid_info_df, file_name)
            fps = int(fps)
            bin_frame_length = self.bin_length * fps
            data_df_lst = [data_df[i:i + bin_frame_length] for i in range(0, data_df.shape[0], bin_frame_length)]
            video_dict[file_name] = {}
            for bin_cnt, df in enumerate(data_df_lst):
                video_dict[file_name][bin_cnt] = {}
                bouts_df = detect_bouts(data_df=df, target_lst=list(self.clf_names), fps=fps)
                bouts_df['Shifted start'] = bouts_df['Start_time'].shift(-1)
                bouts_df['Interval duration'] = bouts_df['Shifted start'] - bouts_df['End Time']
                for clf in self.clf_names:
                    video_dict[file_name][bin_cnt][clf] = defaultdict(list)
                    bout_df = bouts_df.loc[bouts_df['Event'] == clf]
                    if len(bouts_df) > 0:
                        video_dict[file_name][bin_cnt][clf]['First occurance (s)'] = round(bout_df['Start_time'].min(), 3)
                        video_dict[file_name][bin_cnt][clf]['Event count'] = len(bout_df)
                        video_dict[file_name][bin_cnt][clf]['Total event duration (s)'] = round(bout_df['Bout_time'].sum(), 3)
                        video_dict[file_name][bin_cnt][clf]['Mean event duration (s)'] = round(bout_df["Bout_time"].mean(), 3)
                        video_dict[file_name][bin_cnt][clf]['Median event duration (s)'] = round(bout_df['Bout_time'].median(), 3)
                    else:
                        video_dict[file_name][bin_cnt][clf]['First occurance (s)'] = None
                        video_dict[file_name][bin_cnt][clf]['Event count'] = 0
                        video_dict[file_name][bin_cnt][clf]['Total event duration (s)'] = 0
                        video_dict[file_name][bin_cnt][clf]['Mean event duration (s)'] = 0
                        video_dict[file_name][bin_cnt][clf]['Median event duration (s)'] = 0
                    if len(bouts_df) > 1:
                        video_dict[file_name][bin_cnt][clf]['Mean event interval (s)'] = round(bout_df[:-1]['Interval duration'].mean(), 3)
                        video_dict[file_name][bin_cnt][clf]['Median event interval (s)'] = round(bout_df[:-1]['Interval duration'].median(), 3)
                    else:
                        video_dict[file_name][bin_cnt][clf]['Mean event interval (s)'] = None
                        video_dict[file_name][bin_cnt][clf]['Median event interval (s)'] = None

        for video_name, video_info in video_dict.items():
            for bin_number, bin_data in video_info.items():
                data_df = pd.DataFrame.from_dict(bin_data).reset_index().rename(columns={'index':'Measurement'})
                data_df = pd.melt(data_df, id_vars=['Measurement']).rename(columns={'value':'Value', 'variable': 'Classifier'})
                data_df.insert(loc=0, column='Time bin #', value=bin_number)
                data_df.insert(loc=0, column='Video', value=video_name)
                self.out_df_lst.append(data_df)
        out_df = pd.concat(self.out_df_lst, axis=0).sort_values(by=['Video', 'Time bin #']).set_index('Video')
        out_df = out_df[out_df['Measurement'].isin(self.measurements)]
        out_df = out_df[out_df['Classifier'].isin(self.classifiers)]
        save_path = os.path.join(self.project_path, 'logs', 'Time_bins_ML_results_' + self.datetime + '.csv')
        out_df.to_csv(save_path)
        print('SIMBA COMPLETE: Classification time-bins results saved at project_folder/logs/output/{}'.format(str('Time_bins_ML_results_' + self.datetime + '.csv')))

# test = TimeBinsClf(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',
#                    bin_length=2,
#                    measurements=['First occurance (s)', 'Event count', 'Total event duration (s)', 'Mean event duration (s)'])
# test.analyze_timebins_clf()


# test = TimeBinsClf(config_path='/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/project_config.ini',
#                    bin_length=2,
#                    measurements=['First occurance (s)', 'Event count', 'Total event duration (s)', 'Mean event duration (s)'])
# test.analyze_timebins_clf()
