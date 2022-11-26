__author__ = "Simon Nilsson", "JJ Choong"

from datetime import datetime
import pandas as pd
from simba.read_config_unit_tests import read_config_entry, check_file_exist_and_readable, read_config_file
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
import os, glob
from simba.train_model_functions import get_all_clf_names
from simba.drop_bp_cords import get_fn_ext
from simba.misc_tools import detect_bouts
from simba.rw_dfs import read_df


class ClfLogCreator(object):
    """
    Class for creating aggregate statistics from classification data.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    data_measures: list
        Aggregate statistics measures to calculate. OPTIONS: ['Bout count', 'Total event duration',
        'Mean event bout duration (s)', 'Median event bout duration (s)', 'First event occurrence',
        'Mean event bout interval duration (s)', 'Median event bout interval duration (s)']
    classifiers: list
        Classifiers to calculate aggregate statistics for. E.g.,: ['Attack', 'Sniffing']

    Notes
    -----

    Examples
    -----
    >>> clf_log_creator = ClfLogCreator(config_path="MyConfigPath", data_measures=['Bout count', 'Total event duration'], classifiers=['Attack', 'Sniffing'])
    >>> clf_log_creator.analyze_data()
    >>> clf_log_creator.save_results()
    """

    def __init__(self,
                 config_path: str,
                 data_measures: list,
                 classifiers: list):

        self.chosen_measures = data_measures
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.config, self.classifiers = read_config_file(config_path), classifiers
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.files_in_dir = os.path.join(self.project_path, 'csv', 'machine_results')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.files_found = glob.glob(self.files_in_dir + '/*.' + self.file_type)
        self.model_cnt = read_config_entry(self.config, 'SML settings', 'No_targets', data_type='int')
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.model_cnt)
        self.file_save_name = os.path.join(self.project_path, 'logs', 'data_summary_' + str(self.datetime) + '.csv')
        self.files_found = glob.glob(self.files_in_dir + '/*.' + self.file_type)
        print('Analyzing {} files for {} classifiers...'.format(str(len(self.files_found)), str(len(self.clf_names))))

    def analyze_data(self):

        """
        Method to create dataframe of classifier aggregate statistics

        Returns
        -------
        Attribute: pd.Dataframe
            results_df
        """

        self.results_df = pd.DataFrame()
        for file_cnt, file_path in enumerate(self.files_found):
            _, file_name, _ = get_fn_ext(file_path)
            print('Analyzing video {}...'.format(file_name))
            _, _, fps = read_video_info(self.vid_info_df, file_name)
            check_file_exist_and_readable(file_path)
            data_df = read_df(file_path, self.file_type)
            bouts_df = detect_bouts(data_df=data_df, target_lst=self.clf_names, fps=fps)
            bouts_df['Shifted start'] = bouts_df['Start_time'].shift(-1)
            bouts_df['Interval duration'] = bouts_df['Shifted start'] - bouts_df['End Time']
            for clf in self.clf_names:
                clf_results_dict = {}
                clf_data = bouts_df.loc[bouts_df['Event'] == clf]
                if len(clf_data) > 0:
                    clf_results_dict['First event occurrence (s)'] = round(clf_data['Start_time'].min(), 3)
                    clf_results_dict['Bout count'] = len(clf_data)
                    clf_results_dict['Total event duration (s)'] = round(clf_data['Bout_time'].sum(), 3)
                    clf_results_dict['Mean event bout duration (s)'] = round(clf_data['Bout_time'].mean(), 3)
                    clf_results_dict['Median event bout duration (s)'] = round(clf_data['Bout_time'].median(), 3)
                else:
                    clf_results_dict['First event occurrence (s)'] = None
                    clf_results_dict['Bout count'] = None
                    clf_results_dict['Total event duration (s)'] = None
                    clf_results_dict['Mean event bout duration (s)'] = None
                    clf_results_dict['Median event bout duration (s)'] = None
                if len(clf_data) > 1:
                    interval_df = clf_data[:-1].copy()
                    clf_results_dict['Mean event bout interval duration (s)'] = round(interval_df['Interval duration'].mean(), 3)
                    clf_results_dict['Median event bout interval duration (s)'] = round(interval_df['Interval duration'].median(), 3)
                else:
                    clf_results_dict['Mean event bout interval duration (s)'] = None
                    clf_results_dict['Median event bout interval duration (s)'] = None
                video_clf_pd = pd.DataFrame.from_dict(clf_results_dict, orient='index').reset_index().rename(columns={'index': 'Measure', 0: 'Value'})
                video_clf_pd.insert(loc=0, column='Classifier', value=clf)
                video_clf_pd.insert(loc=0, column='Video', value=file_name)
                self.results_df = pd.concat([self.results_df, video_clf_pd], axis=0)

    def save_results(self):
        """
        Method to save classifier aggregate statistics created in :meth:`~simba.ClfLogCreator.analyze_data` to disk.
        Results are stored in the `project_folder/logs` directory of the SimBA project

        Returns
        -------
        None
        """

        results_df = self.results_df[self.results_df['Measure'].isin(self.chosen_measures)].sort_values(by=['Video', 'Classifier', 'Measure']).reset_index(drop=True)
        results_df = results_df[results_df['Classifier'].isin(self.classifiers)].set_index('Video')
        results_df.to_csv(self.file_save_name)
        print('SIMBA COMPLETE: Data log saved at {}'.format(self.file_save_name))


# test = ClfLogCreator(config_path=r"/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini", data_measures=['Bout count', 'Total event duration','Mean event bout duration (s)', 'Median event bout duration (s)', 'First event occurrence', 'Mean event bout interval duration (s)',
#                                                                                                                                                'Median event bout interval duration (s)'])
# test.analyze_data()
# # test.save_results()











