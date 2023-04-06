__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
from simba.read_config_unit_tests import (check_file_exist_and_readable,
                                          check_if_filepath_list_is_empty)
from simba.feature_extractors.unit_tests import read_video_info
import os
from simba.mixins.config_reader import ConfigReader
from simba.drop_bp_cords import get_fn_ext
from simba.misc_tools import detect_bouts
from simba.rw_dfs import read_df

class ClfLogCreator(ConfigReader):
    """
    Class for creating aggregate statistics from classification data.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    data_measures: list
        Aggregate statistics measures to calculate. OPTIONS: ['Bout count', 'Total event duration (s)',
        'Mean event bout duration (s)', 'Median event bout duration (s)', 'First event occurrence (s)',
        'Mean event bout interval duration (s)', 'Median event bout interval duration (s)']
    classifiers: list
        Classifiers to calculate aggregate statistics for. E.g.,: ['Attack', 'Sniffing']

    Notes
    ----------
    `GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results`__.

    Examples
    -----
    >>> clf_log_creator = ClfLogCreator(config_path="MyConfigPath", data_measures=['Bout count', 'Total event duration'], classifiers=['Attack', 'Sniffing'])
    >>> clf_log_creator.run()
    >>> clf_log_creator.save()
    """

    def __init__(self,
                 config_path: str,
                 data_measures: list,
                 classifiers: list):

        super().__init__(config_path=config_path)
        self.chosen_measures, self.classifiers = data_measures, classifiers
        self.file_save_name = os.path.join(self.project_path, 'logs', 'data_summary_' + str(self.datetime) + '.csv')
        check_if_filepath_list_is_empty(filepaths=self.machine_results_paths,
                                        error_msg='SIMBA ERROR: No data files found in the project_folder/csv/machine_results directory. Run classifiers before analysing results.')
        print(f'Analyzing {str(len(self.machine_results_paths))} file(s) for {str(len(self.clf_names))} classifiers...')

    def run(self):

        """
        Method to create dataframe of classifier aggregate statistics

        Returns
        -------
        Attribute: pd.Dataframe
            results_df
        """

        self.results_df = pd.DataFrame()
        for file_cnt, file_path in enumerate(self.machine_results_paths):
            _, file_name, _ = get_fn_ext(file_path)
            print('Analyzing video {}...'.format(file_name))
            _, _, fps = read_video_info(self.video_info_df, file_name)
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

    def save(self):
        """
        Method to save classifier aggregate statistics created in :meth:`~simba.ClfLogCreator.analyze_data` to disk.
        Results are stored in the `project_folder/logs` directory of the SimBA project

        Returns
        -------
        None
        """

        self.results_df = self.results_df[self.results_df['Measure'].isin(self.chosen_measures)].sort_values(by=['Video', 'Classifier', 'Measure']).reset_index(drop=True)
        self.results_df = self.results_df[self.results_df['Classifier'].isin(self.classifiers)].set_index('Video')
        self.results_df.to_csv(self.file_save_name)
        self.timer.stop_timer()
        print(f'SIMBA COMPLETE: Data log saved at {self.file_save_name} (elapsed time: {self.timer.elapsed_time_str}s)')


# test = ClfLogCreator(config_path=r"/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini",
#                      data_measures=['Bout count',
#                                     'Total event duration (s)'],
#                      classifiers=['Attack', 'Sniffing'])
# test.run()
# test.save()











