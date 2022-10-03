__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import (read_config_file,
                                             check_that_column_exist,
                                             check_int,
                                             check_str,
                                             check_float,
                                             read_config_entry)

import os, glob
from datetime import datetime
from simba.rw_dfs import read_df, save_df
from simba.drop_bp_cords import get_fn_ext
import pandas as pd
from copy import deepcopy
from simba.pybursts import kleinberg_burst_detection
import shutil

class KleinbergCalculator(object):
    '''
    Class for smoothing classification results using the Kleinberg burst
    detection algorithm.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    classifier_names: list
        List of classifier names to smooth.
    sigma: float
        Burst detection sigma value. Higher sigma values and fewer, longer, behavioural bursts will be recognised.
    gamma: float
        Burst detection gamma value. Higher gamma values and fewer behavioural bursts will be recognised
    hierarchy: flot
        Burst detection hierarchy level. Higher hierarchy values and fewer behavioural bursts will to be recognised.

    Notes
    ----------
    `GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/kleinberg_filter.md>`__.

    Examples
    ----------
    >>> kleinberg_calculator = KleinbergCalculator(config_path='MySimBAConfigPath', classifier_names=['Attack'], sigma=2, gamma=0.3, hierarchy=2)
    >>> kleinberg_calculator.perform_kleinberg()

    References
    ----------

    .. [1] Kleinberg, Bursty and Hierarchical Structure in Streams, `Data Mining and Knowledge Discovery`,
           vol. 7, pp. 373–397, 2003.
    .. [2] Lee et al., Temporal microstructure of dyadic social behavior during relationship formation in mice, `PLOS One`,
           2019.
    .. [3] Bordes et al., Automatically annotated motion tracking identifies a distinct social behavioral profile
           following chronic social defeat stress, `bioRxiv`, 2022.
    '''

    def __init__(self,
                 config_path: str,
                 classifier_names: list,
                 sigma=2,
                 gamma=0.3,
                 hierarchy=1
                 ):

        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.in_path = os.path.join(self.project_path, 'csv', 'machine_results')
        self.logs_folder = os.path.join(self.project_path, 'logs')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.in_path = os.path.join(self.project_path, 'csv', 'machine_results')
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        sigma, gamma, hierarchy = int(sigma), float(gamma), int(hierarchy)
        check_int(value=sigma, name='sigma', min_value=1)
        check_float(value=gamma, name='gamma', min_value=0)
        check_int(value=hierarchy, name='hierarchy')
        self.sigma, self.gamma, self.hierarchy = float(sigma), float(gamma), float(hierarchy)
        self.files_found = glob.glob(self.in_path + '/*.' + self.file_type)
        if len(self.files_found) == 0:
            print('SIMBA ERROR: No data files found in {}'.format(self.in_path))
            raise ValueError('SIMBA ERROR: No data files found in {}'.format(self.in_path))
        self.clfs = classifier_names
        original_data_files_folder = os.path.join(self.project_path, 'csv', 'machine_results', 'Pre_Kleinberg_{}'.format( self.datetime))
        if not os.path.exists(original_data_files_folder): os.makedirs(original_data_files_folder)
        for file_path in self.files_found:
            _, file_name, ext = get_fn_ext(file_path)
            shutil.copyfile(file_path, os.path.join(original_data_files_folder, file_name + ext))
        print('Processing Kleinberg burst detection for {} file(s)...'.format(str(len(self.files_found))))


    def perform_kleinberg(self):
        '''
        Method to perform Kleinberg smoothing. Results are stored in the `project_folder/csv/targets_inserted` directory.
        Detailed log is saved in the `project_folder/logs/` directory.

        Returns
        ----------
        None

        '''

        detailed_df_lst = []
        for file_cnt, file_path in enumerate(self.files_found):
            _, video_name, _ = get_fn_ext(file_path)
            print('Kleinberg analysis video {}. Video {}/{}...'.format(video_name, str(file_cnt+1), str(len(self.files_found))))
            data_df = read_df(file_path, self.file_type).reset_index(drop=True)
            video_out_df = deepcopy(data_df)
            for clf in self.clfs:
                check_that_column_exist(df=data_df, column_name=clf, file_name=video_name)
                clf_offsets = data_df.index[data_df[clf] == 1].values
                if len(clf_offsets) > 0:
                    video_out_df[clf] = 0
                    kleinberg_bouts = pd.DataFrame(kleinberg_burst_detection(offsets=clf_offsets, s=self.sigma, gamma=self.gamma), columns = ['Hierarchy', 'Start', 'Stop'])
                    kleinberg_bouts['Stop'] += 1
                    kleinberg_bouts.insert(loc=0, column='Classifier', value=clf)
                    kleinberg_bouts.insert(loc=0, column='Video', value=video_name)
                    detailed_df_lst.append(kleinberg_bouts)
                    clf_bouts_in_hierarchy = kleinberg_bouts[kleinberg_bouts['Hierarchy'] == self.hierarchy]
                    hierarchy_idx = list(clf_bouts_in_hierarchy.apply(lambda x: list(range(x['Start'], x['Stop'] + 1)), 1))
                    hierarchy_idx = [x for xs in hierarchy_idx for x in xs]
                    hierarchy_idx = [x for x in hierarchy_idx if x in list(data_df.index)]
                    video_out_df.loc[hierarchy_idx, clf] = 1
            save_df(video_out_df, self.file_type, file_path)

        if len(detailed_df_lst) > 0:
            detailed_df = pd.concat(detailed_df_lst, axis=0)
            detailed_save_path = os.path.join(self.logs_folder, 'Kleinberg_detailed_log_{}.csv'.format(str(self.datetime)))
            detailed_df.to_csv(detailed_save_path)
            print('SIMBA COMPLETE: Kleinberg analysis complete. See {} for details of detected bouts of all classifiers in all hierarchies'.format(detailed_save_path))
        else:
            print('SIMBA WARNING: All behavior bouts removed following kleinberg smoothing.')

#
# test = KleinbergCalculator(config_path='/Users/simon/Desktop/simbapypi_dev/tests/test_data/multi_animal_dlc_two_c57/project_folder/project_config.ini',
#                  classifier_names=['Attack'],
#                  sigma=2,
#                  gamma=0.3,
#                  hierarchy=1)

# test.perform_kleinberg()
# #data = run_kleinberg(r'Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini', ['int'], sigma=2, gamma=0.3, hierarchy=1)