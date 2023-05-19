__author__ = "Simon Nilsson"

import pandas as pd
import numpy as np
import shutil, os
from copy import deepcopy
from typing import List, Union


from simba.utils.checks import (check_that_column_exist,
                                check_int,
                                check_float,
                                check_if_filepath_list_is_empty)
from simba.utils.read_write import write_df, read_df, get_fn_ext
from simba.utils.printing import stdout_success
from simba.mixins.config_reader import ConfigReader
from simba.utils.warnings import KleinbergWarning
from simba.data_processors.pybursts_calculator import kleinberg_burst_detection
from simba.utils.enums import Paths

class KleinbergCalculator(ConfigReader):
    '''
    Smooth classification data using the Kleinberg burst detection algorithm.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter List[str] classifier_names: Classifier names to apply Kleinberg smoothing to.
    :parameter float sigma: Burst detection sigma value. Higher sigma values and fewer, longer, behavioural bursts will be recognised. Default: 2.
    :parameter float gamma: Burst detection gamma value. Higher gamma values and fewer behavioural bursts will be recognised. Default: 0.3.
    :parameter int hierarchy: Burst detection hierarchy level. Higher hierarchy values and fewer behavioural bursts will to be recognised. Default: 1.
    :parameter bool hierarchical_search: See `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/kleinberg_filter.md#hierarchical-search-example>`_ Default: False.



    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/kleinberg_filter.md>`__.

    Examples
    ----------
    >>> kleinberg_calculator = KleinbergCalculator(config_path='MySimBAConfigPath', classifier_names=['Attack'], sigma=2, gamma=0.3, hierarchy=2, hierarchical_search=False)
    >>> kleinberg_calculator.run()

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
                 config_path: Union[str, os.PathLike],
                 classifier_names: List[str],
                 sigma: int = 2,
                 gamma: float = 0.3,
                 hierarchy: int = 1,
                 hierarchical_search: bool = False):

        super().__init__(config_path=config_path)
        self.hierarchical_search, sigma, gamma, hierarchy = hierarchical_search, float(sigma), float(gamma), int(hierarchy)
        check_float(value=sigma, name='sigma', min_value=1.01)
        check_float(value=gamma, name='gamma', min_value=0)
        check_int(value=hierarchy, name='hierarchy')
        self.sigma, self.gamma, self.hierarchy = float(sigma), float(gamma), float(hierarchy)
        check_if_filepath_list_is_empty(filepaths=self.machine_results_paths,
                                        error_msg=f'SIMBA ERROR: No data files found in {self.machine_results_dir}. Cannot perform Kleinberg smooting')
        self.clfs = classifier_names
        original_data_files_folder = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value, 'Pre_Kleinberg_{}'.format( self.datetime))
        if not os.path.exists(original_data_files_folder): os.makedirs(original_data_files_folder)
        for file_path in self.machine_results_paths:
            _, file_name, ext = get_fn_ext(file_path)
            shutil.copyfile(file_path, os.path.join(original_data_files_folder, file_name + ext))
        print(f'Processing Kleinberg burst detection for {str(len(self.machine_results_paths))} file(s)...')

    def hierarchical_searcher(self):
        if (len(self.kleinberg_bouts['Hierarchy']) == 1) and (int(self.kleinberg_bouts.at[0, 'Hierarchy']) == 0):
            self.clf_bouts_in_hierarchy = self.kleinberg_bouts
        else:
            results = []
            kleinberg_df = deepcopy(self.kleinberg_bouts)
            kleinberg_df.loc[kleinberg_df['Hierarchy'] == 0, 'Hierarchy'] = np.inf
            kleinberg_df['prior_hierarchy'] = kleinberg_df['Hierarchy'].shift(1)
            kleinberg_df['hierarchy_difference'] = kleinberg_df['Hierarchy'] - kleinberg_df['prior_hierarchy']
            start_idx = list(kleinberg_df.index[kleinberg_df['hierarchy_difference'] <= 0])
            end_idx = list([x-1 for x in start_idx][1:])
            end_idx_2 = list(kleinberg_df.index[(kleinberg_df["hierarchy_difference"] == 0)|(kleinberg_df["hierarchy_difference"] > 1)])
            end_idx.extend((end_idx_2))
            for start, end in zip(start_idx, end_idx):
                hierarchies_in_bout = kleinberg_df.loc[start:end]
                target_hierarchy_in_hierarchies_bout = hierarchies_in_bout[hierarchies_in_bout['Hierarchy'] == self.hierarchy]
                if len(target_hierarchy_in_hierarchies_bout) == 0:
                    for lower_hierarchy in list(range(int(self.hierarchy-1.0), -1, -1)):
                        lower_hierarchy_in_hierarchies_bout = hierarchies_in_bout[hierarchies_in_bout['Hierarchy'] == lower_hierarchy]
                        if len(lower_hierarchy_in_hierarchies_bout) > 0:
                            target_hierarchy_in_hierarchies_bout = lower_hierarchy_in_hierarchies_bout
                            break
                if len(target_hierarchy_in_hierarchies_bout) > 0:
                    results.append(target_hierarchy_in_hierarchies_bout)
            if len(results) > 0:
                self.clf_bouts_in_hierarchy = pd.concat(results, axis=0).drop(['prior_hierarchy', 'hierarchy_difference'], axis=1)
            else:
                self.clf_bouts_in_hierarchy = pd.DataFrame(columns=['Video', 'Classifier', 'Hierarchy', 'Start', 'Stop'])

    def run(self):
        '''
        Method to perform Kleinberg smoothing. Results are stored in the `project_folder/csv/targets_inserted` directory.
        Detailed log is saved in the `project_folder/logs/` directory.

        Returns
        ----------
        None
        '''

        detailed_df_lst = []
        for file_cnt, file_path in enumerate(self.machine_results_paths):
            _, video_name, _ = get_fn_ext(file_path)
            print(f'Kleinberg analysis video {video_name}. Video { str(file_cnt+1)}/{str(len(self.machine_results_paths))}...')
            data_df = read_df(file_path, self.file_type).reset_index(drop=True)
            video_out_df = deepcopy(data_df)
            for clf in self.clfs:
                check_that_column_exist(df=data_df, column_name=clf, file_name=video_name)
                clf_offsets = data_df.index[data_df[clf] == 1].values
                if len(clf_offsets) > 0:
                    video_out_df[clf] = 0
                    self.kleinberg_bouts = pd.DataFrame(kleinberg_burst_detection(offsets=clf_offsets, s=self.sigma, gamma=self.gamma), columns = ['Hierarchy', 'Start', 'Stop'])
                    self.kleinberg_bouts['Stop'] += 1
                    self.kleinberg_bouts.insert(loc=0, column='Classifier', value=clf)
                    self.kleinberg_bouts.insert(loc=0, column='Video', value=video_name)
                    detailed_df_lst.append(self.kleinberg_bouts)
                    if self.hierarchical_search:
                        print('Applying hierarchical search...')
                        self.hierarchical_searcher()
                    else:
                        self.clf_bouts_in_hierarchy = self.kleinberg_bouts[self.kleinberg_bouts['Hierarchy'] == self.hierarchy]
                    hierarchy_idx = list(self.clf_bouts_in_hierarchy.apply(lambda x: list(range(x['Start'], x['Stop'] + 1)), 1))
                    hierarchy_idx = [x for xs in hierarchy_idx for x in xs]
                    hierarchy_idx = [x for x in hierarchy_idx if x in list(data_df.index)]
                    video_out_df.loc[hierarchy_idx, clf] = 1
            write_df(video_out_df, self.file_type, file_path)

        self.timer.stop_timer()
        if len(detailed_df_lst) > 0:
            detailed_df = pd.concat(detailed_df_lst, axis=0)
            detailed_save_path = os.path.join(self.logs_path, 'Kleinberg_detailed_log_{}.csv'.format(str(self.datetime)))
            detailed_df.to_csv(detailed_save_path)
            stdout_success(msg='Kleinberg analysis complete. See {detailed_save_path} for details of detected bouts of all classifiers in all hierarchies', elapsed_time=self.timer.elapsed_time_str)
        else:
            KleinbergWarning(msg='All behavior bouts removed following kleinberg smoothing')

# test = KleinbergCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals/project_folder/project_config.ini',
#                            classifier_names=['Attack'],
#                            sigma=1.1,
#                            gamma=0.3,
#                            hierarchy=5,
#                            hierarchical_search=False)
#
# test.perform_kleinberg()
# #data = run_kleinberg(r'Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini', ['int'], sigma=2, gamma=0.3, hierarchy=1)