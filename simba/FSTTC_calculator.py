__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import (check_int,
                                          check_str,
                                          check_float,
                                          read_config_entry,
                                          read_config_file,
                                          check_if_filepath_list_is_empty)
import os, glob
import pandas as pd
from datetime import datetime
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.drop_bp_cords import get_fn_ext
from simba.rw_dfs import read_df
from simba.misc_tools import (detect_bouts,
                              SimbaTimer)
from simba.enums import Paths, ReadConfig
import itertools
import seaborn as sns

class FSTTCPerformer(object):
    """
    Class for calculating forward spike-time tiling coefficients between pairs of
    classified behaviors.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    time_window: int
        Integer representing the time window in seconds
    behavior_lst: list
        Behaviors to calculate FSTTC between.
    create_graphs: bool
        If True, created violin plots representing each FSTTC


    Notes
    -----
    `GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/FSTTC.md>`__.

    Examples
    -----
    >>> fsttc_calculator = FSTTCPerformer(config_path='MyConfigPath', time_window=2, behavior_lst=['Attack', 'Sniffing'], create_graphs=True)
    >>> fsttc_calculator.find_sequences()
    >>> fsttc_calculator.calculate_FSTTC()
    >>> fsttc_calculator.save_FSTTC()
    >>> fsttc_calculator.plot_FSTTC()

    References
    ----------
    .. [1] Lee et al., Temporal microstructure of dyadic social behavior during relationship formation in mice, `PLOS One`,
           2019.
    .. [2] Cutts et al., Detecting Pairwise Correlations in Spike Trains: An Objective Comparison of Methods and
           Application to the Study of Retinal Waves, `J Neurosci`, 2014.
    """


    def __init__(self,
                 config_path: str,
                 time_window: int,
                 behavior_lst: list,
                 create_graphs: bool):

        self.timer = SimbaTimer()
        self.timer.start_timer()
        self.time_delta = int(time_window)
        self.behavior_lst = behavior_lst
        self.graph_status = create_graphs
        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.PROJECT_PATH.value, data_type=ReadConfig.FOLDER_PATH.value)
        self.in_path = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.logs_folder = os.path.join(self.project_path, 'logs')
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.file_type = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.FILE_TYPE.value, 'str', 'csv')
        self.video_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.files_found = glob.glob(self.in_path + "/*." + self.file_type)
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: Cannot calculate FSTTC, no data found in {} directory'.format(self.in_path))
        self.clf_permutations = list(itertools.permutations(self.behavior_lst, 2))
        print('Processing FSTTC for {} file(s)...'.format(str(len(self.files_found))))

    def find_sequences(self):
        """
        Method to create list of dataframes holding information on the sequences of behaviors including
        inter-temporal distances.

        Returns
        -------
        Attribute: list
            vide_df_sequence_lst
        """

        self.video_sequence_dict = {}
        out_columns =['Video',
                      'First behaviour',
                        'First behaviour start frame',
                         'First behavior end frame',
                         'Second behaviour',
                         'Second behaviour start frame',
                         'Difference: first behavior start to second behavior start',
                         'Time 2nd behaviour start to time window end']

        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_sequence_dict[self.video_name] = {}
            print('Analyzing behavioral sequences: {}. Video {}/{}'.format(self.video_name, str(file_cnt + 1), str(len(self.files_found))))
            _, _, self.fps = read_video_info(self.video_info_df, self.video_name)
            self.video_sequence_dict[self.video_name]['fps'] = self.fps
            self.frames_in_window = int((self.fps / 1000) * self.time_delta)
            self.data_df = read_df(file_path, self.file_type)[self.behavior_lst]
            self.video_sequence_dict[self.video_name]['session_length_frames'] = len(self.data_df)
            bouts_df = detect_bouts(data_df=self.data_df, target_lst=self.behavior_lst, fps=self.fps)
            bouts_df['Start_frame'] = (bouts_df['Start_time'] * self.fps).astype(int) - 1
            bouts_df = bouts_df[['Event', 'Start_frame', 'End_frame']]
            for first_clf, second_clf in self.clf_permutations:
                self.vide_df_sequence_lst = []
                sequence_name = 'FSTTC {} {}'.format(first_clf, second_clf)
                first_clf_df = bouts_df[bouts_df['Event'] == first_clf].sort_values(by=['Start_frame']).reset_index(drop=True)
                second_clf_df = bouts_df[bouts_df['Event'] == second_clf].sort_values(by=['Start_frame']).reset_index(drop=True)
                for index, row in first_clf_df.iterrows():
                    frame_crtrn_min, frame_crtrn_max = row['Start_frame'] + 1, row['End_frame'] + self.frames_in_window
                    second_clf_df_crtrn = second_clf_df.loc[(second_clf_df['Start_frame'] >= frame_crtrn_min) & (second_clf_df['Start_frame'] <= frame_crtrn_max)]
                    if len(second_clf_df_crtrn) > 0:
                        second_clf_df_crtrn = second_clf_df_crtrn.head(1)
                        frames_between_behaviors = (second_clf_df_crtrn['Start_frame'] - row['Start_frame']).values
                        second_clf_df_crtrn['Frames_between_behaviors'] = (second_clf_df_crtrn['Start_frame'] - row['Start_frame'])
                        if frames_between_behaviors <= 0:
                            frames_between_behaviors = 1
                        second_clf_df_crtrn['Frames_between_behaviors'] = frames_between_behaviors
                        second_clf_df_crtrn['Frames_between_second_behavior_start_to_time_window_end'] = (row['End_frame'] + self.frames_in_window) - second_clf_df_crtrn['Start_frame'].values[0]
                        self.vide_df_sequence_lst.append(pd.DataFrame([[self.video_name, first_clf, row['Start_frame'],
                                                 row['End_frame'], second_clf, second_clf_df_crtrn['Start_frame'].values[0],
                                                 second_clf_df_crtrn['Frames_between_behaviors'].values[0],
                                                 second_clf_df_crtrn['Frames_between_second_behavior_start_to_time_window_end'].values[0]]], columns=out_columns))
                    else:

                        self.vide_df_sequence_lst.append(pd.DataFrame([[self.video_name, first_clf,
                                                               int(row['Start_frame']),
                                                               int(row['End_frame']),
                                                               'None',
                                                               'None',
                                                               'None',
                                                               'None']],
                                                             columns=out_columns))

                if len(self.vide_df_sequence_lst) > 0:
                    video_sequences = pd.concat(self.vide_df_sequence_lst, axis=0).drop_duplicates(subset=['Video', 'First behaviour', 'First behaviour start frame', 'First behavior end frame', 'Second behaviour'], keep='first').reset_index(drop=True)
                    video_sequences['Total_window_frames'] = ((video_sequences['First behavior end frame'] - video_sequences['First behaviour start frame']) + self.frames_in_window)
                    self.video_sequence_dict[self.video_name][sequence_name] = video_sequences
                else:
                    self.video_sequence_dict[self.video_name][sequence_name] = None


    def calculate_FSTTC(self):
        """
        Method to calculate forward spike-time tiling coefficients (FSTTC) using the data computed in :meth:
        :meth:`~simba.FSTTCPerformer.find_sequences`.

        Returns
        -------
        Attribute: dict
            results_dict
        """


        self.results_dict = {}
        for video_name, video_data in self.video_sequence_dict.items():
            self.results_dict[video_name] = {}
            fps, session_frames = video_data['fps'], video_data['session_length_frames']
            for first_clf, second_clf in self.clf_permutations:
                sequence_name = 'FSTTC {} {}'.format(first_clf, second_clf)
                sequence_data = video_data[sequence_name]
                if sequence_data is None:
                    self.results_dict[video_name][sequence_name] = 'No events'
                else:
                    len_clf_1 = len(sequence_data[sequence_data['First behaviour'] == first_clf])
                    len_clf_1_2 = len(sequence_data[(sequence_data['First behaviour'] == first_clf) & (sequence_data['Second behaviour'] == second_clf)])
                    if (len_clf_1 > 0) & (len_clf_1_2 == 0):
                        self.results_dict[video_name][sequence_name] = 0.0
                    else:
                        clf_1_2_df = sequence_data[(sequence_data['First behaviour'] == first_clf) & (sequence_data['Second behaviour'] == second_clf)]
                        P = len_clf_1_2 / len_clf_1
                        Ta = sum(clf_1_2_df['Total_window_frames']) / session_frames
                        Tb = sum(clf_1_2_df['Time 2nd behaviour start to time window end']) / session_frames
                        self.results_dict[video_name][sequence_name] = 0.5 * ((P - Tb) / (1 - (P * Tb)) + ((P - Ta) / (1 - (P * Ta))))

    def save_FSTTC(self):
        """
        Method to save forward spike-time tiling coefficients (FSTTC) to disk within the `project_folder/logs` directory.

        Returns
        -------
        None
        """



        self.out_df = pd.DataFrame()
        for video, data in self.results_dict.items():
            video_df = pd.DataFrame.from_dict(data, orient='index').reset_index().rename(columns={'index': 'FSTTC', 0: 'Value'})
            video_df.insert(loc=0, column='Video', value=video)
            self.out_df = pd.concat([self.out_df,video_df], axis=0)
        file_save_path = os.path.join(self.logs_folder, 'FSTTC_{}.csv'.format(str(self.datetime)))
        self.out_df.to_csv(file_save_path)
        self.timer.stop_timer()
        print('SIMBA COMPLETE: FSTTC data saved at {} (elapsed time: {}s)'.format(file_save_path, self.timer.elapsed_time_str))

    def plot_FSTTC(self):
        """
        Method to visualize forward spike-time tiling coefficients (FSTTC) as png violin plots. Results are stored on
        disk within the `project_folder/logs` directory.

        Returns
        -------
        None
        """

        if self.graph_status:
            data_df = self.out_df[self.out_df['Value'] != 'No events'].reset_index(drop=True)
            data_df['Value'] = pd.to_numeric(data_df['Value'], errors='coerce')
            figure_FSCTT = sns.violinplot(x="FSTTC", y="Value", data=data_df, cut=0)
            figure_FSCTT.set_xticklabels(figure_FSCTT.get_xticklabels(), rotation=45, size=7)
            figure_FSCTT.figure.set_size_inches(13.7, 8.27)
            save_plot_path = os.path.join(self.logs_folder, 'FSTTC_{}.png'.format(str(self.datetime)))
            figure_FSCTT.figure.savefig(save_plot_path, bbox_inches="tight")
            print('FSTTC figure saved at {}'.format(save_plot_path))


# test = FSTTCPerformer(config_path='/Users/simon/Desktop/train_model_project/project_folder/project_config.ini',
#                      time_window=2,
#                      behavior_lst=['Attack', 'Sniffing'],
#                     create_graphs=True)
# test.find_sequences()
# test.calculate_FSTTC()
# test.save_FSTTC()
# # test.plot_FSTTC()
# if __name__ == "__main__":
#     config_path = r"Z:\Classifiers\100620_all\project_folder\project_config.ini"
#     behavior_list = ['Attack', 'pursuit_prediction', 'Escape']
#     time_delta = 2000
#     create_graphs = True
#     FSTCC_performer = FSTTCPerformer(config_path, time_delta, behavior_list, create_graphs)
#     FSTCC_performer.find_sequences()
#     FSTCC_performer.calculate_FSTTC()
#     FSTCC_performer.save_FSTTC()
#     # FSTCC_performer.plot_results()

