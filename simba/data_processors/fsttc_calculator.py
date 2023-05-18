__author__ = "Simon Nilsson"

import os
import pandas as pd
import itertools
import seaborn as sns
from typing import List, Union

from simba.utils.data import detect_bouts
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, read_df
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.errors import CountError
from simba.mixins.config_reader import ConfigReader

class FSTTCCalculator(ConfigReader):
    """
    Compute forward spike-time tiling coefficients between pairs of
    classified behaviors.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter int time_window: FSTTC hyperparameter; Integer representing the time window in seconds.
    :parameter List[str] behavior_lst: Behaviors to calculate FSTTC between. FSTTC will be computed for all combinations of behaviors.
    :parameter bool create_graphs: If True, created violin plots (as below) representing each FSTTC. Default: False.

    .. image:: _static/img/fsttc_violin.png
       :width: 500
       :align: center

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/FSTTC.md>`__.

    Examples
    -----
    >>> fsttc_calculator = FSTTCCalculator(config_path='MyConfigPath', time_window=2, behavior_lst=['Attack', 'Sniffing'], create_graphs=True)
    >>> fsttc_calculator.run()

    References
    ----------
    .. [1] Lee et al., Temporal microstructure of dyadic social behavior during relationship formation in mice, `PLOS One`,
           2019.
    .. [2] Cutts et al., Detecting Pairwise Correlations in Spike Trains: An Objective Comparison of Methods and
           Application to the Study of Retinal Waves, `J Neurosci`, 2014.
    """


    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 time_window: int,
                 behavior_lst: List[str],
                 create_graphs: bool = False):

        super().__init__(config_path=config_path)
        self.time_delta = int(time_window)
        self.behavior_lst = behavior_lst
        if len(self.behavior_lst) < 2:
            raise CountError(msg='FSTCC requires at least two behaviors')
        self.graph_status = create_graphs
        check_if_filepath_list_is_empty(filepaths=self.machine_results_paths,
                                        error_msg=f'Cannot calculate FSTTC, no data found in {self.machine_results_paths} directory')
        self.clf_permutations = list(itertools.permutations(self.behavior_lst, 2))
        print(f'Processing FSTTC for {str(len(self.machine_results_paths))} file(s)...')

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

        for file_cnt, file_path in enumerate(self.machine_results_paths):
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_sequence_dict[self.video_name] = {}
            print('Analyzing behavioral sequences: {}. Video {}/{}'.format(self.video_name, str(file_cnt + 1), str(len(self.machine_results_paths))))
            _, _, self.fps = self.read_video_info(video_name=self.video_name)
            self.video_sequence_dict[self.video_name]['fps'] = self.fps
            self.frames_in_window = int((self.fps / 1000) * self.time_delta)
            self.data_df = read_df(file_path, self.file_type)#[self.behavior_lst]
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

                        self.vide_df_sequence_lst.append(pd.DataFrame([[self.video_name,
                                                                        first_clf,
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


    def run(self):
        """
        Method to calculate forward spike-time tiling coefficients (FSTTC) using the data computed in :meth:
        :meth:`~simba.FSTTCPerformer.find_sequences`.

        Returns
        -------
        Attribute: dict
            results_dict
        """

        self.find_sequences()
        self.results_dict = {}
        for video_name, video_data in self.video_sequence_dict.items():
            self.results_dict[video_name] = {}
            fps, session_frames = video_data['fps'], video_data['session_length_frames']
            for first_clf, second_clf in self.clf_permutations:
                if first_clf not in self.results_dict[video_name].keys():
                    self.results_dict[video_name][first_clf] = {}
                self.results_dict[video_name][first_clf][second_clf] = {}
                sequence_data = video_data[f'FSTTC {first_clf} {second_clf}']
                if sequence_data is None:
                    self.results_dict[video_name][first_clf][second_clf] = 'No events'
                else:
                    len_clf_1 = len(sequence_data[sequence_data['First behaviour'] == first_clf])
                    len_clf_1_2 = len(sequence_data[(sequence_data['First behaviour'] == first_clf) & (sequence_data['Second behaviour'] == second_clf)])
                    if (len_clf_1 > 0) & (len_clf_1_2 == 0):
                        self.results_dict[video_name][first_clf][second_clf] = 0.0
                    else:
                        clf_1_2_df = sequence_data[(sequence_data['First behaviour'] == first_clf) & (sequence_data['Second behaviour'] == second_clf)]
                        P = len_clf_1_2 / len_clf_1
                        Ta = sum(clf_1_2_df['Total_window_frames']) / session_frames
                        Tb = sum(clf_1_2_df['Time 2nd behaviour start to time window end']) / session_frames
                        self.results_dict[video_name][first_clf][second_clf] = 0.5 * ((P - Tb) / (1 - (P * Tb)) + ((P - Ta) / (1 - (P * Ta))))
        self.save()
        if self.graph_status:
            self.plot_FSTTC()

    def plot_FSTTC(self):
        """
        Method to visualize forward spike-time tiling coefficients (FSTTC) as png violin plots. Results are stored on
        disk within the `project_folder/logs` directory.

        Returns
        -------
        None
        """

        self.out_df['BEHAVIOR COMBINATION'] = self.out_df['FIRST BEHAVIOR'].str.cat(self.out_df['SECOND BEHAVIOR'], sep='-')
        data_df = self.out_df[self.out_df['FSTTC'] != 'No events'].reset_index(drop=True)
        data_df['FSTTC'] = pd.to_numeric(data_df['FSTTC'], errors='coerce')
        figure_FSCTT = sns.violinplot(x='BEHAVIOR COMBINATION', y="FSTTC", data=data_df, cut=0)
        figure_FSCTT.set_xticklabels(figure_FSCTT.get_xticklabels(), rotation=45, size=7)
        figure_FSCTT.figure.set_size_inches(13.7, 8.27)
        save_plot_path = os.path.join(self.logs_path, f'FSTTC_{self.datetime}.png')
        figure_FSCTT.figure.savefig(save_plot_path, bbox_inches="tight")
        stdout_success(msg=f'FSTTC figure saved at {save_plot_path}')


    def save(self):
        """
        Method to save forward spike-time tiling coefficients (FSTTC) to disk within the `project_folder/logs` directory.

        Returns
        -------
        None
        """
        self.out_df = pd.DataFrame(columns=['VIDEO', 'FIRST BEHAVIOR', 'SECOND BEHAVIOR', 'FSTTC'])
        for video_name, video_data in self.results_dict.items():
            for first_behavior, first_behavior_data in video_data.items():
                for second_behavior, fsttc in first_behavior_data.items():
                    self.out_df.loc[len(self.out_df)] = [video_name, first_behavior, second_behavior, fsttc]
        file_save_path = os.path.join(self.logs_path, 'FSTTC_{}.csv'.format(str(self.datetime)))
        self.out_df.to_csv(file_save_path)
        self.timer.stop_timer()
        stdout_success(msg=f'FSTTC data saved at {file_save_path}', elapsed_time=self.timer.elapsed_time_str)

# test = FSTTCCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                      time_window=2000,
#                      behavior_lst=['Attack', 'Sniffing'],
#                     create_graphs=False)
# test.run()

#
# test = FSTTCCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini',
#                      time_window=2000,
#                      behavior_lst=['Erratic Turning', 'Bottom', 'Normal Swimming', 'Freezing', 'Wall Bumping'],
#                     create_graphs=True)
# test.run()





