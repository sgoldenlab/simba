import pandas as pd
from simba.read_config_unit_tests import read_config_file, read_config_entry, read_project_path_and_file_type
from simba.features_scripts.unit_tests import read_video_info, read_video_info_csv
from simba.misc_tools import check_if_filepath_list_is_empty, check_multi_animal_status, get_fn_ext, detect_bouts
from simba.enums import Paths, ReadConfig, Dtypes
from simba.rw_dfs import read_df
from simba.drop_bp_cords import create_body_part_dictionary, getBpNames
import os, glob
from datetime import datetime
import seaborn as sns
import matplotlib

class PupRetrieverCalculator(object):
    def __init__(self,
                 config_path: str,
                 settings: dict):

        self.config = read_config_file(ini_path=config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.settings, self.datetime = settings, datetime.now().strftime('%Y%m%d%H%M%S')
        self.animal_cnt = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.ANIMAL_CNT.value, Dtypes.INT.value)
        self.clf_lst = [settings['clf_approach'], settings['clf_carry'], settings['clf_dig']]
        self.distance_pup_core_field = f'{self.settings["core_nest"]} {self.settings["pup_name"]} {"distance"}'
        self.distance_dam_core_field = f'{self.settings["core_nest"]} {self.settings["dam_name"]} {"distance"}'
        self.pup_in_core_field = f'{self.settings["core_nest"]} {self.settings["pup_name"]} {"in zone"}'
        self.pup_in_nest_field = f'{self.settings["nest"]} {self.settings["pup_name"]} {"in zone"}'

        machine_results_path = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.logs_dir_path = os.path.join(self.project_path, 'logs')

        self.data_files = glob.glob(machine_results_path + '/*.' + self.file_type)
        check_if_filepath_list_is_empty(filepaths=self.data_files, error_msg='SIMBA ERROR: NO FILES FOUND IN {}'.format(machine_results_path))
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        multi_animal_status, self.multi_animal_ids = check_multi_animal_status(self.config, self.animal_cnt)
        x_cols, y_cols, p_cols = getBpNames(inifile=config_path)
        self.bp_dict = create_body_part_dictionary(multiAnimalStatus=multi_animal_status,
                                                   multiAnimalIDList=self.multi_animal_ids,
                                                   animalsNo=self.animal_cnt,
                                                   Xcols=x_cols,
                                                   Ycols=y_cols,
                                                   Pcols=p_cols)

    def __get_max_frames(self):
        self.max_frames = int(self.fps * self.settings['max_time'])
        if self.max_frames < len(self.data_df):
            self.data_df = self.data_df.head(self.max_frames)


    def __check_column_names(self):
        for c in [self.distance_pup_core_field, self.distance_dam_core_field, self.pup_in_core_field, self.pup_in_nest_field]:
            if c not in self.data_df.columns:
                print(f'SIMBA ERROR: Could not find anticipated column named {c} in {self.file_path}')
                raise ValueError()

    def correct_in_nest_frames(self):
        if 1 in self.data_df[self.pup_in_nest_field].values:
            for nest_frame in self.data_df[self.data_df[self.pup_in_nest_field] == 1].index.tolist():
                sliced_df = self.data_df[self.settings['clf_carry']].loc[nest_frame - self.carry_frames + 1:nest_frame].tolist()
                if sum(sliced_df) == 0:
                    self.data_df.at[nest_frame, self.pup_in_nest_field] = 0
                else:
                    break

    def __create_log(self):
        log = {'Datetime': self.datetime,
               'Videos_#': len(self.data_files), **self.settings}
        log_df = pd.DataFrame.from_dict(log, orient='index').rename(columns={0: 'VALUES'})
        log_save_path = os.path.join(self.logs_dir_path, f'Log_pup_retrieval_{self.datetime}')
        log_df.to_csv(log_save_path)
        print('Pup retreival log saved at {}...'.format(log_save_path))



    def __generate_figure(self,
                        data: pd.DataFrame,
                        y_col: str,
                        x_lbl: str,
                        y_lbl: str,
                        title: str,
                        hue: str,
                        video_name: str):

        current_figure = sns.scatterplot(x=data.index, y=data[y_col], hue=data[hue], legend=False, palette='Set1')
        current_figure.set(xlabel=x_lbl, ylabel=y_lbl, title=title)
        save_plot_name = f'{title} {video_name} {self.datetime}.png'
        save_plot_path = os.path.join(self.logs_dir_path, video_name)
        if not os.path.exists(save_plot_path): os.makedirs(save_plot_path)
        image_save_path = os.path.join(save_plot_path, save_plot_name)
        current_figure.figure.savefig(image_save_path, bbox_inches="tight")
        current_figure.clear()


    def __create_swarm_plot(self):
        figure_df = self.out_df.copy()
        figure_df['Experiment'] = 1
        swarm_plot = sns.swarmplot(x="Experiment", y="PUP IN NEST (S)", data=figure_df, color="grey")
        swarm_plot.set(xlabel='', ylabel="Pup in nest (s)", title='Summary - pup retrieval time (s)')
        swarm_plot_name = f'Summary_pup_retrieval_times_{self.datetime}.png'
        save_plot_path = os.path.join(self.logs_dir_path, swarm_plot_name)
        swarm_plot.figure.savefig(save_plot_path, bbox_inches="tight")
        swarm_plot.clear()
        print(f'Swarm plot saved @ {save_plot_path}...')

    def run(self):
        self.out = []
        for file_cnt, file_path in enumerate(self.data_files):
            self.results = {}
            self.file_path = file_path
            _, video_name, _ = get_fn_ext(filepath=file_path)
            _, _, self.fps = read_video_info(vid_info_df=self.vid_info_df, video_name=video_name)
            self.data_df = read_df(file_path=file_path, file_type=self.file_type).fillna(method='ffill')
            self.__get_max_frames()
            self.carry_frames = int(self.fps * self.settings['carry_time'])

            self.data_df['mean_p_mother'] = self.data_df[self.bp_dict[self.settings['dam_name']]['P_bps']].mean(axis=1)
            self.data_df['pup_p_mean'] = self.data_df[self.bp_dict[self.settings['pup_name']]['P_bps']].mean(axis=1)
            self.data_df['cumsum_nest_pup'] = self.data_df[self.pup_in_nest_field].cumsum()

            if self.settings['distance_plots']:
                self.__generate_figure(data=self.data_df,
                                       y_col=self.distance_dam_core_field,
                                       x_lbl='frame number',
                                       y_lbl='distance (mm)',
                                       title='distance between mother and corenest - BEFORE pre-processing',
                                       hue='cumsum_nest_pup',
                                       video_name=video_name)

                self.__generate_figure(data=self.data_df,
                                       y_col=self.distance_pup_core_field,
                                       x_lbl='frame number',
                                       y_lbl='distance (mm)',
                                       title='distance between pup and corenest - BEFORE pre-processing',
                                       hue='cumsum_nest_pup',
                                       video_name=video_name)

            for clf in self.clf_lst:
                self.data_df.loc[self.data_df['mean_p_mother'] < self.settings['dam_track_p'], clf] = 0

            first_row = self.data_df[self.data_df[self.distance_pup_core_field] > self.settings['start_distance_criterion']].index[0]
            self.data_df.loc[0:first_row, self.distance_pup_core_field] = self.data_df.loc[first_row, self.distance_pup_core_field]
            self.data_df.loc[0:first_row, self.pup_in_core_field] = 0
            self.data_df.loc[0:first_row, self.pup_in_nest_field] = 0
            self.correct_in_nest_frames()

            rows_with_low_mean_pup_prob = self.data_df[self.data_df['pup_p_mean'] < self.settings['pup_track_p']].index.tolist()
            self.data_df.loc[rows_with_low_mean_pup_prob, self.pup_in_core_field] = 0
            self.data_df.loc[rows_with_low_mean_pup_prob, self.pup_in_nest_field] = 0

            if self.settings['smooth_function'] == 'gaussian':
                self.data_df[self.distance_pup_core_field] = self.data_df[self.distance_pup_core_field].rolling(window=int(self.fps), win_type='gaussian', center=True).mean(std=self.settings['smooth_factor']).fillna(self.data_df[self.distance_pup_core_field])
                self.data_df[self.distance_dam_core_field] = self.data_df[self.distance_dam_core_field].rolling(window=int(self.fps), win_type='gaussian', center=True).mean(std=self.settings['smooth_factor']).fillna(self.data_df[self.distance_dam_core_field])

            if self.settings['distance_plots']:
                self.__generate_figure(data=self.data_df,
                                       y_col=self.distance_dam_core_field,
                                       x_lbl='frame number',
                                       y_lbl='distance (mm)',
                                       title='distance between mother and corenest - AFTER pre-processing',
                                       hue='cumsum_nest_pup',
                                       video_name=video_name)

                self.__generate_figure(data=self.data_df,
                                       y_col=self.distance_pup_core_field,
                                       x_lbl='frame number',
                                       y_lbl='distance (mm)',
                                       title='distance between pup and core-nest - AFTER pre-processing',
                                       hue='cumsum_nest_pup',
                                       video_name=video_name)

            closest_dist_between_pup_and_zone = round(self.data_df[self.distance_pup_core_field].min(), 3)
            if 1 in self.data_df[self.pup_in_nest_field].values:
                frame_when_pup_is_in_zone = self.data_df[self.data_df[self.pup_in_nest_field] == 1].index.min()
                time_seconds_until_zone = round(frame_when_pup_is_in_zone / self.fps, 3)
                reason_zone = "Pup in nest"
            else:
                frame_when_pup_is_in_zone = len(self.data_df)
                time_seconds_until_zone = round(frame_when_pup_is_in_zone / self.fps, 3)
                reason_zone = "Pup not retrieved"

            if 1 in self.data_df[self.pup_in_core_field].values:
                frame_when_pup_is_in_core_nest = self.data_df[self.data_df[self.pup_in_core_field] == 1].index.min()
                time_seconds_until_corenest = round(frame_when_pup_is_in_core_nest / self.fps, 3)
                reason_corenest = "Pup in core-nest"
            else:
                frame_when_pup_is_in_core_nest = len(self.data_df)
                time_seconds_until_corenest = round(frame_when_pup_is_in_core_nest / self.fps, 3)
                reason_corenest = "Pup not in core-nest"

            latencies, total_times, before_retrieval_time = {}, {}, {}
            for clf in self.clf_lst:
                total_times[clf] = round(self.data_df[clf].sum() / self.fps, 3)
                before_retrieval_time[clf] = round(self.data_df.loc[0: frame_when_pup_is_in_zone, clf].sum() / self.fps, 3)
                latencies[clf] = round(self.data_df[self.data_df[clf] == 1].index.min() / self.fps, 3)

            event_counter, time_between_events, mean_duration = {}, {}, {}
            bouts = detect_bouts(data_df=self.data_df, target_lst=self.clf_lst, fps=self.fps)
            for clf in self.clf_lst:
                clf_bouts = bouts[bouts['Event'] == clf].reset_index(drop=True)
                event_counter[clf] = len(clf_bouts)
                mean_duration[clf] = round(clf_bouts['Bout_time'].mean() / self.fps, 3)
                clf_bouts['Start_time'] = clf_bouts['Start_time'].shift(-1)
                clf_bouts.drop(clf_bouts.tail(1).index, inplace=True)
                clf_bouts['TIME BETWEEN'] = clf_bouts['Start_time'] - clf_bouts['End Time']
                time_between_events[clf] = round(clf_bouts['TIME BETWEEN'].mean() / self.fps, 3)

            before_enter_core_df = self.data_df.loc[0: frame_when_pup_is_in_core_nest - 1]
            before_core_bouts = detect_bouts(data_df=before_enter_core_df, target_lst=self.clf_lst, fps=self.fps)
            event_counter_before_corenest, mean_event_length_before_corenest = {}, {}
            for clf in self.clf_lst:
                clf_bouts = before_core_bouts[before_core_bouts['Event'] == clf].reset_index(drop=True)
                mean_event_length_before_corenest[clf] = round(clf_bouts['Bout_time'].mean(), 3)
                event_counter_before_corenest[clf] = len(clf_bouts)

            retrieval_frame = self.data_df[self.data_df[self.pup_in_nest_field] == 1].index.min()
            retrieval_frame = 160
            first_approach = self.data_df[self.data_df[self.settings['clf_approach']] == 1].index.min()
            dig_bouts = detect_bouts(data_df=before_enter_core_df, target_lst=[self.settings['clf_dig']], fps=self.fps)
            dig_bouts_in_window = dig_bouts.loc[(dig_bouts['End_frame'] > first_approach) & (dig_bouts['End_frame'] < retrieval_frame)]
            dig_bouts_in_window_seconds = round(dig_bouts_in_window['Bout_time'].sum(), 3)

            self.results['VIDEO'] = video_name
            self.results['PUP IN NEST (FRAME)'] = frame_when_pup_is_in_zone
            self.results['PUP IN NEST (S)'] = time_seconds_until_zone
            self.results['MINIMUM DISTANCE (PUP TO CORENEST)'] = closest_dist_between_pup_and_zone
            self.results['REASON (PUP IN NEST)'] = reason_zone
            self.results['PUP IN CORE-NEST (FRAME)'] = frame_when_pup_is_in_core_nest
            self.results['PUP IN CORE-NEST (S)'] = time_seconds_until_corenest
            self.results['REASON (PUP IN CORE-NEST)'] = reason_corenest


            for clf in self.clf_lst:
                self.results[clf + ' (TOTAL TIME)'] = total_times[clf]
                self.results[clf + ' (BEFORE RETRIEVAL)'] = event_counter_before_corenest[clf]
                self.results[clf + ' (LATENCY TO FIRST EVENT)'] = latencies[clf]
                self.results[clf + ' (EVENT COUNT)'] = event_counter[clf]
                self.results[clf + ' (MEAN DURATION)'] = mean_duration[clf]
                self.results[clf + ' (MEAN INTERVAL)'] = time_between_events[clf]

            self.results['DIG TIME AFTER APPROACH AND BEFORE RETRIEVAL (S)'] = dig_bouts_in_window_seconds
            self.results['DIG EVENTS AFTER APPROACH'] = len(dig_bouts_in_window)
            self.results['MEAN DIG DURATION AFTER APPROACH'] = round(dig_bouts_in_window['Bout_time'].mean(), 3)
            self.out.append(pd.DataFrame.from_dict(self.results, orient='index').T)

        self.out_df = pd.concat(self.out, axis=0).reset_index(drop=True)

        if self.settings['log']:
            self.__create_log()

        if self.settings['swarm_plot']:
            self.__create_swarm_plot()

    def save_results(self):
        file_path = os.path.join(self.logs_dir_path, f'Log_pup_retrieval_{self.datetime}.csv')
        self.out_df.to_csv(file_path)
        print(f'SIMBA COMPLETE: Summary data saved at {file_path}.')




# settings = {'pup_track_p': 0.025, 'dam_track_p': 0.5, 'start_distance_criterion': 80.0, 'carry_frames': 90.0,
#      'core_nest': 'corenest', 'nest': 'nest', 'dam_name': '1_mother', 'pup_name': '2_pup',
#      'smooth_function': 'gaussian', 'smooth_factor': 5, 'max_time': 90.0, 'clf_carry': 'carry',
#      'clf_approach': 'approach', 'clf_dig': 'digging', 'distance_plots': True, 'log': True, 'swarm_plot': True}
# config_path = '/Users/simon/Downloads/Automated PRT_test/project_folder/project_config.ini'
#
# test = PupRetrieverCalculator(config_path=config_path, settings=settings)
# test.run()