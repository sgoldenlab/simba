__author__ = "Simon Nilsson"

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import os, glob
from simba.utils.printing import stdout_success, SimbaTimer
from simba.utils.enums import ConfigKey, Dtypes
from simba.utils.checks import check_that_column_exist, check_if_filepath_list_is_empty, check_int
from simba.utils.read_write import get_fn_ext, read_df, read_config_entry
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin

class TimeBinsMovementCalculator(ConfigReader, FeatureExtractionMixin):
    """
    Class for calculating and aggregating movement statistics into user-defined time-bins.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    bin_length: int
        Integer representing the time bin size in seconds

    Example
    ----------

    >>> timebin_movement_analyzer = TimeBinsMovementCalculator(config_path='MyConfigPath', bin_length=15)
    >>> timebin_movement_analyzer.analyze_movement()

    """

    def __init__(self,
                 config_path: str,
                 bin_length: int,
                 plots: bool):

        ConfigReader.__init__(self, config_path=config_path)
        self.bin_length, self.plots = bin_length, plots
        check_int(name='TIME BIN', value=bin_length, min_value=1)
        self.col_headers = []
        for animal_no in range(self.animal_cnt):
            animal_bp = read_config_entry(self.config, ConfigKey.PROCESS_MOVEMENT_SETTINGS.value, 'animal_{}_bp'.format(str(animal_no + 1)), Dtypes.STR.value)
            self.col_headers.extend((animal_bp + '_x', animal_bp + '_y'))

        check_if_filepath_list_is_empty(filepaths=self.outlier_corrected_paths,
                                        error_msg=f'SIMBA ERROR: Cannot analyze movement in time-bins, data directory {self.outlier_corrected_dir} is empty.')
        self.animal_combinations = list(itertools.combinations(self.animal_bp_dict, 2))
        print('Processing {} video(s)...'.format(str(len(self.outlier_corrected_paths))))

    def __create_plots(self):
        print('Creating time-bin movement plots...')
        sns.set_style("whitegrid", {'grid.linestyle': '--'})
        self.video_df['Time bin #'] = self.video_df['Time bin #'].astype(str)
        plots_dir = os.path.join(self.project_path, 'logs', f'time_bin_movement_plots_{self.datetime}')
        if not os.path.exists(plots_dir): os.makedirs(plots_dir)
        for video_name in self.video_df.index.unique():
            video_df = self.video_df[self.video_df.index == video_name].reset_index(drop=True)
            video_movement_df = video_df[video_df['Measurement'].isin(list(self.movement_cols))]
            line_plot = sns.lineplot(data=video_movement_df, x="Time bin #", y="Value", hue='Measurement')
            line_plot.figure.savefig(os.path.join(plots_dir, f'{video_name}.png'))
            plt.close()
        stdout_success(msg='Time bin movement plots saved in {plots_dir}...')

    def run(self):
        """
        Method for running the movement time-bin analysis. Results are stored in the ``project_folder/logs`` directory
        of the SimBA project.

        Returns
        ----------
        None
        """
        video_dict, self.out_df_lst = {}, []
        for file_cnt, file_path in enumerate(self.outlier_corrected_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(file_path)
            print(f'Processing time-bin movements for video {video_name} ({str(file_cnt+1)}/{str(len(self.outlier_corrected_paths))})...')
            result_df = pd.DataFrame()
            video_dict[video_name] = {}
            video_settings, px_per_mm, fps = self.read_video_info(video_name=video_name)
            fps, self.movement_cols, self.velocity_cols = int(fps), set(), set()
            bin_length_frames = int(fps * self.bin_length)
            data_df = read_df(file_path, self.file_type)
            data_df_sliced = pd.DataFrame()
            for animal, data in self.animal_bp_dict.items():
                check_that_column_exist(df=data_df, column_name=data['X_bps'][0], file_name=file_path)
                check_that_column_exist(df=data_df, column_name=data['Y_bps'][0], file_name=file_path)
                data_df_sliced[data['X_bps'][0]] = data_df[data['X_bps'][0]]
                data_df_sliced[data['Y_bps'][0]] = data_df[data['Y_bps'][0]]
            data_df_shifted = data_df.shift(periods=1).add_suffix('_shifted').fillna(0)
            data_df = pd.concat([data_df_sliced, data_df_shifted], axis=1, join='inner').fillna(0).reset_index(drop=True)
            for animal, data in self.animal_bp_dict.items():
                movement_col_name = 'Movement {}'.format(animal)
                x_col, y_col = data['X_bps'][0], data['Y_bps'][0]
                bp_time_1 = data_df[[x_col, y_col]].values
                bp_time_2 = data_df[[x_col + '_shifted', y_col + '_shifted']].values
                result_df[movement_col_name] = pd.Series(self.framewise_euclidean_distance(location_1=bp_time_1, location_2=bp_time_2, px_per_mm=px_per_mm, centimeter=True))
                result_df.loc[0, movement_col_name] = 0
            for animal_c in self.animal_combinations:
                distance_col_name = 'Distance {} {}'.format(animal_c[0], animal_c[1])
                bp_1_x, bp_1_y = self.animal_bp_dict[animal_c[0]]['X_bps'][0], self.animal_bp_dict[animal_c[0]]['Y_bps'][0]
                bp_2_x, bp_2_y = self.animal_bp_dict[animal_c[0]]['X_bps'][0], self.animal_bp_dict[animal_c[1]]['Y_bps'][0]
                bp_1 = data_df[[bp_1_x, bp_1_y]].values.astype(float)
                bp_2 = data_df[[bp_2_x, bp_2_y]].values.astype(float)
                result_df[distance_col_name] = pd.Series(self.framewise_euclidean_distance(location_1=bp_1, location_2=bp_2, px_per_mm=px_per_mm))
            results_df_lists = [result_df[i:i + bin_length_frames] for i in range(0, result_df.shape[0], bin_length_frames)]
            indexed_df_lst = []
            for bin, results in enumerate(results_df_lists):
                time_bin_per_s = [results[i:i + fps] for i in range(0, results.shape[0], fps)]
                for second, df in enumerate(time_bin_per_s):
                    df['Time bin #'], df['Second'] = bin, second
                    indexed_df_lst.append(df)
            indexed_df = pd.concat(indexed_df_lst, axis=0)
            movement_cols = [x for x in indexed_df.columns if x.startswith('Movement ')]
            distance_cols = [x for x in indexed_df.columns if x.startswith('Distance ')]
            for movement_col in movement_cols:
                movement_sum = indexed_df.groupby(['Time bin #'])[movement_col].sum().reset_index()
                movement_velocity = indexed_df.groupby(['Time bin #', 'Second'])[movement_col].sum().reset_index()
                movement_velocity = movement_velocity.groupby(['Time bin #'])[movement_col].mean().reset_index()
                video_dict[video_name][movement_col + ' (cm)'] = movement_sum
                self.movement_cols.add(movement_col + ' (cm)')
                video_dict[video_name][movement_col + ' velocity (cm/s)'] = movement_velocity
                self.velocity_cols.add(movement_col + ' velocity (cm/s)')
            for distance_col in distance_cols:
                video_dict[video_name][distance_col + ' (cm)'] = indexed_df.groupby(['Time bin #'])[distance_col].mean().reset_index()
            video_timer.stop_timer()
            print(f'Video {video_name} complete (elapsed time: {video_timer.elapsed_time_str}s)...')

        for video_name, video_info in video_dict.items():
            for measurement, bin_data in video_info.items():
                data_df = pd.DataFrame.from_dict(bin_data).reset_index(drop=True)
                data_df.columns = ['Time bin #', measurement]
                data_df = pd.melt(data_df, id_vars=['Time bin #']).drop('variable', axis=1).rename(columns={'value': 'Value'})
                data_df.insert(loc=0, column='Measurement', value=measurement)
                data_df.insert(loc=0, column='Video', value=video_name)
                self.out_df_lst.append(data_df)
        self.video_df = pd.concat(self.out_df_lst, axis= 0).sort_values(by=['Video', 'Time bin #']).set_index('Video')
        if self.plots:
            self.__create_plots()

        save_path = os.path.join(self.project_path, 'logs', 'Time_bins_movement_results_' + self.datetime + '.csv')
        self.video_df.to_csv(save_path)
        self.timer.stop_timer()
        stdout_success(msg=f'Movement time-bins results saved at {save_path}', elapsed_time=self.timer.elapsed_time_str)

# test = TimeBinsMovementAnalyzer(config_path='/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini',
#                                 bin_length=60, plots=True)
# test.analyze_movement()

# test = TimeBinsMovementAnalyzer(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini',
#                                 bin_length=1, plots=True)
# test.analyze_movement()