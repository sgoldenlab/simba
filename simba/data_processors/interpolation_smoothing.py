__author__ = "Simon Nilsson"

from scipy.signal import savgol_filter
import glob, os
import numpy as np
import pandas as pd
import shutil
from simba.mixins.config_reader import ConfigReader
from simba.utils.printing import stdout_success, SimbaTimer
from simba.utils.read_write import read_df, write_df, get_video_meta_data, get_fn_ext, find_video_of_file
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.enums import Methods
from simba.utils.errors import NoFilesFoundError


class Interpolate(ConfigReader):
    def __init__(self,
                 input_path: str,
                 config_path: str,
                 method: str,
                 initial_import_multi_index: bool = False):

        super().__init__(config_path=config_path, read_video_info=False)
        self.interpolation_type, self.interpolation_method = method.split(':')[0], method.split(':')[1].replace(" ", "").lower()
        if os.path.isdir(input_path):
            self.files_found = glob.glob(input_path + '/*' + self.file_type)
            self.input_dir = input_path
            check_if_filepath_list_is_empty(filepaths=self.files_found, error_msg=f"SIMBA ERROR: {self.input_dir} does not contain any {self.file_type} files.")
        else:
            self.files_found = [input_path]
            self.input_dir = os.path.dirname(input_path)
        if not initial_import_multi_index:
            self.save_dir = os.path.join(self.input_dir, f'Pre_{self.interpolation_method}_{self.interpolation_type}_interpolation_{self.datetime}')
            if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.initial_import_multi_index = initial_import_multi_index
        if self.interpolation_type == 'Animal(s)':
            self.animal_interpolator()
        if self.interpolation_type == 'Body-parts':
            self.body_part_interpolator()
        self.timer.stop_timer()
        stdout_success(msg=f'{str(len(self.files_found))} data file(s) interpolated)', elapsed_time=self.timer.elapsed_time_str)

    def animal_interpolator(self):
        for file_path in self.files_found:
            video_timer = SimbaTimer()
            video_timer.start_timer()
            _, video_name, _ = get_fn_ext(filepath=file_path)
            df = read_df(file_path=file_path, file_type=self.file_type, check_multiindex=self.initial_import_multi_index)
            if self.initial_import_multi_index:
                df.columns = self.bp_col_names
            for animal_name, animal_bps in self.animal_bp_dict.items():
                animal_df = df[animal_bps['X_bps'] + animal_bps['Y_bps']].fillna(0).astype(int)
                idx = list(animal_df[animal_df.eq(animal_df.iloc[:, 0], axis=0).all(axis='columns')].index)
                animal_df.loc[idx, :] = np.nan
                animal_df = animal_df.interpolate(method=self.interpolation_method, axis=0).ffill().bfill()
                df.update(animal_df)
            if not self.initial_import_multi_index:
                shutil.move(src=file_path, dst=os.path.join(self.save_dir, os.path.basename(file_path)))
            else:
                multi_idx_header = []
                for i in range(len(df.columns)): multi_idx_header.append(('IMPORTED_POSE', 'IMPORTED_POSE', list(df.columns)[i]))
                df.columns = pd.MultiIndex.from_tuples(multi_idx_header)
            write_df(df=df, file_type=self.file_type, save_path=file_path, multi_idx_header=self.initial_import_multi_index)
            video_timer.stop_timer()
            print(f'Video {video_name} interpolated (elapsed time {video_timer.elapsed_time_str})...')

    def body_part_interpolator(self):
        for file_path in self.files_found:
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            df = read_df(file_path=file_path, file_type=self.file_type, check_multiindex=True)
            if self.initial_import_multi_index:
                df.columns = self.bp_col_names
            for animal in self.animal_bp_dict:
                for x_bps_name, y_bps_name in zip(self.animal_bp_dict[animal]['X_bps'], self.animal_bp_dict[animal]['Y_bps']):
                    idx = (df[(df[x_bps_name] == 0) & (df[y_bps_name] == 0)].index.tolist())
                    df.loc[idx, [x_bps_name, y_bps_name]] = np.nan
                    df[x_bps_name] = df[x_bps_name].interpolate(method=self.interpolation_method, axis=0).ffill().bfill()
                    df[y_bps_name] = df[y_bps_name].interpolate(method=self.interpolation_method, axis=0).ffill().bfill()
            if not self.initial_import_multi_index:
                shutil.move(src=file_path, dst=os.path.join(self.save_dir, os.path.basename(file_path)))
            else:
                multi_idx_header = []
                for i in range(len(df.columns)): multi_idx_header.append(('IMPORTED_POSE', 'IMPORTED_POSE', list(df.columns)[i]))
                df.columns = pd.MultiIndex.from_tuples(multi_idx_header)
            write_df(df=df, file_type=self.file_type, save_path=file_path, multi_idx_header=self.initial_import_multi_index)
            video_timer.stop_timer()
            print(f'Video {video_name} interpolated (elapsed time {video_timer.elapsed_time_str}) ...')

class Smooth(ConfigReader):
    def __init__(self,
                 config_path: str,
                 input_path: str,
                 time_window: int,
                 smoothing_method: str,
                 initial_import_multi_index: bool = False):

        super().__init__(config_path=config_path, read_video_info=False)
        if os.path.isdir(input_path):
            self.files_found = glob.glob(input_path + '/*' + self.file_type)
            self.input_dir = input_path
            check_if_filepath_list_is_empty(filepaths=self.files_found, error_msg=f"SIMBA ERROR: {self.input_dir} does not contain any {self.file_type} files.")
        else:
            self.files_found = [input_path]
            self.input_dir = os.path.dirname(input_path)
        if not initial_import_multi_index:
            self.save_dir = os.path.join(self.input_dir, f'Pre_{smoothing_method}_interpolation_{self.datetime}')
            if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.time_window, self.initial_import_multi_index = time_window, initial_import_multi_index
        if smoothing_method == Methods.SAVITZKY_GOLAY.value:
            self.savgol_smoother()
        elif smoothing_method == Methods.GAUSSIAN.value:
            self.gaussian_smoother()
        self.timer.stop_timer()
        stdout_success(msg=f'{str(len(self.files_found))} data file(s) smoothened', elapsed_time=self.timer.elapsed_time_str)

    def savgol_smoother(self):
        for file_path in self.files_found:
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            df = read_df(file_path=file_path, file_type=self.file_type, check_multiindex=self.initial_import_multi_index)
            video_path = find_video_of_file(video_dir=self.video_dir, filename=video_name)
            if not video_path:
                raise NoFilesFoundError(msg=f'No video for file {video_name} found in SimBA project. Import the video before doing smoothing. To perform smoothing, SimBA needs the video in order to read the video FPS.')
            video_meta_data = get_video_meta_data(video_path=video_path)
            frames_in_time_window = int(self.time_window / (1000 / video_meta_data['fps']))
            if (frames_in_time_window % 2) == 0:
                frames_in_time_window = frames_in_time_window - 1
            if (frames_in_time_window % 2) <= 3:
                frames_in_time_window = 5
            for c in df.columns:
                df[c] = savgol_filter(x=df[c].to_numpy(), window_length=frames_in_time_window, polyorder=3, mode='nearest')
            if not self.initial_import_multi_index:
                shutil.move(src=file_path, dst=os.path.join(self.save_dir, os.path.basename(file_path)))
            else:
                multi_idx_header = []
                for i in range(len(df.columns)): multi_idx_header.append(('IMPORTED_POSE', 'IMPORTED_POSE', list(df.columns)[i]))
                df.columns = pd.MultiIndex.from_tuples(multi_idx_header)
            write_df(df=df, file_type=self.file_type, save_path=file_path, multi_idx_header=self.initial_import_multi_index)
            video_timer.stop_timer()
            print(f'Video {video_name} smoothed (Savitzky Golay: {str(self.time_window)}ms) (elapsed time {video_timer.elapsed_time_str})...')

    def gaussian_smoother(self):
        for file_path in self.files_found:
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            df = read_df(file_path=file_path, file_type=self.file_type, check_multiindex=self.initial_import_multi_index)
            video_path = find_video_of_file(video_dir=self.video_dir, filename=video_name)
            if not video_path:
                raise NoFilesFoundError(msg=f'No video for file {video_name} found in SimBA project. Import the video before doing Gaussian smoothing. To perform smoothing, SimBA needs the video in order to read the video FPS.')
            video_meta_data = get_video_meta_data(video_path=video_path)
            frames_in_time_window = int(self.time_window / (1000 / video_meta_data['fps']))
            for c in df.columns:
                df[c] = df[c].rolling(window=int(frames_in_time_window), win_type='gaussian', center=True).mean(std=5).fillna(df[c]).abs()
            if not self.initial_import_multi_index:
                shutil.move(src=file_path, dst=os.path.join(self.save_dir, os.path.basename(file_path)))
            else:
                multi_idx_header = []
                for i in range(len(df.columns)): multi_idx_header.append(('IMPORTED_POSE', 'IMPORTED_POSE', list(df.columns)[i]))
                df.columns = pd.MultiIndex.from_tuples(multi_idx_header)
            write_df(df=df, file_type=self.file_type, save_path=file_path, multi_idx_header=self.initial_import_multi_index)
            video_timer.stop_timer()
            print(f'Video {video_name} smoothed (Gaussian: {str(self.time_window)}ms) (elapsed time {video_timer.elapsed_time_str})...')



# Interpolate(input_path='/Users/simon/Desktop/envs/troubleshooting/DLC_2_Black_animals/project_folder/csv/input_csv',
#             config_path='/Users/simon/Desktop/envs/troubleshooting/DLC_2_Black_animals/project_folder/project_config.ini',
#             method='Animal(s):Quadratic',
#             initial_import_multi_index=True)

# PostHocInterpolate(config_path='/Users/simon/Desktop/envs/troubleshooting/ddddfff/project_folder/project_config.ini',
#                    input_dir='/Users/simon/Desktop/envs/troubleshooting/ddddfff/project_folder/csv/outlier_corrected_movement_location',
#                    method='Animal(s): Nearest')


# Smooth(config_path='/Users/simon/Desktop/envs/troubleshooting/DLC_2_Black_animals/project_folder/project_config.ini',
#               input_path='/Users/simon/Desktop/envs/troubleshooting/DLC_2_Black_animals/project_folder/csv/input_csv',
#               time_window=100,
#               smoothing_method=Methods.SAVITZKY_GOLAY.value,
#               initial_import_multi_index=True)




# class PostHocSmooth(ConfigReader):
#     def __init__(self,
#                  config_path: str,
#                  input_dir: str,
#                  time_window: int,
#                  smoothing_method: str):