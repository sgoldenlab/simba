import os
import pandas as pd
import itertools
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
from simba.utils.printing import stdout_success
from simba.utils.enums import ConfigKey, DirNames, Dtypes
from simba.mixins.config_reader import ConfigReader
from simba.utils.read_write import get_fn_ext, read_config_entry, read_df

class ROITimebinCalculator(ConfigReader):
    """
    Class for calulating how much time and how many entries animals are making into user-defined ROIs
    in user-defined time bins. Results are stored in the `project_folder/logs` directory of the SimBA project.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    bin_length: int
        length of time bins in seconds.

    Notes
    ----------

    Examples
    ----------
    >>> roi_time_bin_calculator = ROITimebinCalculator(config_path='MySimBaConfigPath', bin_length=15)
    >>> roi_time_bin_calculator.run()
    >>> roi_time_bin_calculator.save_results()
    """

    def __init__(self,
                 config_path: str,
                 bin_length: int):

        ConfigReader.__init__(self, config_path=config_path)
        self.bin_length = bin_length
        self.roi_animal_cnt = read_config_entry(config=self.config, section=ConfigKey.ROI_SETTINGS.value, option=ConfigKey.ROI_ANIMAL_CNT.value, data_type=Dtypes.INT.value)
        self.probability_threshold = read_config_entry(config=self.config, section=ConfigKey.ROI_SETTINGS.value, option=ConfigKey.PROBABILITY_THRESHOLD.value, data_type=Dtypes.FLOAT.value, default_value=0.00)
        self.save_path_time = os.path.join(self.logs_path, 'ROI_time_bins_{}s_time_data_{}.csv'.format(str(bin_length), self.datetime))
        self.save_path_entries = os.path.join(self.logs_path, 'ROI_time_bins_{}s_entry_data_{}.csv'.format(str(bin_length), self.datetime))
        self.roi_analyzer = ROIAnalyzer(ini_path=self.config_path, data_path=DirNames.OUTLIER_MOVEMENT_LOCATION.value, calculate_distances=False)
        self.roi_analyzer.run()
        self.shape_names = self.roi_analyzer.shape_names
        self.animal_names = list(self.roi_analyzer.bp_dict.keys())
        self.entries_exits_df = self.roi_analyzer.detailed_df

    def run(self):
        self.files_found = self.roi_analyzer.files_found
        self.out_dict_time, self.out_dict_entries = {}, {}
        print('Analyzing time-bin data for {} videos...'.format(str(len(self.files_found))))
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            self.out_dict_time[self.video_name], self.out_dict_entries[self.video_name] = {}, {}
            _, _, fps = self.read_video_info(video_name=self.video_name)
            frames_per_bin = int(fps * self.bin_length)
            video_frms = list(range(0, len(read_df(file_path=file_path, file_type=self.file_type))))
            frame_bins = [video_frms[i * frames_per_bin:(i + 1) * frames_per_bin] for i in range((len(video_frms) + frames_per_bin - 1) // frames_per_bin )]
            self.video_data = self.entries_exits_df[self.entries_exits_df['VIDEO'] == self.video_name]
            for animal_name, shape_name in list(itertools.product(self.animal_names, self.shape_names)):
                self.results_time, self.results_entries = {}, {}
                self.results_time[shape_name], self.results_entries[shape_name] = {}, {}
                self.results_time[shape_name][animal_name], self.results_entries[shape_name][animal_name] = {}, {}
                data_df = self.video_data.loc[(self.video_data['SHAPE'] == shape_name) & (self.video_data['ANIMAL'] == animal_name)]
                entry_frms = list(data_df['ENTRY FRAMES'])
                inside_shape_frms = [list(range(x, y)) for x, y in zip(list(data_df['ENTRY FRAMES'].astype(int)), list(data_df['EXIT FRAMES'].astype(int) + 1))]
                inside_shape_frms = [i for s in inside_shape_frms for i in s]
                for bin_cnt, bin_frms in enumerate(frame_bins):
                    frms_inside_roi_in_timebin = [x for x in inside_shape_frms if x in bin_frms]
                    entry_roi_in_timebin = [x for x in entry_frms if x in bin_frms]
                    self.results_time[shape_name][animal_name][bin_cnt] = len(frms_inside_roi_in_timebin) / fps
                    self.results_entries[shape_name][animal_name][bin_cnt] = len(entry_roi_in_timebin)
                self.out_dict_time[self.video_name].update(self.results_time)
                self.out_dict_entries[self.video_name].update(self.results_entries)

    def save_results(self):
        results_time_df = pd.DataFrame(columns=['VIDEO', 'SHAPE', 'ANIMAL', 'TIME BIN', 'TIME INSIDE SHAPE (S)'])
        results_entries_df = pd.DataFrame(columns=['VIDEO', 'SHAPE', 'ANIMAL', 'TIME BIN', 'ENTRY COUNT'])
        for video_name, video_data in self.out_dict_time.items():
            for shape_name, shape_data in video_data.items():
                for animal_name, animal_data in shape_data.items():
                    for bin_name, bin_data in animal_data.items():
                        results_time_df.loc[len(results_time_df)] = [video_name, shape_name, animal_name, bin_name, bin_data]
        results_time_df['TIME INSIDE SHAPE (S)'] = results_time_df['TIME INSIDE SHAPE (S)'].round(6)
        results_time_df.to_csv(self.save_path_time)
        stdout_success(msg=f'ROI time bin time data saved at {self.save_path_time}')
        for video_name, video_data in self.out_dict_entries.items():
            for shape_name, shape_data in video_data.items():
                for animal_name, animal_data in shape_data.items():
                    for bin_name, bin_data in animal_data.items():
                        results_entries_df.loc[len(results_entries_df)] = [video_name, shape_name, animal_name, bin_name, bin_data]
        results_entries_df.to_csv(self.save_path_entries)
        self.timer.stop_timer()
        stdout_success(msg=f'ROI time bin entry data saved at {self.save_path_entries}', elapsed_time=self.timer.elapsed_time_str)


# test = ROITimebinCalculator(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',bin_length=5)
# test.analyze_time_bins()
# test.save_results()


# test = ROITimebinCalculator(config_path=r"/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini",bin_length=5)
# test.run()
# test.save_results()