from datetime import datetime
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          read_project_path_and_file_type)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
import os
import pandas as pd
from simba.ROI_analyzer import ROIAnalyzer
from simba.misc_tools import (get_fn_ext,
                              SimbaTimer)
from simba.enums import ReadConfig, Paths, DirNames, Dtypes
from simba.rw_dfs import read_df
import itertools

class ROITimebinCalculator(object):
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
    >>> roi_time_bin_calculator.analyze_time_bins()
    >>> roi_time_bin_calculator.save_results()
    """

    def __init__(self,
                 config_path: str,
                 bin_length: int):

        self.config_path, self.bin_length = config_path, bin_length
        self.date_time = datetime.now().strftime('%Y%m%d%H%M%S')
        self.config = read_config_file(config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.roi_animal_cnt = read_config_entry(config=self.config, section=ReadConfig.ROI_SETTINGS.value, option=ReadConfig.ROI_ANIMAL_CNT.value, data_type=Dtypes.INT.value)
        self.probability_threshold = read_config_entry(config=self.config, section=ReadConfig.ROI_SETTINGS.value, option=ReadConfig.PROBABILITY_THRESHOLD.value, data_type=Dtypes.FLOAT.value,default_value=0.00)
        self.logs_path = os.path.join(self.project_path, 'logs')
        self.save_path_time = os.path.join(self.logs_path, 'ROI_time_bins_{}s_time_data_{}.csv'.format(str(bin_length), self.date_time))
        self.save_path_entries = os.path.join(self.logs_path, 'ROI_time_bins_{}s_entry_data_{}.csv'.format(str(bin_length), self.date_time))
        self.video_info_df = read_video_info_csv(file_path=os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.roi_analyzer = ROIAnalyzer(ini_path=self.config_path, data_path=DirNames.OUTLIER_MOVEMENT_LOCATION.value, calculate_distances=False)
        self.roi_analyzer.read_roi_dfs()
        self.roi_analyzer.analyze_ROIs()
        self.shape_names = self.roi_analyzer.shape_names
        self.animal_names = list(self.roi_analyzer.bp_dict.keys())
        self.entries_exits_df = pd.concat(self.roi_analyzer.entry_exit_df_lst, axis=0)
        self.timer = SimbaTimer()
        self.timer.start_timer()

    def analyze_time_bins(self):
        self.files_found = self.roi_analyzer.files_found
        self.out_dict_time, self.out_dict_entries = {}, {}
        print('Analyzing {} videos...'.format(str(len(self.files_found))))
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            self.out_dict_time[self.video_name], self.out_dict_entries[self.video_name] = {}, {}
            _, _, fps = read_video_info(vid_info_df=self.video_info_df, video_name=self.video_name)
            frames_per_bin = int(fps * self.bin_length)
            video_frms = list(range(0, len(read_df(file_path=file_path, file_type=self.file_type))))
            frame_bins = [video_frms[i * frames_per_bin:(i + 1) * frames_per_bin] for i in range((len(video_frms) + frames_per_bin - 1) // frames_per_bin )]
            self.video_data = self.entries_exits_df[self.entries_exits_df['Video'] == self.video_name]
            for animal_name, shape_name in list(itertools.product(self.animal_names, self.shape_names)):
                self.results_time, self.results_entries = {}, {}
                self.results_time[shape_name], self.results_entries[shape_name] = {}, {}
                self.results_time[shape_name][animal_name], self.results_entries[shape_name][animal_name] = {}, {}
                data_df = self.video_data.loc[(self.video_data['Shape'] == shape_name) & (self.video_data['Animal'] == animal_name)]
                entry_frms = list(data_df['Entry_times'])
                inside_shape_frms = [list(range(x, y)) for x, y in zip(list(data_df['Entry_times'].astype(int)), list(data_df['Exit_times'].astype(int) + 1))]
                inside_shape_frms = [i for s in inside_shape_frms for i in s]
                for bin_cnt, bin_frms in enumerate(frame_bins):
                    frms_inside_roi_in_timebin = [x for x in inside_shape_frms if x in bin_frms]
                    entry_roi_in_timebin = [x for x in entry_frms if x in bin_frms]
                    self.results_time[shape_name][animal_name][bin_cnt] = len(frms_inside_roi_in_timebin) / fps
                    self.results_entries[shape_name][animal_name][bin_cnt] = len(entry_roi_in_timebin)
                self.out_dict_time[self.video_name].update(self.results_time)
                self.out_dict_entries[self.video_name].update(self.results_entries)

    def save_results(self):

        results_time_df = pd.DataFrame(columns=['Video', 'Shape', 'Animal', 'Time bin', 'Time inside shape (s)'])
        results_entries_df = pd.DataFrame(columns=['Video', 'Shape', 'Animal', 'Time bin', 'Entry count'])
        for video_name, video_data in self.out_dict_time.items():
            for shape_name, shape_data in video_data.items():
                for animal_name, animal_data in shape_data.items():
                    for bin_name, bin_data in animal_data.items():
                        results_time_df.loc[len(results_time_df)] = [video_name, shape_name, animal_name, bin_name, bin_data]
        results_time_df.to_csv(self.save_path_time)
        print('SIMBA COMPLETE: ROI time bin time data saved at {}'.format(self.save_path_time))
        for video_name, video_data in self.out_dict_entries.items():
            for shape_name, shape_data in video_data.items():
                for animal_name, animal_data in shape_data.items():
                    for bin_name, bin_data in animal_data.items():
                        results_entries_df.loc[len(results_entries_df)] = [video_name, shape_name, animal_name, bin_name, bin_data]
        results_entries_df.to_csv(self.save_path_entries)
        self.timer.stop_timer()
        print('SIMBA COMPLETE: ROI time bin entry data saved at {} (elapsed time {}s)'.format(self.save_path_entries, self.timer.elapsed_time_str))


# test = ROITimebinCalculator(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',bin_length=5)
# test.analyze_time_bins()
# test.save_results()