from simba.read_config_unit_tests import (read_config_entry, check_int, check_str, insert_default_headers_for_feature_extraction, check_file_exist_and_readable, read_config_file)
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info, check_minimum_roll_windows
import os, glob
from _legacy.feature_methods import (convex_hull_area,
                                     euclid_distance_between_two_body_parts,
                                     aggregate_statistics_of_hull_area,
                                     rolling_window_aggregation,
                                     three_point_angle)

class ExtractFeaturesFrom14bps(object):
    def __init__(self,
                 config_path: str):

        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.data_in_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.save_dir = os.path.join(self.project_path, 'csv', 'features_extracted')
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        roll_windows_values = [2, 5, 6, 7.5, 15]
        self.roll_windows_values = check_minimum_roll_windows(roll_windows_values, self.vid_info_df['fps'].min())
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        self.simba_working_dir = os.getcwd()
        self.in_headers = ["Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y",
                           "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y", "Center_1_p",
                           "Lat_left_1_x", "Lat_left_1_y",
                           "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x",
                           "Tail_base_1_y", "Tail_base_1_p", "Ear_left_2_x", "Ear_left_2_y", "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p",
                           "Nose_2_x", "Nose_2_y", "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p", "Lat_left_2_x"
                           ,"Lat_left_2_y",
                           "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x",
                           "Tail_base_2_y", "Tail_base_2_p"]
        self.mouse_1_headers, self.mouse_2_headers = self.in_headers[0:21], self.in_headers[21:]
        self.mouse_2_p_headers = [x for x in self.mouse_2_headers if x[-2:] == '_p']
        self.mouse_1_p_headers = [x for x in self.mouse_1_headers if x[-2:] == '_p']
        self.mouse_1_headers = [x for x in self.mouse_1_headers if x[-2:] != '_p']
        self.mouse_2_headers = [x for x in self.mouse_2_headers if x[-2:] != '_p']
        print('Extracting features from {} file(s)...'.format(str(len(self.files_found))))

    def extract_features(self):
        for file_cnt, file_path in enumerate(self.files_found):
            roll_windows = []
            _, self.video_name, _ = get_fn_ext(file_path)
            video_settings, self.px_per_mm, fps = read_video_info(self.vid_info_df, self.video_name)
            for window in self.roll_windows_values:
                roll_windows.append(int(fps / window))




extractor = ExtractFeaturesFrom14bps(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')