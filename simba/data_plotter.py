__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
from joblib import Parallel, delayed
from simba.read_config_unit_tests import read_config_entry, check_that_column_exist, read_config_file
import os, glob
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.movement_processor import MovementProcessor
from simba.misc_tools import check_multi_animal_status, get_fn_ext, get_video_meta_data, find_video_of_file, find_core_cnt
from simba.drop_bp_cords import createColorListofList
import numpy as np
import cv2


class DataPlotter(object):
    """
    Class for tabular data visualizations of animal movement and distances in the current frame and their aggregate
    statistics.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    -----

    Examples
    -----
    >>> data_plotter = DataPlotter(config_path='MyConfigPath')
    >>> data_plotter.process_movement()
    >>> data_plotter.create_data_plots()
    """
    def __init__(self,
                 config_path: str,
                 body_part: str=None):

        self.config, self.config_path = read_config_file(config_path), config_path
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.save_dir = os.path.join(self.project_path, 'frames', 'output', 'live_data_table')
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.in_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.video_dir = os.path.join(self.project_path, 'videos')
        self.files_found = glob.glob(self.in_dir + '/*.' + self.file_type)
        self.multi_animal_id_status, self.multi_animal_id_list = check_multi_animal_status(self.config, self.no_animals)
        self.color_lst_of_lst = createColorListofList(self.no_animals, 3)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.cpu_cnt, self.cpu_to_use = find_core_cnt()
        self.__compute_spacings()
        print('Processing {} video(s)...'.format(str(len(self.files_found))))

    def __compute_spacings(self):
        self.loc_dict = {}
        self.loc_dict['Animal'] = (50, 20)
        self.loc_dict['total_movement_header'] = (250, 20)
        self.loc_dict['current_velocity_header'] = (475, 20)
        self.loc_dict['animals'] = {}
        y_cord, x_cord = 75, 15
        for animal_cnt, animal_name in enumerate(self.multi_animal_id_list):
            self.loc_dict['animals'][animal_name] = {}
            self.loc_dict['animals'][animal_name]['index_loc'] = (50, y_cord)
            self.loc_dict['animals'][animal_name]['total_movement_loc'] = (250, y_cord)
            self.loc_dict['animals'][animal_name]['current_velocity_loc'] = (475, y_cord)
            y_cord += 50

    def process_movement(self):
        """
        Method to create movement data for visualization

        Returns
        -------
        Attribute: pd.Dataframe
            movement
        """

        movement_processor = MovementProcessor(config_path=self.config_path)
        movement_processor.process_movement()
        self.movement = movement_processor.movement_dict

    def create_data_plots(self):
        """
        Method to create and save visualizations on disk from data created in
        :meth:`~simba.DataPlotter.process_movement`. Results are stored in the `project_folder/frames/output/live_data_table`.

        Returns
        -------
        None
        """




        def multiprocess_img_creation(video_data_slice=None, location_dict=None, animal_ids=None, video_data=None, color_lst=None):
            img = np.zeros((480, 640, 3))
            cv2.putText(img, 'Animal', location_dict['Animal'], cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, 'Total movement (cm)', location_dict['total_movement_header'], cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, 'Velocity (cm/s)',location_dict['current_velocity_header'], cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
            for animal_cnt, animal_name in enumerate(animal_ids):
                clr = self.color_lst_of_lst[animal_cnt][0]
                total_movement = str(round(video_data[animal_name].iloc[0:video_data_slice.index.max()].sum() / 10, 2))
                current_velocity = str(round(video_data_slice[animal_name].sum() / 10, 2))
                cv2.putText(img, animal_name, location_dict['animals'][animal_name]['index_loc'], cv2.FONT_HERSHEY_TRIPLEX, 0.5, clr, 1)
                cv2.putText(img, total_movement, location_dict['animals'][animal_name]['total_movement_loc'], cv2.FONT_HERSHEY_TRIPLEX, 0.5, clr, 1)
                cv2.putText(img, current_velocity, location_dict['animals'][animal_name]['current_velocity_loc'], cv2.FONT_HERSHEY_TRIPLEX, 0.5, clr, 1)
            return img

        for file_cnt, file_path in enumerate(self.files_found):
            _, video_name, _ = get_fn_ext(file_path)
            video_data = pd.DataFrame(self.movement[video_name])
            video_path = find_video_of_file(video_dir=self.video_dir,filename=video_name)
            video_meta_data = get_video_meta_data(video_path)
            _, _, self.fps = read_video_info(vid_info_df=self.vid_info_df, video_name=video_name)
            video_save_path = os.path.join(self.save_dir, video_name + '.mp4')
            self.writer = cv2.VideoWriter(video_save_path, self.fourcc, self.fps, (640, 480))
            video_data_lst = np.array_split(pd.DataFrame(video_data), int(video_meta_data['frame_count'] / self.fps))
            self.imgs = Parallel(n_jobs=self.cpu_to_use, verbose=1, backend="threading")(delayed(multiprocess_img_creation)(x, self.loc_dict, self.multi_animal_id_list, video_data, self.color_lst_of_lst) for x in video_data_lst)
            frm_cnt = 0
            for img_cnt, img in enumerate(self.imgs):
                for frame_cnt in range(int(self.fps)):
                    self.writer.write(np.uint8(img))
                    frm_cnt += 1
                    print('Frame: {} / {}. Video: {} ({}/{})'.format(str(frm_cnt), str(video_meta_data['frame_count']),
                                                                   video_name, str(file_cnt + 1),
                                                                   len(self.files_found)))

            print('Data tables created for video {}...'.format(video_name))
            self.writer.release()

        print('SIMBA COMPLETE: All data table videos created inside {}'.format(self.save_dir))


# test = DataPlotter(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
# # test.process_movement()
# # test.create_data_plots()