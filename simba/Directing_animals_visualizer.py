__author__ = "Simon Nilsson", "JJ Choong"

from simba.Directing_animals_analyzer import DirectingOtherAnimalsAnalyzer
from simba.read_config_unit_tests import read_config_entry, check_that_column_exist, read_config_file
import os, glob
from simba.drop_bp_cords import get_fn_ext, getBpNames, createColorListofList, create_body_part_dictionary
from simba.misc_tools import find_video_of_file, get_video_meta_data, check_multi_animal_status
from simba.features_scripts.unit_tests import read_video_info_csv
import cv2
import numpy as np
from simba.rw_dfs import read_df


class DirectingOtherAnimalsVisualizer(object):
    """
    Class for visualizing when animals are directing towards body-parts of other animals.

    > Note: Requires the pose-estimation data for the left ear, right ears and nose of individual animals.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    -----
    Requires the pose-estimation data for the ``left ear``, ``right ear`` and ``nose`` of
    each individual animals. `YouTube example of expected output <https://youtu.be/4WXs3sKu41I>`__.

    Examples
    -----
    >>> directing_visualizer = DirectingOtherAnimalsVisualizer(config_path='MyProjectConfig')
    >>> directing_visualizer.visualize_results()
    """

    def __init__(self,
                 config_path: str):

        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.direction_analyzer = DirectingOtherAnimalsAnalyzer(config_path=config_path)
        self.direction_analyzer.process_directionality()
        self.direction_analyzer.create_directionality_dfs()
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.color_lst_of_lst = createColorListofList(self.no_animals, int(len(self.x_cols) + 1))
        self.data_in_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.data_dict = self.direction_analyzer.directionality_df_dict
        self.video_directory = os.path.join(self.project_path, 'videos')
        self.save_directory = os.path.join(self.project_path, 'frames', 'output', 'ROI_directionality_visualize')
        if not os.path.exists(self.save_directory): os.makedirs(self.save_directory)
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, self.x_cols, self.y_cols, [], self.color_lst_of_lst)
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        print('Processing {} video(s)...'.format(str(len(self.files_found))))

    def visualize_results(self):
        """
        Method to create directionality videos. Results are stored in
        the `project_folder/frames/output/ROI_directionality_visualize` directory of the SimBA project

        Returns
        ----------
        None
        """

        for file_cnt, file_path in enumerate(self.files_found):
            self.file_cnt = file_cnt
            _, self.video_name, _ = get_fn_ext(file_path)
            self.data_df = read_df(file_path, file_type=self.file_type)
            self.save_path = os.path.join(self.save_directory, self.video_name + '.mp4')
            video_path = find_video_of_file(self.video_directory, self.video_name)
            self.cap = cv2.VideoCapture(video_path)
            self.video_meta_data = get_video_meta_data(video_path)
            self.video_data = self.data_dict[self.video_name]
            self.writer = cv2.VideoWriter(self.save_path, self.fourcc, self.video_meta_data['fps'], (self.video_meta_data['width'], self.video_meta_data['height']))
            if self.video_name in list(self.video_data['Video']):
                self.__create_video()
            else:
                print('SimBA skipping video {}: No animals are directing each other in the video.')
        print('SIMBA COMPLETE: All directionality visualizations saved in {}'.format(self.save_directory))

    def __create_video(self):
        img_cnt = 0
        while (self.cap.isOpened()):
            ret, img = self.cap.read()
            if ret:
                bp_data = self.data_df.iloc[img_cnt]
                for cnt, (animal_name, animal_bps) in enumerate(self.animal_bp_dict.items()):
                    for bp in zip(animal_bps['X_bps'], animal_bps['Y_bps']):
                        x_bp, y_bp = bp_data[bp[0]], bp_data[bp[1]]
                        cv2.circle(img, (int(x_bp), int(y_bp)), 3, self.animal_bp_dict[animal_name]['colors'][cnt], 3)

                if img_cnt in list(self.video_data['Frame_#'].unique()):
                    img_data = self.video_data[self.video_data['Frame_#'] == img_cnt]
                    unique_animals = img_data['Animal_1'].unique()
                    for animal in unique_animals:
                        animal_img_data = img_data[img_data['Animal_1'] == animal].reset_index(drop=True)
                        clr_list = self.animal_bp_dict[animal]['colors']
                        for i, r in animal_img_data.iterrows():
                            cv2.line(img, (int(r['Eye_x']), int(r['Eye_y'])), (int(r['Animal_2_bodypart_x']), int(r['Animal_2_bodypart_y'])), clr_list[i], 4)
                img = np.uint8(img)
                img_cnt += 1
                self.writer.write(img)
                print('Frame: {} / {}. Video: {} ({}/{})'.format(str(img_cnt), str(self.video_meta_data['frame_count']),
                                                                 self.video_name, str(self.file_cnt + 1),
                                                                 len(self.files_found)))
            else:
                self.cap.release()
                self.writer.release()
                print('Directionality video {} saved in {} directory ...'.format(self.video_name, self.save_directory))


# test = DirectingOtherAnimalsVisualizer(config_path='/Users/simon/Desktop/train_model_project/project_folder/project_config.ini')
# test.visualize_results()
