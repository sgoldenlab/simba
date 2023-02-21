__author__ = "Simon Nilsson", "JJ Choong"

from simba.ROI_analyzer import ROIAnalyzer
from simba.drop_bp_cords import get_fn_ext, createColorListofList
from simba.features_scripts.unit_tests import read_video_info
import pandas as pd
import os
import itertools
import cv2
from simba.enums import Paths, Formats
from simba.misc_tools import (get_video_meta_data,
                              add_missing_ROI_cols,
                              SimbaTimer)
import numpy as np


class ROIPlot(object):
    """
    Class for visualizing the ROI data (number of entries/exits, time-spent-in etc)

    Parameters
    ----------
    config_path: str
        Path to SimBA project config file in Configparser format
    video_path: str
        Name of video to create ROI visualizations for

    Notes
    ----------
    `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    Examples
    ----------
    >>> roi_visualizer = ROIPlot(ini_path=r'MyProjectConfig', video_path="MyVideo.mp4")
    >>> roi_visualizer.insert_data()
    >>> roi_visualizer.visualize_ROI_data()
    """

    def __init__(
            self,
            ini_path: str,
            video_path: str,
            style_attr: dict):

        self.roi_analyzer = ROIAnalyzer(ini_path=ini_path, data_path='outlier_corrected_movement_location')
        self.video_dir_path = os.path.join(self.roi_analyzer.project_path, 'videos')
        self.roi_analyzer.read_roi_dfs()
        self.video_path = os.path.join(self.video_dir_path, video_path)
        _, self.video_name, _ = get_fn_ext(video_path)
        self.roi_analyzer.files_found = [os.path.join(self.roi_analyzer.input_folder, self.video_name + '.' + self.roi_analyzer.file_type)]
        if not os.path.isfile(self.roi_analyzer.files_found[0]):
            print('SIMBA ERROR: Could not find the file at path {}. Please make sure you have corrected body-part outliers or indicated that you want to skip outlier correction'.format(self.roi_analyzer.files_found[0]))
            raise FileNotFoundError()
        self.roi_analyzer.analyze_ROIs()
        self.roi_entries_df = pd.concat(self.roi_analyzer.entry_exit_df_lst, axis=0)
        self.data_df, self.style_attr = self.roi_analyzer.data_df, style_attr
        self.video_shapes = list(itertools.chain(self.roi_analyzer.video_recs['Name'].unique(), self.roi_analyzer.video_circs['Name'].unique(),self.roi_analyzer.video_polys['Name'].unique()))
        if len(list(set(self.video_shapes))) != len(self.video_shapes):
            print('SIMBA ERROR: Some SIMBA ROI shapes have identical names. Please use unique names to visualize ROI data.')
            raise AttributeError()
        self.roi_analyzer.video_recs = add_missing_ROI_cols(self.roi_analyzer.video_recs)
        self.roi_analyzer.video_circs = add_missing_ROI_cols(self.roi_analyzer.video_circs)
        self.roi_analyzer.video_polys = add_missing_ROI_cols(self.roi_analyzer.video_polys)

        self.shape_dicts = {}
        for df in [self.roi_analyzer.video_recs, self.roi_analyzer.video_circs, self.roi_analyzer.video_polys]:
            if not df['Name'].is_unique:
                df = df.drop_duplicates(subset=['Name'], keep='first')
                print('SIMBA WARNING: Some of your ROIs with the same shape has the same names. E.g., you have two rectangles named "My rectangle". SimBA prefers ROI shapes with unique names. SimBA will keep one of the unique shape names and drop the rest.')
            d = df.set_index('Name').to_dict(orient='index')
            self.shape_dicts = {**self.shape_dicts, **d}

        self.shape_columns = []
        for animal in self.roi_analyzer.multiAnimalIDList:
            for shape_name in self.video_shapes:
                self.data_df[animal + '_' + shape_name] = 0
                self.shape_columns.append(animal + '_' + shape_name)
        self.bp_dict = self.roi_analyzer.bp_dict
        self.output_folder = os.path.join(self.roi_analyzer.project_path, Paths.ROI_ANALYSIS.value)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.timer = SimbaTimer()
        self.timer.start_timer()

    def insert_data(self):
        """
        Method to concat ROI and pose-estimation data.

        Returns
        -------
        Attribute: pd.Dataframe
            data_df

        """

        self.roi_entries_dict =self.roi_entries_df[['Animal', 'Shape', 'Entry_times', 'Exit_times']].to_dict(orient='records')
        for entry_dict in self.roi_entries_dict:
            entry, exit = int(entry_dict['Entry_times']), int(entry_dict['Exit_times'])
            entry_dict['frame_range'] = list(range(entry, exit + 1))
            col_name = entry_dict['Animal'] + '_' + entry_dict['Shape']
            self.data_df[col_name][self.data_df.index.isin(entry_dict['frame_range'])] = 1

    def __calc_text_locs(self):
        add_spacer = 2
        self.loc_dict = {}
        for animal_cnt, animal_name in enumerate(self.roi_analyzer.multiAnimalIDList):
            self.loc_dict[animal_name] = {}
            for shape in self.video_shapes:
                self.loc_dict[animal_name][shape] = {}
                self.loc_dict[animal_name][shape]['timer_text'] = '{} {} {}'.format(shape, animal_name, 'timer:')
                self.loc_dict[animal_name][shape]['entries_text'] = '{} {} {}'.format(shape, animal_name, 'entries:')
                self.loc_dict[animal_name][shape]['timer_text_loc'] = ((self.video_meta_data['width'] + 5), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.spacing_scaler * add_spacer))
                self.loc_dict[animal_name][shape]['timer_data_loc'] = (int(self.border_img_w-(self.border_img_w/8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.spacing_scaler*add_spacer))
                add_spacer += 1
                self.loc_dict[animal_name][shape]['entries_text_loc'] = ((self.video_meta_data['width'] + 5), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.spacing_scaler * add_spacer))
                self.loc_dict[animal_name][shape]['entries_data_loc'] = (int(self.border_img_w - (self.border_img_w / 8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.spacing_scaler * add_spacer))
                add_spacer += 1

    def __insert_texts(self, shape_df):
        for animal_name in self.roi_analyzer.multiAnimalIDList:
            for _, shape in shape_df.iterrows():
                shape_name, shape_color = shape['Name'], shape['Color BGR']
                cv2.putText(self.border_img, self.loc_dict[animal_name][shape_name]['timer_text'], self.loc_dict[animal_name][shape_name]['timer_text_loc'], self.font, self.font_size, shape_color, 1)
                cv2.putText(self.border_img, self.loc_dict[animal_name][shape_name]['entries_text'], self.loc_dict[animal_name][shape_name]['entries_text_loc'], self.font, self.font_size, shape_color, 1)

    def __create_counters(self):
        self.cnt_dict = {}
        for animal_cnt, animal_name in enumerate(self.roi_analyzer.multiAnimalIDList):
            self.cnt_dict[animal_name] = {}
            for shape in self.video_shapes:
                self.cnt_dict[animal_name][shape] = {}
                self.cnt_dict[animal_name][shape]['timer'] = 0
                self.cnt_dict[animal_name][shape]['entries'] = 0
                self.cnt_dict[animal_name][shape]['entry_status'] = False

    def __update_video_meta_data(self):
        new_cap = cv2.VideoCapture(self.video_path)
        new_cap.set(1, 1)
        _, img = self.cap.read()
        bordered_img = cv2.copyMakeBorder(img, 0, 0, 0, int(self.video_meta_data['width']), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        self.border_img_h, self.border_img_w = bordered_img.shape[0], bordered_img.shape[1]
        new_cap.release()

    def visualize_ROI_data(self):
        """
        Method to visualize ROI data. Results are stored in the `project_folder/frames/output/ROI_analysis`
        directroy of the SimBA project.

        Returns
        -------
        None
        """

        save_path = os.path.join(self.output_folder, self.video_name + '.avi')
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_meta_data = get_video_meta_data(self.video_path)
        video_settings, pix_per_mm, fps = read_video_info(self.roi_analyzer.vid_info_df, self.video_name)
        self.space_scale, radius_scale, res_scale, font_scale = 25, 10, 1500, 0.8
        max_dim = max(self.video_meta_data['width'], self.video_meta_data['height'])
        self.font = cv2.FONT_HERSHEY_TRIPLEX
        draw_scale, self.font_size = int(radius_scale / (res_scale / max_dim)), float(font_scale / (res_scale / max_dim))
        fourcc = cv2.VideoWriter_fourcc(*Formats.AVI_CODEC.value)
        self.spacing_scaler = int(self.space_scale / (res_scale / max_dim))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (self.video_meta_data['width'] * 2, self.video_meta_data['height']))
        color_lst = createColorListofList(self.roi_analyzer.no_animals, int((len(self.roi_analyzer.bp_columns) / 3)))[0]
        self.__update_video_meta_data()
        self.__calc_text_locs()
        self.__create_counters()

        frame_cnt = 0
        while (self.cap.isOpened()):
            ret, img = self.cap.read()
            try:
                if ret:
                    self.border_img = cv2.copyMakeBorder(img, 0, 0, 0, int(self.video_meta_data['width']), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    self.border_img_h, self.border_img_w = self.border_img.shape[0], self.border_img.shape[1]
                    self.__insert_texts(self.roi_analyzer.video_recs)
                    self.__insert_texts(self.roi_analyzer.video_circs)
                    self.__insert_texts(self.roi_analyzer.video_polys)

                    for _, row in self.roi_analyzer.video_recs.iterrows():
                        top_left_x, top_left_y, shape_name = row['topLeftX'], row['topLeftY'], row['Name']
                        bottom_right_x, bottom_right_y = row['Bottom_right_X'], row['Bottom_right_Y']
                        thickness, color = row['Thickness'], row['Color BGR']
                        cv2.rectangle(self.border_img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)

                    for _, row in self.roi_analyzer.video_circs.iterrows():
                        center_x, center_y, radius, shape_name = row['centerX'], row['centerY'], row['radius'], row['Name']
                        thickness, color = row['Thickness'], row['Color BGR']
                        cv2.circle(self.border_img, (center_x, center_y), radius, color, thickness)

                    for _, row in self.roi_analyzer.video_polys.iterrows():
                        vertices, shape_name = row['vertices'], row['Name']
                        thickness, color = row['Thickness'], row['Color BGR']
                        cv2.polylines(self.border_img, [vertices], True, color, thickness=thickness)

                    for animal_cnt, animal_name in enumerate(self.roi_analyzer.multiAnimalIDList):
                        bp_data = self.data_df.loc[frame_cnt, self.bp_dict[animal_name]].values
                        if self.roi_analyzer.p_thresh < bp_data[2]:
                            if self.style_attr['Show_body_part']:
                                cv2.circle(self.border_img, (int(bp_data[0]), int(bp_data[1])), draw_scale, color_lst[animal_cnt], -1)
                            if self.style_attr['Show_animal_name']:
                                cv2.putText(self.border_img, animal_name, (int(bp_data[0]), int(bp_data[1])), self.font, self.font_size, color_lst[animal_cnt], 1)
                            for shape_name in self.video_shapes:
                                if self.data_df.loc[frame_cnt, animal_name + '_' + shape_name] == 1:
                                    self.cnt_dict[animal_name][shape_name]['timer'] += (1 / fps)
                                    if not self.cnt_dict[animal_name][shape_name]['entry_status']:
                                        self.cnt_dict[animal_name][shape_name]['entry_status'] = True
                                        self.cnt_dict[animal_name][shape_name]['entries'] += 1
                                else:
                                    self.cnt_dict[animal_name][shape_name]['entry_status'] = False

                                cv2.putText(self.border_img, str(round(self.cnt_dict[animal_name][shape_name]['timer'], 2)), self.loc_dict[animal_name][shape_name]['timer_data_loc'], self.font, self.font_size, self.shape_dicts[shape_name]['Color BGR'], 1)
                                cv2.putText(self.border_img, str(self.cnt_dict[animal_name][shape_name]['entries']), self.loc_dict[animal_name][shape_name]['entries_data_loc'], self.font, self.font_size, self.shape_dicts[shape_name]['Color BGR'], 1)

                    writer.write(np.uint8(self.border_img))
                    # cv2.imshow('Window', self.border_img)
                    # key = cv2.waitKey(3000)
                    # if key == 27:
                    #     cv2.destroyAllWindows()
                    print('Frame: {} / {}, Video: {}.'.format(str(frame_cnt), str(self.video_meta_data['frame_count']), self.video_name))
                    frame_cnt += 1

                if img is None:
                    print('Video ' + str(self.video_name) + ' saved at ' + save_path)
                    self.cap.release()
                    break

            except Exception as e:
                writer.release()
                print(e.args)
                print('NOTE: index error / keyerror. Some frames of the video may be missing. Make sure you are running latest version of SimBA with pip install simba-uw-tf-dev')
                break

        writer.release()


# test = ROIPlot(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini', video_path="termite_test.mp4")
# test.insert_data()
# test.visualize_ROI_data()

# test = ROIPlot(ini_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', video_path=r"Together_1.avi")
# test.insert_data()
# test.visualize_ROI_data()

# test = ROIPlot(ini_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\ROI_2_animals\project_folder\project_config.ini", video_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\ROI_2_animals\project_folder\videos\Video7.mp4")
# test.insert_data()
# test.visualize_ROI_data()