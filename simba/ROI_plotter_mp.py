__author__ = "Simon Nilsson", "JJ Choong"

from simba.ROI_analyzer import ROIAnalyzer
from simba.drop_bp_cords import get_fn_ext, createColorListofList
from simba.features_scripts.unit_tests import read_video_info
import pandas as pd
import os
import itertools
import cv2
import platform
import multiprocessing
import functools
from simba.enums import ReadConfig, Formats, Paths, Defaults
from simba.misc_tools import (get_video_meta_data,
                              add_missing_ROI_cols,
                              SimbaTimer,
                              detect_bouts,
                              concatenate_videos_in_folder)
import numpy as np

pd.options.mode.chained_assignment = None
def _img_creator(data: pd.DataFrame,
                 loc_dict: dict,
                 scalers: dict,
                 video_meta_data: dict,
                 save_temp_directory: str,
                 shape_meta_data: dict,
                 video_shape_names: list,
                 input_video_path: str,
                 body_part_dict: dict,
                 roi_analyzer_data: object,
                 colors: list,
                 style_attr: dict):

    def __insert_texts(shape_df):
        for animal_name in roi_analyzer_data.multiAnimalIDList:
            for _, shape in shape_df.iterrows():
                shape_name, shape_color = shape['Name'], shape['Color BGR']
                cv2.putText(border_img, loc_dict[animal_name][shape_name]['timer_text'], loc_dict[animal_name][shape_name]['timer_text_loc'], font, scalers['font_size'], shape_color, 1)
                cv2.putText(border_img, loc_dict[animal_name][shape_name]['entries_text'], loc_dict[animal_name][shape_name]['entries_text_loc'], font, scalers['font_size'], shape_color, 1)

        return border_img

    font = cv2.FONT_HERSHEY_TRIPLEX
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    group_cnt = int(data['group'].values[0])
    start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
    save_path = os.path.join(save_temp_directory, '{}.mp4'.format(str(group_cnt)))
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'], (video_meta_data['width'] * 2, video_meta_data['height']))
    cap = cv2.VideoCapture(input_video_path)
    cap.set(1, start_frm)

    while current_frm < end_frm:
        ret, img = cap.read()
        border_img = cv2.copyMakeBorder(img, 0, 0, 0, int(video_meta_data['width']), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        border_img = __insert_texts(roi_analyzer_data.video_recs)
        border_img = __insert_texts(roi_analyzer_data.video_circs)
        border_img = __insert_texts(roi_analyzer_data.video_polys)

        for _, row in roi_analyzer_data.video_recs.iterrows():
            top_left_x, top_left_y, shape_name = row['topLeftX'], row['topLeftY'], row['Name']
            bottom_right_x, bottom_right_y = row['Bottom_right_X'], row['Bottom_right_Y']
            thickness, color = row['Thickness'], row['Color BGR']
            cv2.rectangle(border_img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)

        for _, row in roi_analyzer_data.video_circs.iterrows():
            center_x, center_y, radius, shape_name = row['centerX'], row['centerY'], row['radius'], row['Name']
            thickness, color = row['Thickness'], row['Color BGR']
            cv2.circle(border_img, (center_x, center_y), radius, color, thickness)

        for _, row in roi_analyzer_data.video_polys.iterrows():
            vertices, shape_name = row['vertices'], row['Name']
            thickness, color = row['Thickness'], row['Color BGR']
            cv2.polylines(border_img, [vertices], True, color, thickness=thickness)

        for animal_cnt, animal_name in enumerate(roi_analyzer_data.multiAnimalIDList):
            if style_attr['Show_body_part'] or style_attr['Show_animal_name']:
                bp_data = data.loc[current_frm, body_part_dict[animal_name]].values
                if roi_analyzer_data.p_thresh < bp_data[2]:
                    if style_attr['Show_body_part']:
                        cv2.circle(border_img, (int(bp_data[0]), int(bp_data[1])), scalers['circle_size'], colors[animal_cnt], -1)
                    if style_attr['Show_animal_name']:
                        cv2.putText(border_img, animal_name, (int(bp_data[0]), int(bp_data[1])), font, scalers['font_size'], colors[animal_cnt], 1)

            for shape_name in video_shape_names:
                timer = round(data.loc[current_frm, '{}_{}_cum_sum_time'.format(animal_name, shape_name)], 2)
                entries = data.loc[current_frm, '{}_{}_cum_sum_entries'.format(animal_name, shape_name)]
                cv2.putText(border_img, str(timer), loc_dict[animal_name][shape_name]['timer_data_loc'], font, scalers['font_size'], shape_meta_data[shape_name]['Color BGR'], 1)
                cv2.putText(border_img, str(entries), loc_dict[animal_name][shape_name]['entries_data_loc'], font, scalers['font_size'], shape_meta_data[shape_name]['Color BGR'], 1)

        writer.write(border_img)
        current_frm += 1
        print('Multi-processing video frame {} on core {}...'.format(str(current_frm), str(group_cnt)))

    cap.release()
    writer.release()

    return group_cnt



class ROIPlotMultiprocess(object):
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
            core_cnt: int,
            style_attr: dict):


        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        self.roi_analyzer = ROIAnalyzer(ini_path=ini_path, data_path='outlier_corrected_movement_location')
        self.video_dir_path = os.path.join(self.roi_analyzer.project_path, 'videos')
        self.roi_analyzer.read_roi_dfs()
        self.video_path = os.path.join(self.video_dir_path, video_path)
        _, self.video_name, _ = get_fn_ext(video_path)
        self.core_cnt = core_cnt
        self.maxtasksperchild = Defaults.MAX_TASK_PER_CHILD.value
        self.chunksize = Defaults.CHUNK_SIZE.value
        self.roi_analyzer.files_found = [os.path.join(self.roi_analyzer.input_folder, self.video_name + '.' + self.roi_analyzer.file_type)]
        if not os.path.isfile(self.roi_analyzer.files_found[0]):
            print('SIMBA ERROR: Could not find the file at path {}. Please make sure you have corrected body-part outliers or indicated that you want to skip outlier correction'.format(self.roi_analyzer.files_found[0]))
            raise FileNotFoundError()
        self.roi_analyzer.analyze_ROIs()
        self.roi_entries_df = pd.concat(self.roi_analyzer.entry_exit_df_lst, axis=0)
        self.data_df, self.style_attr = self.roi_analyzer.data_df, style_attr
        self.out_parent_dir = os.path.join(self.roi_analyzer.project_path, Paths.ROI_ANALYSIS.value)
        if not os.path.exists(self.out_parent_dir): os.makedirs(self.out_parent_dir)
        self.video_save_path = os.path.join(self.out_parent_dir, self.video_name + '.mp4')
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
                self.loc_dict[animal_name][shape]['timer_text_loc'] = ((self.video_meta_data['width'] + 5), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.scalers['space_size'] * add_spacer))
                self.loc_dict[animal_name][shape]['timer_data_loc'] = (int(self.border_img_w-(self.border_img_w/8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.scalers['space_size']*add_spacer))
                add_spacer += 1
                self.loc_dict[animal_name][shape]['entries_text_loc'] = ((self.video_meta_data['width'] + 5), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.scalers['space_size'] * add_spacer))
                self.loc_dict[animal_name][shape]['entries_data_loc'] = (int(self.border_img_w - (self.border_img_w / 8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.scalers['space_size'] * add_spacer))
                add_spacer += 1



    def __create_counters(self):
        self.cnt_dict = {}
        for animal_cnt, animal_name in enumerate(self.roi_analyzer.multiAnimalIDList):
            self.cnt_dict[animal_name] = {}
            for shape in self.video_shapes:
                self.cnt_dict[animal_name][shape] = {}
                self.cnt_dict[animal_name][shape]['timer'] = 0
                self.cnt_dict[animal_name][shape]['entries'] = 0
                self.cnt_dict[animal_name][shape]['entry_status'] = False


    def __calculate_cumulative(self):
        for animal in self.roi_analyzer.multiAnimalIDList:
            for shape in self.video_shapes:
                self.data_df['{}_{}_cum_sum_time'.format(animal, shape)] = self.data_df['{}_{}'.format(animal, shape)].cumsum() / self.fps
                roi_bouts = list(detect_bouts(data_df=self.data_df, target_lst=['{}_{}'.format(animal, shape)], fps=self.fps)['Start_frame'])
                self.data_df['{}_{}_entry'.format(animal, shape)] = 0
                self.data_df.loc[roi_bouts, '{}_{}_entry'.format(animal, shape)] = 1
                self.data_df['{}_{}_cum_sum_entries'.format(animal, shape)] = self.data_df['{}_{}_entry'.format(animal, shape)].cumsum()


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

        video_timer = SimbaTimer()
        video_timer.start_timer()
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_meta_data = get_video_meta_data(self.video_path)
        video_settings, pix_per_mm, self.fps = read_video_info(self.roi_analyzer.vid_info_df, self.video_name)
        self.space_scale, radius_scale, res_scale, font_scale = 25, 10, 1500, 0.8
        max_dim = max(self.video_meta_data['width'], self.video_meta_data['height'])
        self.scalers = {}
        self.scalers['circle_size'], self.scalers['font_size'] = int(radius_scale / (res_scale / max_dim)), float(font_scale / (res_scale / max_dim))
        self.scalers['space_size'] = int(self.space_scale / (res_scale / max_dim))
        color_lst = createColorListofList(self.roi_analyzer.no_animals, int((len(self.roi_analyzer.bp_columns) / 3)))[0]
        self.temp_folder = os.path.join(self.out_parent_dir, self.video_name, 'temp')
        if not os.path.exists(self.temp_folder): os.makedirs(self.temp_folder)
        self.__update_video_meta_data()
        self.__calc_text_locs()
        self.__create_counters()
        self.__calculate_cumulative()

        data_arr = np.array_split(self.data_df, self.core_cnt)
        for df_cnt in range(len(data_arr)):
            data_arr[df_cnt]['group'] = df_cnt
        frm_per_core = len(data_arr[0])

        print('Creating ROI images, multiprocessing (determined chunksize: {}, cores: {})...'.format(str(self.chunksize), str(self.core_cnt)))
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(_img_creator,
                                          loc_dict=self.loc_dict,
                                          scalers=self.scalers,
                                          video_meta_data=self.video_meta_data,
                                          save_temp_directory= self.temp_folder,
                                          body_part_dict=self.bp_dict,
                                          input_video_path=self.video_path,
                                          roi_analyzer_data=self.roi_analyzer,
                                          video_shape_names=self.video_shapes,
                                          colors=color_lst,
                                          shape_meta_data=self.shape_dicts,
                                          style_attr=self.style_attr)
            for cnt, result in enumerate(pool.imap(constants, data_arr, chunksize=self.chunksize)):
                print('Image {}/{}, Video {}...'.format(str(int(frm_per_core * (result + 1))), str(len(self.data_df)), self.video_name))
            print('Joining {} multiprocessed video...'.format(self.video_name))
            concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.video_save_path, video_format='mp4')

            video_timer.stop_timer()
            pool.terminate()
            pool.join()
            print('SIMBA COMPLETE: Video {} created. Video saved in project_folder/frames/output/ROI_analysis (elapsed time: {}s).'.format(self.video_name, video_timer.elapsed_time_str))

# test = ROIPlot(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini',
#                video_path="termite_test.mp4",
#                core_cnt=5,
#                style_attr={'Show_body_part': True, 'Show_animal_name': True})
# test.insert_data()
# test.visualize_ROI_data()

# test = ROIPlot(ini_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', video_path=r"Together_1.avi")
# test.insert_data()
# test.visualize_ROI_data()

# test = ROIPlot(ini_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\ROI_2_animals\project_folder\project_config.ini", video_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\ROI_2_animals\project_folder\videos\Video7.mp4")
# test.insert_data()
# test.visualize_ROI_data()