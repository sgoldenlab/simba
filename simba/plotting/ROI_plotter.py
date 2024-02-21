__author__ = "Simon Nilsson"

import os
import itertools
import cv2
import numpy as np
from copy import deepcopy
from typing import Union, Optional, List


from simba.utils.enums import Paths, Formats, TagNames, TextOptions
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.errors import DuplicationError
from simba.utils.warnings import DuplicateNamesWarning
from simba.utils.errors import NoFilesFoundError
from simba.utils.printing import stdout_success, SimbaTimer, log_event
from simba.utils.read_write import get_fn_ext, get_video_meta_data, read_config_entry
from simba.utils.data import create_color_palettes


class ROIPlot(ConfigReader, PlottingMixin):
    """
    Visualize the ROI data (number of entries/exits, time-spent in ROIs etc).

    .. note::
       `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

       Use :meth:`simba.plotting.ROI_plotter_mp.ROIPlotMultiprocess` for improved run-time.



    :param str config_path: Path to SimBA project config file in Configparser format
    :param str video_path: Name of video to create ROI visualizations for
    :param dict style_attr: User-defined visualization settings.

    Examples
    ----------
    >>> settings = {'Show_body_part': True, 'Show_animal_name': True}
    >>> roi_visualizer = ROIPlot(ini_path=r'MyProjectConfig', video_path="MyVideo.mp4", settings=settings)
    >>> roi_visualizer.insert_data()
    >>> roi_visualizer.run()
    """

    def __init__(
            self,
            ini_path: Union[str, os.PathLike],
            video_path: Union[str, os.PathLike],
            style_attr: dict,
            body_parts: Optional[dict] = None,
            threshold: Optional[float] = None):

        ConfigReader.__init__(self, config_path=ini_path)
        PlottingMixin.__init__(self)
        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))

        settings = None
        if body_parts:
            settings = {'body_parts': body_parts}
            if threshold: settings['threshold'] = threshold
            else: settings['threshold'] = 0.0
            self.animal_id_lst = list(body_parts.keys())
        if threshold: self.threshold = threshold
        else: self.threshold = 0.0

        self.roi_analyzer = ROIAnalyzer(ini_path=ini_path, data_path='outlier_corrected_movement_location', settings=settings)
        if not body_parts:
            self.animal_id_lst = self.roi_analyzer.multi_animal_id_list
        self.video_dir_path = os.path.join(self.roi_analyzer.project_path, 'videos')
        self.video_path = os.path.join(self.video_dir_path, video_path)
        _, self.video_name, _ = get_fn_ext(video_path)
        self.roi_analyzer.files_found = [os.path.join(self.roi_analyzer.input_folder, self.video_name + '.' + self.roi_analyzer.file_type)]
        if not os.path.isfile(self.roi_analyzer.files_found[0]):
            raise NoFilesFoundError(msg='SIMBA ERROR: Could not find the file at path {}. Please make sure you have corrected body-part outliers or indicated that you want to skip outlier correction'.format(self.roi_analyzer.files_found[0]), source=self.__class__.__name__)
        self.roi_analyzer.run()
        self.roi_entries_df = self.roi_analyzer.detailed_df
        self.data_df, self.style_attr = self.roi_analyzer.data_df, style_attr
        self.video_shapes = list(itertools.chain(self.roi_analyzer.video_recs['Name'].unique(), self.roi_analyzer.video_circs['Name'].unique(),self.roi_analyzer.video_polys['Name'].unique()))
        if len(list(set(self.video_shapes))) != len(self.video_shapes):
            raise DuplicationError(msg='Some SIMBA ROI shapes have identical names. Please use unique names to visualize ROI data.', source=self.__class__.__name__)
        self.roi_analyzer.video_recs = self.add_missing_ROI_cols(self.roi_analyzer.video_recs)
        self.roi_analyzer.video_circs = self.add_missing_ROI_cols(self.roi_analyzer.video_circs)
        self.roi_analyzer.video_polys = self.add_missing_ROI_cols(self.roi_analyzer.video_polys)

        self.shape_dicts = {}
        for df in [self.roi_analyzer.video_recs, self.roi_analyzer.video_circs, self.roi_analyzer.video_polys]:
            if not df['Name'].is_unique:
                df = df.drop_duplicates(subset=['Name'], keep='first')
                DuplicateNamesWarning(msg='Some of your ROIs with the same shape (i.e., circle, rectangle, polygon) has the same names. E.g., you have two rectangles named "My rectangle". SimBA prefers ROI shapes with unique names. SimBA will keep one of the unique shape names and drop the rest.', source=self.__class__.__name__)
            d = df.set_index('Name').to_dict(orient='index')
            self.shape_dicts = {**self.shape_dicts, **d}

        self.shape_columns = []
        for animal in self.animal_id_lst:
            for shape_name in self.video_shapes:
                self.data_df[animal + '_' + shape_name] = 0
                self.shape_columns.append(animal + '_' + shape_name)
        self.bp_dict = self.roi_analyzer.bp_dict
        self.output_folder = os.path.join(self.roi_analyzer.project_path, Paths.ROI_ANALYSIS.value)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.timer = SimbaTimer(start=True)

    def insert_data(self):
        """
        Method to concat ROI and pose-estimation data.

        Returns
        -------
        Attribute: pd.Dataframe
            data_df

        """
        self.roi_entries_dict =self.roi_entries_df[['ANIMAL', 'SHAPE', 'ENTRY FRAMES', 'EXIT FRAMES']].to_dict(orient='records')
        for entry_dict in self.roi_entries_dict:
            entry, exit = int(entry_dict['ENTRY FRAMES']), int(entry_dict['EXIT FRAMES'])
            entry_dict['frame_range'] = list(range(entry, exit + 1))
            col_name = entry_dict['ANIMAL'] + '_' + entry_dict['SHAPE']
            self.data_df[col_name][self.data_df.index.isin(entry_dict['frame_range'])] = 1

    def __calc_text_locs(self):
        line_spacer = deepcopy(TextOptions.FIRST_LINE_SPACING.value)
        self.loc_dict = {}
        for animal_cnt, animal_name in enumerate(self.animal_id_lst):
            self.loc_dict[animal_name] = {}
            for shape in self.video_shapes:
                self.loc_dict[animal_name][shape] = {}
                self.loc_dict[animal_name][shape]['timer_text'] = '{} {} {}'.format(shape, animal_name, 'timer:')
                self.loc_dict[animal_name][shape]['entries_text'] = '{} {} {}'.format(shape, animal_name, 'entries:')
                self.loc_dict[animal_name][shape]['timer_text_loc'] = ((self.video_meta_data['width'] + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data['height'] - (self.video_meta_data['height'] + TextOptions.BORDER_BUFFER_Y.value) + self.spacing_scaler * line_spacer))
                self.loc_dict[animal_name][shape]['timer_data_loc'] = (int(self.border_img_w-(self.border_img_w/8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + TextOptions.BORDER_BUFFER_Y.value) + self.spacing_scaler*line_spacer))
                line_spacer += 1
                self.loc_dict[animal_name][shape]['entries_text_loc'] = ((self.video_meta_data['width'] + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data['height'] - (self.video_meta_data['height'] + TextOptions.BORDER_BUFFER_Y.value) + self.spacing_scaler * line_spacer))
                self.loc_dict[animal_name][shape]['entries_data_loc'] = (int(self.border_img_w - (self.border_img_w / 8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + TextOptions.BORDER_BUFFER_Y.value) + self.spacing_scaler * line_spacer))
                line_spacer += 1

    def __insert_texts(self, shape_df):
        for animal_name in self.animal_id_lst:
            for _, shape in shape_df.iterrows():
                shape_name, shape_color = shape['Name'], shape['Color BGR']
                cv2.putText(self.border_img, self.loc_dict[animal_name][shape_name]['timer_text'], self.loc_dict[animal_name][shape_name]['timer_text_loc'], self.font, self.font_size, shape_color, TextOptions.TEXT_THICKNESS.value)
                cv2.putText(self.border_img, self.loc_dict[animal_name][shape_name]['entries_text'], self.loc_dict[animal_name][shape_name]['entries_text_loc'], self.font, self.font_size, shape_color, TextOptions.TEXT_THICKNESS.value)

    def __create_counters(self):
        self.cnt_dict = {}
        for animal_cnt, animal_name in enumerate(self.animal_id_lst):
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

    def run(self):
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
        video_settings, pix_per_mm, fps = self.read_video_info(video_name=self.video_name)
        self.space_scale, radius_scale, res_scale, font_scale = 25, 10, 1500, 0.8
        max_dim = max(self.video_meta_data['width'], self.video_meta_data['height'])
        self.font = Formats.FONT.value
        draw_scale = int(TextOptions.RADIUS_SCALER.value / (TextOptions.RESOLUTION_SCALER.value / max_dim))
        self.font_size = float(TextOptions.FONT_SCALER.value / (TextOptions.RESOLUTION_SCALER.value / max_dim))
        self.spacing_scaler = int(TextOptions.SPACE_SCALER.value / (TextOptions.RESOLUTION_SCALER.value / max_dim))
        fourcc = cv2.VideoWriter_fourcc(*Formats.AVI_CODEC.value)
        writer = cv2.VideoWriter(save_path, fourcc, fps, (self.video_meta_data['width'] * 2, self.video_meta_data['height']))
        color_lst = create_color_palettes(self.roi_analyzer.animal_cnt, int((len(self.roi_analyzer.bp_names) / 3)))[0]
        self.__update_video_meta_data()
        self.__calc_text_locs()
        self.__create_counters()

        frame_cnt = 0
        while (self.cap.isOpened()):
            ret, img = self.cap.read()
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
                    cv2.rectangle(self.border_img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), color, int(thickness))
                for _, row in self.roi_analyzer.video_circs.iterrows():
                    center_x, center_y, radius, shape_name = row['centerX'], row['centerY'], row['radius'], row['Name']
                    thickness, color = row['Thickness'], row['Color BGR']
                    cv2.circle(self.border_img, (center_x, center_y), radius, color, int(thickness))
                for _, row in self.roi_analyzer.video_polys.iterrows():
                    vertices, shape_name = row['vertices'], row['Name']
                    thickness, color = row['Thickness'], row['Color BGR']
                    cv2.polylines(self.border_img, [vertices], True, color, thickness=int(thickness))
                for animal_cnt, animal_name in enumerate(self.animal_id_lst):
                    bp_data = self.data_df.loc[frame_cnt, self.bp_dict[animal_name]].fillna(0.0).values
                    if self.threshold < bp_data[2]:
                        if self.style_attr['Show_body_part']:
                            cv2.circle(self.border_img, (int(bp_data[0]), int(bp_data[1])), draw_scale, color_lst[animal_cnt], -1)
                        if self.style_attr['Show_animal_name']:
                            cv2.putText(self.border_img, animal_name, (int(bp_data[0]), int(bp_data[1])), self.font, self.font_size, color_lst[animal_cnt], TextOptions.TEXT_THICKNESS.value)
                        for shape_name in self.video_shapes:
                            if self.data_df.loc[frame_cnt, animal_name + '_' + shape_name] == 1:
                                self.cnt_dict[animal_name][shape_name]['timer'] += (1 / fps)
                                if not self.cnt_dict[animal_name][shape_name]['entry_status']:
                                    self.cnt_dict[animal_name][shape_name]['entry_status'] = True
                                    self.cnt_dict[animal_name][shape_name]['entries'] += 1
                            else:
                                self.cnt_dict[animal_name][shape_name]['entry_status'] = False
                for animal_cnt, animal_name in enumerate(self.animal_id_lst):
                    for shape_name in self.video_shapes:
                        cv2.putText(self.border_img, str(round(self.cnt_dict[animal_name][shape_name]['timer'], 2)), self.loc_dict[animal_name][shape_name]['timer_data_loc'], self.font, self.font_size, self.shape_dicts[shape_name]['Color BGR'], TextOptions.TEXT_THICKNESS.value)
                        cv2.putText(self.border_img, str(self.cnt_dict[animal_name][shape_name]['entries']), self.loc_dict[animal_name][shape_name]['entries_data_loc'], self.font, self.font_size, self.shape_dicts[shape_name]['Color BGR'], TextOptions.TEXT_THICKNESS.value)
                writer.write(np.uint8(self.border_img))
                # cv2.imshow('Window', self.border_img)
                # key = cv2.waitKey(3000)
                # if key == 27:
                #     cv2.destroyAllWindows()
                print('Frame: {} / {}, Video: {}.'.format(str(frame_cnt), str(self.video_meta_data['frame_count']), self.video_name))
                frame_cnt += 1

            if img is None:
                self.timer.stop_timer()
                print(f'SIMBA COMPLETE: Video {self.video_name} saved at {save_path} (elapsed time: {self.timer.elapsed_time_str}s).')
                self.cap.release()
                break

            # except Exception as e:
            #     writer.release()
            #     print(e.args)
            #     print('NOTE: index error / keyerror. Some frames of the video may be missing. Make sure you are running latest version of SimBA with pip install simba-uw-tf-dev')
            #     break

        writer.release()
        stdout_success(msg=f'Video {self.video_name} created. Video saved in project_folder/frames/output/ROI_analysis', source=self.__class__.__name__)

# test = ROIPlot(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini',
#                video_path="termite_test.mp4",
#                style_attr={'Show_body_part': True, 'Show_animal_name': True})
# test.insert_data()
# test.visualize_ROI_data()


# test = ROIPlot(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini', video_path="termite_test.mp4")
# test.insert_data()
# test.visualize_ROI_data()

# test = ROIPlot(ini_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', video_path=r"Together_1.avi")
# test.insert_data()
# test.visualize_ROI_data()


# test = ROIPlot(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                video_path="Together_1.avi",
#                style_attr={'Show_body_part': True, 'Show_animal_name': False},
#                body_parts={f'Simon': 'Ear_left_1'})
# test.insert_data()
# test.run()

# test = ROIPlot(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini',
#                video_path="termite_test.mp4",
#                style_attr={'Show_body_part': True, 'Show_animal_name': True},
#                body_parts={f'Simon': 'Termite_1_Head_1'})
# test.insert_data()
# test.run()