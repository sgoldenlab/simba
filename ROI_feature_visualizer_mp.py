__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd

from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_file_exist_and_readable,
                                          read_project_path_and_file_type)
import os
from simba.features_scripts.unit_tests import read_video_info_csv
from simba.misc_tools import (check_multi_animal_status,
                              get_video_meta_data,
                              SimbaTimer,
                              split_and_group_df,
                              remove_a_folder,
                              concatenate_videos_in_folder)
from simba.drop_bp_cords import (getBpNames,
                                 createColorListofList,
                                 create_body_part_dictionary,
                                 get_fn_ext)
from simba.enums import ReadConfig, Formats, Paths, Defaults, Dtypes
from simba.ROI_feature_analyzer import ROIFeatureCreator
import cv2
from simba.rw_dfs import read_df
import itertools
import multiprocessing
import functools
import numpy as np
import platform

def _img_creator(data: pd.DataFrame,
                 text_locations: dict,
                 scalers: dict,
                 save_temp_dir: str,
                 video_meta_data: dict,
                 shape_info: dict,
                 style_attr: dict,
                 directing_viable: bool,
                 video_path: str,
                 animal_names: list,
                 tracked_bps: dict,
                 animal_bps: dict,
                 directing_data: pd.DataFrame):

    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    font = cv2.FONT_HERSHEY_COMPLEX

    def __insert_texts(shape_info: dict, img: np.array):
        for shape_name, shape_info in shape_info.items():
            for animal_name in animal_names:
                shape_color = shape_info['Color BGR']
                cv2.putText(img, text_locations[animal_name][shape_name]['in_zone_text'], text_locations[animal_name][shape_name]['in_zone_text_loc'], font, scalers['font_size'], shape_color, 1)
                cv2.putText(img, text_locations[animal_name][shape_name]['distance_text'], text_locations[animal_name][shape_name]['distance_text_loc'], font, scalers['font_size'], shape_color, 1)
                if directing_viable and style_attr['Directionality']:
                    cv2.putText(img, text_locations[animal_name][shape_name]['directing_text'], text_locations[animal_name][shape_name]['directing_text_loc'], font, scalers['font_size'], shape_color, 1)
        return img

    def __insert_shapes(img: np.array, shape_info: dict):
        for shape_name, shape_info in shape_info.items():
            if shape_info['Shape_type'] == 'Rectangle':
                cv2.rectangle(img, (int(shape_info['topLeftX']), int(shape_info['topLeftY'])), (int(shape_info['Bottom_right_X']), int(shape_info['Bottom_right_Y'])), shape_info['Color BGR'], int(shape_info['Thickness']))
                if style_attr['ROI_centers']:
                    center_cord = ((int(shape_info['topLeftX'] + (shape_info['width'] / 2))), (int(shape_info['topLeftY'] + (shape_info['height'] / 2))))
                    cv2.circle(img, center_cord, scalers['circle_size'], shape_info['Color BGR'], -1)
                if style_attr['ROI_ear_tags']:
                    for tag_data in shape_info['Tags'].values():
                        cv2.circle(img, tag_data, scalers['circle_size'], shape_info['Color BGR'], -1)


            if shape_info['Shape_type'] == 'Circle':
                cv2.circle(img, (int(shape_info['centerX']), int(shape_info['centerY'])), shape_info['radius'], shape_info['Color BGR'], int(shape_info['Thickness']))
                if style_attr['ROI_centers']:
                    cv2.circle(img, (int(shape_info['centerX']), int(shape_info['centerY'])), scalers['circle_size'], shape_info['Color BGR'], -1)
                if style_attr['ROI_ear_tags']:
                    for tag_data in shape_info['Tags'].values():
                        cv2.circle(img, tag_data, scalers['circle_size'], shape_info['Color BGR'], -1)

            if shape_info['Shape_type'] == 'Polygon':
                cv2.polylines(img, [shape_info['vertices']], True, shape_info['Color BGR'], thickness=int(shape_info['Thickness']))
                if style_attr['ROI_centers']:
                    cv2.circle(img, shape_info['Tags']['Center_tag'], scalers['circle_size'], shape_info['Color BGR'], -1)
                if style_attr['ROI_ear_tags']:
                    for tag_data in shape_info['Tags'].values():
                        cv2.circle(img, tag_data, scalers['circle_size'], shape_info['Color BGR'], -1)

        return img


    def __insert_directing_line(directing_data: pd.DataFrame,
                                shape_name: str,
                                frame_cnt: int,
                                shape_info: dict,
                                img: np.array,
                                video_name: str,
                                style_attr: dict):

        r = directing_data.loc[(directing_data['Video'] == video_name) & (directing_data['ROI'] == shape_name) & (directing_data['Animal'] == animal_name) & (directing_data['Frame'] == frame_cnt)]
        clr = shape_info[shape_name]['Color BGR']
        thickness = shape_info[shape_name]['Thickness']
        if style_attr['Directionality_style'] == 'Funnel':
            convex_hull_arr = np.array([[r['ROI_edge_1_x'], r['ROI_edge_1_y']],
                                        [r['ROI_edge_2_x'], r['ROI_edge_2_y']],
                                        [r['Eye_x'], r['Eye_y']]]).reshape(-1, 2).astype(int)
            cv2.fillPoly(img, [convex_hull_arr], clr)

        if style_attr['Directionality_style'] == 'Lines':
            cv2.line(img, (int(r['Eye_x']), int(r['Eye_y'])), (int(r['ROI_x']), int(r['ROI_y'])), clr, int(thickness))

        return img

    group_cnt = int(data['group'].values[0])
    start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
    save_path = os.path.join(save_temp_dir, '{}.mp4'.format(str(group_cnt)))
    _, video_name, _ = get_fn_ext(filepath=video_path)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'], (video_meta_data['width'] * 2, video_meta_data['height']))
    cap = cv2.VideoCapture(video_path)
    cap.set(1, start_frm)

    while current_frm < end_frm:
        ret, img = cap.read()
        img = cv2.copyMakeBorder(img, 0, 0, 0, int(video_meta_data['width']), borderType=cv2.BORDER_CONSTANT, value=style_attr['Border_color'])
        img = __insert_texts(shape_info=shape_info, img=img)
        if style_attr['Pose_estimation']:
            for animal, animal_bp_name in tracked_bps.items():
                bp_cords = data.loc[current_frm, animal_bp_name].values
                cv2.circle(img, (int(bp_cords[0]), int(bp_cords[1])), 0, animal_bps[animal]['colors'][0], scalers['circle_size'])
                cv2.putText(img, animal, (int(bp_cords[0]), int(bp_cords[1])), font, scalers['font_size'], animal_bps[animal]['colors'][0], 1)

        img = __insert_shapes(img=img, shape_info=shape_info)

        for animal_name, shape_name in itertools.product(animal_names,shape_info):
            in_zone_col_name = '{} {} {}'.format(shape_name, animal_name, 'in zone')
            distance_col_name = '{} {} {}'.format(shape_name, animal_name, 'distance')
            in_zone_value = str(bool(data.loc[current_frm, in_zone_col_name]))
            distance_value = str(round(data.loc[current_frm, distance_col_name], 2))
            cv2.putText(img, in_zone_value, text_locations[animal_name][shape_name]['in_zone_data_loc'], font, scalers['font_size'], shape_info[shape_name]['Color BGR'], 1)
            cv2.putText(img, distance_value, text_locations[animal_name][shape_name]['distance_data_loc'], font, scalers['font_size'], shape_info[shape_name]['Color BGR'], 1)
            if directing_viable and style_attr['Directionality']:
                facing_col_name = '{} {} {}'.format(shape_name, animal_name, 'facing')
                facing_value = bool(data.loc[current_frm, facing_col_name])
                cv2.putText(img, str(facing_value), text_locations[animal_name][shape_name]['directing_data_loc'], font, scalers['font_size'], shape_info[shape_name]['Color BGR'], 1)
                if facing_value:
                    img = __insert_directing_line(directing_data=directing_data,
                                                  shape_name=shape_name,
                                                  frame_cnt=current_frm,
                                                  shape_info=shape_info,
                                                  img=img,
                                                  video_name=video_name,
                                                  style_attr=style_attr)
        writer.write(img)
        current_frm += 1
        print('Multi-processing video frame {} on core {}...'.format(str(current_frm), str(group_cnt)))
    cap.release()
    writer.release()

    return group_cnt

class ROIfeatureVisualizerMultiprocess(object):
    """
    Class for visualizing features that depend on the relationships between the location of the animals and user-defined
    ROIs. E.g., distances to centroids of ROIs, cumulative time spent in ROIs, if animals are directing towards ROIs
    etc.

    Parameters
    ----------
    config_path: str
        Path to SimBA project config file in Configparser format
    video_name: str
        Name of video to create feature visualizations for.
    style_attr: dict
        Image user attributes
    core_cnt: int
        Number of cores to parallelize over.

    Notes
    ----------
    `Tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-5-visualizing-roi-features>`__.

    Examples
    ----------
    style_attr = {'ROI_centers': True, 'ROI_ear_tags': True, 'Directionality': True, 'Directionality_style': 'Funnel', 'Border_color': (0, 128, 0), 'Pose_estimation': True}
    roi_feature_visualizer = ROIfeatureVisualizerMultiprocess(config_path='test_/project_folder/project_config.ini', video_name='Together_1.avi', style_attr=style_attr, core_cnt=3)
    roi_feature_visualizer.create_visualization()
    """

    def __init__(self,
                 config_path: str,
                 video_name: str,
                 core_cnt: int,
                 style_attr: dict):


        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        self.config = read_config_file(config_path)
        _, self.video_name, _ = get_fn_ext(video_name)
        self.core_cnt, self.style_attr = core_cnt, style_attr
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.no_animals = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.ANIMAL_CNT.value, Dtypes.INT.value)
        self.data_in_dir = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.save_folder = os.path.join(self.project_path, Paths.ROI_FEATURES.value)
        self.save_path = os.path.join(self.save_folder, self.video_name + '.mp4')
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        self.save_temp_dir = os.path.join(self.save_folder, 'temp')
        if os.path.exists(self.save_temp_dir): remove_a_folder(folder_dir=self.save_temp_dir)
        os.makedirs(self.save_temp_dir)

        self.logs_path = os.path.join(self.project_path, 'logs')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.x_cols, self.y_cols, self.p_cols = getBpNames(config_path)
        c_map_size = int(len(self.x_cols) + 1)
        color_lst_lst = createColorListofList(self.no_animals, c_map_size)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, self.x_cols, self.y_cols, self.p_cols, color_lst_lst)
        self.roi_feature_creator = ROIFeatureCreator(config_path=config_path)
        self.file_in_path = os.path.join(self.data_in_dir, self.video_name + '.' + self.file_type)
        self.video_path = os.path.join(self.project_path, 'videos', video_name)
        check_file_exist_and_readable(file_path=self.file_in_path)
        self.roi_feature_creator.features_files = [self.file_in_path]
        self.roi_feature_creator.files_found = [self.file_in_path]
        self.roi_feature_creator.analyze_ROI_data()
        self.video_meta_data = get_video_meta_data(self.video_path)
        self.scalers = {}
        self.maxtasksperchild = Defaults.MAX_TASK_PER_CHILD.value
        self.chunksize = Defaults.CHUNK_SIZE.value
        self.space_scale, self.radius_scale, self.res_scale, self.font_scale = 25, 10, 1500, 0.8
        self.max_dim = max(self.video_meta_data['width'], self.video_meta_data['height'])
        self.scalers['circle_size'] = int(self.radius_scale / (self.res_scale / self.max_dim))
        self.scalers['font_size'] = float(self.font_scale / (self.res_scale / self.max_dim))
        self.scalers['spacing_size'] = int(self.space_scale / (self.res_scale / self.max_dim))
        self.data_df = read_df(self.file_in_path,self.file_type)
        self.bp_names = self.roi_feature_creator.roi_analyzer.bp_dict
        self.video_recs = self.roi_feature_creator.roi_analyzer.video_recs
        self.video_circs = self.roi_feature_creator.roi_analyzer.video_circs
        self.video_polys = self.roi_feature_creator.roi_analyzer.video_polys
        self.shape_dicts = {}
        for df in [self.video_recs, self.video_circs, self.video_polys]:
            d = df.set_index('Name').to_dict(orient='index')
            self.shape_dicts = {**self.shape_dicts, **d}
        self.video_shapes = list(itertools.chain(self.video_recs['Name'].unique(), self.video_circs['Name'].unique(),self.video_polys['Name'].unique()))
        self.roi_directing_viable = self.roi_feature_creator.roi_directing_viable
        if self.roi_directing_viable:
            self.directing_data = self.roi_feature_creator.directing_analyzer.results_df
            self.directing_data = self.directing_data[self.directing_data['Video'] == self.video_name]
        else:
            self.directing_data = None
        self.roi_feature_creator.out_df.fillna(0, inplace=True)
        self.timer = SimbaTimer()
        self.timer.start_timer()

    def __calc_text_locs(self):
        add_spacer = 2
        self.loc_dict = {}
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap.set(0, 1)
        ret, img = self.cap.read()
        self.img_w_border = cv2.copyMakeBorder(img, 0, 0, 0, int(self.video_meta_data['width']), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        self.img_w_border_h, self.img_w_border_w = self.img_w_border.shape[0], self.img_w_border.shape[1]

        for animal_cnt, animal_name in enumerate(self.multi_animal_id_lst):
            self.loc_dict[animal_name] = {}
            for shape in self.video_shapes:
                self.loc_dict[animal_name][shape] = {}
                self.loc_dict[animal_name][shape]['in_zone_text'] = '{} {} {}'.format(shape, animal_name, 'in zone')
                self.loc_dict[animal_name][shape]['distance_text'] = '{} {} {}'.format(shape, animal_name, 'distance')
                self.loc_dict[animal_name][shape]['in_zone_text_loc'] = ((self.video_meta_data['width'] + 5), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.scalers['spacing_size'] * add_spacer))
                self.loc_dict[animal_name][shape]['in_zone_data_loc'] = (int(self.img_w_border_w -(self.img_w_border_w/8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.scalers['spacing_size'] * add_spacer))
                add_spacer += 1
                self.loc_dict[animal_name][shape]['distance_text_loc'] = ((self.video_meta_data['width'] + 5), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.scalers['spacing_size'] * add_spacer))
                self.loc_dict[animal_name][shape]['distance_data_loc'] = (int(self.img_w_border_w- (self.img_w_border_w / 8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.scalers['spacing_size'] * add_spacer))
                add_spacer += 1
                if self.roi_directing_viable and self.style_attr['Directionality']:
                    self.loc_dict[animal_name][shape]['directing_text'] = '{} {} {}'.format(shape, animal_name, 'facing')
                    self.loc_dict[animal_name][shape]['directing_text_loc'] = ((self.video_meta_data['width'] + 5), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.scalers['spacing_size'] * add_spacer))
                    self.loc_dict[animal_name][shape]['directing_data_loc'] = (int(self.img_w_border_w - (self.img_w_border_w / 8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.scalers['spacing_size'] * add_spacer))
                    add_spacer += 1

    def create_visualization(self):
        """
        Creates and saves visualizations of ROI-based features. Results are stored in the ``project_folder/frames/
        output/ROI_features`` directory  of the SimBA project.

        Returns
        ----------
        None
        """

        self.__calc_text_locs()
        data_arr, frm_per_core = split_and_group_df(df=self.roi_feature_creator.out_df, splits=self.core_cnt, include_split_order=True)

        print('Creating ROI feature images, multiprocessing (determined chunksize: {}, cores: {})...'.format(str(self.chunksize), str(self.core_cnt)))
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(_img_creator,
                                          text_locations=self.loc_dict,
                                          scalers=self.scalers,
                                          video_meta_data=self.video_meta_data,
                                          shape_info=self.shape_dicts,
                                          style_attr=self.style_attr,
                                          save_temp_dir=self.save_temp_dir,
                                          directing_data=self.directing_data,
                                          video_path=self.video_path,
                                          directing_viable=self.roi_directing_viable,
                                          animal_names=self.multi_animal_id_lst,
                                          tracked_bps=self.bp_names,
                                          animal_bps=self.animal_bp_dict)
            for cnt, result in enumerate(pool.imap(constants, data_arr, chunksize=self.chunksize)):
                print('Image {}/{}, Video {}...'.format(str(int(frm_per_core * (result + 1))), str(len(self.data_df)), self.video_name))

            print('Joining {} multiprocessed video...'.format(self.video_name))
            concatenate_videos_in_folder(in_folder=self.save_temp_dir, save_path=self.save_path, video_format='mp4')

            self.timer.stop_timer()
            pool.terminate()
            pool.join()
            print('Video {} complete (elapsed time: {}s). Video saved in project_folder/frames/output/ROI_features.'.format(self.video_name, self.timer.elapsed_time_str))


# style_attr = {'ROI_centers': True, 'ROI_ear_tags': True, 'Directionality': True, 'Directionality_style': 'Funnel', 'Border_color': (0, 128, 0), 'Pose_estimation': True}
# roi_feature_visualizer = ROIfeatureVisualizerMultiprocess(config_path='/Users/simon/Desktop/envs/simba_dev/tests/test_data/mouse_open_field/project_folder/project_config.ini', video_name='Video1.mp4', style_attr=style_attr, core_cnt=3)
# roi_feature_visualizer.create_visualization()