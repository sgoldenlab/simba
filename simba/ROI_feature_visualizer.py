__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import read_config_entry, read_config_file, check_file_exist_and_readable
import os, glob
from simba.features_scripts.unit_tests import read_video_info_csv
from simba.misc_tools import check_multi_animal_status, get_video_meta_data
from simba.drop_bp_cords import getBpNames, createColorListofList, create_body_part_dictionary, get_fn_ext
from simba.ROI_feature_analyzer import ROIFeatureCreator
import cv2
from simba.rw_dfs import read_df
import itertools
import numpy as np

class ROIfeatureVisualizer(object):
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

    Notes
    ----------
    `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    Examples
    ----------
    >>> roi_feature_visualizer = ROIfeatureVisualizer(config_path='MyProjectConfig', video_name='MyVideo.mp4')
    >>> roi_feature_visualizer.create_visualization()
    """

    def __init__(self,
                 config_path: str,
                 video_name: str
                 ):

        self.config = read_config_file(config_path)
        _, self.video_name, _ = get_fn_ext(video_name)
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.data_in_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.save_folder = os.path.join(self.project_path, 'frames', 'output', 'ROI_features')
        self.save_path = os.path.join(self.save_folder, self.video_name + '.mp4')
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        self.logs_path = os.path.join(self.project_path, 'logs')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
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
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.cap = cv2.VideoCapture(self.video_path)
        self.space_scale, self.radius_scale, self.res_scale, self.font_scale = 60, 12, 1500, 1.5
        self.max_dim = max(self.video_meta_data['width'], self.video_meta_data['height'])
        self.circle_scale = int(self.radius_scale / (self.res_scale / self.max_dim))
        self.font_size = float(self.font_scale / (self.res_scale / self.max_dim))
        self.spacing_scale = int(self.space_scale / (self.res_scale / self.max_dim))
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
        self.roi_feature_creator.out_df.fillna(0, inplace=True)

    def __calc_text_locs(self):
        add_spacer = 2
        self.loc_dict = {}
        for animal_cnt, animal_name in enumerate(self.multi_animal_id_lst):
            self.loc_dict[animal_name] = {}
            for shape in self.video_shapes:
                self.loc_dict[animal_name][shape] = {}
                self.loc_dict[animal_name][shape]['in_zone_text'] = '{} {} {}'.format(shape, animal_name, 'in zone')
                self.loc_dict[animal_name][shape]['distance_text'] = '{} {} {}'.format(shape, animal_name, 'distance')
                self.loc_dict[animal_name][shape]['in_zone_text_loc'] = ((self.video_meta_data['width'] + 5), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.spacing_scale * add_spacer))
                self.loc_dict[animal_name][shape]['in_zone_data_loc'] = (int(self.img_w_border_w-(self.img_w_border_w/8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.spacing_scale * add_spacer))
                add_spacer += 1
                self.loc_dict[animal_name][shape]['distance_text_loc'] = ((self.video_meta_data['width'] + 5), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.spacing_scale * add_spacer))
                self.loc_dict[animal_name][shape]['distance_data_loc'] = (int(self.img_w_border_w - (self.img_w_border_w / 8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.spacing_scale * add_spacer))
                add_spacer += 1
                if self.roi_directing_viable:
                    self.loc_dict[animal_name][shape]['directing_text'] = '{} {} {}'.format(shape, animal_name, 'facing')
                    self.loc_dict[animal_name][shape]['directing_text_loc'] = ((self.video_meta_data['width'] + 5), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.spacing_scale * add_spacer))
                    self.loc_dict[animal_name][shape]['directing_data_loc'] = (int(self.img_w_border_w - (self.img_w_border_w / 8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + 10) + self.spacing_scale * add_spacer))
                    add_spacer += 1

    def __insert_texts(self, shape_df):
        for animal_name in self.multi_animal_id_lst:
            for _, shape in shape_df.iterrows():
                shape_name, shape_color = shape['Name'], shape['Color BGR']
                cv2.putText(self.img_w_border, self.loc_dict[animal_name][shape_name]['in_zone_text'], self.loc_dict[animal_name][shape_name]['in_zone_text_loc'], self.font, self.font_size, shape_color, 1)
                cv2.putText(self.img_w_border, self.loc_dict[animal_name][shape_name]['distance_text'], self.loc_dict[animal_name][shape_name]['distance_text_loc'], self.font, self.font_size, shape_color, 1)
                if self.roi_directing_viable:
                    cv2.putText(self.img_w_border, self.loc_dict[animal_name][shape_name]['directing_text'], self.loc_dict[animal_name][shape_name]['directing_text_loc'], self.font, self.font_size, shape_color, 1)

    def __insert_directing_line(self):
        r = self.directing_data.loc[(self.directing_data['ROI'] == self.shape_name) &
                                           (self.directing_data['Animal'] == self.animal_name) &
                                           (self.directing_data['Frame'] == self.frame_cnt)]
        clr = self.shape_dicts[self.shape_name]['Color BGR']
        thickness =  self.shape_dicts[self.shape_name]['Thickness']
        cv2.line(self.img_w_border, (int(r['Eye_x']), int(r['Eye_y'])), (int(r['ROI_x']), int(r['ROI_y'])), clr, thickness)

    def create_visualization(self):
        """
        Creates and saves visualizations of ROI-based features. Results are stored in the ``project_folder/frames/
        output/ROI_features`` directory  of the SimBA project.

        Returns
        ----------
        None
        """

        self.frame_cnt = 0
        while (self.cap.isOpened()):
            ret, self.frame = self.cap.read()
            if ret:
                self.img_w_border = cv2.copyMakeBorder(self.frame, 0, 0, 0, int(self.video_meta_data['width']), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
                self.img_w_border_h, self.img_w_border_w = self.img_w_border.shape[0], self.img_w_border.shape[1]
                if self.frame_cnt == 0:
                    self.__calc_text_locs()
                    self.writer = cv2.VideoWriter(self.save_path, self.fourcc, self.video_meta_data['fps'], (self.img_w_border_w, self.img_w_border_h))
                self.__insert_texts(self.video_recs)
                self.__insert_texts(self.video_circs)
                self.__insert_texts(self.video_polys)
                for animal, animal_bp_name in self.bp_names.items():
                    bp_cords = self.data_df.loc[self.frame_cnt, animal_bp_name].values
                    cv2.circle(self.img_w_border, (int(bp_cords[0]), int(bp_cords[1])), 0, self.animal_bp_dict[animal]['colors'][0], self.circle_scale)

                for _, row in self.video_recs.iterrows():
                    cv2.rectangle(self.img_w_border, (row['topLeftX'], row['topLeftY']), (row['Bottom_right_X'], row['Bottom_right_Y']), row['Color BGR'], row['Thickness'])
                    center_cord = ((int(row['topLeftX'] + (row['width'] / 2))),
                                   (int(row['topLeftY'] + (row['height'] / 2))))
                    cv2.circle(self.img_w_border, center_cord, self.circle_scale, row['Color BGR'], -1)

                for _, row in self.video_circs.iterrows():
                    cv2.circle(self.img_w_border, (row['centerX'], row['centerY']), row['radius'], row['Color BGR'], row['Thickness'])
                    cv2.circle(self.img_w_border, (row['centerX'], row['centerY']), self.circle_scale, row['Color BGR'], -1)

                for _, row in self.video_polys.iterrows():
                    cv2.polylines(self.img_w_border, [row['vertices']], True, row['Color BGR'], thickness=row['Thickness'])
                    cv2.circle(self.img_w_border, (row['Center_X'], row['Center_Y']), self.circle_scale, row['Color BGR'], -1)

                for animal_name, shape_name in itertools.product(self.multi_animal_id_lst, self.video_shapes):
                    self.animal_name, self.shape_name = animal_name, shape_name
                    in_zone_col_name = '{} {} {}'.format(shape_name, animal_name, 'in zone')
                    distance_col_name = '{} {} {}'.format(shape_name, animal_name, 'distance')
                    in_zone_value = str(bool(self.roi_feature_creator.out_df.loc[self.frame_cnt, in_zone_col_name]))
                    distance_value = round(self.roi_feature_creator.out_df.loc[self.frame_cnt, distance_col_name], 2)
                    cv2.putText(self.img_w_border, in_zone_value, self.loc_dict[animal_name][shape_name]['in_zone_data_loc'], self.font, self.font_size, self.shape_dicts[shape_name]['Color BGR'], 1)
                    cv2.putText(self.img_w_border, str(distance_value), self.loc_dict[animal_name][shape_name]['distance_data_loc'], self.font, self.font_size, self.shape_dicts[shape_name]['Color BGR'], 1)
                    if self.roi_directing_viable:
                        facing_col_name = '{} {} {}'.format(shape_name, animal_name, 'facing')
                        facing_value = self.roi_feature_creator.out_df.loc[self.frame_cnt, facing_col_name]
                        cv2.putText(self.img_w_border, str(bool(facing_value)), self.loc_dict[animal_name][shape_name]['directing_data_loc'], self.font, self.font_size, self.shape_dicts[shape_name]['Color BGR'], 1)
                        if facing_value:
                            self.__insert_directing_line()

                self.frame_cnt += 1
                self.writer.write(np.uint8(self.img_w_border))
                print('Frame: {} / {}. Video: {}'.format(str(self.frame_cnt), str(self.video_meta_data['frame_count']),
                                                                 self.video_name))

                # cv2.imshow('Window', self.img_w_border)
                # key = cv2.waitKey(3000)
                # if key == 27:
                #     cv2.destroyAllWindows()

            else:
                self.cap.release()
                self.writer.release()
                print('Feature video {} saved in {} directory ...'.format(self.video_name, self.save_path))

            #break
            # cv2.imshow('Window', self.img_w_border)
            # key = cv2.waitKey(3000)
            # if key == 27:
            #     cv2.destroyAllWindows()


#                self.loc_dict[animal_name][shape]['in_zone_data_loc']



            # for inside_roi_col in self.roi_feature_creator.inside_roi_columns:
            #     print(inside_roi_col)
            #     self.roi_feature_creator.out_df.loc[frame_cnt, inside_roi_col]
            #


















# test = ROIfeatureVisualizer(config_path='/Users/simon/Desktop/train_model_project/project_folder/project_config.ini', video_name='Together_1.avi')
# test.create_visualization()
# test.save_new_features_files()

