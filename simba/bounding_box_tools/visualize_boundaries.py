import pandas as pd

from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry,
                                          read_project_path_and_file_type)
from simba.read_config_unit_tests import check_file_exist_and_readable
from simba.misc_tools import (check_multi_animal_status,
                              find_video_of_file,
                              get_video_meta_data,
                              find_core_cnt, remove_a_folder)
from simba.drop_bp_cords import (create_body_part_dictionary,
                                 getBpNames,
                                 createColorListofList)
from simba.enums import Paths, ReadConfig, Dtypes
from simba.rw_dfs import read_df
import numpy as np
import pickle
import functools
import cv2
import pathlib, re
import multiprocessing
from multiprocessing import pool
import platform
import os, glob

def _image_creator(frm_range: list,
                   polygon_data: dict,
                   animal_bp_dict: dict,
                   data_df: pd.DataFrame or None,
                   intersection_data_df: pd.DataFrame or None,
                   roi_attributes: dict,
                   video_path: str,
                   key_points: bool,
                   greyscale: bool):

    cap, current_frame = cv2.VideoCapture(video_path), frm_range[0]

    cap.set(1, frm_range[0])
    img_lst = []
    while current_frame < frm_range[-1]:
        ret, frame = cap.read()
        if ret:
            if key_points:
                frm_data = data_df.iloc[current_frame]
            if greyscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            for animal_cnt, (animal, animal_data) in enumerate(animal_bp_dict.items()):
                if key_points:
                    for bp_cnt, (x_col, y_col) in enumerate(zip(animal_data['X_bps'], animal_data['Y_bps'])):
                        cv2.circle(frame, (frm_data[x_col], frm_data[y_col]), 0, roi_attributes[animal]['bbox_clr'], roi_attributes[animal]['keypoint_size'])
                animal_polygon = np.array(list(polygon_data[animal][current_frame].convex_hull.exterior.coords)).astype(int)
                if intersection_data_df is not None:
                    intersect = intersection_data_df.loc[current_frame, intersection_data_df.columns.str.startswith(animal)].sum()
                    if intersect > 0:
                        cv2.polylines(frame, [animal_polygon], 1, roi_attributes[animal]['highlight_clr'], roi_attributes[animal]['highlight_clr_thickness'])
                cv2.polylines(frame, [animal_polygon], 1, roi_attributes[animal]['bbox_clr'], roi_attributes[animal]['bbox_thickness'])
            img_lst.append(frame)
            current_frame += 1
        else:
            print('SIMBA WARNING: SimBA tried to grab frame number {} from video {}, but could not find it. The video has {} frames.'.format(str(current_frame), video_path, str(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    return img_lst

class BoundaryVisualizer(object):
    """
    Class visualizing user-specified animal-anchored ROI boundaries. Results are stored in the
    `project_folder/frames/output/anchored_rois` directory of teh SimBA project

    Parameters
    ----------
    config_path: str
        Path to SimBA project config file in Configparser format
    video_name: str
        Name of the video in the SimBA project to create bounding box video for
    include_key_points: bool
        If True, includes pose-estimated body-parts in the visualization.
    greyscale: bool
        If True, converts the video (but not the shapes/keypoints) to greyscale.
    show_intersections: bool or None
        If True, then produce highlight boundaries/keypoints to signify present intersections.

    Notes
    ----------
    `Bounding boxes tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/anchored_rois.md__.

    Examples
    ----------
    >>> boundary_visualizer = BoundaryVisualizer(config_path='MySimBAConfig', video_name='MyVideoName', include_key_points=True, greyscale=True)
    >>> boundary_visualizer.run_visualization()
    """

    def __init__(self,
                 config_path: str,
                 video_name: str,
                 include_key_points: bool,
                 greyscale: bool,
                 show_intersections: bool or None,
                 roi_attributes: dict or None):

        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        self.config, self.config_path = read_config_file(ini_path=config_path), config_path
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.polygon_path = os.path.join(self.project_path, 'logs', 'anchored_rois.pickle')
        self.video_name, self.include_key_points, self.greyscale, self.roi_attributes = video_name, include_key_points, greyscale, roi_attributes
        print(roi_attributes)
        self.show_intersections, self.intersection_data_folder = show_intersections, os.path.join(self.project_path, 'csv', 'anchored_roi_data')
        check_file_exist_and_readable(file_path=self.polygon_path)
        self.intersections_df = None
        if self.show_intersections: self._find_intersection_data()
        with open(self.polygon_path, 'rb') as fp: self.polygons = pickle.load(fp)
        self.input_dir = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.video_dir = os.path.join(self.project_path, 'videos')
        self.video_path = find_video_of_file(video_dir=self.video_dir, filename=video_name)
        self.save_parent_dir = os.path.join(self.project_path, 'frames', 'output', 'anchored_rois')
        self.save_video_path = os.path.join(self.save_parent_dir, video_name + '.mp4')
        if not os.path.exists(self.save_parent_dir): os.makedirs(self.save_parent_dir)
        self.no_animals = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.ANIMAL_CNT.value, Dtypes.INT.value)
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.color_lst_of_lst = createColorListofList(self.no_animals, int(len(list(self.x_cols)) + 1))
        self.cpu_cnt, self.cpu_to_use = find_core_cnt()
        self.maxtasksperchild, self.chunksize = 10, 1
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, list(self.multi_animal_id_lst), self.no_animals, list(self.x_cols), list(self.y_cols), [], self.color_lst_of_lst)

    def _find_intersection_data(self):
        self.intersection_path = None
        for p in [os.path.join(self.intersection_data_folder, self.video_name + x) for x in ['.pickle', '.csv', '.parquet']]:
            if os.path.isfile(p):
                self.intersection_path = p
        if self.intersection_path is None:
            print('SIMBA WARNING: No ROI intersection data found for video {} in directory {}. Skipping intersection visualizations'.format(self.video_name, self.intersection_data_folder))
            self.show_intersections = False
            self.intersections_df = None
        else:
            if self.intersection_path.endswith('pickle'):
                self.intersections_df = pd.read_pickle(self.intersection_path)
            elif self.intersection_path.endswith('parquet'):
                self.intersections_df = pd.read_parquet(self.intersection_path)
            elif self.intersection_path.endswith('csv'):
                self.intersections_df = pd.read_csv(self.intersection_path)

    def run_visualization(self, chunk_size=50):
        if self.include_key_points:
            self.data_df_path = os.path.join(self.input_dir, self.video_name + '.' + self.file_type)
            if not os.path.isfile(self.data_df_path):
                print('SIMBA ERROR: No keypoint data found in {} for video {}. Untick key-point checkbox or import pose-estimation data.')
                raise FileNotFoundError()
            self.data_df = read_df(file_path=self.data_df_path, file_type=self.file_type).astype(int).reset_index(drop=True)
        else:
            self.data_df = None
        print('Creating visualization for video {}...'.format(self.video_name))
        video_path = find_video_of_file(video_dir=self.video_dir, filename=self.video_name)
        video_meta_data = get_video_meta_data(video_path=video_path)
        self.max_dim = max(video_meta_data['width'], video_meta_data['height'])
        self.space_scale, self.radius_scale, self.res_scale, self.font_scale = 60, 12, 1500, 1.1
        if self.roi_attributes is None:
            self.roi_attributes = {}
            for animal_name, animal_data in self.animal_bp_dict.items():
                self.roi_attributes[animal_name] = {}
                self.roi_attributes[animal_name]['bbox_clr'] = animal_data['colors'][0]
                self.roi_attributes[animal_name]['bbox_thickness'] = 2
                self.roi_attributes[animal_name]['keypoint_size'] = int(self.radius_scale / (self.res_scale / self.max_dim))
                self.roi_attributes[animal_name]['highlight_clr'] = (0, 0, 255)
                self.roi_attributes[animal_name]['highlight_clr_thickness'] = 10

        self.video_save_path = os.path.join(self.save_parent_dir, self.video_name + '.mp4')
        self.temp_folder = os.path.join(self.save_parent_dir, self.video_name)
        if not os.path.exists(self.temp_folder): os.makedirs(self.temp_folder)
        frame_chunks = [[i, i + chunk_size] for i in range(0, video_meta_data['frame_count'], chunk_size)]
        frame_chunks[-1][-1] = min(frame_chunks[-1][-1], video_meta_data['frame_count'])
        functools.partial(_image_creator, b=self.data_df)
        with pool.Pool(self.cpu_to_use, maxtasksperchild=self.maxtasksperchild) as p:
            constants = functools.partial(_image_creator,
                                          data_df=self.data_df,
                                          polygon_data=self.polygons[self.video_name],
                                          animal_bp_dict=self.animal_bp_dict,
                                          roi_attributes=self.roi_attributes,
                                          video_path=video_path,
                                          key_points=self.include_key_points,
                                          greyscale=self.greyscale,
                                          intersection_data_df=self.intersections_df)
            for cnt, result in enumerate(p.imap(constants, frame_chunks, chunksize=self.chunksize)):
                save_path = os.path.join(self.temp_folder, str(cnt) + '.mp4')
                writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), video_meta_data['fps'], (video_meta_data['width'], video_meta_data['height']))
                for img in result:
                    writer.write(img)
                writer.release()
                if int(chunk_size * cnt) < video_meta_data['frame_count']:
                    print('Image {}/{}...'.format(str(int(chunk_size * cnt)), str(video_meta_data['frame_count'])))
            p.terminate()
            p.join()

        files = glob.glob(self.temp_folder + '/*.mp4')
        files.sort(key=lambda f: int(re.sub('\D', '', f)))
        temp_txt_path = pathlib.Path(self.temp_folder, 'files.txt')
        with open(temp_txt_path, 'w') as f:
            for file in files:
                f.write("file '" + str(pathlib.Path(file)) + "'\n")
        if os.path.exists(self.save_video_path): os.remove(self.save_video_path)
        returned = os.system('ffmpeg -f concat -safe 0 -i "{}" "{}" -hide_banner -loglevel error'.format(temp_txt_path, self.save_video_path))
        while True:
            if returned != 0:
                pass
            else:
                remove_a_folder(folder_dir=self.temp_folder)
                break
        print('SIMBA COMPLETE: Anchored ROI video created at {}'.format(self.save_video_path))

# boundary_visualizer = BoundaryVisualizer(config_path='/Users/simon/Desktop/troubleshooting/termites/project_folder/project_config.ini',
#                                          video_name='termites_test',
#                                          include_key_points=True,
#                                          greyscale=True,
#                                          show_intersections=True)
# boundary_visualizer.run_visualization()
