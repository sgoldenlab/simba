import pandas as pd

from simba.read_config_unit_tests import read_config_file, read_config_entry
from simba.read_config_unit_tests import check_file_exist_and_readable
from simba.misc_tools import check_multi_animal_status, find_video_of_file, get_video_meta_data, find_core_cnt
from simba.drop_bp_cords import create_body_part_dictionary, get_fn_ext, getBpNames, createColorListofList
from simba.rw_dfs import read_df
import numpy as np
import pickle
import functools
import cv2
from multiprocessing import pool
import os, glob

def _image_creator(frm_range: list,
                   polygon_data: dict,
                   animal_bp_dict: dict,
                   data_df: pd.DataFrame,
                   circle_size: int,
                   video_path: str):
    cap, current_frame = cv2.VideoCapture(video_path), frm_range[0]
    cap.set(1, frm_range[0])
    while current_frame < frm_range[-1]:
        _, frame = cap.read()
        frm_data = data_df.iloc[current_frame]
        for animal_cnt, (animal, animal_data) in enumerate(animal_bp_dict.items()):
            for bp_cnt, (x_col, y_col) in enumerate(zip(animal_data['X_bps'], animal_data['Y_bps'])):
                cv2.circle(frame, (frm_data[x_col], frm_data[y_col]), 0, animal_data['colors'][bp_cnt], circle_size)
            animal_polygon = np.array(list(polygon_data[animal][current_frame].convex_hull.exterior.coords)).astype(int)
            cv2.polylines(frame, [animal_polygon], 1, animal_data['colors'][animal_cnt], 2)
        cv2.imshow('sdfsdf', frame)
        cv2.waitKey(2000)
        current_frame += 1

class BoundaryVisualizer(object):
    def __init__(self,
                 config_path: str):

        self.config, self.config_path = read_config_file(ini_path=config_path), config_path
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.polygon_path = os.path.join(self.project_path, 'logs', 'anchored_rois.pickle')
        check_file_exist_and_readable(file_path=self.polygon_path)
        with open(self.polygon_path, 'rb') as fp: self.polygons = pickle.load(fp)
        self.input_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.video_dir = os.path.join(self.project_path, 'videos')
        self.save_parent_dir = os.path.join(self.project_path, 'frames', 'output', 'anchored_rois')
        if not os.path.exists(self.save_parent_dir): os.makedirs(self.save_parent_dir)
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.files_found = glob.glob(self.input_dir + '/*.' + self.file_type)
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.color_lst_of_lst = createColorListofList(self.no_animals, int(len(list(self.x_cols)) + 1))
        self.cpu_cnt, self.cpu_to_use = find_core_cnt()
        self.maxtasksperchild, self.chunksize = 10, 1
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, list(self.multi_animal_id_lst), self.no_animals, list(self.x_cols), list(self.y_cols), [], self.color_lst_of_lst)

    def run_visualization(self, chunk_size=1000):
        for file_cnt, file_path in enumerate(self.files_found):
            _, file_name, _ = get_fn_ext(filepath=file_path)
            self.data_df = read_df(file_path=file_path, file_type=self.file_type).astype(int).reset_index(drop=True)
            video_path = find_video_of_file(video_dir=self.video_dir, filename=file_name)
            video_meta_data = get_video_meta_data(video_path=video_path)
            self.max_dim = max(video_meta_data['width'], video_meta_data['height'])
            self.space_scale, self.radius_scale, self.res_scale, self.font_scale = 60, 12, 1500, 1.1
            self.circle_scale = int(self.radius_scale / (self.res_scale / self.max_dim))
            self.font_size = float(self.font_scale / (self.res_scale / self.max_dim))
            self.spacing_scale = int(self.space_scale / (self.res_scale / self.max_dim))
            self.video_save_path = os.path.join(self.save_parent_dir, file_name + '.mp4')
            self.temp_folder = os.path.join(self.save_parent_dir, file_name)
            if not os.path.exists(self.temp_folder): os.makedirs(self.temp_folder)
            frame_chunks = [[i, i + chunk_size] for i in range(0, video_meta_data['frame_count'], chunk_size)]
            frame_chunks[-1][-1] = min(frame_chunks[-1][-1], video_meta_data['frame_count'])
            functools.partial(_image_creator, b=self.data_df)
            with pool.Pool(self.cpu_to_use, maxtasksperchild=self.maxtasksperchild) as p:
                constants = functools.partial(_image_creator,
                                              data_df=self.data_df,
                                              polygon_data=self.polygons[file_name],
                                              circle_size=self.circle_scale,
                                              animal_bp_dict=self.animal_bp_dict,
                                              video_path=video_path)
                for cnt, result in enumerate(p.imap(constants, frame_chunks, chunksize=self.chunksize)):
                    save_path = os.path.join(self.temp_folder, str(cnt) + '.mp4')

test = BoundaryVisualizer(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
test.run_visualization()