from simba.read_config_unit_tests import read_config_file, read_config_entry
from simba.rw_dfs import read_df
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.drop_bp_cords import create_body_part_dictionary, get_fn_ext, getBpNames
import os, glob
import itertools
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from joblib import Parallel, delayed
import pickle
from scipy.spatial import ConvexHull
from simba.misc_tools import check_multi_animal_status

class AnimalBoundaryFinder(object):
    def __init__(self,
                 config_path: str,
                 roi_type: str,
                 force_rectangle: bool,
                 body_parts: dict or None,
                 parallel_offset: int or None):

        self.config, self.config_path = read_config_file(ini_path=config_path), config_path
        self.parallel_offset_mm, self.roi_type, self.force_rectangle = parallel_offset, roi_type.lower(), force_rectangle
        self.body_parts = body_parts
        self.project_path= read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.input_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.files_found = glob.glob(self.input_dir + '/*.' + self.file_type)
        if len(self.files_found) == 0:
            print('SIMBA ERROR: ZERO files found')
            raise ValueError()
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.save_path = os.path.join(self.project_path, 'logs', 'anchored_rois.pickle')
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, list(self.multi_animal_id_lst),  self.no_animals, list(self.x_cols), list(self.y_cols), [], [])
        if (self.roi_type == 'circle') or (self.roi_type == 'square'):
            self.center_bp_names = {}
            for animal, body_part in self.body_parts.items():
                self.center_bp_names[animal] = [body_part + '_x', body_part + '_y']

    def _save_results(self):
        with open(self.save_path, 'wb') as path:
            pickle.dump(self.polygons, path, pickle.HIGHEST_PROTOCOL)

    def minimum_bounding_rectangle(self, points):

        pi2 = np.pi / 2.
        hull_points = points[ConvexHull(points).vertices]
        edges = hull_points[1:] - hull_points[:-1]
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)
        rotations = np.vstack([np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2), np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))
        rot_points = np.dot(rotations, hull_points.T)
        min_x, max_x = np.nanmin(rot_points[:, 0], axis=1), np.nanmax(rot_points[:, 0], axis=1)
        min_y, max_y = np.nanmin(rot_points[:, 1], axis=1), np.nanmax(rot_points[:, 1], axis=1)
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)
        x1, x2 = max_x[best_idx], min_x[best_idx]
        y1, y2 = max_y[best_idx], min_y[best_idx]
        r = rotations[best_idx]
        rval = np.zeros((4, 2))
        rval[0], rval[1] = np.dot([x1, y2], r), np.dot([x2, y2], r)
        rval[2], rval[3] = np.dot([x2, y1], r), np.dot([x1, y1], r)
        return rval

    def _find_polygons(self, point_array: np.array):
        if self.roi_type == 'polygon':
            animal_shape = LineString(point_array.tolist()).buffer(self.offset_px)
        elif self.roi_type == 'circle':
            animal_shape = Point(point_array).buffer(self.offset_px)
        elif self.roi_type == 'square':
            top_left = Point(int(point_array[0] - self.offset_px), int(point_array[1] - self.offset_px))
            top_right = Point(int(point_array[0] + self.offset_px), int(point_array[1] - self.offset_px))
            bottom_left = Point(int(point_array[0] - self.offset_px), int(point_array[1] + self.offset_px))
            bottom_right = Point(int(point_array[0] + self.offset_px), int(point_array[1] + self.offset_px))
            animal_shape = Polygon([top_left, top_right, bottom_left, bottom_right])
        if self.force_rectangle:
            animal_shape = Polygon(self.minimum_bounding_rectangle(points=np.array(animal_shape.exterior.coords)))
        return animal_shape

    def find_boundaries(self):
        self.polygons = {}
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            _, px_per_mm, _ = read_video_info(self.vid_info_df, self.video_name)
            self.offset_px = px_per_mm * self.parallel_offset_mm
            self.polygons[self.video_name] = {}
            self.data_df = read_df(file_path=file_path,file_type=self.file_type).astype(int)
            for animal in self.animal_bp_dict.keys():
                if self.roi_type == 'polygon':
                    animal_x_cols, animal_y_cols = self.animal_bp_dict[animal]['X_bps'], self.animal_bp_dict[animal]['Y_bps']
                    animal_df = self.data_df[[x for x in itertools.chain.from_iterable(itertools.zip_longest(animal_x_cols,animal_y_cols)) if x]]
                    animal_arr = np.reshape(animal_df.values, (-1, len(animal_x_cols), 2))
                if (self.roi_type == 'circle') or (self.roi_type == 'square'):
                    animal_arr = self.data_df[self.center_bp_names[animal]].values
                self.polygons[self.video_name][animal] = Parallel(n_jobs=5, verbose=2, backend="loky")(delayed(self._find_polygons)(x) for x in animal_arr)
        self._save_results()

test = AnimalBoundaryFinder(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',
                            roi_type='polygon',
                            body_parts={'Animal_1': 'Nose_1', 'Animal_2': 'Nose_2'},
                            force_rectangle=True,
                            parallel_offset=30)
test.find_boundaries()
