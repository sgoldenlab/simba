import os.path
from copy import deepcopy
import numpy as np

import pandas as pd

from simba.utils.read_write import read_df, get_fn_ext, write_df, find_core_cnt
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.abstract_classes import AbstractFeatureExtraction
from simba.utils.errors import NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.plotting.geometry_plotter import GeometryPlotter
import argparse

WHITE = 'white'
BLACK = 'black'

CORE_CNT = -1
VISUALIZE = True
BUFFER = 20
RECTANGLES = True

class PiotrFeatureExtractor(ConfigReader,
                            FeatureExtractionMixin,
                            GeometryMixin,
                            AbstractFeatureExtraction):

    def __init__(self,
                 config_path: str):

        ConfigReader.__init__(self, config_path=config_path)
        FeatureExtractionMixin.__init__(self)
        GeometryMixin.__init__(self)
        if len(self.outlier_corrected_paths) == 0:
            raise NoFilesFoundError(msg=f'No files found in {self.outlier_corrected_dir}')
        self.session_timer = SimbaTimer(start=True)
        self.config_path = config_path

    def run(self):
        print(f'Used core count: {CORE_CNT}')
        for file_cnt, file_path in enumerate(self.outlier_corrected_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            print(f'Processing {video_name}...')
            _, pixels_per_mm, _ = self.read_video_info(video_name=video_name)
            data_df = read_df(file_path=file_path, file_type=self.file_type)
            data_df = data_df.apply(pd.to_numeric, errors='coerce').fillna(0).replace(0, np.nan)
            data_df = data_df.interpolate(method='nearest').bfill().ffill()
            nan_cols = data_df.columns[data_df.isnull().all(0)] # FIND COLUMN NAMES THAT ARE ALL NaNs
            #data_df = data_df.drop(nan_cols, axis=1) # DROP COLUMNS THAT ARE ALL NaNs FROM THE DATA
            results = deepcopy(data_df)
            save_path = os.path.join(self.features_dir, f'{video_name}.{self.file_type}')
            white_animal_bp_names, black_animal_bp_names = self.animal_bp_dict[WHITE], self.animal_bp_dict[BLACK]
            white_animal_cols, black_animal_cols = [], []
            for x, y in zip(white_animal_bp_names['X_bps'], white_animal_bp_names['Y_bps']): white_animal_cols.extend((x, y))
            for x, y in zip(black_animal_bp_names['X_bps'], black_animal_bp_names['Y_bps']): black_animal_cols.extend((x, y))
            black_animal_cols = [x for x in black_animal_cols if x not in nan_cols] # DROP COLUMN NAMES THAT ARE ALL NaNs FROM THE BLACK ANIMAL BODY-PART NAMES
            white_animal_cols = [x for x in white_animal_cols if x not in nan_cols] # DROP COLUMN NAMES THAT ARE ALL NaNs FROM THE WHITE ANIMAL BODY-PART NAMES
            white_animal_df, black_animal_df = data_df[white_animal_cols], data_df[black_animal_cols]
            white_animal_df_arr = white_animal_df.values.reshape(len(white_animal_df), -1 , 2)
            black_animal_df_arr = black_animal_df.values.reshape(len(black_animal_df), -1,  2)
            white_animal_polygons = GeometryMixin().multiframe_bodyparts_to_polygon(data=white_animal_df_arr, pixels_per_mm=pixels_per_mm, parallel_offset=BUFFER, verbose=True, video_name=video_name, animal_name='white', core_cnt=CORE_CNT)
            black_animal_polygons = GeometryMixin().multiframe_bodyparts_to_polygon(data=black_animal_df_arr, pixels_per_mm=pixels_per_mm, parallel_offset=BUFFER, verbose=True, video_name=video_name, animal_name='black', core_cnt=CORE_CNT)
            if RECTANGLES:
                white_animal_polygons = GeometryMixin().multiframe_minimum_rotated_rectangle(shapes=white_animal_polygons, video_name=video_name, animal_name='white', verbose=True, core_cnt=CORE_CNT)
                black_animal_polygons = GeometryMixin().multiframe_minimum_rotated_rectangle(shapes=black_animal_polygons, video_name=video_name, animal_name='black', verbose=True, core_cnt=CORE_CNT)
            results['polygon_pct_overlap'] = GeometryMixin().multiframe_compute_pct_shape_overlap(shape_1=white_animal_polygons, shape_2=black_animal_polygons, animal_names='black_white', video_name=video_name, verbose=True, core_cnt=CORE_CNT)
            combined_list = [list(pair) for pair in list(zip(white_animal_polygons, black_animal_polygons))]
            difference = GeometryMixin().multiframe_difference(shapes=combined_list, verbose=True, animal_names='white_black', video_name=video_name, core_cnt=CORE_CNT)
            results['difference_area'] = GeometryMixin().multiframe_area(shapes=difference, pixels_per_mm=pixels_per_mm, verbose=True, video_name=video_name, core_cnt=CORE_CNT)
            self.save(data=results, save_path=save_path)
            if VISUALIZE:
                geometry_plotter = GeometryPlotter(config_path=self.config_path, geometries=[white_animal_polygons, black_animal_polygons], video_name=video_name, core_cnt=CORE_CNT)
                geometry_plotter.run()
            video_timer.stop_timer()
            stdout_success(msg=f'{video_name} complete!', elapsed_time=video_timer.elapsed_time_str)

        self.session_timer.stop_timer()
        stdout_success(msg=f'{len(self.outlier_corrected_paths)} data files saved in {self.features_dir}')

    def save(self,
             data: pd.DataFrame,
             save_path: str):

        write_df(df=data, file_type=self.file_type, save_path=save_path)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='SimBA Custom Feature Extractor')
#     parser.add_argument('--config_path', type=str, help='SimBA project config path')
#     args = parser.parse_args()
#     feature_extractor = PiotrFeatureExtractor(config_path=args.config_path)
#     feature_extractor.run()


feature_extractor = PiotrFeatureExtractor(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/piotr/project_folder/project_config.ini')
feature_extractor.run()