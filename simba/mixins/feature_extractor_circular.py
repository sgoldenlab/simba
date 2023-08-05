import glob
import os
from typing import Union, List
import pandas as pd
import numpy as np

from simba.mixins.feature_extraction_circular_mixin import FeatureExtractionCircularMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_if_filepath_list_is_empty, check_if_dir_exists
from simba.utils.read_write import read_df, get_fn_ext, write_df
from simba.utils.errors import DataHeaderError, CountError
from simba.utils.lookups import cardinality_to_integer_lookup

class CircularFeatureExtractor(FeatureExtractionCircularMixin,
                               ConfigReader):

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 angle_fields: dict,
                 time_windows: np.ndarray,
                 save_dir: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path)
        FeatureExtractionCircularMixin.__init__(self)
        check_if_filepath_list_is_empty(filepaths=self.outlier_corrected_paths, error_msg=f'No data files detected in {self.outlier_corrected_dir}')
        check_if_dir_exists(in_dir=save_dir); self.cardinal_to_int_lk = cardinality_to_integer_lookup()
        for k, v in angle_fields.items():
            if len(v) is not 2 and not 3:
                raise CountError(f'The angle has to be computed using 2 or 3 fields. The input for animal {k} is {len(v)}')
        self.angle_fields, self.time_windows, self.save_dir = angle_fields, time_windows, save_dir

    def run(self):
        for file_cnt, file_path in enumerate(self.outlier_corrected_paths):
            _, video_name, _ = get_fn_ext(filepath=file_path)
            self.save_path = os.path.join(self.save_dir, f'{video_name}.{self.file_type}')
            self.df = read_df(file_path=file_path, file_type=self.file_type)
            _, px_per_mm, fps = self.read_video_info(video_name=video_name)
            if len(self.df.columns) != len(self.bp_headers):
                raise DataHeaderError(msg=f'The file {file_path} contains {len(self.df.columns)} columns. But your SimBA project suggest you should have {len(self.bp_headers)} body-part columns.')
            self.df.columns = self.bp_headers
            self.results = pd.DataFrame()
            self.angle_data = self.compute_directions()
            for animal_name, animal_angles in self.angle_data.items():
                self.results[f'Angle_degrees_{animal_name}'] = animal_angles
                mean_dispersions = self.rolling_mean_dispersion(data=animal_angles, time_windows=self.time_windows, fps=fps)
                for i in range(mean_dispersions.shape[1]):
                    self.results[f'Rolling_mean_dispersion_{animal_name}_time_window_{self.time_windows[i]}s'] = mean_dispersions[:, i]
                resultant_vector_length = self.rolling_resultant_vector_length(data=animal_angles, fps=fps, time_windows=self.time_windows)
                for i in range(resultant_vector_length.shape[1]):
                    self.results[f'Rolling_resultant_vector_length_{animal_name}_time_window_{self.time_windows[i]}s'] = resultant_vector_length[:, i]
                self.results[f'Compass_cardinal_{animal_name}'] = self.degrees_to_compass_cardinal(degree_angles=animal_angles)
                self.results[f'Compass_cardinal_{animal_name}'] = self.results[f'Compass_cardinal_{animal_name}'].map(self.cardinal_to_int_lk)

            write_df(df=pd.concat([self.df, self.results], axis=1), file_type=self.file_type, save_path=self.save_path)

    def compute_directions(self):
        angle_data = {}
        for k, v in self.angle_fields.items():
            if len(v) == 2:
                angle_data[k] = self.direction_two_bps(bp_x=self.df[[f'{v[0]}_x', f'{v[0]}_y']].values,
                                                       bp_y=self.df[[f'{v[1]}_x', f'{v[1]}_y']].values).astype(int)
        return angle_data



# test = CircularFeatureExtractor(config_path='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/project_config.ini',
#                                 angle_fields= {'Animal_1': ['SwimBladder', 'Tail1']},
#                                 time_windows=np.array([1.0, 2.0]),
#                                 save_dir='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/csv/circular_features')
# test.run()