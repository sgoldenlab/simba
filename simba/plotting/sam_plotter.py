import os
from typing import List, Optional, Tuple, Union

import numpy as np

from simba.mixins.geometry_mixin import GeometryMixin
from simba.plotting.geometry_plotter import GeometryPlotter
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_if_valid_rgb_tuple,
                                check_instance, check_int,
                                check_valid_dataframe)
from simba.utils.enums import Formats
from simba.utils.errors import CountError, NoFilesFoundError
from simba.utils.lookups import get_current_time
from simba.utils.read_write import (find_all_videos_in_directory,
                                    find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df)

VERTICE = 'VERTICE'
FRAME = 'FRAME'
NAME = 'NAME'


class SamVisualizer():
    """
    :example:
    >>> r = SamVisualizer(data_path=r"C:\troubleshooting\sam_results\10B_Mouse_5-choice_MustTouchTrainingNEWFINAL_a7_clipped_3.csv", video_path=r"D:\platea\platea_videos\videos\clipped\10B_Mouse_5-choice_MustTouchTrainingNEWFINAL_a7_clipped_3.mp4", save_dir='D:\cvat_annotations\sam_videos', color=[(255, 255, 1)])
    >>> r.run()
    """

    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 core_cnt: int = -1,
                 color: Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]] = (178, 102, 255),
                 shape_opacity: float = 0.5,
                 bg_opacity: float = 1.0):

        if os.path.isdir(data_path):
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=['.csv'],
                                                                   raise_error=True, as_dict=True)
        elif os.path.isfile(data_path):
            check_file_exist_and_readable(file_path=data_path)
            self.data_paths = {get_fn_ext(filepath=data_path)[1]: data_path}
        else:
            raise NoFilesFoundError(msg=f'{data_path} is not a valid file path or directory.',
                                    source=self.__class__.__name__)
        if os.path.isdir(video_path):
            self.video_paths = find_all_videos_in_directory(directory=video_path, raise_error=True, as_dict=True)
        elif os.path.isfile(video_path):
            check_file_exist_and_readable(file_path=video_path)
            self.video_paths = {get_fn_ext(filepath=video_path)[1]: video_path}
        else:
            raise NoFilesFoundError(msg=f'{video_path} is not a valid directory of file path.')
        for k, v in self.data_paths.items():
            if k not in self.video_paths.keys():
                raise NoFilesFoundError(
                    msg=f'Could not find a video for SAM tracking data file {k} in the {video_path} directory',
                    source=self.__class__.__name__)
        check_if_dir_exists(in_dir=save_dir, create_if_not_exist=True)
        check_int(name='core count', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        self.core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
        check_float(name='shape_opacity', value=shape_opacity, max_value=1.0, min_value=0.1, raise_error=True)
        check_float(name='bg_opacity', value=bg_opacity, max_value=1.0, min_value=0.1, raise_error=True)
        check_instance(source=f'{self.__class__.__name__} color', accepted_types=(list, tuple,), instance=color)
        if isinstance(color, (tuple,)):
            check_if_valid_rgb_tuple(data=color, raise_error=True, source=self.__class__.__name__)
            color = [color]
        else:
            for c in color:
                check_if_valid_rgb_tuple(data=c)
        self.bg_opacity, self.shape_opacity, self.colors = bg_opacity, shape_opacity, color
        self.save_dir = save_dir

    def run(self):
        print(
f'Running SAM visualization for {len(list(self.data_paths.keys()))} video(s) (start time: {get_current_time()})...')
        for file_name, file_path in self.data_paths.items():
            print(f'Running SAM visualization for video {file_name}...')
            geometries = []
            video_path = self.video_paths[file_name]
            df = read_df(file_path=file_path, file_type='csv')
            animal_names = df[NAME].unique()
            animal_names = [x for x in animal_names if x != -1]
            if len(animal_names) != len(self.colors):
                raise CountError(msg=f'{len(animal_names)} animal names ({animal_names}) found in data {file_path} but {len(self.colors)} color(s) passed: {self.colors}', source=self.__class__.__name__)
            # check_video_and_data_frm_count_align(video=video_path, data=df, name=file_name, raise_error=True)
            check_valid_dataframe(df=df, source='', valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=[FRAME, NAME])
            for animal_name in animal_names:
                animal_df = df[df[NAME] == animal_name].reset_index(drop=True)
                vertice_cols = [x for x in animal_df.columns if VERTICE in x]
                animal_vertice_arr = animal_df[vertice_cols].values.reshape(len(animal_df), int(len(vertice_cols) / 2), 2).astype(np.int32)
                geometries.append(GeometryMixin().bodyparts_to_polygon(data=animal_vertice_arr, simplify_tolerance=2, convex_hull=False))

            plotter = GeometryPlotter(geometries=geometries,
                                      video_name=video_path,
                                      save_dir=self.save_dir,
                                      core_cnt=self.core_cnt,
                                      palette=None,
                                      colors=self.colors,
                                      bg_opacity=self.bg_opacity,
                                      shape_opacity=self.shape_opacity,
                                      circle_size=None)
            plotter.run()
            #


r = SamVisualizer(
    data_path=r"D:\cvat_annotations\sam_yolo_data\s34-drinking.csv",
    video_path=r"D:\cvat_annotations\videos\mp4_20250624155703\s34-drinking.mp4",
    save_dir='D:\cvat_annotations\sam_videos', color=[(255, 255, 1)])
r.run()