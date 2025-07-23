import os
from typing import Optional, Tuple, Union

import numpy as np

from simba.mixins.geometry_mixin import GeometryMixin
from simba.plotting.geometry_plotter import GeometryPlotter
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_if_valid_rgb_tuple,
                                check_int, check_valid_dataframe,
                                check_video_and_data_frm_count_align)
from simba.utils.enums import Formats
from simba.utils.errors import NoFilesFoundError
from simba.utils.lookups import get_current_time
from simba.utils.read_write import (find_all_videos_in_directory,
                                    find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df)

VERTICE = 'vertice'
CENTER_X = 'center_x'
CENTER_Y = 'center_y'
NOSE_X = 'nose_x'
NOSE_Y = 'nose_y'
TAIL_X = 'tail_x'
TAIL_Y = 'tail_y'
LEFT_X = 'left_x'
LEFT_Y = 'left_y'
RIGHT_X = 'right_x'
RIGHT_Y = 'right_y'

BP_COLS = [NOSE_X, NOSE_Y, TAIL_X, TAIL_Y, LEFT_X, LEFT_Y, RIGHT_X, RIGHT_Y]

class BlobVisualizer():
    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 core_cnt: int = -1,
                 shape_opacity: float = 0.5,
                 bg_opacity: float = 1.0,
                 circle_size: Optional[int] = None,
                 hull: Optional[Tuple[int, int, int]] = (178, 102, 255),
                 anterior: Optional[Tuple[int, int, int]] = (0, 0, 255),
                 posterior: Optional[Tuple[int, int, int]] = (0, 128, 0),
                 center: Optional[Tuple[int, int, int]] = (0, 165, 255),
                 left: Optional[Tuple[int, int, int]] = (255, 51, 153),
                 right: Optional[Tuple[int, int, int]] = (255, 255, 102)):

        if os.path.isdir(data_path):
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=['.csv'], raise_error=True, as_dict=True)
        elif os.path.isfile(data_path):
            check_file_exist_and_readable(file_path=data_path)
            self.data_paths = {get_fn_ext(filepath=data_path)[1]: data_path}
        else:
            raise NoFilesFoundError(msg=f'{data_path} is not a valid file path or directory.', source=self.__class__.__name__)
        if os.path.isdir(video_path):
            self.video_paths = find_all_videos_in_directory(directory=video_path, raise_error=True, as_dict=True)
        elif os.path.isfile(video_path):
            check_file_exist_and_readable(file_path=video_path)
            self.video_paths = {get_fn_ext(filepath=video_path)[1]: video_path}
        else:
            raise NoFilesFoundError(msg=f'{video_path} is not a valid directory of file path.')
        for k, v in self.data_paths.items():
            if k not in self.video_paths.keys():
                raise NoFilesFoundError(msg=f'Could not find a video for blob tracking data file {k} in the {video_path} directory', source=self.__class__.__name__)
        check_if_dir_exists(in_dir=save_dir, create_if_not_exist=True)
        check_int(name='core count', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        self.core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
        check_float(name='shape_opacity', value=shape_opacity, max_value=1.0, min_value=0.1, raise_error=True)
        check_float(name='bg_opacity', value=bg_opacity, max_value=1.0, min_value=0.1, raise_error=True)
        if circle_size is not None:
            check_int(name='circle_size', value=circle_size, min_value=1)
        self.circle_size = circle_size
        self.bg_opacity, self.shape_opacity = bg_opacity, shape_opacity
        self.save_dir = save_dir
        for i in [hull, anterior, posterior, center, left, right]:
            if i is not None:
                check_if_valid_rgb_tuple(data=i)
        self.anterior, self.posterior, self.center, self.left, self.right, self.hull = anterior, posterior, center, left, right, hull
        self.colors = [x for x in [self.hull, self.anterior, self.posterior, self.center, self.left, self.right] if x is not None]

    def run(self):
        print(f'Running blob visualization for {len(list(self.data_paths.keys()))} video(s) (start time: {get_current_time()})...')
        for file_name, file_path in self.data_paths.items():
            print(f'Running blob visualization for video {file_name}...')
            geometries = []
            video_path = self.video_paths[file_name]
            df = read_df(file_path=file_path, file_type='csv')
            check_video_and_data_frm_count_align(video=video_path, data=df, name=file_name, raise_error=True)
            check_valid_dataframe(df=df, source='', valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=BP_COLS)
            if self.hull:
                vertice_cols = [x for x in df.columns if VERTICE in x]
                vertice_arr = df[vertice_cols].values.reshape(len(df), int(len(vertice_cols) / 2), 2).astype(np.int32)
                geometries.append(GeometryMixin().bodyparts_to_polygon(data=vertice_arr, simplify_tolerance=2, convex_hull=False))
            if self.anterior is not None:
                geometries.append(GeometryMixin.bodyparts_to_points(data=df[[NOSE_X, NOSE_Y]].values))
            if self.posterior is not None:
                geometries.append(GeometryMixin.bodyparts_to_points(data=df[[TAIL_X, TAIL_Y]].values))
            if self.center is not None:
                geometries.append(GeometryMixin.bodyparts_to_points(data=df[[CENTER_X, CENTER_Y]].values))
            if self.left is not None:
                geometries.append(GeometryMixin.bodyparts_to_points(data=df[[LEFT_X, LEFT_Y]].values))
            if self.right is not None:
                geometries.append(GeometryMixin.bodyparts_to_points(data=df[[RIGHT_X, RIGHT_Y]].values))

            plotter = GeometryPlotter(geometries=geometries,
                                      video_name=video_path,
                                      save_dir=self.save_dir,
                                      core_cnt=self.core_cnt,
                                      palette=None,
                                      colors=self.colors,
                                      bg_opacity=self.bg_opacity,
                                      shape_opacity=self.shape_opacity,
                                      circle_size=self.circle_size)
            plotter.run()
            # #

            # img = read_frm_of_video(video_path=video_path, frame_index=0)
            #
            # cv2.imshow('asdasdas', img)
            # cv2.waitKey(5000)



# visualizer = BlobVisualizer(data_path=r'D:\troubleshooting\netholabs\original_videos\whiskers\results',
#                             video_path=r'D:\troubleshooting\netholabs\original_videos\whiskers',
#                             save_dir=r"D:\troubleshooting\netholabs\original_videos\whiskers\out_video",
#                             core_cnt=15,
#                             posterior=None,
#                             left=None,
#                             right=None)
# visualizer.run()

# visualizer = BlobVisualizer(data_path=r'D:\EPM_4\data', video_path=r'D:\EPM_4\original', save_dir=r"D:\EPM_4\out_videos")
# visualizer.run()


# visualizer = BlobVisualizer(data_path=r'D:\water_maze\data', video_path=r'D:\water_maze\original', save_dir=r"D:\water_maze\out_videos")
# visualizer.run()
#
# #
#visualizer = BlobVisualizer(data_path=r'D:\open_field_3\sample\blob_data', video_path=r'D:\open_field_3\sample', save_dir=r"D:\open_field_3\sample\out_videos")
#visualizer = BlobVisualizer(data_path=r'D:\EPM\large_sample\out', video_path=r'D:\EPM\large_sample', save_dir=r'D:\EPM\videos_out')

#visualizer = BlobVisualizer(data_path=r'D:\EPM\sampled\data', video_path=r'D:\EPM\sampled', save_dir=r"D:\EPM\sampled\out_videos")
# visualizer = BlobVisualizer(data_path=r'D:\open_field_2\sample\clipped_10min\data', video_path=r'D:\open_field_2\sample\clipped_10min', save_dir=r"D:\open_field_2\sample\clipped_10min\videos_out")

#visualizer = BlobVisualizer(data_path=r'D:\open_field_4\data', video_path=r'D:\open_field_4', save_dir="D:\open_field_4\out_videos")
#visualizer = BlobVisualizer(data_path=r'D:\open_field\data', video_path=r'D:\open_field', save_dir=r"D:\open_field\videos")
#visualizer = BlobVisualizer(data_path=r'D:\EPM_3\out', video_path=r'D:\EPM_3', save_dir=r"D:\EPM_3\out_videos")

# visualizer = BlobVisualizer(data_path=r'D:\FST_SIDE\out', video_path=r'D:\FST_SIDE', save_dir=r"D:\FST_SIDE\videos")
# #
# # #visualizer = BlobVisualizer(data_path=r'D:\EPM\data', video_path=r'D:\EPM\sampled', save_dir=r"D:\EPM\videos_out")
# visualizer.run()

# visualizer = BlobVisualizer(data_path=r'D:\OF_7\data', video_path=r'D:\OF_7\original', save_dir=r'D:\OF_7\out_videos')
# visualizer.run()