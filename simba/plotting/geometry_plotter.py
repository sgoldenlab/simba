import functools
import multiprocessing
import os
from typing import List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import (LineString, MultiLineString, MultiPolygon, Point,
                              Polygon)

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_float, check_if_dir_exists,
                                check_instance, check_int,
                                check_iterable_length, check_valid_boolean)
from simba.utils.data import create_color_palettes
from simba.utils.enums import Defaults, Formats
from simba.utils.errors import InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, find_video_of_file,
                                    get_fn_ext, get_video_meta_data)
from simba.utils.warnings import FrameRangeWarning

ACCEPTED_TYPES = [Polygon, LineString, MultiPolygon, MultiLineString, Point]
FRAME_COUNT = "frame_count"

def geometry_visualizer(data: np.ndarray,
                        video_path: Union[str, os.PathLike],
                        video_temp_dir: Union[str, os.PathLike],
                        video_meta_data: dict,
                        thickness: int,
                        verbose: bool,
                        bg_opacity: float,
                        palette: str,
                        circle_size: int,
                        shape_cnt: int):

    group = int(data[0][-1])
    colors = create_color_palettes(no_animals=1, map_size=shape_cnt, cmaps=[palette])
    colors = [x for xs in colors for x in xs]
    start_frm, end_frm = data[0][-2], data[-1][-2]
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    video_save_path = os.path.join(video_temp_dir, f"{group}.mp4")
    video_writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
    cap = cv2.VideoCapture(video_path)
    for frm_cnt, frm_id in enumerate(range(start_frm, end_frm + 1)):
        cap.set(1, int(frm_id))
        _, img = cap.read()
        if bg_opacity != 1.0:
            opacity = 1 - bg_opacity
            h, w, clr = img.shape[:3]
            opacity_image = np.ones((h, w, clr), dtype=np.uint8) * int(255 * opacity)
            img = cv2.addWeighted(img.astype(np.uint8), 1 - opacity, opacity_image.astype(np.uint8), opacity, 0)
        for shape_cnt, shape in enumerate(data[frm_cnt][0:-2]):
            if isinstance(shape, Polygon):
                cv2.polylines(img, [np.array(shape.exterior.coords).astype(np.int32)], True, (colors[shape_cnt][::-1]), thickness=thickness)
                interior_coords = [np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2)) for interior in shape.interiors]
                for interior in interior_coords:
                    cv2.polylines(img, [interior], isClosed=True, color=(colors[shape_cnt][::-1]), thickness=thickness)
            elif isinstance(shape, LineString):
                cv2.polylines(img, [np.array(shape.coords, dtype=np.int32)], False, (colors[shape_cnt][::-1]), thickness=thickness)
            elif isinstance(shape, MultiPolygon):
                for polygon_cnt, polygon in enumerate(shape.geoms):
                    cv2.polylines(img, [np.array((polygon.convex_hull.exterior.coords), dtype=np.int32)], True, (colors[shape_cnt + polygon_cnt + 1][::-1]), thickness=thickness)
            elif isinstance(shape, MultiLineString):
                for line_cnt, line in enumerate(shape.geoms):
                    cv2.polylines(img, [np.array(shape[line_cnt].coords, dtype=np.int32)], False, (colors[shape_cnt][::-1]), thickness=thickness)
            elif isinstance(shape, Point):
                arr = np.array((shape.coords)).astype(np.int32)
                x, y = arr[0][0], arr[0][1]
                cv2.circle(img,(x, y),circle_size, colors[shape_cnt][::-1], thickness)
        video_writer.write(img.astype(np.uint8))
        if verbose:
            print(f"Creating frame {frm_id} (CPU core: {group})")
    video_writer.release()
    cap.release()

    return group


class GeometryPlotter(ConfigReader, PlottingMixin):
    """
    A class for creating overlay geometry visualization videos based on provided geometries and video name.

    .. seealso::
       To quickly create static geometries on a white background (useful for troubleshooting unexpected geometries), see :func:`simba.mixins.geometry_mixin.GeometryMixin.view_shapes`
       and :func:`:func:`simba.mixins.geometry_mixin.GeometryMixin.geometry_video`

    .. image:: _static/img/GeometryPlotter.gif
       :width: 600
       :align: center

    :param config_path: Union[str, os.PathLike]: Path to SimBA configuration file.
    :param geometries: List[List[Union[Polygon, LineString, MultiPolygon, MultiLineString, Point]]]: List of lists of geometries for each frame.
    :param video_name: str: Name of the input video.
    :param core_cnt: int, optional: Number of CPU cores to use for parallel processing. Defaults to -1 represnting all available cores
    :raises InvalidInputError: If the provided geometries contain invalid data types.
    :raises CountError: If the number of shapes in the geometries does not match the number of frames in the video.
    """

    def __init__(self,
                 geometries: List[List[Union[Polygon, LineString, MultiPolygon, MultiLineString, Point]]],
                 video_name: Union[str, os.PathLike],
                 config_path: Optional[Union[str, os.PathLike]] = None,
                 core_cnt: Optional[int] = -1,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 thickness: Optional[int] = None,
                 circle_size: Optional[int] = None,
                 bg_opacity: Optional[float] = 1,
                 palette: Optional[str] = 'jet',
                 verbose: Optional[bool] = True):

        PlottingMixin.__init__(self)
        check_instance(source=self.__class__.__name__, instance=geometries, accepted_types=list)
        check_iterable_length(source=self.__class__.__name__, val=len(geometries), min=1)
        if circle_size is not None:
            check_int(name="circle_size", value=circle_size, min_value=1)
        if thickness is not None:
            check_int(name="thickness", value=thickness, min_value=1)
        check_float(name="video_opacity", value=bg_opacity, min_value=0.0, max_value=1.0)
        check_valid_boolean(value=verbose, source='verbose', raise_error=True)
        check_int(name="CORE COUNT", value=core_cnt, min_value=-1, raise_error=True, unaccepted_vals=[0])
        self.core_cnt = core_cnt
        if core_cnt == -1 or core_cnt > find_core_cnt()[0]:
            self.core_cnt = core_cnt
        if config_path is not None:
            ConfigReader.__init__(self, config_path=config_path)
        if os.path.isfile(video_name):
            self.video_path = video_name
        else:
            if config_path is None:
                raise InvalidInputError(msg=f'When providing a non-path video name, pass config_path')
            self.video_path = find_video_of_file(video_dir=self.video_dir, filename=video_name, raise_error=True)
        video_name = get_fn_ext(filepath=self.video_path)[1]
        self.video_meta_data = get_video_meta_data(video_path=self.video_path)
        if circle_size is None:
            circle_size = self.get_optimal_circle_size(frame_size=(self.video_meta_data['width'], self.video_meta_data['height']), circle_frame_ratio=100)
        if thickness is None:
            thickness = circle_size
        for i in range(len(geometries)):
            if len(geometries[i]) != self.video_meta_data[FRAME_COUNT]:
                FrameRangeWarning(msg=f"Geometry {i+1} contains {len(geometries[i])} shapes but video has {self.video_meta_data[FRAME_COUNT]} frames")
        self.geometries, self.video_name, self.thickness = geometries, video_name, thickness
        if config_path is None:
            check_if_dir_exists(in_dir=save_dir)
            self.save_dir = save_dir
        else:
            self.save_dir = os.path.join(self.frames_output_dir, "geometry_visualization")
        if not os.path.isdir(self.save_dir): os.makedirs(self.save_dir)
        self.temp_dir = os.path.join(self.save_dir, video_name, "temp")
        if not os.path.isdir(self.temp_dir): os.makedirs(self.temp_dir)
        self.save_path = os.path.join(self.save_dir, f'{video_name}.mp4')
        self.verbose, self.bg_opacity, self.palette = verbose, bg_opacity, palette
        self.circles_size = circle_size

    def run(self):
        video_timer = SimbaTimer(start=True)
        data = pd.DataFrame(np.array(self.geometries).T)
        data, obs_per_split = self.split_and_group_df(df=data, splits=self.core_cnt, include_row_index=True, include_split_order=False)

        for i in range(len(data)):
            new_col = np.full(len(data[i]), fill_value=i).reshape(-1, 1)
            data[i] = np.concatenate((data[i], new_col), axis=1)

        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(geometry_visualizer,
                                          video_path=self.video_path,
                                          video_temp_dir=self.temp_dir,
                                          video_meta_data=self.video_meta_data,
                                          thickness=self.thickness,
                                          verbose=self.verbose,
                                          bg_opacity=self.bg_opacity,
                                          palette=self.palette,
                                          circle_size=self.circles_size,
                                          shape_cnt=len(self.geometries))
            for cnt, result in enumerate(pool.imap(constants, data, chunksize=1)):
                print(f"Section {result}/{len(data)} complete...")
            pool.terminate()
            pool.join()

        print(f"Joining {self.video_name} geometry video...")
        concatenate_videos_in_folder(in_folder=self.temp_dir, save_path=self.save_path, remove_splits=True)
        video_timer.stop_timer()
        stdout_success(msg=f"Geometry video {self.save_path} complete!", elapsed_time=video_timer.elapsed_time_str, source=self.__class__.__name__)
