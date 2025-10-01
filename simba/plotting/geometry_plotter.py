import functools
import multiprocessing
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import (LineString, MultiLineString, MultiPolygon, Point,
                              Polygon)

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_float, check_if_dir_exists,
                                check_if_valid_rgb_tuple, check_instance,
                                check_int, check_iterable_length, check_str,
                                check_valid_boolean, check_valid_lst)
from simba.utils.data import create_color_palettes
from simba.utils.enums import Defaults, Formats
from simba.utils.errors import InvalidInputError
from simba.utils.lookups import get_color_dict
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, find_video_of_file,
                                    get_fn_ext, get_video_meta_data,
                                    remove_a_folder)
from simba.utils.warnings import FrameRangeWarning

ACCEPTED_TYPES = [Polygon, LineString, MultiPolygon, MultiLineString, Point]
FRAME_COUNT = "frame_count"

def geometry_visualizer(data: Tuple[int, pd.DataFrame],
                        video_path: Union[str, os.PathLike],
                        video_temp_dir: Union[str, os.PathLike],
                        video_meta_data: dict,
                        thickness: int,
                        verbose: bool,
                        intersection_clr: Optional[Tuple[int, int, int]],
                        bg_opacity: float,
                        colors: list,
                        circle_size: int,
                        shape_opacity: float):

    group, idx = int(data[0]), data[1].index.tolist()
    start_frm, end_frm = idx[0], idx[-1]
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    video_save_path = os.path.join(video_temp_dir, f"{group}.mp4")
    video_writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
    cap = cv2.VideoCapture(video_path)
    batch_shapes = data[1].values.reshape(len(data[1]), -1)
    for frm_cnt, frm_id in enumerate(range(start_frm, end_frm + 1)):
        cap.set(1, int(frm_id))
        ret, img = cap.read()
        if ret:
            img_cpy = img.copy()
            if bg_opacity != 1.0:
                opacity = 1 - bg_opacity
                h, w, clr = img.shape[:3]
                opacity_image = np.ones((h, w, clr), dtype=np.uint8) * int(255 * opacity)
                img = cv2.addWeighted(img.astype(np.uint8), 1 - opacity, opacity_image.astype(np.uint8), opacity, 0)
            for shape_cnt, shape in enumerate(batch_shapes[frm_cnt]):
                shape_clr = colors[shape_cnt]
                if intersection_clr is not None and shape is not None:
                    current_shapes = batch_shapes[frm_cnt]
                    for i in range(len(current_shapes)):
                        if i != shape_cnt and current_shapes[i] is not None and shape.intersects(current_shapes[i]):
                            shape_clr = intersection_clr
                            break
                if isinstance(shape, Polygon):
                    img_cpy = cv2.fillPoly(img_cpy, [np.array(shape.exterior.coords).astype(np.int32)], color=shape_clr)
                    interior_coords = [np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2)) for interior in shape.interiors]
                    for interior in interior_coords:
                        img_cpy = cv2.fillPoly(img_cpy, [interior], color=(shape_clr[::-1]))
                elif isinstance(shape, LineString):
                    img_cpy = cv2.fillPoly(img_cpy, [np.array(shape.coords, dtype=np.int32)], color=shape_clr)
                elif isinstance(shape, MultiPolygon):
                    for polygon_cnt, polygon in enumerate(shape.geoms):
                        img_cpy = cv2.fillPoly(img_cpy, [np.array((polygon.convex_hull.exterior.coords), dtype=np.int32)], color=shape_clr)
                elif isinstance(shape, MultiLineString):
                    for line_cnt, line in enumerate(shape.geoms):
                        img_cpy = cv2.fillPoly(img_cpy,[np.array(shape[line_cnt].coords, dtype=np.int32)], color=shape_clr)
                elif isinstance(shape, Point):
                    arr = np.array((shape.coords)).astype(np.int32)
                    x, y = arr[0][0], arr[0][1]
                    img_cpy = cv2.circle(img_cpy,(x, y), circle_size, shape_clr, thickness)
            if shape_opacity is not None:
                img = cv2.addWeighted(img_cpy, shape_opacity, img, 1 - shape_opacity, 0, img)
            else:
                img = np.copy(img_cpy)
            video_writer.write(img.astype(np.uint8))
            if verbose:
                print(f"Creating frame {frm_id} / {video_meta_data['frame_count']} (CPU core: {group}, video name: {video_meta_data['video_name']})")
        else:
            FrameRangeWarning(msg=f'Frame {frm_id} in video {video_meta_data["video_name"]} could not be read.')
            pass
    video_writer.release()
    cap.release()

    return group


class GeometryPlotter(ConfigReader, PlottingMixin):
    """
    A class for creating overlay geometry visualization videos based on provided geometries and video name.

    .. seealso::
       To quickly create static geometries on a white background (useful for troubleshooting unexpected geometries), see :func:`simba.mixins.geometry_mixin.GeometryMixin.view_shapes`
       and :func:`simba.mixins.geometry_mixin.GeometryMixin.geometry_video`

    .. image:: _static/img/GeometryPlotter.gif
       :width: 600
       :align: center

    .. video:: _static/img/GeometryPlotter_1.webm
       :width: 900
       :autoplay:
       :loop:

    .. video:: _static/img/GeometryPlotter_2.webm
       :width: 600
       :autoplay:
       :loop:

    :param List[List[Union[Polygon, LineString, MultiPolygon, MultiLineString, Point]]] geometries: List of lists of geometries for each frame. Each list contains as many entries as frames. Each list may represent a track or unique tracked object.
    :param Union[str, os.PathLike] video_name: Name of the input video.
    :param Optional[Union[str, os.PathLike]] config_path: Path to SimBA configuration file. Default: None.
    :param Optional[int] core_cnt: Number of CPU cores to use for parallel processing. Default: -1 (all available cores).
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save output videos. Default: None.
    :param Optional[int] thickness: Thickness of geometry outlines in pixels. Default: None.
    :param Optional[int] circle_size: Size of circles for Point geometries. Default: None.
    :param Optional[float] bg_opacity: Background video opacity (0.0-1.0). Default: 1.0.
    :param float shape_opacity: Shape fill opacity (0.0-1.0). Default: 0.3.
    :param Optional[str] palette: Color palette name for geometries. Default: None. Provide either a `palette` name, or a list of `colors`. If both are passed, `palette` is used.
    :param Optional[List[Union[str, Tuple[int, int, int]]]] colors: Custom colors for geometries. Default: None.
    :param Optional[Tuple[int, int, int]] intersection_clr: Color for geometries that intersect other geometries. Default to None (which means intersecting geometries maintain the original color while intersecting.
    :param Optional[bool] verbose: Print progress information. Default: True.
    :raises InvalidInputError: If the provided geometries contain invalid data types or if neither palette nor colors are provided.
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
                 intersection_clr: Optional[Tuple[int, int, int]] = None,
                 bg_opacity: Optional[float] = 1,
                 shape_opacity: float = 0.3,
                 palette: Optional[str] = None,
                 colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
                 verbose: Optional[bool] = True):

        PlottingMixin.__init__(self)
        check_instance(source=self.__class__.__name__, instance=geometries, accepted_types=list)
        check_iterable_length(source=self.__class__.__name__, val=len(geometries), min=1)
        if circle_size is not None:
            check_int(name="circle_size", value=circle_size, min_value=1)
        if thickness is not None:
            check_int(name="thickness", value=thickness, min_value=1)
        check_float(name="video_opacity", value=bg_opacity, min_value=0.0, max_value=1.0)
        check_float(name="shape_opacity", value=shape_opacity, min_value=0.0, max_value=1.0)
        check_valid_boolean(value=verbose, source='verbose', raise_error=True)
        self.color_dict = get_color_dict()
        check_int(name="CORE COUNT", value=core_cnt, min_value=-1, raise_error=True, unaccepted_vals=[0])
        self.core_cnt = core_cnt
        if palette is None and colors is None:
            raise InvalidInputError(msg='Pass palette or colors', source=self.__class__.__name__)
        if core_cnt == -1 or core_cnt > find_core_cnt()[0]:
            self.core_cnt = find_core_cnt()[0]
        if config_path is not None:
            ConfigReader.__init__(self, config_path=config_path)
        if os.path.isfile(video_name):
            self.video_path = video_name
        else:
            if config_path is None:
                raise InvalidInputError(msg=f'When providing a non-path video name, pass config_path')
            self.video_path = find_video_of_file(video_dir=self.video_dir, filename=video_name, raise_error=True)
        if intersection_clr is not None:
            check_if_valid_rgb_tuple(data=intersection_clr, raise_error=True, source=f'{self.__class__.__name__} intersection_clr')
        video_name = get_fn_ext(filepath=self.video_path)[1]
        self.video_meta_data = get_video_meta_data(video_path=self.video_path)
        self.shape_opacity = shape_opacity
        if circle_size is None:
            circle_size = self.get_optimal_circle_size(frame_size=(self.video_meta_data['width'], self.video_meta_data['height']), circle_frame_ratio=50)
        if thickness is None:
            thickness = circle_size
        if palette is None:
            self.colors = []
            check_valid_lst(data=colors, source=f'{self.__class__.__name__} colors', valid_dtypes=(str, tuple), exact_len=len(geometries))
            for clr in colors:
                if isinstance(clr, str):
                    check_str(name=f'{self.__class__.__name__} colors', value=clr, options=list(self.color_dict.keys()))
                else:
                    check_if_valid_rgb_tuple(data=clr)
                self.colors.append(clr)
        else:
            colors = create_color_palettes(no_animals=1, map_size=len(geometries) + 1, cmaps=[palette])
            self.colors = [x for xs in colors for x in xs]
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
        if os.path.isdir(self.temp_dir): remove_a_folder(folder_dir=self.temp_dir, ignore_errors=False)
        os.makedirs(self.temp_dir)
        self.save_path = os.path.join(self.save_dir, f'{video_name}.mp4')
        self.verbose, self.bg_opacity, self.palette = verbose, bg_opacity, palette
        self.circles_size, self.intersection_clr = circle_size, intersection_clr

    def run(self):
        video_timer = SimbaTimer(start=True)
        data = pd.DataFrame(self.geometries).T
        #data = data.head(100)
        data = np.array_split(data, self.core_cnt)
        data_splits = []
        for i in range(len(data)): data_splits.append((i, data[i]))
        del data

        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(geometry_visualizer,
                                          video_path=self.video_path,
                                          video_temp_dir=self.temp_dir,
                                          video_meta_data=self.video_meta_data,
                                          thickness=self.thickness,
                                          verbose=self.verbose,
                                          bg_opacity=self.bg_opacity,
                                          intersection_clr=self.intersection_clr,
                                          colors=self.colors,
                                          circle_size=self.circles_size,
                                          shape_opacity=self.shape_opacity)
            for cnt, result in enumerate(pool.imap(constants, data_splits, chunksize=1)):
                print(f"Section {result}/{len(data_splits)} complete...")
            pool.terminate()
            pool.join()

        print(f"Joining {self.video_name} geometry video...")
        concatenate_videos_in_folder(in_folder=self.temp_dir, save_path=self.save_path, remove_splits=True)
        video_timer.stop_timer()
        stdout_success(msg=f"Geometry video {self.save_path} complete!", elapsed_time=video_timer.elapsed_time_str, source=self.__class__.__name__)



# if __name__ == '__main__':
#     from simba.utils.read_write import read_pickle
#     VIDEO_PATH = r"D:\ares\data\termite_2\termite.mp4"
#     DATA_PATH = r"D:\ares\data\termite_2\termite_2_geometries.pickle"
#     data = read_pickle(data_path=DATA_PATH)
#
#     get_video_meta_data(video_path=VIDEO_PATH)
#
#     geos = []
#
#     for k, v in data.items():
#         geos.append(list(v.values()))
#         #geos.append([subdict[i] for subdict in data.values()])
#
#     plotter = GeometryPlotter(geometries=geos, video_name=VIDEO_PATH, save_dir=r"D:\ares\data\termite_2\video", palette='Set1', core_cnt=32, bg_opacity=1)
#     plotter.run()
#     #max_frm = 9000
