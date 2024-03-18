import functools
import multiprocessing
import os
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import (LineString, MultiLineString, MultiPolygon, Point,
                              Polygon)

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_instance, check_int,
                                check_iterable_length, check_str)
from simba.utils.enums import Defaults, Formats, TextOptions
from simba.utils.errors import CountError, InvalidInputError
from simba.utils.lookups import get_color_dict
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, find_video_of_file,
                                    get_video_meta_data)
from simba.utils.warnings import FrameRangeWarning

ACCEPTED_TYPES = [Polygon, LineString, MultiPolygon, MultiLineString, Point]
FRAME_COUNT = "frame_count"


def geometry_visualizer(
    data: np.ndarray,
    video_path: Union[str, os.PathLike],
    video_temp_dir: Union[str, os.PathLike],
    video_meta_data: dict,
):
    group = int(data[0][-1])
    colors = list(get_color_dict().values())
    start_frm, end_frm = data[0][-2], data[-1][-2]
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    video_save_path = os.path.join(video_temp_dir, "{}.mp4".format(str(group)))
    video_writer = cv2.VideoWriter(
        video_save_path,
        fourcc,
        video_meta_data["fps"],
        (video_meta_data["width"], video_meta_data["height"]),
    )
    cap = cv2.VideoCapture(video_path)
    for frm_cnt, frm_id in enumerate(range(start_frm, end_frm + 1)):
        cap.set(1, int(frm_id))
        _, img = cap.read()
        for shape_cnt, shape in enumerate(data[frm_cnt][0:-2]):
            if isinstance(shape, Polygon):
                cv2.polylines(
                    img,
                    [np.array(shape.exterior.coords).astype(np.int)],
                    True,
                    (colors[shape_cnt][::-1]),
                    thickness=TextOptions.LINE_THICKNESS.value,
                )
                interior_coords = [
                    np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
                    for interior in shape.interiors
                ]
                for interior in interior_coords:
                    cv2.polylines(
                        img,
                        [interior],
                        isClosed=True,
                        color=(colors[shape_cnt][::-1]),
                        thickness=TextOptions.LINE_THICKNESS.value,
                    )
            if isinstance(shape, LineString):
                cv2.polylines(
                    img,
                    [np.array(shape.coords, dtype=np.int32)],
                    False,
                    (colors[shape_cnt][::-1]),
                    thickness=TextOptions.LINE_THICKNESS.value,
                )
            if isinstance(shape, MultiPolygon):
                for polygon_cnt, polygon in enumerate(shape.geoms):
                    cv2.polylines(
                        img,
                        [
                            np.array(
                                (polygon.convex_hull.exterior.coords), dtype=np.int32
                            )
                        ],
                        True,
                        (colors[shape_cnt + polygon_cnt + 1][::-1]),
                        thickness=TextOptions.LINE_THICKNESS.value,
                    )
            if isinstance(shape, MultiLineString):
                for line_cnt, line in enumerate(shape.geoms):
                    cv2.polylines(
                        img,
                        [np.array(shape[line_cnt].coords, dtype=np.int32)],
                        False,
                        (colors[shape_cnt][::-1]),
                        thickness=TextOptions.LINE_THICKNESS.value,
                    )
            if isinstance(shape, Point):
                cv2.circle(
                    img,
                    (
                        int(np.array(shape.centroid)[0]),
                        int(np.array(shape.centroid)[1]),
                    ),
                    0,
                    colors[shape_cnt][::-1],
                    TextOptions.LINE_THICKNESS.value,
                )

        video_writer.write(img.astype(np.uint8))
        print(f"Creating frame {frm_id} (core: {group})")
    video_writer.release()
    cap.release()

    return group


class GeometryPlotter(ConfigReader, PlottingMixin):
    """
    A class for creating overlay geometry visualization videos based on provided geometries and video name.

    .. note::
       To quickly create static geometries on a white background (useful for troubleshooting unexpected geometries), use
       :meth:`simba.mixins.geometry_mixin.GeometryMixin.view_shapes`.

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

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        geometries: List[List[Union[Polygon, LineString]]],
        video_name: str,
        core_cnt: Optional[int] = -1,
        save_path: Optional[Union[str, os.PathLike]] = None,
    ):
        check_file_exist_and_readable(file_path=config_path)
        check_instance(
            source=self.__class__.__name__, instance=geometries, accepted_types=list
        )
        check_iterable_length(
            source=self.__class__.__name__, val=len(geometries), min=1
        )
        check_str(name="Video name", value=video_name)
        shape_types = set()
        for i in geometries:
            shape_types.update(set([type(x) for x in i]))
        for i in shape_types:
            if i not in ACCEPTED_TYPES:
                raise InvalidInputError(
                    msg=f"geometries contain an invalid datatype {i}. Accepted: {ACCEPTED_TYPES}",
                    source=self.__class__.__name__,
                )
        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        self.video_path = find_video_of_file(
            video_dir=self.video_dir, filename=video_name
        )
        self.video_meta_data = get_video_meta_data(video_path=self.video_path)
        for i in range(len(geometries)):
            if len(geometries[i]) != self.video_meta_data[FRAME_COUNT]:
                FrameRangeWarning(
                    msg=f"Geometry {i+1} contains {len(geometries[i])} shapes but video has {self.video_meta_data[FRAME_COUNT]} frames"
                )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        self.geometries, self.core_cnt, self.video_name = (
            geometries,
            core_cnt,
            video_name,
        )
        self.geometry_dir = os.path.join(
            self.frames_output_dir, "geometry_visualization"
        )
        if not os.path.isdir(self.geometry_dir):
            os.makedirs(self.geometry_dir)
        self.temp_dir = os.path.join(
            self.frames_output_dir, self.geometry_dir, video_name, "temp"
        )
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)
        if save_path is None:
            self.save_path = os.path.join(self.geometry_dir, f"{video_name}.mp4")
        else:
            check_if_dir_exists(in_dir=os.path.dirname(save_path))
            self.save_path = save_path

    def run(self):
        video_timer = SimbaTimer(start=True)
        data = pd.DataFrame(np.array(self.geometries).T)
        print(data[0][0], data[0][1])
        data, obs_per_split = self.split_and_group_df(
            df=data,
            splits=self.core_cnt,
            include_row_index=True,
            include_split_order=False,
        )

        for i in range(len(data)):
            new_col = np.full(len(data[i]), fill_value=i).reshape(-1, 1)
            data[i] = np.concatenate((data[i], new_col), axis=1)
        print(data)
        with multiprocessing.Pool(
            self.core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                geometry_visualizer,
                video_path=self.video_path,
                video_temp_dir=self.temp_dir,
                video_meta_data=self.video_meta_data,
            )
            for cnt, result in enumerate(pool.imap(constants, data, chunksize=1)):
                print(f"Section {result}/{len(data)} complete...")
            pool.terminate()
            pool.join()

        print(f"Joining {self.video_name} geometry video...")
        concatenate_videos_in_folder(
            in_folder=self.temp_dir, save_path=self.save_path, remove_splits=True
        )
        video_timer.stop_timer()
        stdout_success(
            msg=f"Geometry video {self.save_path} complete!",
            elapsed_time=video_timer.elapsed_time_str,
            source=self.__class__.__name__,
        )
