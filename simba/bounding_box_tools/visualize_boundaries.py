__author__ = "Simon Nilsson"

import functools
import multiprocessing
import os
import pickle
import platform
from multiprocessing import pool
from typing import Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.errors import NoFilesFoundError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder, get_fn_ext,
                                    get_video_meta_data, read_df)

#
# def _image_creator(frm_range: list,
#                    polygon_data: dict,
#                    animal_bp_dict: dict,
#                    data_df: pd.DataFrame or None,
#                    intersection_data_df: pd.DataFrame or None,
#                    roi_attributes: dict,
#                    video_path: str,
#                    key_points: bool,
#                    greyscale: bool):
#
#     cap, current_frame = cv2.VideoCapture(video_path), frm_range[0]
#     cap.set(1, frm_range[0])
#     img_lst = []
#     while current_frame < frm_range[-1]:
#         ret, frame = cap.read()
#         if ret:
#             if key_points:
#                 frm_data = data_df.iloc[current_frame]
#             if greyscale:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#             for animal_cnt, (animal, animal_data) in enumerate(animal_bp_dict.items()):
#                 if key_points:
#                     for bp_cnt, (x_col, y_col) in enumerate(zip(animal_data['X_bps'], animal_data['Y_bps'])):
#                         cv2.circle(frame, (frm_data[x_col], frm_data[y_col]), 0, roi_attributes[animal]['bbox_clr'], roi_attributes[animal]['keypoint_size'])
#                 animal_polygon = np.array(list(polygon_data[animal][current_frame].convex_hull.exterior.coords)).astype(int)
#                 if intersection_data_df is not None:
#                     intersect = intersection_data_df.loc[current_frame, intersection_data_df.columns.str.startswith(animal)].sum()
#                     if intersect > 0:
#                         cv2.polylines(frame, [animal_polygon], 1, roi_attributes[animal]['highlight_clr'], roi_attributes[animal]['highlight_clr_thickness'])
#                 cv2.polylines(frame, [animal_polygon], 1, roi_attributes[animal]['bbox_clr'], roi_attributes[animal]['bbox_thickness'])
#             img_lst.append(frame)
#             current_frame += 1
#         else:
#             print('SIMBA WARNING: SimBA tried to grab frame number {} from video {}, but could not find it. The video has {} frames.'.format(str(current_frame), video_path, str(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
#     return img_lst


class BoundaryVisualizer(ConfigReader, PlottingMixin):
    """
    Visualisation of user-specified animal-anchored ROI boundaries. Results are stored in the
    ``project_folder/frames/output/anchored_rois`` directory of the SimBA project

    :parameter str config_path: Path to SimBA project config file in Configparser format
    :parameter str video_name: Name of the video in the SimBA project to create bounding box video for
    :parameter bool include_key_points: If True, includes pose-estimated body-parts in the video.
    :parameter bool greyscale:  If True, converts the video (but not the shapes/keypoints) to greyscale.
    :parameter bool show_intersections:  If True, then produce highlight boundaries/keypoints to signify present intersections.
                                         See `this example for highlighted intersections <https://github.com/sgoldenlab/simba/blob/master/images/termites_video_3.gif>`_

    .. note::
        `Bounding boxes tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/anchored_rois.md>`_.

    Examples
    ----------
    >>> boundary_visualizer = BoundaryVisualizer(config_path='MySimBAConfig', video_name='MyVideoName', include_key_points=True, greyscale=True)
    >>> boundary_visualizer.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        video_name: str,
        include_key_points: bool,
        greyscale: bool,
        show_intersections: bool or None,
        roi_attributes: dict or None,
    ):
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)

        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        self.polygon_path = os.path.join(
            self.project_path, "logs", "anchored_rois.pickle"
        )
        check_file_exist_and_readable(file_path=self.polygon_path)
        (
            self.video_name,
            self.include_key_points,
            self.greyscale,
            self.roi_attributes,
        ) = (video_name, include_key_points, greyscale, roi_attributes)
        (
            self.show_intersections,
            self.intersection_data_folder,
        ) = show_intersections, os.path.join(
            self.project_path, "csv", "anchored_roi_data"
        )
        self.intersections_df = None
        if self.show_intersections:
            self._find_intersection_data()
        with open(self.polygon_path, "rb") as fp:
            self.polygons = pickle.load(fp)
        self.video_path = self.find_video_of_file(
            video_dir=self.video_dir, filename=video_name
        )
        self.save_parent_dir = os.path.join(
            self.project_path, "frames", "output", "anchored_rois"
        )
        self.save_video_path = os.path.join(self.save_parent_dir, video_name + ".mp4")
        if not os.path.exists(self.save_parent_dir):
            os.makedirs(self.save_parent_dir)

    def _find_intersection_data(self):
        self.intersection_path = None
        for p in [
            os.path.join(self.intersection_data_folder, self.video_name + x)
            for x in [".pickle", ".csv", ".parquet"]
        ]:
            if os.path.isfile(p):
                self.intersection_path = p
        if self.intersection_path is None:
            print(
                "SIMBA WARNING: No ROI intersection data found for video {} in directory {}. Skipping intersection visualizations".format(
                    self.video_name, self.intersection_data_folder
                )
            )
            self.show_intersections = False
            self.intersections_df = None
        else:
            _, _, ext = get_fn_ext(filepath=self.intersection_path)
            self.intersections_df = read_df(
                file_path=self.intersection_path, file_type=ext[1:]
            )

    def run(self, chunk_size=50):
        if self.include_key_points:
            self.data_df_path = os.path.join(
                self.outlier_corrected_dir, self.video_name + "." + self.file_type
            )
            if not os.path.isfile(self.data_df_path):
                raise NoFilesFoundError(
                    msg=f"SIMBA ERROR: No keypoint data found in {self.data_df_path}. Untick key-point checkbox or import pose-estimation data."
                )
            self.data_df = (
                read_df(file_path=self.data_df_path, file_type=self.file_type)
                .astype(int)
                .reset_index(drop=True)
            )
        else:
            self.data_df = None
        print("Creating visualization for video {}...".format(self.video_name))
        video_path = self.find_video_of_file(
            video_dir=self.video_dir, filename=self.video_name
        )
        video_meta_data = get_video_meta_data(video_path=video_path)
        self.max_dim = max(video_meta_data["width"], video_meta_data["height"])
        self.space_scale, self.radius_scale, self.res_scale, self.font_scale = (
            60,
            12,
            1500,
            1.1,
        )
        if self.roi_attributes is None:
            self.roi_attributes = {}
            for animal_name, animal_data in self.animal_bp_dict.items():
                self.roi_attributes[animal_name] = {}
                self.roi_attributes[animal_name]["bbox_clr"] = animal_data["colors"][0]
                self.roi_attributes[animal_name]["bbox_thickness"] = 2
                self.roi_attributes[animal_name]["keypoint_size"] = int(
                    self.radius_scale / (self.res_scale / self.max_dim)
                )
                self.roi_attributes[animal_name]["highlight_clr"] = (0, 0, 255)
                self.roi_attributes[animal_name]["highlight_clr_thickness"] = 10

        self.video_save_path = os.path.join(
            self.save_parent_dir, self.video_name + ".mp4"
        )
        self.temp_folder = os.path.join(self.save_parent_dir, self.video_name)
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        frame_chunks = [
            [i, i + chunk_size]
            for i in range(0, video_meta_data["frame_count"], chunk_size)
        ]
        frame_chunks[-1][-1] = min(frame_chunks[-1][-1], video_meta_data["frame_count"])
        functools.partial(self.bbox_mp, b=self.data_df)
        with pool.Pool(self.cpu_to_use, maxtasksperchild=self.maxtasksperchild) as p:
            constants = functools.partial(
                self.bbox_mp,
                data_df=self.data_df,
                polygon_data=self.polygons[self.video_name],
                animal_bp_dict=self.animal_bp_dict,
                roi_attributes=self.roi_attributes,
                video_path=video_path,
                key_points=self.include_key_points,
                greyscale=self.greyscale,
                intersection_data_df=self.intersections_df,
            )
            for cnt, result in enumerate(
                p.imap(constants, frame_chunks, chunksize=self.multiprocess_chunksize)
            ):
                save_path = os.path.join(self.temp_folder, str(cnt) + ".mp4")
                writer = cv2.VideoWriter(
                    save_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    video_meta_data["fps"],
                    (video_meta_data["width"], video_meta_data["height"]),
                )
                for img in result:
                    writer.write(img)
                writer.release()
                if int(chunk_size * cnt) < video_meta_data["frame_count"]:
                    print(
                        "Image {}/{}...".format(
                            str(int(chunk_size * cnt)),
                            str(video_meta_data["frame_count"]),
                        )
                    )
            p.terminate()
            p.join()

        concatenate_videos_in_folder(
            in_folder=self.temp_folder,
            save_path=self.save_video_path,
            video_format="mp4",
            remove_splits=True,
        )
        stdout_success(msg=f"Anchored ROI video created at {self.save_video_path}")


# boundary_visualizer = BoundaryVisualizer(config_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini',
#                                          video_name='Testing_Video_3',
#                                          include_key_points=True,
#                                          greyscale=True,
#                                          show_intersections=True,
#                                          roi_attributes=None)
# boundary_visualizer.run_visualization()

# boundary_visualizer = BoundaryVisualizer(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                          video_name='Together_3',
#                                          include_key_points=True,
#                                          greyscale=True,
#                                          show_intersections=True,
#                                          roi_attributes=None)
# boundary_visualizer.run_visualization()
