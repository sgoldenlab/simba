__author__ = "Simon Nilsson"

import os
from itertools import permutations
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from shapely.geometry import MultiPoint

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.unsupervised.enums import Clustering, Unsupervised
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_keys_exist_in_dict,
                                check_if_valid_rgb_tuple, check_int, check_str)
from simba.utils.data import sample_df_n_by_unique
from simba.utils.read_write import read_pickle


class ClusterVideoVisualizer(ConfigReader, UnsupervisedMixin):
    """
    Class for creating video examples of cluster assignments.

    :param Union[str, os.PathLike] config_path: Path to SimBA project configuration file.
    :param Union[str, os.PathLike] data_path: Path to pickle file containing unsupervised results.
    :param Optional[Union[int, None]] max_videos: Maximum number of videos to create for each cluster. Defaults to None.
    :param Optional[int] speed: Speed of the generated videos. Defaults to 1.0.
    :param Optional[Tuple[int, int, int]] bg_clr: Background color of the videos as RGB tuple. Defaults to white (255, 255, 255).
    :param Optional[Literal] plot_type: Type of plot to generate ('VIDEO', 'HULL', 'SKELETON', 'POINTS'). Defaults to 'SKELETON'.

    :example:
    >>> config_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini'
    >>> data_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/cluster_mdls/hopeful_khorana.pickle'
    >>> visualizer = ClusterVideoVisualizer(config_path=config_path, data_path=data_path, bg_clr=(0, 0, 255), max_videos=20, speed=0.2, plot_type='POINTS')
    >>> visualizer.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        data_path: Union[str, os.PathLike],
        max_videos: Optional[Union[int, None]] = None,
        speed: Optional[int] = 1.0,
        bg_clr: Optional[Tuple[int, int, int]] = (255, 255, 255),
        plot_type: Optional[
            Literal["VIDEO", "HULL", "SKELETON", "POINTS"]
        ] = "SKELETON",
    ):

        check_file_exist_and_readable(file_path=data_path)
        check_file_exist_and_readable(file_path=config_path)
        check_if_valid_rgb_tuple(data=bg_clr)
        if max_videos != None:
            check_int(name="max_videos", value=max_videos, min_value=1)
        check_float(name="speed", value=speed, min_value=0.1)
        check_str(
            name="plot_type",
            value=plot_type,
            options=("VIDEO", "HULL", "SKELETON", "POINTS"),
        )
        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        self.data = read_pickle(data_path=data_path)
        check_if_keys_exist_in_dict(
            data=self.data,
            key=[Clustering.CLUSTER_MODEL.value, Unsupervised.DATA.value],
            name=data_path,
        )
        self.max_videos, self.speed, self.plot_type, self.bg_clr = (
            max_videos,
            speed,
            plot_type,
            bg_clr,
        )
        self.cl_mdl_name = self.data[Clustering.CLUSTER_MODEL.value][
            Unsupervised.HASHED_NAME.value
        ]
        self.animal_bp_cols, self.skeleton_perm = {}, {}
        self.save_dir = os.path.join(
            self.project_path, self.frames_output_dir, "clusters"
        )
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        for animal_nme in self.animal_bp_dict.keys():
            animal_bp_cols = []
            for x, y in zip(
                self.animal_bp_dict[animal_nme]["X_bps"],
                self.animal_bp_dict[animal_nme]["Y_bps"],
            ):
                animal_bp_cols.extend((x, y))
            self.animal_bp_cols[animal_nme] = animal_bp_cols
            bp_names = [x[:-2] for x in self.animal_bp_dict[animal_nme]["X_bps"]]
            self.skeleton_perm[animal_nme] = list(permutations(bp_names, 2))

    def run(self):
        cluster_idx = self.data[Unsupervised.DATA.value][
            Unsupervised.BOUTS_FEATURES.value
        ].index
        cluster_lbls = self.data[Clustering.CLUSTER_MODEL.value][
            Unsupervised.MODEL.value
        ].labels_
        cluster_df = pd.DataFrame(cluster_lbls, columns=["CLUSTER"], index=cluster_idx)
        if self.max_videos != None:
            cluster_df = sample_df_n_by_unique(
                df=cluster_df, field="CLUSTER", n=self.max_videos
            )
            cluster_df = cluster_df[~cluster_df.index.duplicated(keep="first")]
        for cluster_id in sorted(cluster_df["CLUSTER"].unique()):
            event_idx = list(cluster_df[cluster_df["CLUSTER"] == cluster_id].index)
            print(f"Creating {len(event_idx)} videos for cluster {cluster_id}...")
            for event in event_idx:
                video_pose = (
                    self.data[Unsupervised.DATA.value][Unsupervised.FRAME_POSE.value]
                    .loc[event[0], :]
                    .reset_index()
                )
                event_df = video_pose[
                    (video_pose["FRAME"] >= event[1])
                    & (video_pose["FRAME"] <= event[2])
                ].astype(np.int64)
                save_path = os.path.join(
                    self.save_dir, f"{event[0]}_{event[1]}_{event[2]}_{cluster_id}.mp4"
                )
                video_info, _, fps = self.read_video_info(video_name=event[0])
                out_fps = int(fps * self.speed)
                if out_fps > 1:
                    out_fps = 1
                w, h = int(video_info["Resolution_width"]), int(
                    video_info["Resolution_height"].astype(int)
                )
                bg_img = np.full((w, h, 3), self.bg_clr, dtype=np.uint8)
                shapes = []
                for name, bps in self.animal_bp_cols.items():
                    if self.plot_type == "HULL":
                        animal_event_pose_data = event_df[bps].values
                        animal_event_pose_data = animal_event_pose_data.reshape(
                            len(animal_event_pose_data), -1, 2
                        )
                        shapes.append(
                            GeometryMixin().multiframe_bodyparts_to_polygon(
                                data=animal_event_pose_data,
                                pixels_per_mm=1,
                                parallel_offset=1,
                                verbose=False,
                                core_cnt=-1,
                            )
                        )
                    elif self.plot_type == "SKELETON":
                        animal_skeleton_bps = self.skeleton_perm[name]
                        shapes.append(
                            GeometryMixin().multiframe_bodyparts_to_multistring_skeleton(
                                data_df=event_df,
                                skeleton=animal_skeleton_bps,
                                core_cnt=-1,
                                verbose=False,
                            )
                        )
                    elif self.plot_type == "POINTS":
                        animal_event_pose_data = event_df[bps].values
                        animal_event_pose_data = animal_event_pose_data.reshape(
                            len(animal_event_pose_data), -1, 2
                        )
                        results = GeometryMixin().multiframe_bodypart_to_point(
                            data=animal_event_pose_data
                        )
                        multi_points = []
                        for frm in range(len(results)):
                            multi_points.append(MultiPoint(results[frm]))
                        shapes.append(multi_points)
                _ = GeometryMixin.geometry_video(
                    shapes=shapes,
                    save_path=save_path,
                    fps=out_fps,
                    size=(w, h),
                    bg_img=bg_img,
                )
                print(f"Cluster video saved at {save_path}...")


# config_path = '/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini'
# data_path = '/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/small_clusters/adoring_hoover.pickle'
# visualizer = ClusterVideoVisualizer(config_path=config_path,
#                                     data_path=data_path,
#                                     bg_clr=(0, 0, 255),
#                                     max_videos=3,
#                                     speed=1.0,
#                                     plot_type='HULL')
# visualizer.run()
