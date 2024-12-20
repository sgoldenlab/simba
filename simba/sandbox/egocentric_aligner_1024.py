import math
import os
from copy import deepcopy
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_valid_dataframe)
from simba.utils.enums import Formats
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df,
                                    read_frm_of_video, read_video_info,
                                    write_df)
from simba.video_processors.video_processing import create_average_frm


class EgocentricalAligner(ConfigReader):

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 anchor_1: Optional[str] = 'tail_base',
                 anchor_2: Optional[str] = 'nose',
                 direction: int = 0,
                 anchor_location: Optional[Tuple[int, int]] = (250, 250),
                 rotate_video: Optional[bool] = False):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=False)
        if data_dir is None:
            self.data_paths = find_files_of_filetypes_in_directory(directory=self.outlier_corrected_dir, extensions=['.csv'])
        else:
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'])
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        self.anchor_1_cols = [f'{anchor_1}_x'.lower(), f'{anchor_1}_y'.lower()]
        self.anchor_2_cols = [f'{anchor_2}_x'.lower(), f'{anchor_2}_y'.lower()]
        self.rotate_video, self.save_dir = rotate_video, save_dir
        self.anchor_1, self.anchor_2 = anchor_1, anchor_2
        self.target_angle = np.deg2rad(direction)
        self.anchor_location = anchor_location

    def run(self):
        for file_cnt, file_path in enumerate(self.data_paths):
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            save_path = os.path.join(self.save_dir, f'{self.video_name}.{self.file_type}')
            df = read_df(file_path=file_path, file_type=self.file_type)
            original_cols = list(df.columns)
            df.columns = [x.lower() for x in list(df.columns)]
            self.body_parts_lst = [x.lower() for x in self.body_parts_lst]
            bp_cols = [x for x in df.columns if not x.endswith('_p')]
            anchor_1_idx = self.body_parts_lst.index(self.anchor_1)
            anchor_2_idx = self.body_parts_lst.index(self.anchor_2)
            data_arr = df[bp_cols].values.reshape(len(df), len(self.body_parts_lst), 2).astype(np.int32)
            results_arr = np.zeros_like(data_arr)
            self.rotation_angles, self.rotation_vectors, self.centers, self.deltas = [], [], [], []
            for frame_index in range(data_arr.shape[0]):
                frame_points = data_arr[frame_index]
                frame_anchor_1 = frame_points[anchor_1_idx]
                self.centers.append(tuple(frame_anchor_1))
                frame_anchor_2 = frame_points[anchor_2_idx]
                delta_x, delta_y = frame_anchor_2[0] - frame_anchor_1[0], frame_anchor_2[1] - frame_anchor_1[1]
                self.deltas.append((delta_x, delta_x))
                current_angle = np.arctan2(delta_y, delta_x)
                rotate_angle = self.target_angle - current_angle
                self.rotation_angles.append(rotate_angle)
                cos_theta, sin_theta = np.cos(rotate_angle), np.sin(rotate_angle)
                R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
                self.rotation_vectors.append(R)
                keypoints_translated = frame_points - frame_anchor_1
                keypoints_rotated = np.dot(keypoints_translated, R.T)
                anchor_1_position_after_rotation = keypoints_rotated[anchor_1_idx]
                translation_to_target = np.array(self.anchor_location) - anchor_1_position_after_rotation
                keypoints_aligned = keypoints_rotated + translation_to_target
                results_arr[frame_index] = keypoints_aligned

            results_arr = results_arr.reshape(len(df), len(bp_cols))
            self.out_df = pd.DataFrame(results_arr, columns=bp_cols)
            df.update(self.out_df)
            df.columns = original_cols
            write_df(df=df, file_type=self.file_type, save_path=save_path)
            if self.rotate_video:
                self.run_video_rotation()

    def run_video_rotation(self):
        video_path = find_video_of_file(video_dir=self.video_dir, filename=self.video_name, raise_error=True)
        video_meta = get_video_meta_data(video_path=video_path)
        save_path = os.path.join(self.save_dir, f'{self.video_name}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*f'{Formats.MP4_CODEC.value}')
        writer = cv2.VideoWriter(save_path, fourcc, video_meta['fps'], (video_meta['width'], video_meta['height']))
        cap = cv2.VideoCapture(video_path)
        target_x, target_y = self.anchor_location

        for frm_idx in range(video_meta['frame_count']):
            img = read_frm_of_video(video_path=cap, frame_index=frm_idx)
            # Get the rotation matrix and translation info for this frame
            R = self.rotation_vectors[frm_idx]  # 2x2 rotation matrix
            center = self.centers[frm_idx]  # Center to rotate around

            # Apply rotation to the image
            M_rotate = np.hstack([R, np.array([[-center[0] * R[0, 0] - center[1] * R[0, 1] + center[0]], [-center[0] * R[1, 0] - center[1] * R[1, 1] + center[1]]])])
            rotated_frame = cv2.warpAffine(img, M_rotate, (video_meta['width'], video_meta['height']))

            # Calculate translation to move rotated anchor to target location
            translation_x = target_x - center[0]
            translation_y = target_y - center[1]
            M_translate = np.float32([[1, 0, translation_x], [0, 1, translation_y]])

            # Apply translation to keep the anchor point at target location
            final_frame = cv2.warpAffine(rotated_frame, M_translate, (video_meta['width'], video_meta['height']))

            writer.write(final_frame)
            print(frm_idx, save_path)

        # Release resources
        cap.release()
        writer.release()


EgocentricalAligner(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini",
                    rotate_video=True,
                    anchor_1='tail_base',
                    anchor_2='nose',
                    data_dir=r'D:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location',
                    save_dir=r"D:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\rotated").run()


