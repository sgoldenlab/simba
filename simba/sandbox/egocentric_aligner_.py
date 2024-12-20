import functools
import multiprocessing
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_int,
    check_valid_dataframe)
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df,
                                    read_frm_of_video, remove_a_folder,
                                    write_df)
from simba.utils.warnings import FrameRangeWarning


def _egocentric_aligner(frm_range: np.ndarray,
                        video_path: Union[str, os.PathLike],
                        temp_dir: Union[str, os.PathLike],
                        save_dir: Union[str, os.PathLike],
                        centers: List[Tuple[int, int]],
                        rotation_vectors: np.ndarray,
                        target: Tuple[int, int]):

    video_meta = get_video_meta_data(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    batch, frm_range = frm_range[0], frm_range[1]
    save_path = os.path.join(temp_dir, f'{batch}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*f'{Formats.MP4_CODEC.value}')
    writer = cv2.VideoWriter(save_path, fourcc, video_meta['fps'], (video_meta['width'], video_meta['height']))

    for frm_cnt, frm_id in enumerate(frm_range):
        img = read_frm_of_video(video_path=cap, frame_index=frm_id)
        R, center = rotation_vectors[frm_id], centers[frm_id]
        M_rotate = np.hstack([R, np.array([[-center[0] * R[0, 0] - center[1] * R[0, 1] + center[0]], [-center[0] * R[1, 0] - center[1] * R[1, 1] + center[1]]])])
        rotated_frame = cv2.warpAffine(img, M_rotate, (video_meta['width'], video_meta['height']))
        translation_x = target[0] - center[0]
        translation_y = target[1] - center[1]
        M_translate = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        final_frame = cv2.warpAffine(rotated_frame, M_translate, (video_meta['width'], video_meta['height']))
        writer.write(final_frame)
        print(f'Creating frame {frm_id} (CPU core: {batch+1}).')

    cap.release()
    writer.release()
    return batch+1

class EgocentricalAligner(ConfigReader):

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 anchor_1: Optional[str] = 'tail_base',
                 anchor_2: Optional[str] = 'nose',
                 direction: int = 0,
                 anchor_location: Optional[Tuple[int, int]] = (250, 250),
                 rotate_video: Optional[bool] = False,
                 cores: Optional[int] = -1):
        """
        Aligns and rotates movement data and associated video frames based on specified anchor points to produce an egocentric view of the subject. The class aligns frames around a selected anchor point, optionally rotating the subject to a consistent direction and saving the output video.

        .. video:: _static/img/EgocentricalAligner.webm
           :width: 800
           :autoplay:
           :loop:

        :param Union[str, os.PathLike] config_path: Path to the configuration file.
        :param Union[str, os.PathLike] save_dir: Directory where the processed output will be saved.
        :param Optional[Union[str, os.PathLike]] data_dir: Directory containing CSV files with movement data.
        :param Optional[str] anchor_1: Primary anchor point (e.g., 'tail_base') around which the alignment centers.
        :param Optional[str] anchor_2: Secondary anchor point (e.g., 'nose') defining the alignment direction.
        :param int direction: Target angle, in degrees, for alignment; e.g., `0` aligns along the x-axis.
        :param Optional[Tuple[int, int]] anchor_location: Pixel location in the output where `anchor_1`  should appear; default is `(250, 250)`.
        :param Optional[bool] rotate_video: Whether to rotate the video to align with the specified direction.
        :param Optional[int] cores: Number of CPU cores to use for video rotation; `-1` uses all available cores.

        :example:
        >>> aligner = EgocentricalAligner(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", rotate_video=True, anchor_1='tail_base', anchor_2='nose', data_dir=r'C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\test', save_dir=r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\test\bg_temp\rotated")
        >>> aligner.run()
        """

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
        check_int(name='cores', value=cores, min_value=-1, max_value=find_core_cnt()[0])
        if cores == -1:
            self.cores = find_core_cnt()[0]
        else:
            self.cores = cores

    def run(self):
        for file_cnt, file_path in enumerate(self.data_paths):
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            save_path = os.path.join(self.save_dir, f'{self.video_name}.{self.file_type}')
            df = read_df(file_path=file_path, file_type=self.file_type)
            original_cols, self.file_path = list(df.columns), file_path
            df.columns = [x.lower() for x in list(df.columns)]
            check_valid_dataframe(df=df, source=self.__class__.__name__, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.anchor_1_cols + self.anchor_2_cols)
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
        video_timer = SimbaTimer(start=True)
        video_path = find_video_of_file(video_dir=self.video_dir, filename=self.video_name, raise_error=True)
        video_meta = get_video_meta_data(video_path=video_path)
        save_path = os.path.join(self.save_dir, f'{self.video_name}.mp4')
        temp_dir = os.path.join(self.save_dir, 'temp')
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        else:
            remove_a_folder(folder_dir=temp_dir)
            os.makedirs(temp_dir)
        if video_meta['frame_count'] != len(self.out_df):
            FrameRangeWarning(msg=f'The video {video_path} contains {video_meta["frame_count"]} frames while the file {self.file_path} contains {len(self.out_df)} frames', source=self.__class__.__name__)
        frm_list = np.arange(0, video_meta['frame_count'])
        frm_list = np.array_split(frm_list, self.cores)
        frm_list = [(cnt, x) for cnt, x in enumerate(frm_list)]
        print(f"Creating rotated videos, multiprocessing (chunksize: {self.multiprocess_chunksize}, cores: {self.cores})...")
        with multiprocessing.Pool(self.cores, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(_egocentric_aligner,
                                          save_dir=self.save_dir,
                                          temp_dir=temp_dir,
                                          video_path=video_path,
                                          centers=self.centers,
                                          rotation_vectors=self.rotation_vectors,
                                          target=self.anchor_location)
            for cnt, result in enumerate(pool.imap(constants, frm_list, chunksize=self.multiprocess_chunksize)):
                print(f"Rotate batch {result}/{self.cores} complete...")
            pool.terminate()
            pool.join()

        concatenate_videos_in_folder(in_folder=temp_dir, save_path=save_path, remove_splits=True, gpu=False)
        video_timer.stop_timer()
        print(f"Egocentric rotation video {save_path} complete (elapsed time: {video_timer.elapsed_time_str}s) ...")


# if __name__ == "__main__":
#     aligner = EgocentricalAligner(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                                   rotate_video=True,
#                                   anchor_1='tail_base',
#                                   anchor_2='nose',
#                                   data_dir=r'C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\test',
#                                   save_dir=r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\test\bg_temp\rotated")
#     aligner.run()
#

