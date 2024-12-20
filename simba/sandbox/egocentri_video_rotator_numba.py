import os
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from simba.mixins.image_mixin import ImageMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_if_valid_rgb_tuple,
                                check_int, check_valid_array,
                                check_valid_boolean, check_valid_tuple)
from simba.utils.data import (align_target_warpaffine_vectors,
                              center_rotation_warpaffine_vectors,
                              egocentric_frm_rotator,
                              egocentrically_align_pose)
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext,
                                    get_video_meta_data, read_df,
                                    read_frm_of_video,
                                    read_img_batch_from_video_gpu,
                                    remove_a_folder)


class EgocentricVideoRotatorAccelerated():
    """
    Perform egocentric rotation of a video using CPU multiprocessing.

    .. video:: _static/img/EgocentricalAligner_2.webm
       :width: 800
       :autoplay:
       :loop:

    .. seealso::
       To perform joint egocentric alignment of both pose and video, or pose only, use :func:`~simba.data_processors.egocentric_aligner.EgocentricalAligner`.
       To produce rotation vectors, use :func:`~simba.utils.data.egocentrically_align_pose_numba` or :func:`~simba.utils.data.egocentrically_align_pose`.

    :param Union[str, os.PathLike] video_path: Path to a video file.
    :param np.ndarray centers: A 2D array of shape `(num_frames, 2)` containing the original locations of `anchor_1_idx` in each frame before alignment. Returned by :func:`~simba.utils.data.egocentrically_align_pose_numba` or :func:`~simba.utils.data.egocentrically_align_pose`.
    :param np.ndarray rotation_vectors: A 3D array of shape `(num_frames, 2, 2)` containing the rotation matrices applied to each frame. Returned by :func:`~simba.utils.data.egocentrically_align_pose_numba` or :func:`~simba.utils.data.egocentrically_align_pose`.
    :param bool verbose: If True, prints progress. Deafult True.
    :param Tuple[int, int, int] fill_clr: The color of the additional pixels. Deafult black. (0, 0, 0).
    :param int core_cnt: Number of CPU cores to use for video rotation; `-1` uses all available cores.
    :param Optional[Union[str, os.PathLike]] save_path: The location where to store the rotated video. If None, saves the video as the same dir as the input video with the `_rotated` suffix.

    :example:
    >>> DATA_PATH = "C:\501_MA142_Gi_Saline_0513.csv"
    >>> VIDEO_PATH = "C:\501_MA142_Gi_Saline_0513.mp4"
    >>> SAVE_PATH = "C:\501_MA142_Gi_Saline_0513_rotated.mp4"
    >>> ANCHOR_LOC = np.array([250, 250])

    >>> df = read_df(file_path=DATA_PATH, file_type='csv')
    >>> bp_cols = [x for x in df.columns if not x.endswith('_p')]
    >>> data = df[bp_cols].values.reshape(len(df), int(len(bp_cols)/2), 2).astype(np.int32)
    >>> _, centers, rotation_vectors = egocentrically_align_pose(data=data, anchor_1_idx=6, anchor_2_idx=2, anchor_location=ANCHOR_LOC, direction=0)
    >>> rotater = EgocentricVideoRotator(video_path=VIDEO_PATH, centers=centers, rotation_vectors=rotation_vectors, anchor_location=ANCHOR_LOC, save_path=SAVE_PATH)
    >>> rotater.run()
    """

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 centers: np.ndarray,
                 rotation_vectors: np.ndarray,
                 anchor_location: Tuple[int, int],
                 verbose: bool = True,
                 fill_clr: Tuple[int, int, int] = (0, 0, 0),
                 core_cnt: int = -1,
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 batch_size: Optional[int] = 500,
                 gpu: Optional[bool] = True):

        check_file_exist_and_readable(file_path=video_path)
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        check_valid_array(data=centers, source=f'{self.__class__.__name__} centers', accepted_ndims=(2,), accepted_axis_1_shape=[2, ], accepted_axis_0_shape=[self.video_meta_data['frame_count']], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=rotation_vectors, source=f'{self.__class__.__name__} rotation_vectors', accepted_ndims=(3,), accepted_axis_0_shape=[self.video_meta_data['frame_count']], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_tuple(x=anchor_location, source=f'{self.__class__.__name__} anchor_location', accepted_lengths=(2,), valid_dtypes=(int,))
        for i in anchor_location: check_int(name=f'{self.__class__.__name__} anchor_location', value=i, min_value=1)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose')
        check_if_valid_rgb_tuple(data=fill_clr)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        if core_cnt > find_core_cnt()[0] or core_cnt == -1:
            self.core_cnt = find_core_cnt()[0]
        else:
            self.core_cnt = core_cnt
        video_dir, self.video_name, _ = get_fn_ext(filepath=video_path)
        if save_path is not None:
            self.save_dir = os.path.dirname(save_path)
            check_if_dir_exists(in_dir=self.save_dir, source=f'{self.__class__.__name__} save_path')
        else:
            self.save_dir = video_dir
            save_path = os.path.join(video_dir, f'{self.video_name}_rotated.mp4')
        self.video_path, self.save_path, self.gpu = video_path, save_path, gpu
        self.centers, self.rotation_vectors, self.batch_size = centers, rotation_vectors, batch_size
        self.verbose, self.fill_clr, self.anchor_loc = verbose, fill_clr, anchor_location
        fourcc = cv2.VideoWriter_fourcc(*f'{Formats.MP4_CODEC.value}')
        self.writer = cv2.VideoWriter(save_path, fourcc, self.video_meta_data['fps'], (self.video_meta_data['width'], self.video_meta_data['height']))


    def run(self):
        center_rotations = center_rotation_warpaffine_vectors(rotation_vectors=rotation_vectors, centers=self.centers)
        target_rotations = align_target_warpaffine_vectors(centers=self.centers, target=np.array(self.anchor_loc))
        frm_idx = np.arange(0, self.video_meta_data['frame_count'])
        frm_idx = np.array_split(frm_idx, range(self.batch_size, len(frm_idx), self.batch_size))
        timer = time.time()
        for frm_batch_cnt, frm_batch in enumerate(frm_idx):
            print(frm_batch_cnt, len(frm_idx))
            sample_center_rotations = center_rotations[frm_batch[0]:frm_batch[-1],]
            sample_target_rotations = target_rotations[frm_batch[0]:frm_batch[-1],]
            start = time.time()
            if not self.gpu:
                sample_imgs = ImageMixin.read_img_batch_from_video(video_path=self.video_path, start_frm=frm_batch[0], end_frm=frm_batch[-1])
            else:
                sample_imgs = read_img_batch_from_video_gpu(video_path=self.video_path, start_frm=frm_batch[0], end_frm=frm_batch[-1])
            sample_imgs = np.stack(list(sample_imgs.values()), axis=1)
            sample_imgs = egocentric_frm_rotator(frames=sample_imgs, rotation_matrices=sample_center_rotations)
            sample_imgs = egocentric_frm_rotator(frames=sample_imgs, rotation_matrices=sample_target_rotations)
            for img in sample_imgs:
                cv2.imshow('sasdasd', img)
                cv2.waitKey(60)

                self.writer.write(img.astype(np.uint8))
            if frm_batch_cnt == 2:
                self.writer.release()
                break


        print(f'Total time: {time.time() - timer}')

        pass

DATA_PATH = r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\data\501_MA142_Gi_Saline_0513.csv"
VIDEO_PATH = r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos\501_MA142_Gi_Saline_0513.mp4"
SAVE_PATH = r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos\501_MA142_Gi_Saline_0513_rotated.mp4"
ANCHOR_LOC = np.array([250, 250])

df = read_df(file_path=DATA_PATH, file_type='csv')
bp_cols = [x for x in df.columns if not x.endswith('_p')]
data = df[bp_cols].values.reshape(len(df), int(len(bp_cols) / 2), 2).astype(np.int32)
_, centers, rotation_vectors = egocentrically_align_pose(data=data, anchor_1_idx=6, anchor_2_idx=2, anchor_location=ANCHOR_LOC, direction=0)

rotater = EgocentricVideoRotatorAccelerated(video_path=VIDEO_PATH, centers=centers, rotation_vectors=rotation_vectors, anchor_location=(250, 250), save_path=SAVE_PATH)
rotater.run()


#CHECK IF THIS ONE ALSO GIVE SNONE
