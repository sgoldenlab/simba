import functools
import multiprocessing
import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_if_valid_rgb_tuple,
                                check_int, check_valid_array,
                                check_valid_boolean)
from simba.utils.data import egocentrically_align_pose
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext,
                                    get_video_meta_data, read_df,
                                    read_frm_of_video, remove_a_folder)


def egocentric_video_aligner(frm_range: np.ndarray,
                             video_path: Union[str, os.PathLike],
                             temp_dir: Union[str, os.PathLike],
                             video_name: str,
                             centers: np.ndarray,
                             rotation_vectors: np.ndarray,
                             target: Tuple[int, int],
                             fill_clr: Tuple[int, int, int] = (255, 255, 255),
                             verbose: bool = False):


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
        rotated_frame = cv2.warpAffine(img, M_rotate, (video_meta['width'], video_meta['height']), borderValue=fill_clr)
        translation_x = target[0] - center[0]
        translation_y = target[1] - center[1]
        M_translate = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        final_frame = cv2.warpAffine(rotated_frame, M_translate, (video_meta['width'], video_meta['height']), borderValue=fill_clr)
        writer.write(final_frame)
        if verbose:
            print(f'Creating frame {frm_id} ({video_name}, CPU core: {batch+1}).')

    writer.release()
    return batch+1

    pass

class EgocentricVideoRotator():

    """



    :param Union[str, os.PathLike] video_path: Path to a video file.
    :param np.ndarray centers: A 2D array of shape `(num_frames, 2)` containing the original locations of `anchor_1_idx` in each frame before alignment. Returned by ``
    :param np.ndarray rotation_vectors:
    :param bool verbose:
    :param Tuple[int, int, int] fill_clr:
    :param int core_cnt:
    :param Optional[Union[str, os.PathLike]] save_path:

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
                 anchor_location: np.ndarray,
                 verbose: bool = True,
                 fill_clr: Tuple[int, int, int] = (0, 0, 0),
                 core_cnt: int = -1,
                 save_path: Optional[Union[str, os.PathLike]] = None):

        check_file_exist_and_readable(file_path=video_path)
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        check_valid_array(data=centers, source=f'{self.__class__.__name__} centers', accepted_ndims=(2,), accepted_axis_1_shape=[2,], accepted_axis_0_shape=[self.video_meta_data['frame_count']], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=rotation_vectors, source=f'{self.__class__.__name__} rotation_vectors', accepted_ndims=(3,), accepted_axis_0_shape=[self.video_meta_data['frame_count']], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=anchor_location, source=f'{self.__class__.__name__} anchor_location', accepted_ndims=(1,), accepted_axis_0_shape=[2], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose')
        check_if_valid_rgb_tuple(data=fill_clr)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        if core_cnt > find_core_cnt()[0] or core_cnt == -1: self.core_cnt = find_core_cnt()[0]
        else: self.core_cnt = core_cnt
        video_dir, self.video_name, _ = get_fn_ext(filepath=video_path)
        if save_path is not None:
            self.save_dir = os.path.dirname(save_path)
            check_if_dir_exists(in_dir=self.save_dir, source=f'{self.__class__.__name__} save_path')
        else:
            save_path = os.path.join(video_dir, f'{self.video_name}_rotated.mp4')
        self.video_path, self.save_path = video_path, save_path
        self.centers, self.rotation_vectors = centers, rotation_vectors
        self.verbose, self.fill_clr, self.anchor_loc = verbose, fill_clr, anchor_location

    def run(self):
        video_timer = SimbaTimer(start=True)
        temp_dir = os.path.join(self.save_dir, 'temp')
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        else:
            remove_a_folder(folder_dir=temp_dir)
            #os.makedirs(temp_dir)
        frm_list = np.arange(0, self.video_meta_data['frame_count'])
        frm_list = np.array_split(frm_list, self.core_cnt)
        frm_list = [(cnt, x) for cnt, x in enumerate(frm_list)]
        print(f"Creating rotated video {self.video_name}, multiprocessing (chunksize: {1}, cores: {self.core_cnt})...")
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=100) as pool:
            constants = functools.partial(egocentric_video_aligner,
                                          temp_dir=temp_dir,
                                          video_name=self.video_name,
                                          video_path=self.video_path,
                                          centers=self.centers,
                                          rotation_vectors=self.rotation_vectors,
                                          target=self.anchor_loc,
                                          verbose=self.verbose,
                                          fill_clr=self.fill_clr)
            for cnt, result in enumerate(pool.imap(constants, frm_list, chunksize=1)):
                print(f"Rotate batch {result}/{self.core_cnt} complete...")
            pool.terminate()
            pool.join()

        concatenate_videos_in_folder(in_folder=temp_dir, save_path=self.save_path, remove_splits=True, gpu=False)
        video_timer.stop_timer()
        stdout_success(msg=f"Egocentric rotation video {self.save_path} complete", elapsed_time=video_timer.elapsed_time_str, source=self.__class__.__name__)




# if __name__ == "__main__":
#     DATA_PATH = r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\data\501_MA142_Gi_Saline_0513.csv"
#     VIDEO_PATH = r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos\501_MA142_Gi_Saline_0513.mp4"
#     SAVE_PATH = r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos\501_MA142_Gi_Saline_0513_rotated.mp4"
#     ANCHOR_LOC = np.array([250, 250])
#
#     df = read_df(file_path=DATA_PATH, file_type='csv')
#     bp_cols = [x for x in df.columns if not x.endswith('_p')]
#     data = df[bp_cols].values.reshape(len(df), int(len(bp_cols)/2), 2).astype(np.int32)
#
#     _, centers, rotation_vectors = egocentrically_align_pose(data=data, anchor_1_idx=6, anchor_2_idx=2, anchor_location=ANCHOR_LOC, direction=0)
#     rotater = EgocentricVideoRotator(video_path=VIDEO_PATH, centers=centers, rotation_vectors=rotation_vectors, anchor_location=ANCHOR_LOC, save_path=SAVE_PATH)
#     rotater.run()

#
#


