import functools
import multiprocessing
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_if_dir_exists,
    check_if_valid_rgb_tuple, check_instance, check_int, check_str,
    check_valid_array, check_valid_boolean, check_valid_dataframe,
    check_valid_tuple)
from simba.utils.data import egocentrically_align_pose
from simba.utils.enums import Formats, Options
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import (bgr_to_rgb_tuple,
                                    concatenate_videos_in_folder,
                                    find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df,
                                    read_frm_of_video, read_video_info_csv,
                                    remove_a_folder, write_df)
from simba.utils.warnings import FrameRangeWarning


def _egocentric_aligner(frm_range: np.ndarray,
                        video_path: Union[str, os.PathLike],
                        temp_dir: Union[str, os.PathLike],
                        video_name: str,
                        centers: List[Tuple[int, int]],
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

class EgocentricalAligner():

    """
    Aligns and rotates movement data and associated video frames based on specified anchor points to produce an egocentric view of the subject. The class aligns frames around a selected anchor point, optionally rotating the subject to a consistent direction and saving the output video.

    .. video:: _static/img/EgocentricalAligner.webm
       :width: 800
       :autoplay:
       :loop:

    .. video:: _static/img/EgocentricalAligner_2.webm
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
    :param Tuple[int, int, int] fill_clr: If rotating the videos, the color of the additional pixels.
    :param Optional[bool] rotate_video: Whether to rotate the video to align with the specified direction.
    :param Optional[int] cores: Number of CPU cores to use for video rotation; `-1` uses all available cores.

    :example:
     >>> aligner = EgocentricalAligner(rotate_video=True, anchor_1='tail_base', anchor_2='nose', data_dir=r"C:/Users/sroni/OneDrive/Desktop/rotate_ex/data", videos_dir=r'C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos', save_dir=r"C:\troubleshooting\mitra\project_folder\videos\additional/examples/rotated", video_info=r"C:\troubleshooting\mitra\project_folder\logs\video_info.csv", direction=0, anchor_location=(250, 250), fill_clr=(0, 0, 0))
     >>> aligner.run()
    """

    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 anchor_1: str = 'tail_base',
                 anchor_2: str = 'nose',
                 direction: int = 0,
                 anchor_location: Tuple[int, int] = (250, 250),
                 core_cnt: int = -1,
                 rotate_video: bool = False,
                 fill_clr: Tuple[int, int, int] = (250, 250, 255),
                 verbose: bool = True,
                 videos_dir: Optional[Union[str, os.PathLike]] = None,
                 video_info: Optional[Union[str, os.PathLike, pd.DataFrame]] = None):

        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'])
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        check_str(name=f'{self.__class__.__name__} anchor_1', value=anchor_1)
        check_str(name=f'{self.__class__.__name__} anchor_2', value=anchor_2)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], unaccepted_vals=[0])
        if core_cnt == -1: self.core_cnt = find_core_cnt()[0]
        check_int(name=f'{self.__class__.__name__} direction', value=direction, min_value=0, max_value=360)
        check_valid_tuple(x=anchor_location, source=f'{self.__class__.__name__} anchor_location', accepted_lengths=(2,), valid_dtypes=(int,))
        for i in anchor_location: check_int(name=f'{self.__class__.__name__} anchor_location', value=i, min_value=1)
        check_valid_boolean(value=[rotate_video, verbose], source=f'{self.__class__.__name__} rotate_video')
        if rotate_video:
            check_if_valid_rgb_tuple(data=fill_clr)
            fill_clr = bgr_to_rgb_tuple(value=fill_clr)
            check_if_dir_exists(in_dir=videos_dir, source=f'{self.__class__.__name__} videos_dir')
            check_instance(source=f'{self.__class__.__name__} video_info', accepted_types=(str, pd.DataFrame), instance=video_info)
            if isinstance(video_info, str): video_info = read_video_info_csv(file_path=video_info)
            else: check_valid_dataframe(df=video_info, source=f'{self.__class__.__name__} video_info', required_fields=Formats.EXPECTED_VIDEO_INFO_COLS.value)
            self.video_paths = find_files_of_filetypes_in_directory(directory=videos_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value)
            for file_path in self.data_paths:
                find_video_of_file(video_dir=videos_dir, filename=get_fn_ext(file_path)[1], raise_error=True)
            check_all_file_names_are_represented_in_video_log(video_info_df=video_info, data_paths=self.data_paths)
        self.anchor_1_cols = [f'{anchor_1}_x'.lower(), f'{anchor_1}_y'.lower()]
        self.anchor_2_cols = [f'{anchor_2}_x'.lower(), f'{anchor_2}_y'.lower()]
        self.anchor_1, self.anchor_2, self.videos_dir = anchor_1, anchor_2, videos_dir
        self.rotate_video, self.save_dir, self.verbose = rotate_video, save_dir, verbose
        self.anchor_location, self.direction, self.fill_clr = np.array(anchor_location), direction, fill_clr

    def run(self):
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            if self.verbose:
                print(f'Analyzing video {self.video_name}... ({file_cnt+1}/{len(self.data_paths)})')
            save_path = os.path.join(self.save_dir, f'{self.video_name}.{Formats.CSV.value}')
            df = read_df(file_path=file_path, file_type=Formats.CSV.value)
            original_cols, self.file_path = list(df.columns), file_path
            df.columns = [x.lower() for x in list(df.columns)]
            check_valid_dataframe(df=df, source=self.__class__.__name__, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.anchor_1_cols + self.anchor_2_cols)

            bp_cols = [x for x in df.columns if not x.endswith('_p')]
            body_parts_lst = []
            _= [body_parts_lst.append(x[:-2]) for x in bp_cols if x[:-2] not in body_parts_lst]
            anchor_1_idx, anchor_2_idx = body_parts_lst.index(self.anchor_1), body_parts_lst.index(self.anchor_2)
            data_arr = df[bp_cols].values.reshape(len(df), len(body_parts_lst), 2).astype(np.int32)
            results_arr, self.centers, self.rotation_vectors = egocentrically_align_pose(data=data_arr, anchor_1_idx=anchor_1_idx, anchor_2_idx=anchor_2_idx, direction=self.direction, anchor_location=self.anchor_location)
            results_arr = results_arr.reshape(len(df), len(bp_cols))
            self.out_df = pd.DataFrame(results_arr, columns=bp_cols)
            df.update(self.out_df)
            df.columns = original_cols
            write_df(df=df, file_type=Formats.CSV.value, save_path=save_path)
            video_timer.stop_timer()
            print(f'{self.video_name} complete, saved at {save_path} (elapsed time: {video_timer.elapsed_time_str}s)')
            if self.rotate_video:
                self.out_df = self.out_df.head(500)
                self.run_video_rotation()



    def run_video_rotation(self):
        video_timer = SimbaTimer(start=True)
        video_path = find_video_of_file(video_dir=self.videos_dir, filename=self.video_name, raise_error=False)
        video_meta = get_video_meta_data(video_path=video_path)
        save_path = os.path.join(self.save_dir, f'{self.video_name}.mp4')
        temp_dir = os.path.join(self.save_dir, 'temp')
        if not (os.path.isdir(temp_dir)):
            os.makedirs(temp_dir)
        else:
            remove_a_folder(folder_dir=temp_dir)
            os.makedirs(temp_dir)
        if video_meta['frame_count'] != len(self.out_df):
            FrameRangeWarning(msg=f'The video {video_path} contains {video_meta["frame_count"]} frames while the file {self.file_path} contains {len(self.out_df)} frames', source=self.__class__.__name__)
        frm_list = np.arange(0, video_meta['frame_count'])
        frm_list = np.arange(0, 500)
        frm_list = np.array_split(frm_list, self.core_cnt)
        frm_list = [(cnt, x) for cnt, x in enumerate(frm_list)]
        print(f"Creating rotated video {self.video_name}, multiprocessing (chunksize: {1}, cores: {self.core_cnt})...")
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=100) as pool:
            constants = functools.partial(_egocentric_aligner,
                                          temp_dir=temp_dir,
                                          video_name=self.video_name,
                                          video_path=video_path,
                                          centers=self.centers,
                                          rotation_vectors=self.rotation_vectors,
                                          target=self.anchor_location,
                                          verbose=self.verbose,
                                          fill_clr=self.fill_clr)
            for cnt, result in enumerate(pool.imap(constants, frm_list, chunksize=1)):
                print(f"Rotate batch {result}/{self.core_cnt} complete...")
            pool.terminate()
            pool.join()

        concatenate_videos_in_folder(in_folder=temp_dir, save_path=save_path, remove_splits=True, gpu=False)
        video_timer.stop_timer()
        print(f"Egocentric rotation video {save_path} complete (elapsed time: {video_timer.elapsed_time_str}s) ...")

# if __name__ == "__main__":
#     aligner = EgocentricalAligner(rotate_video=True,
#                                   anchor_1='tail_base',
#                                   anchor_2='nose',
#                                   data_dir=r'C:\Users\sroni\OneDrive\Desktop\rotate_ex\data',
#                                   videos_dir=r'C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos',
#                                   save_dir=r"C:\troubleshooting\mitra\project_folder\videos\additional\examples\rotated",
#                                   video_info=r"C:\troubleshooting\mitra\project_folder\logs\video_info.csv",
#                                   direction=0,
#                                   anchor_location=(250, 250),
#                                   fill_clr=(0, 0, 0))
#     aligner.run()

    # aligner = EgocentricalAligner(rotate_video=True,
    #                               anchor_1='tail_base',
    #                               anchor_2='nose',
    #                               data_dir=r'C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location',
    #                               videos_dir=r'C:\troubleshooting\mitra\project_folder\videos',
    #                               save_dir=r"C:\troubleshooting\mitra\project_folder\videos\additional\bg_removed\rotated",
    #                               video_info=r"C:\troubleshooting\mitra\project_folder\logs\video_info.csv")
    # aligner.run()


