import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from simba.utils.checks import (check_if_dir_exists, check_if_valid_rgb_tuple,
                                check_int, check_str, check_valid_boolean,
                                check_valid_dataframe, check_valid_tuple)
from simba.utils.data import egocentrically_align_pose_numba
from simba.utils.enums import Formats, Options
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (bgr_to_rgb_tuple, find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_fn_ext, read_df,
                                    write_df)
from simba.video_processors.egocentric_video_rotator import \
    EgocentricVideoRotator


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

    .. seealso::
       To produce rotation vectors, uses :func:`~simba.utils.data.egocentrically_align_pose_numba` or :func:`~simba.utils.data.egocentrically_align_pose`.
       To rotate video only, see :func:`~simba.video_processors.egocentric_video_rotator.EgocentricVideoRotator`


    :param Union[str, os.PathLike] config_path: Path to the configuration file.
    :param Union[str, os.PathLike] save_dir: Directory where the processed output will be saved.
    :param Optional[Union[str, os.PathLike]] data_dir: Directory containing CSV files with movement data.
    :param Optional[str] anchor_1: Primary anchor point (e.g., 'tail_base') around which the alignment centers.
    :param Optional[str] anchor_2: Secondary anchor point (e.g., 'nose') defining the alignment direction.
    :param int direction: Target angle, in degrees, for alignment; e.g., `0` aligns east
    :param Optional[Tuple[int, int]] anchor_location: Pixel location in the output where `anchor_1`  should appear; default is `(250, 250)`.
    :param Tuple[int, int, int] fill_clr: If rotating the videos, the color of the additional pixels.
    :param Optional[bool] rotate_video: Whether to rotate the video to align with the specified direction.
    :param Optional[int] core_cnt: Number of CPU cores to use for video rotation; `-1` uses all available cores.

    :example:
     >>> aligner = EgocentricalAligner(rotate_video=True, anchor_1='tail_base', anchor_2='nose', data_dir=r"/data_dir", videos_dir=r'/videos_dir', save_dir=r"/save_dir", video_info=r"C:\troubleshooting\mitra\project_folder\logs\video_info.csv", direction=0, anchor_location=(250, 250), fill_clr=(0, 0, 0))
     >>> aligner.run()
    """

    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 anchor_1: str = 'tail_base',
                 anchor_2: str = 'nose',
                 direction: int = 0,
                 anchor_location: Union[Tuple[int, int], str] = (250, 250),
                 core_cnt: int = -1,
                 fill_clr: Tuple[int, int, int] = (250, 250, 255),
                 verbose: bool = True,
                 gpu: bool = False,
                 videos_dir: Optional[Union[str, os.PathLike]] = None):

        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'])
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        check_str(name=f'{self.__class__.__name__} anchor_1', value=anchor_1)
        check_str(name=f'{self.__class__.__name__} anchor_2', value=anchor_2)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], unaccepted_vals=[0])
        if core_cnt == -1: self.core_cnt = find_core_cnt()[0]
        check_int(name=f'{self.__class__.__name__} direction', value=direction, min_value=0, max_value=360)
        check_valid_tuple(x=anchor_location, source=f'{self.__class__.__name__} anchor_location', accepted_lengths=(2,), valid_dtypes=(int,))
        check_valid_boolean(value=[gpu], source=f'{self.__class__.__name__} gpu')
        for i in anchor_location:
            check_int(name=f'{self.__class__.__name__} anchor_location', value=i, min_value=1)
        if videos_dir is not None:
            check_if_valid_rgb_tuple(data=fill_clr)
            fill_clr = bgr_to_rgb_tuple(value=fill_clr)
            check_if_dir_exists(in_dir=videos_dir, source=f'{self.__class__.__name__} videos_dir')
            self.video_paths = find_files_of_filetypes_in_directory(directory=videos_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value)
            for file_path in self.data_paths:
                find_video_of_file(video_dir=videos_dir, filename=get_fn_ext(file_path)[1], raise_error=True)
        self.anchor_1_cols = [f'{anchor_1}_x'.lower(), f'{anchor_1}_y'.lower()]
        self.anchor_2_cols = [f'{anchor_2}_x'.lower(), f'{anchor_2}_y'.lower()]
        self.anchor_1, self.anchor_2, self.videos_dir = anchor_1.lower(), anchor_2.lower(), videos_dir
        self.save_dir, self.verbose, self.gpu = save_dir, verbose, gpu
        self.anchor_location, self.direction, self.fill_clr = anchor_location, direction, fill_clr
        self.rotate_video = [True if videos_dir is not None else False]

    def run(self):
        timer = SimbaTimer(start=True)
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
            results_arr, self.centers, self.rotation_vectors = egocentrically_align_pose_numba(data=data_arr, anchor_1_idx=anchor_1_idx, anchor_2_idx=anchor_2_idx, direction=self.direction, anchor_location=np.array(self.anchor_location).astype(np.int64))
            results_arr = results_arr.reshape(len(df), len(bp_cols))
            self.out_df = pd.DataFrame(results_arr, columns=bp_cols)
            df.update(self.out_df)
            df.columns = original_cols
            write_df(df=df, file_type=Formats.CSV.value, save_path=save_path)
            video_timer.stop_timer()
            if self.verbose:
                print(f'{self.video_name} complete, saved at {save_path} (elapsed time: {video_timer.elapsed_time_str}s)')
            if self.rotate_video:
                if self.verbose:
                    print(f'Rotating video {self.video_name}...')
                video_path = find_video_of_file(video_dir=self.videos_dir, filename=self.video_name, raise_error=False)
                save_path = os.path.join(self.save_dir, f'{self.video_name}.mp4')
                video_rotator = EgocentricVideoRotator(video_path=video_path,
                                                       centers=self.centers,
                                                       rotation_vectors=self.rotation_vectors,
                                                       anchor_location=self.anchor_location,
                                                       verbose=self.verbose,
                                                       gpu=self.gpu,
                                                       fill_clr=self.fill_clr,
                                                       core_cnt=self.core_cnt,
                                                       save_path=save_path)
                video_rotator.run()
            if self.verbose:
                print(f'Rotated data for video {self.video_name} ({file_cnt+1}/{len(self.data_paths)}) saved in {self.save_dir}.')
        timer.stop_timer()
        stdout_success(msg=f'Egocentrically aligned data for {len(self.data_paths)} files saved in {self.save_dir}', elapsed_time=timer.elapsed_time_str)


# if __name__ == "__main__":
#     aligner = EgocentricalAligner(anchor_1='butt/proximal tail',
#                                   anchor_2='snout',
#                                   data_dir=r'C:\troubleshooting\open_field_below\project_folder\csv\outlier_corrected_movement_location',
#                                   videos_dir=r'C:\troubleshooting\open_field_below\project_folder\videos',
#                                   save_dir=r"C:\troubleshooting\open_field_below\project_folder\videos\rotated",
#                                   direction=0,
#                                   gpu=True,
#                                   anchor_location=(600, 300),
#                                   fill_clr=(128,128,128))
#     aligner.run()

#     aligner = EgocentricalAligner(anchor_1='tail_base',
#                                   anchor_2='nose',
#                                   data_dir=r'C:\Users\sroni\OneDrive\Desktop\rotate_ex\data',
#                                   videos_dir=r'C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos',
#                                   save_dir=r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\rotated",
#                                   direction=0,
#                                   gpu=True,
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


