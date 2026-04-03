import functools
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_all_file_names_are_represented_in_video_log,
                                check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_if_valid_rgb_tuple,
                                check_int, check_str, check_valid_boolean,
                                check_valid_lst, check_valid_tuple)
from simba.utils.data import (create_color_palette,
                              egocentrically_align_pose_numba, get_cpu_pool,
                              terminate_cpu_pool)
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    create_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, read_video_info_csv)


def _pose_video_worker(frm_range: tuple,
                       data_arr: np.ndarray,
                       save_dir: str,
                       resolution: Tuple[int, int],
                       fps: float,
                       bg_bgr: Tuple[int, int, int],
                       colors: list,
                       draw_bp_idxs: list,
                       skeleton_idxs: list,
                       circle_size: int,
                       line_thickness: int,
                       p_arr: np.ndarray,
                       bp_threshold: float = 0.0,
                       verbose: bool = False):

    batch_id, frame_rng = frm_range[0], frm_range[1]
    start_frm, end_frm, total_frms = frame_rng[0], frame_rng[-1], len(data_arr)
    video_save_path = os.path.join(save_dir, f'{batch_id}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_save_path, fourcc, fps, resolution)
    for frm_idx in range(start_frm, end_frm + 1):
        if verbose:
            stdout_information(msg=f'Batch {batch_id}, frame {frm_idx}/{total_frms}...')
        frame = np.full((resolution[1], resolution[0], 3), bg_bgr, dtype=np.uint8)
        pts = data_arr[frm_idx].astype(np.int32)
        probs = p_arr[frm_idx]
        if skeleton_idxs is not None:
            for idx1, idx2 in skeleton_idxs:
                if probs[idx1] >= bp_threshold and probs[idx2] >= bp_threshold:
                    cv2.line(frame, tuple(pts[idx1]), tuple(pts[idx2]), (105, 105, 105), line_thickness)
        for bp_idx in draw_bp_idxs:
            if probs[bp_idx] >= bp_threshold:
                cv2.circle(frame, tuple(pts[bp_idx]), circle_size, colors[bp_idx], -1)
        writer.write(frame)
    writer.release()
    return batch_id


class SkeletonVideoCreator:
    """
    Create pose-estimation videos rendered on a solid RGB background from SimBA CSV data.

    Reads outlier-corrected pose CSV files (one row per frame), extracts body-part x/y columns,
    and renders keypoints and optional skeleton segments on a blank canvas—no source video is
    required. FPS for each output file is taken from ``video_info.csv`` for the matching video name.

    Alignment modes (at most one applies; egocentric alignment takes precedence if both are set):

    * **Egocentric** (``ego_anchor_1`` + ``ego_anchor_2``): rotates/translates the pose so the
      segment from anchor 1 → anchor 2 matches ``ego_direction`` (see parameter).
    * **Center anchor** (``anchor_bp`` only, no egocentric anchors): each frame, shifts all
      keypoints so ``anchor_bp`` sits at the image center; no rotation.

    Input CSVs must list body parts as ``<bp>_x`` / ``<bp>_y`` columns. Optional ``<bp>_p``
    probability columns gate drawing; if any are missing, probabilities default to 1.0 for all
    body-parts. Skeleton edges are drawn in a fixed gray; keypoint disks use ``palette``.

    .. video:: _static/img/SkeletonVideoCreator.mp4
       :width: 800
       :autoplay:
       :loop:
       :muted:
       :align: center

    .. video:: _static/img/SkeletonVideoCreator_2.mp4
       :width: 800
       :autoplay:
       :loop:
       :muted:
       :align: center

    .. seealso::

       :class:`~simba.plotting.pose_plotter_mp.PosePlotterMultiProcess` — overlay pose on the original recording instead of a blank background.
       :func:`~simba.video_processors.video_processing.superimpose_overlay_video` — inset one video on another (for example, a skeleton clip over the raw recording).

    :param Optional[Union[str, os.PathLike]] config_path: Path to SimBA project ``project_config.ini``.
        If set, ``data_path``, ``save_dir``, and ``video_info_path`` default from the project unless
        overridden. Required unless all three of those are provided explicitly.
    :param Optional[Union[str, os.PathLike]] data_path: Path to one pose CSV or a directory of ``.csv``
        files. If ``None`` and ``config_path`` is set, uses the project's outlier-corrected movement directory.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory for output ``<video_name>.mp4`` files.
        If ``None`` and ``config_path`` is set, uses ``<project>/frames/output/pose_videos`` (created if needed).
    :param Optional[Union[str, os.PathLike]] video_info_path: Path to ``logs/video_info.csv`` (fps and video names).
        If ``None`` and ``config_path`` is set, uses the project's video info path.
    :param Tuple[int, int] resolution: Output size ``(width, height)`` in pixels. Default ``(500, 500)``.
    :param Tuple[int, int, int] bg_color: Background color as **RGB** ``(R, G, B)``, each 0–255.
        Default ``(0, 0, 0)`` (black).
    :param Optional[str] anchor_bp: Body-part name whose location is pinned to the frame center each
        frame (case-insensitive match to CSV names). Ignored if egocentric anchors are set. Default None.
    :param Optional[List[Tuple[str, str]]] skeleton: Pairs of body-part names ``(from, to)`` for line
        segments. Omitted or skipped pairs involving ``omit_bps``. If None, only keypoints are drawn.
    :param Optional[int] circle_size: Keypoint circle radius in pixels. If None, scaled from ``resolution``.
    :param Optional[int] line_thickness: Skeleton line thickness in pixels. If None, scaled from ``resolution``.
    :param Optional[str] ego_anchor_1: First anchor body-part for egocentric alignment (e.g. ``tail_base``).
        Must be given together with ``ego_anchor_2``.
    :param Optional[str] ego_anchor_2: Second anchor; together with ``ego_anchor_1`` defines the forward
        axis before rotation.
    :param int ego_direction: Desired compass heading in degrees for the vector from ``ego_anchor_1`` to
        ``ego_anchor_2`` after alignment: 0 = north/up, 90 = east/right, 180 = south/down, 270 = west/left.
        Default 0.
    :param Optional[List[str]] omit_bps: Body-part names to exclude from dots and skeleton (lowercased internally).
    :param str palette: Matplotlib qualitative palette name for per-body-part keypoint colors. Default ``Set1``.
    :param float bp_threshold: Minimum per-frame probability to draw a keypoint or use it in a skeleton edge.
        Default ``0.0``.
    :param int core_cnt: Worker processes for frame batches; ``-1`` uses all CPUs. Default ``-1``.
    :param bool verbose: Print batch and file progress. Default True.

    :raises InvalidInputError: If neither ``config_path`` nor the triple
        (``data_path``, ``save_dir``, ``video_info_path``) is satisfactorily provided; or if only one
        of ``ego_anchor_1`` / ``ego_anchor_2`` is set.
    :raises NoFilesFoundError: If ``data_path`` is not a valid file or directory.

    :example:
    >>> creator = SkeletonVideoCreator(
    ...     config_path=r'E:/project/project_config.ini',
    ...     resolution=(500, 500),
    ...     bg_color=(0, 0, 0),
    ...     anchor_bp='tail_base',
    ...     skeleton=[('nose', 'left_ear'), ('nose', 'right_ear'), ('left_ear', 'center'), ('right_ear', 'center'), ('center', 'left_side'), ('center', 'right_side'), ('center', 'tail_base'), ('tail_base', 'tail_mid'), ('tail_mid', 'tail_end')],
    ...     ego_anchor_1='tail_base',
    ...     ego_anchor_2='nose',
    ... )
    >>> creator.run()
    """

    def __init__(self,
                 config_path: Optional[Union[str, os.PathLike]] = None,
                 data_path: Optional[Union[str, os.PathLike]] = None,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 video_info_path: Optional[Union[str, os.PathLike]] = None,
                 resolution: Tuple[int, int] = (500, 500),
                 bg_color: Tuple[int, int, int] = (0, 0, 0),
                 anchor_bp: Optional[str] = None,
                 skeleton: Optional[List[Tuple[str, str]]] = None,
                 circle_size: Optional[int] = None,
                 line_thickness: Optional[int] = None,
                 ego_anchor_1: Optional[str] = None,
                 ego_anchor_2: Optional[str] = None,
                 ego_direction: int = 0,
                 omit_bps: Optional[List[str]] = None,
                 palette: str = 'Set1',
                 bp_threshold: float = 0.0,
                 core_cnt: int = -1,
                 verbose: bool = True):

        if config_path is None and (data_path is None or save_dir is None or video_info_path is None):
            raise InvalidInputError(msg='Either config_path or all of data_path, save_dir, and video_info_path must be provided.', source=self.__class__.__name__)

        if config_path is not None:
            check_file_exist_and_readable(file_path=config_path)
            config = ConfigReader(config_path=config_path, read_video_info=False)
            if data_path is None:
                data_path = config.outlier_corrected_movement_dir
            if save_dir is None:
                save_dir = os.path.join(config.frames_output_dir, 'pose_videos')
                if not os.path.isdir(save_dir):
                    create_directory(paths=[save_dir])
            if video_info_path is None:
                video_info_path = config.video_info_path

        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        check_file_exist_and_readable(file_path=video_info_path)
        check_valid_tuple(x=resolution, source=f'{self.__class__.__name__} resolution', accepted_lengths=(2,), valid_dtypes=(int,))
        check_int(name=f'{self.__class__.__name__} resolution width', value=resolution[0], min_value=10)
        check_int(name=f'{self.__class__.__name__} resolution height', value=resolution[1], min_value=10)
        check_if_valid_rgb_tuple(data=bg_color)
        check_int(name=f'{self.__class__.__name__} ego_direction', value=ego_direction, min_value=0, max_value=360)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        if circle_size is not None:
            check_int(name=f'{self.__class__.__name__} circle_size', value=circle_size, min_value=1)
        if line_thickness is not None:
            check_int(name=f'{self.__class__.__name__} line_thickness', value=line_thickness, min_value=1)
        if anchor_bp is not None:
            check_str(name=f'{self.__class__.__name__} anchor_bp', value=anchor_bp)
        if ego_anchor_1 is not None or ego_anchor_2 is not None:
            if ego_anchor_1 is None or ego_anchor_2 is None:
                raise InvalidInputError(msg='Both ego_anchor_1 and ego_anchor_2 must be provided for egocentric alignment.', source=self.__class__.__name__)
            check_str(name=f'{self.__class__.__name__} ego_anchor_1', value=ego_anchor_1)
            check_str(name=f'{self.__class__.__name__} ego_anchor_2', value=ego_anchor_2)
        if skeleton is not None:
            check_valid_lst(data=skeleton, source=f'{self.__class__.__name__} skeleton', valid_dtypes=(tuple,), min_len=1)
        if omit_bps is not None:
            check_valid_lst(data=omit_bps, source=f'{self.__class__.__name__} omit_bps', valid_dtypes=(str,), min_len=1)
        check_str(name=f'{self.__class__.__name__} palette', value=palette)
        check_float(name=f'{self.__class__.__name__} bp_threshold', value=bp_threshold, min_value=0.0, max_value=1.0)

        if circle_size is None:
            circle_size = max(1, PlottingMixin().get_optimal_circle_size(frame_size=resolution, circle_frame_ratio=25))
        if line_thickness is None:
            line_thickness = max(1, PlottingMixin().get_optimal_circle_size(frame_size=resolution, circle_frame_ratio=100))

        self.save_dir, self.resolution, self.core_cnt = save_dir, resolution, core_cnt
        self.bg_color, self.anchor_bp, self.skeleton = bg_color, anchor_bp, skeleton
        self.circle_size, self.line_thickness, self.verbose = circle_size, line_thickness, verbose
        self.ego_anchor_1, self.ego_anchor_2, self.ego_direction = ego_anchor_1, ego_anchor_2, ego_direction
        self.omit_bps = [x.lower() for x in omit_bps] if omit_bps is not None else None
        self.palette, self.bp_threshold = palette, bp_threshold

        self.video_info_df = read_video_info_csv(file_path=video_info_path)
        if os.path.isfile(data_path):
            check_file_exist_and_readable(file_path=data_path)
            self.data_paths = [data_path]
        elif os.path.isdir(data_path):
            check_if_dir_exists(in_dir=data_path, source=f'{self.__class__.__name__} data_path')
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=['.csv'], raise_error=True)
        else:
            raise NoFilesFoundError(msg=f'{data_path} is not a valid file or directory path.', source=self.__class__.__name__)
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)

    def run(self):
        """
        Render skeleton videos for every CSV in ``data_paths``.

        For each file, splits frames across a process pool, writes batch MP4s under
        ``<save_dir>/<video_name>/temp/``, concatenates them to ``<save_dir>/<video_name>.mp4``,
        then prints a completion summary. Does not return a value.
        """
        timer = SimbaTimer(start=True)
        pool = get_cpu_pool(core_cnt=self.core_cnt, verbose=self.verbose, source=self.__class__.__name__)
        core_cnt = pool._processes
        for file_cnt, file_path in enumerate(self.data_paths):
            _, video_name, _ = get_fn_ext(filepath=file_path)
            fps = float(self.video_info_df.loc[self.video_info_df['Video'] == video_name, 'fps'].values[0])

            df = read_df(file_path=file_path, file_type='csv')
            df.columns = [x.lower() for x in df.columns]
            bp_cols = [x for x in df.columns if not x.endswith('_p')]
            body_parts = []
            for c in bp_cols:
                bp_name = c[:-2]
                if bp_name not in body_parts:
                    body_parts.append(bp_name)

            if self.verbose:
                stdout_information(msg=f'Processing {file_cnt + 1}/{len(self.data_paths)} ({video_name}): {len(df)} frames, {len(body_parts)} body-parts, {fps} fps...')

            data_arr = df[bp_cols].values.reshape(len(df), len(body_parts), 2).astype(np.float64)
            p_cols = [f'{bp}_p' for bp in body_parts]
            p_arr = df[p_cols].values.astype(np.float64) if all(c in df.columns for c in p_cols) else np.ones((len(df), len(body_parts)), dtype=np.float64)

            if self.ego_anchor_1 is not None:
                anchor_1_idx = body_parts.index(self.ego_anchor_1.lower())
                anchor_2_idx = body_parts.index(self.ego_anchor_2.lower())
                center = np.array([self.resolution[0] // 2, self.resolution[1] // 2], dtype=np.int64)
                numba_direction = (270 + self.ego_direction) % 360
                data_arr, _, _ = egocentrically_align_pose_numba(data=data_arr.astype(np.int32), anchor_1_idx=anchor_1_idx, anchor_2_idx=anchor_2_idx, direction=numba_direction, anchor_location=center)
                data_arr = data_arr.astype(np.float64)
            elif self.anchor_bp is not None:
                anchor_idx = body_parts.index(self.anchor_bp.lower())
                center_x, center_y = self.resolution[0] // 2, self.resolution[1] // 2
                for frm_idx in range(len(data_arr)):
                    anchor_x, anchor_y = data_arr[frm_idx, anchor_idx, 0], data_arr[frm_idx, anchor_idx, 1]
                    data_arr[frm_idx, :, 0] += center_x - anchor_x
                    data_arr[frm_idx, :, 1] += center_y - anchor_y

            draw_bp_idxs = list(range(len(body_parts)))
            if self.omit_bps is not None:
                draw_bp_idxs = [i for i, bp in enumerate(body_parts) if bp not in self.omit_bps]
            colors = create_color_palette(pallete_name=self.palette, increments=len(body_parts), as_int=True)

            skeleton_idxs = None
            if self.skeleton is not None:
                skeleton_idxs = []
                for bp1, bp2 in self.skeleton:
                    bp1_l, bp2_l = bp1.lower(), bp2.lower()
                    if self.omit_bps is not None and (bp1_l in self.omit_bps or bp2_l in self.omit_bps):
                        continue
                    idx1, idx2 = body_parts.index(bp1_l), body_parts.index(bp2_l)
                    skeleton_idxs.append((idx1, idx2))

            bg_bgr = (self.bg_color[2], self.bg_color[1], self.bg_color[0])
            video_temp_dir = os.path.join(self.save_dir, video_name, 'temp')
            save_path = os.path.join(self.save_dir, f'{video_name}.mp4')
            create_directory(paths=video_temp_dir)

            frm_batches = np.array_split(list(range(len(data_arr))), core_cnt)
            frm_batches = [(i, j) for i, j in enumerate(frm_batches)]
            constants = functools.partial(_pose_video_worker,
                                         data_arr=data_arr,
                                         save_dir=video_temp_dir,
                                         resolution=self.resolution,
                                         fps=fps,
                                         bg_bgr=bg_bgr,
                                         colors=colors,
                                         draw_bp_idxs=draw_bp_idxs,
                                         skeleton_idxs=skeleton_idxs,
                                         circle_size=self.circle_size,
                                         line_thickness=self.line_thickness,
                                         p_arr=p_arr,
                                         bp_threshold=self.bp_threshold,
                                         verbose=self.verbose)
            for cnt, result in enumerate(pool.imap(constants, frm_batches, chunksize=1)):
                if self.verbose:
                    stdout_information(msg=f'{video_name}: batch {result + 1}/{core_cnt} complete...')

            concatenate_videos_in_folder(in_folder=video_temp_dir, save_path=save_path, gpu=True)
            if self.verbose:
                stdout_information(msg=f'{video_name} saved at {save_path}')

        terminate_cpu_pool(pool=pool, source=self.__class__.__name__)
        timer.stop_timer()
        stdout_success(msg=f'Pose videos for {len(self.data_paths)} files saved in {self.save_dir}', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)

if __name__ == "__main__":
    creator = SkeletonVideoCreator(
        config_path=r'E:\troubleshooting\mitra_emergence_hour\project_folder\project_config.ini',
        data_path=r"E:\troubleshooting\mitra_emergence_hour\project_folder\csv\outlier_corrected_movement_location\Box3_180mISOcontrol_Females.csv",
        resolution=(375, 375),
        bg_color=(255, 255, 255),
        bp_threshold=0.1,
        anchor_bp='center',
        skeleton=[('nose', 'left_ear'), ('nose', 'right_ear'), ('left_ear', 'right_ear'), ('left_ear', 'center'), ('right_ear', 'center'), ('center', 'left_side'),  ('left_ear', 'left_side'), ('center', 'right_side'), ('right_ear', 'right_side'), ('left_side', 'tail_base'), ('right_side', 'tail_base'), ('center', 'tail_base'), ('tail_base', 'tail_end')],
        ego_anchor_1='center',
        ego_anchor_2='nose',
        core_cnt=8,
        omit_bps=['tail_center',])
    creator.run()
