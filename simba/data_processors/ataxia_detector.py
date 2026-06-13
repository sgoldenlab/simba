import os
from typing import Optional, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_float,
    check_if_dir_exists, check_int, check_str, check_valid_dataframe)
from simba.utils.data import detect_bouts, plug_holes_shortest_bout
from simba.utils.enums import Formats
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_current_time, get_fn_ext, read_df,
                                    read_video_info)

ATAXIA = 'ATAXIA'


class AtaxiaDetector(ConfigReader):

    """
    Detect ataxia / loss-of-balance using a heuristic vote across five signals from top-down pose-estimation.

    .. important::

        No single body metric separates ataxia from normal behavior - body-width/splay, an early candidate, is in
        fact non-discriminative (it only catches the rare dramatic topple). A supervised leave-one-animal-out
        analysis showed the behavior is a *multivariate* vestibular/dysmetria picture; the five signals below,
        each summarized over a sliding ``window`` and thresholded, recover most of the achievable separation
        (heuristic sensitivity index d' ~ 1.5 at ``min_votes=3``, against a full-model ceiling of ~1.6 - that
        ceiling is set by the pose data, not by the choice of features).

        For each frame, over a centered sliding window:

        * **ear asymmetry** (window max) - ``|dist(left_ear, center) - dist(right_ear, center)|`` > ``ear_asym_threshold`` (left/right imbalance; strongest single signal).
        * **compactness** (window min) - mean distance of nose/tail-base/sides/tail-tip from center > ``compact_threshold`` (abnormally uncompact body).
        * **tail length** (window max) - ``dist(tail_base, tail_tip)`` > ``tail_length_threshold`` (stiff, extended tail).
        * **head-tilt** (window mean) - angle between head axis (ear-midpoint->nose) and body axis (tail-base->nose) > ``headtilt_threshold``.
        * **jerk** (window std) - std of center-speed jerk (3rd derivative) > ``jerk_threshold`` (non-smooth, uncoordinated movement).

        A frame is flagged ``present`` when at least ``min_votes`` of the five signals are tripped: ``min_votes=2``
        is sensitive (wide net to review), ``min_votes=3`` is specific. ``Probability_ATAXIA`` stores the vote
        fraction (n_tripped / 5) as a graded confidence.

        Note: the analysis caps at d' ~ 1.6 at the bout level (CNO and Saline behavior intrinsically overlap), but
        per-session separation is near-perfect (5/5 animals). Higher bout-level resolution requires better pose
        tracking during the behavior or appearance/CNN-based features rather than more keypoint features.

    .. note::

       We pass the nose, left-ear, right-ear, left-side, right-side, tail-base, tail-tip and center body-parts.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file.
    :param Optional[Union[str, os.PathLike]] data_dir: Directory with pose CSVs. If None, uses the project ``outlier_corrected_movement_location`` directory.
    :param Optional[float] window: Sliding-window length in seconds over which each signal is summarized. Defaults to 2.0.
    :param Optional[int] min_votes: Number of the five signals (1-5) that must be tripped to flag a frame. Defaults to 2.
    :param Optional[float] ear_asym_threshold: Ear asymmetry threshold in mm. Defaults to 34.0.
    :param Optional[float] compact_threshold: Compactness (mean extremity-to-center distance) threshold in mm. Defaults to 55.0.
    :param Optional[float] tail_length_threshold: Tail-base to tail-tip distance threshold in mm. Defaults to 99.0.
    :param Optional[float] headtilt_threshold: Head-vs-body angle threshold in degrees. Defaults to 47.0.
    :param Optional[float] jerk_threshold: Center-speed jerk standard-deviation threshold. Defaults to 6.0.
    :param int shortest_bout: Shortest allowed ataxia bout in milliseconds; shorter detections are removed and gaps smaller than this are stitched. Defaults to 1000.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory for results. If None, a time-stamped directory in the project ``logs`` directory.

    References
    ----------
    .. [1] Lazaro et al., Brainwide Genetic Capture for Conscious State Transitions, `biorxiv`, doi: https://doi.org/10.1101/2025.03.28.646066

    :example:
    >>> detector = AtaxiaDetector(config_path=r"H:\projects\brainwide_trap\brainwide_trap\project_folder\project_config.ini", min_votes=3)
    >>> detector.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 nose_name: str = 'nose',
                 left_ear_name: str = 'left_ear',
                 right_ear_name: str = 'right_ear',
                 left_side_name: str = 'left_side',
                 right_side_name: str = 'right_side',
                 tail_base_name: str = 'tail_base',
                 tail_tip_name: str = 'tail_tip',
                 center_name: str = 'center',
                 window: Optional[float] = 2.0,
                 min_votes: Optional[int] = 2,
                 ear_asym_threshold: Optional[float] = 34.0,
                 compact_threshold: Optional[float] = 55.0,
                 tail_length_threshold: Optional[float] = 99.0,
                 headtilt_threshold: Optional[float] = 47.0,
                 jerk_threshold: Optional[float] = 6.0,
                 shortest_bout: int = 1000,
                 save_dir: Optional[Union[str, os.PathLike]] = None):

        self.bps = {'nose': nose_name, 'left_ear': left_ear_name, 'right_ear': right_ear_name,
                    'left_side': left_side_name, 'right_side': right_side_name, 'tail_base': tail_base_name,
                    'tail_tip': tail_tip_name, 'center': center_name}
        for bp_name in self.bps.values():
            check_str(name='body part name', value=bp_name, allow_blank=False)
        for v_name, v in [('window', window), ('ear_asym_threshold', ear_asym_threshold),
                          ('compact_threshold', compact_threshold), ('tail_length_threshold', tail_length_threshold),
                          ('headtilt_threshold', headtilt_threshold), ('jerk_threshold', jerk_threshold)]:
            check_float(name=v_name, value=v, min_value=0.0)
        check_int(name='min_votes', value=min_votes, min_value=1, max_value=5)
        check_int(name='shortest_bout', value=shortest_bout, min_value=0)
        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=False)
        if data_dir is not None:
            check_if_dir_exists(in_dir=data_dir)
        else:
            data_dir = self.outlier_corrected_dir
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'])
        self.heads = {k: [f'{v}_x'.lower(), f'{v}_y'.lower()] for k, v in self.bps.items()}
        self.required_field = [h for hs in self.heads.values() for h in hs]
        self.window, self.min_votes, self.shortest_bout = window, min_votes, shortest_bout
        self.thresholds = {'ear_asym': ear_asym_threshold, 'compact': compact_threshold,
                           'tail_length': tail_length_threshold, 'headtilt': headtilt_threshold,
                           'jerk': jerk_threshold}
        self.save_dir = save_dir
        if self.save_dir is None:
            self.save_dir = os.path.join(self.logs_path, f'ataxia_data_vote{min_votes}_{self.datetime}')
            os.makedirs(self.save_dir)
        else:
            check_if_dir_exists(in_dir=self.save_dir)

    def run(self):
        agg_results = pd.DataFrame(columns=['VIDEO', 'ATAXIA FRAMES', 'ATAXIA TIME (S)', 'ATAXIA BOUT COUNTS', 'ATAXIA PCT OF SESSION', 'VIDEO TOTAL FRAMES', 'VIDEO TOTAL TIME (S)'])
        agg_results_path = os.path.join(self.save_dir, 'aggregate_ataxia_results.csv')
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            video_name = get_fn_ext(filepath=file_path)[1]
            print(f'[{get_current_time()}] Analyzing ataxia {video_name}... (video {file_cnt+1}/{len(self.data_paths)})')
            save_file_path = os.path.join(self.save_dir, f'{video_name}.csv')
            df = read_df(file_path=file_path, file_type='csv').reset_index(drop=True)
            _, px_per_mm, fps = read_video_info(video_info_df=self.video_info_df, video_name=video_name)
            df.columns = [str(x).lower() for x in df.columns]
            check_valid_dataframe(df=df, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.required_field)
            p = {k: df[h].values.astype(np.float32) for k, h in self.heads.items()}
            d = lambda a, b: np.linalg.norm(p[a] - p[b], axis=1) / px_per_mm

            # per-frame signals
            ear_asym = np.abs(d('left_ear', 'center') - d('right_ear', 'center'))
            compact = np.mean([d(bp, 'center') for bp in ('nose', 'tail_base', 'left_side', 'right_side', 'tail_tip')], axis=0)
            tail_len = d('tail_base', 'tail_tip')
            ear_mid = (p['left_ear'] + p['right_ear']) / 2.0
            body_ang = np.degrees(np.arctan2(p['nose'][:, 1] - p['tail_base'][:, 1], p['nose'][:, 0] - p['tail_base'][:, 0]))
            head_ang = np.degrees(np.arctan2(p['nose'][:, 1] - ear_mid[:, 1], p['nose'][:, 0] - ear_mid[:, 0]))
            headtilt = np.abs((head_ang - body_ang + 180) % 360 - 180)
            speed = np.r_[0, np.linalg.norm(np.diff(p['center'], axis=0), axis=1)] / px_per_mm * fps
            jerk = np.abs(np.r_[0, np.diff(np.r_[0, np.diff(speed)])])

            # sliding-window summaries (centered), one per signal, matching the validated aggregations
            w = max(1, int(self.window * fps))
            roll = lambda s: pd.Series(s).rolling(w, center=True, min_periods=1)
            votes = ((roll(ear_asym).max().values > self.thresholds['ear_asym']).astype(int)
                     + (roll(compact).min().values > self.thresholds['compact']).astype(int)
                     + (roll(tail_len).max().values > self.thresholds['tail_length']).astype(int)
                     + (roll(headtilt).mean().values > self.thresholds['headtilt']).astype(int)
                     + (roll(jerk).std().values > self.thresholds['jerk']).astype(int))
            ataxia_idx = np.argwhere(votes >= self.min_votes).astype(np.int32).flatten()

            df[f'Probability_{ATAXIA}'] = votes / 5.0
            df[ATAXIA] = 0
            df.loc[ataxia_idx, ATAXIA] = 1
            df = plug_holes_shortest_bout(data_df=df, clf_name=ATAXIA, fps=fps, shortest_bout=self.shortest_bout)
            bouts = detect_bouts(data_df=df, target_lst=[ATAXIA], fps=fps)
            df[ATAXIA] = 0
            if len(bouts) > 0:
                ataxia_idx = [x for xs in bouts.apply(lambda r: list(range(int(r["Start_frame"]), int(r["End_frame"]) + 1)), 1) for x in xs]
                df.loc[ataxia_idx, ATAXIA] = 1
            else:
                ataxia_idx = []

            df.to_csv(save_file_path)
            agg_results.loc[len(agg_results)] = [video_name, len(ataxia_idx), round(len(ataxia_idx) / fps, 4), len(bouts), round((len(ataxia_idx) / len(df)) * 100, 4), len(df), round(len(df) / fps, 2)]

        agg_results.to_csv(agg_results_path)
        stdout_success(msg=f'Results saved in {self.save_dir} directory.')


# detector = AtaxiaDetector(config_path=r"H:\projects\brainwide_trap\brainwide_trap\project_folder\project_config.ini", min_votes=3)
# detector.run()
