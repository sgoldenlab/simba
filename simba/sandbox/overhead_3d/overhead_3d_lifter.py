import functools
import json
import os
from typing import Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.sandbox.overhead_3d.overhead_3d_utils import (METHOD_OPTIONS,
                                                         REQUIRED_ROLES, ROLES,
                                                         backproject,
                                                         body_scale, box_calib,
                                                         calibrate_box,
                                                         check_keypoint_layout,
                                                         derive_center,
                                                         estimate_wall_mm,
                                                         fix_handedness, lift,
                                                         manual_calib,
                                                         match_roles,
                                                         normalise_proportions,
                                                         scaled_calib)
from simba.utils.checks import (check_float, check_if_dir_exists, check_int,
                                check_str, check_valid_boolean)
from simba.utils.data import get_cpu_pool, terminate_cpu_pool
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.printing import (SimbaTimer, stdout_information,
                                  stdout_success, stdout_warning)
from simba.utils.read_write import (find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, write_df)

CALIBRATION_OPTIONS = ['camera_height', 'box']
DEFAULT_CAMERA_MM = 300.0
MAG_TOL = 0.10


def _calibrate_box_worker(video_path: str, pose_path: str, floor_size_mm: float, file_type: str, roles: dict):
    """Box calibration + this video's wall-height estimate from body reach (first animal). Returns a small dict."""
    try:
        box = calibrate_box(video_path, floor_size_mm=floor_size_mm)
        px, conf = next(iter(_read_pose_worker(pose_path, file_type, roles).values()))
        box['wall_mm_estimate'] = estimate_wall_mm(px, conf, box)
        return box
    except Exception as e:
        return {'error': str(e)}


def _read_pose_worker(file_path: str, file_type: str, roles: Dict[str, Dict[str, Optional[str]]]):
    """Pose file -> {animal: (px (T, 9, 2), conf (T, 9))} in lifter-role order (runs in worker processes)."""
    df = read_df(file_path=file_path, file_type=file_type).reset_index(drop=True)
    out = {}
    for animal, rmap in roles.items():
        T = len(df)
        px, conf = np.zeros((T, 9, 2)), np.zeros((T, 9))
        for k, role in enumerate(ROLES):
            bp = rmap[role]
            if bp is not None:
                px[:, k] = df[[f'{bp}_x', f'{bp}_y']].values
                conf[:, k] = df[f'{bp}_p'].values
        mapped = np.array([rmap[r] is not None for r in ROLES])
        if conf.max() == 0 and (px != 0).any():                              # tracker wrote no confidences
            conf[:, mapped] = 1.0
        conf[(px == 0).all(-1)] = 0                                            # (0, 0) = missing in SimBA
        if rmap['Center'] is None:                                             # no Center body-part: derive it
            px, conf = derive_center(px, conf)
        for k in range(9):                                                     # absent roles: placeholder at Center
            if conf[:, k].max() == 0:
                px[:, k] = px[:, ROLES.index('Center')]
        out[animal] = (px.astype(np.float64), conf.astype(np.float64))
    return out


def _file_body_parts(file_path: str, file_type: str) -> List[str]:
    cols = pd.read_csv(file_path, nrows=0, index_col=0).columns if file_type == 'csv' else pd.read_parquet(file_path).columns
    return [c[:-2] for c in cols if c.endswith('_x')]


def _write_worker(data: Dict[str, np.ndarray], file_type: str, save_path: str):
    write_df(df=pd.DataFrame(data).round(2), file_type=file_type, save_path=save_path)
    return save_path


class Overhead3DLifter(ConfigReader):
    """
    Lift the 2D pose-estimation of a SimBA project, recorded with ONE camera looking straight down, to 3D keypoints
    (mm; x, y on the floor plane, z = height above the floor) for every animal and frame.

    Body-parts are mapped onto the lifter roles Nose, Left_Ear, Right_Ear, Left_Side, Right_Side, Center, Tail_Base,
    Tail_Center, Tail_End (Nose, ears, Center, Tail_Base required) by name, or explicitly with ``bp_map``. Before lifting,
    each animal's keypoint layout is checked against anatomy (warns when body-part names do not match the data),
    left/right ears and flanks are swapped if labelled the other way round, and (lifter input only) segment proportions
    are normalised to the lifter's training domain.

    .. note::
       Accuracy (height MAE): ~4-5 mm on held-out synthetic and real 3D mouse data for ``method='lifter'``; see the README.
       The 3D is a *trunk/head* reconstruction: no limbs, the tail lies on the floor. Rats are outside the training data.

    .. note::
       Use the 3D as a *measurement* (head/nose height in mm, rear height and duration, head-up time, body pitch), not
       as a classifier booster. The heights are computed from the same 2D keypoints, so a rich SimBA 2D feature set
       already holds most of that information. On the mitra project (9 mice, leave-one-mouse-out
       HistGradientBoosting), adding 3D features changed best F1 from 0.525 to 0.539 for rearing and from 0.538 to 0.529 for
       grooming, and raised immobility from 0.291 to 0.363 (0.388 with a temporal lifter). Per-video normalisation of the
       3D features gave no gain. Classifier accuracy there is limited by generalisation to unseen animals (rearing F1 0.78 on
       random frames vs 0.52 on held-out mice), which no keypoint-derived feature fixes.

    .. video:: _static/img/Overhead3DLifter_ressler.webm
       :width: 800
       :autoplay:
       :loop:
       :muted:
       :align: center

    .. video:: _static/img/Overhead3DLifter_maplight.webm
       :width: 800
       :autoplay:
       :loop:
       :muted:
       :align: center

    :param Union[str, os.PathLike] config_path: Path to the SimBA project_config.ini.
    :param Optional[Union[str, os.PathLike]] data_dir: Directory with pose files. Default: project ``csv/outlier_corrected_movement_location``.
    :param Optional[Union[str, os.PathLike]] save_dir: Output directory. Default: project ``csv/pose_3d``.
    :param Literal['lifter', 'fused', 'geometry'] method: Lifting method. ``lifter`` (default) is the most accurate on real 3D data and robust to missing tail points.
    :param Literal['camera_height', 'box'] calibration: ``camera_height``: camera height above the floor is known (``camera_height_mm``), pixels/mm from the project video_info, optical centre = image centre. ``box``: auto-calibrate from a square box whose video is cropped at the wall rim (``box_floor_mm``, ``wall_height_mm`` or estimated from body reach).
    :param Optional[float] camera_height_mm: Camera height above the floor (mm) for ``calibration='camera_height'``. None (unknown): 300 mm for a lifter-size mouse, scaled per video by the animals' apparent size (this also absorbs a wrong pixels/mm); flagged ``camera_height_assumed`` in the summary.
    :param float box_floor_mm: Floor side length (mm) for ``calibration='box'``. Default 156.
    :param Optional[float] wall_height_mm: Wall height (mm) for ``calibration='box'``. None: estimated from body reach (median over videos).
    :param Optional[Dict[str, str]] bp_map: Role -> body-part name (without animal prefix), e.g. ``{'Nose': 'RIGHT_EAR', 'Center': 'LEFT_SIDE'}``. Applied to all animals. Unmapped roles are matched by name.
    :param bool fix_left_right: Swap left/right ears/flanks when labelled the other way round. Default True.
    :param bool normalise: Normalise segment proportions of the lifter input. Default True.
    :param bool size_normalise: Animals > 15% larger/smaller than the lifter's mouse (e.g. rats) are lifted in a world scaled to mouse size and the heights scaled back. Default True.
    :param bool height_error: Write ``<animal>_<role>_z_err``: the expected height error (mm) of every keypoint and frame, from the disagreement of the lifter's 5 ensemble models, converted to mm with a fit of the real error on held-out 3D data (real mice: predicted 1.9-7.8 vs actual 2.3-8.3 mm MAE from the most to least certain fifth of frames). The ranking (which frames to trust) is reliable; the mm value is approximate and does not include errors from a wrong camera height or wrong 2D keypoints. ``method='lifter'`` only; ignored otherwise. Default True.
    :param bool gpu: Run the lifter on the GPU if available. Default True.
    :param int core_cnt: Worker processes for reading/writing pose files and box calibration. -1 = min(all cores, 8). Default -1.
    :param bool verbose: Print progress. Default True.

    :example:
    >>> Overhead3DLifter(config_path=r'I:/mitra/nick_ressler/project_folder/project_config.ini', calibration='box').run()
    >>> Overhead3DLifter(config_path=r'D:/maplight_ri/project_folder/project_config.ini', calibration='camera_height', camera_height_mm=300,
    ...                  bp_map={'Nose': 'RIGHT_EAR', 'Left_Ear': 'NOSE', 'Right_Ear': 'LEFT_EAR', 'Center': 'LEFT_SIDE', 'Left_Side': 'CENTER'}).run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 method: Literal['lifter', 'fused', 'geometry'] = 'lifter',
                 calibration: Literal['camera_height', 'box'] = 'camera_height',
                 camera_height_mm: Optional[float] = None,
                 box_floor_mm: float = 156.0,
                 wall_height_mm: Optional[float] = None,
                 bp_map: Optional[Dict[str, str]] = None,
                 fix_left_right: bool = True,
                 size_normalise: bool = True,
                 normalise: bool = True,
                 height_error: bool = True,
                 gpu: bool = True,
                 core_cnt: int = -1,
                 verbose: bool = True):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=False)
        check_str(name=f'{self.__class__.__name__} method', value=method, options=METHOD_OPTIONS)
        check_str(name=f'{self.__class__.__name__} calibration', value=calibration, options=CALIBRATION_OPTIONS)
        if calibration == 'camera_height' and camera_height_mm is not None:
            check_float(name=f'{self.__class__.__name__} camera_height_mm', value=camera_height_mm, min_value=50.0)
        check_float(name=f'{self.__class__.__name__} box_floor_mm', value=box_floor_mm, min_value=10.0)
        if wall_height_mm is not None:
            check_float(name=f'{self.__class__.__name__} wall_height_mm', value=wall_height_mm, min_value=5.0)
        for v, n in ((fix_left_right, 'fix_left_right'), (normalise, 'normalise'), (height_error, 'height_error'), (gpu, 'gpu'), (verbose, 'verbose')):
            check_valid_boolean(value=v, source=f'{self.__class__.__name__} {n}', raise_error=True)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        if bp_map is not None:
            bad = [k for k in bp_map if k not in ROLES]
            if bad:
                raise InvalidInputError(msg=f'bp_map keys {bad} are not roles; valid roles: {ROLES}', source=self.__class__.__name__)
        self.data_dir = self.outlier_corrected_dir if data_dir is None else data_dir
        check_if_dir_exists(in_dir=self.data_dir)
        self.save_dir = os.path.join(self.project_path, 'csv', 'pose_3d') if save_dir is None else save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.method, self.calibration, self.camera_height_mm = method, calibration, camera_height_mm
        self.box_floor_mm, self.wall_height_mm, self.bp_map = box_floor_mm, wall_height_mm, bp_map
        self.fix_left_right, self.normalise, self.gpu, self.verbose = fix_left_right, normalise, gpu, verbose
        self.size_normalise = size_normalise
        self.height_error = height_error and method == 'lifter'
        self.core_cnt = min(find_core_cnt()[0], 8) if core_cnt == -1 else core_cnt   # IO-bound workers; more mostly costs RAM
        self.files = find_files_of_filetypes_in_directory(directory=self.data_dir, extensions=[f'.{self.file_type}'], raise_error=True, sort_alphabetically=True)
        known = set(self.video_info_df['Video'].astype(str))
        skipped = [get_fn_ext(f)[1] for f in self.files if get_fn_ext(f)[1] not in known]
        if skipped:
            stdout_warning(msg=f'{len(skipped)} pose file(s) have no row in video_info.csv and are SKIPPED: {skipped}')
            self.files = [f for f in self.files if get_fn_ext(f)[1] in known]
        if not self.files:
            raise NoFilesFoundError(msg=f'No pose file in {self.data_dir} has a row in video_info.csv', source=self.__class__.__name__)
        self.roles = self._map_roles()

    def _map_roles(self) -> Dict[str, Dict[str, Optional[str]]]:
        roles = {}
        file_bps = _file_body_parts(self.files[0], self.file_type)
        for i, (animal, d) in enumerate(self.animal_bp_dict.items()):
            names = [x[:-2] for x in d['X_bps']]
            if not all(n in file_bps for n in names):                         # config body-parts are not the file's columns
                if len(self.animal_bp_dict) == 1:
                    names = file_bps
                else:
                    names = [n for n in file_bps if n.endswith(f'_{i + 1}')] or [n for n in file_bps if n.startswith(f'{animal}_')]
                if not names:
                    raise InvalidInputError(msg=f'Animal {animal}: project body-parts {[x[:-2] for x in d["X_bps"]]} are not in {self.files[0]} and the file columns cannot be assigned to animals.', source=self.__class__.__name__)
                stdout_warning(msg=f'{animal}: the project_config body-parts do not match the pose-file columns; using the file columns {names}')
            prefix = f'{animal}_' if all(n.startswith(f'{animal}_') for n in names) else ''
            short = [n[len(prefix):] for n in names]
            m = match_roles(short, self.bp_map)
            missing = [r for r in REQUIRED_ROLES if r != 'Center' and (m[r] is None or m[r] not in short)]
            if missing:
                raise InvalidInputError(msg=f'Animal {animal}: could not map required roles {missing} onto body-parts {short}. Pass bp_map.', source=self.__class__.__name__)
            roles[animal] = {r: (f'{prefix}{b}' if b is not None and b in short else None) for r, b in m.items()}
            if self.verbose:
                stdout_information(msg=f'{animal}: ' + ', '.join(f'{r}={b}' for r, b in roles[animal].items() if b is not None))
        return roles

    def run(self):
        timer = SimbaTimer(start=True)
        names = [get_fn_ext(f)[1] for f in self.files]
        boxes, fallback = {}, {}
        pool = get_cpu_pool(core_cnt=min(self.core_cnt, len(names)), source=self.__class__.__name__)
        if self.calibration == 'box':
            videos = {n: self.find_video_of_file(video_dir=self.video_dir, filename=n) for n in names}
            if self.verbose:
                stdout_information(msg=f'Calibrating {len(names)} boxes on {self.core_cnt} cores...')
            res = pool.starmap(functools.partial(_calibrate_box_worker, floor_size_mm=self.box_floor_mm, file_type=self.file_type, roles=self.roles),
                               [(videos[n], f) for n, f in zip(names, self.files)])
            boxes = dict(zip(names, res))
            ok_m = [b['magnification'] for b in boxes.values() if 'error' not in b]
            if not ok_m:
                raise NoFilesFoundError(msg='Box calibration failed for every video.', source=self.__class__.__name__)
            mag_med = float(np.median(ok_m))
            for n, b in boxes.items():
                fallback[n] = 'error' in b or abs(b['magnification'] - mag_med) / mag_med > MAG_TOL
            self._mag_med = mag_med

        wall = self.wall_height_mm
        if self.calibration == 'box' and wall is None:
            ests = [boxes[n]['wall_mm_estimate'] for n in names if not fallback[n]]
            ests = [e for e in ests if np.isfinite(e)]
            if not ests:
                raise InvalidInputError(msg='Could not estimate the wall height from body reach; pass wall_height_mm.', source=self.__class__.__name__)
            wall = float(np.median(ests))
            if self.verbose:
                stdout_information(msg=f'Wall height from body reach: {wall:.0f} mm (median of {len(ests)} videos)')

        rows, writes = [], []
        reader = pool.imap(functools.partial(_read_pose_worker, file_type=self.file_type, roles=self.roles), self.files)   # streamed, in order
        for i, (n, animals) in enumerate(zip(names, reader)):
            v_timer = SimbaTimer(start=True)
            video_settings, px_per_mm, fps = self.read_video_info(video_name=n)          # raises if the video is not in video_info
            if self.calibration == 'camera_height':
                w, h = int(video_settings['Resolution_width'].values[0]), int(video_settings['Resolution_height'].values[0])
                H = self.camera_height_mm
                if H is None:                                                  # unknown: 300 mm for a lifter-size mouse, scaled
                    sv = np.nanmedian([body_scale(p_, c_, px_per_mm, tol=1.0) for p_, c_ in animals.values()])   # by apparent size (also
                    H = DEFAULT_CAMERA_MM * (sv if np.isfinite(sv) else 1.0)        # absorbs a wrong px/mm)
                calib = manual_calib(w, h, px_per_mm, H)
            else:
                b = boxes[n]
                if fallback[n]:
                    if 'error' in b:
                        raise InvalidInputError(msg=f'{n}: box calibration failed ({b["error"]})', source=self.__class__.__name__)
                    calib = box_calib(b, wall, self._mag_med, b['floor'].mean(0))
                else:
                    calib = box_calib(b, wall)
            out = {}
            for a, (px, conf) in animals.items():
                for msg in check_keypoint_layout(px, conf, calib['px_per_mm_floor']):
                    stdout_warning(msg=f'{n} {a}: {msg}')
                swapped = []
                if self.fix_left_right:
                    px, conf, swapped = fix_handedness(px, conf)
                    if swapped and self.verbose:
                        stdout_information(msg=f'{n} {a}: swapped left/right {" and ".join(swapped)} to match the lifter convention')
                s = body_scale(px, conf, calib['px_per_mm_floor']) if self.size_normalise else 1.0
                s = 1.0 if not np.isfinite(s) else s
                c_lift = calib if s == 1.0 else scaled_calib(calib, s)            # e.g. rats: lift at mouse size, scale back
                px_in = normalise_proportions(px, conf, c_lift['px_per_mm_floor'])[0] if (self.normalise and self.method != 'geometry') else None
                P = lift(self.method, px, conf, c_lift, fps, gpu=self.gpu, px_in=px_in, return_err=self.height_error)
                P, z_err = P if self.height_error else (P, None)
                if s != 1.0:
                    P = backproject(px, P[..., 2] * s, calib)
                    z_err = None if z_err is None else z_err * s
                ok = conf[:, [0, 1, 2, 5, 6]].min(1) >= 0.5
                for k, role in enumerate(ROLES):
                    if self.roles[a][role] is None and role != 'Center':      # a derived Center is written
                        continue
                    for j, ax in enumerate('xyz'):
                        out[f'{a}_{role}_{ax}'] = P[:, k, j]
                    out[f'{a}_{role}_p'] = conf[:, k]
                    if z_err is not None:
                        out[f'{a}_{role}_z_err'] = z_err[:, k]
                out[f'{a}_quality_ok'] = ok.astype(np.int8)
                rows.append({'video': n, 'animal': a, 'method': self.method, 'calibration': self.calibration, 'frames': len(P),
                             'quality_ok_pct': round(100 * ok.mean(), 1), 'swapped': '+'.join(swapped), 'size_scale': round(s, 3), 'center_derived': int(self.roles[a]['Center'] is None),
                             'head_z_median_mm': round(float(np.median(P[ok][:, :3, 2])), 1) if ok.any() else np.nan,
                             'head_z_err_median_mm': round(float(np.median(z_err[ok][:, :3])), 1) if (z_err is not None and ok.any()) else np.nan,
                             'centre_x_px': calib['centre'][0], 'centre_y_px': calib['centre'][1], 'focal_px': calib['focal_px'],
                             'camera_height_mm': calib['camera_height_mm'], 'camera_height_assumed': int(self.calibration == 'camera_height' and self.camera_height_mm is None), 'px_per_mm_floor': calib['px_per_mm_floor'], 'fps': fps,
                             'wall_height_mm': wall if self.calibration == 'box' else np.nan, 'calibration_fallback': int(fallback.get(n, False)),
                             'floor_px': json.dumps(np.round(calib['floor'], 1).tolist()) if calib.get('floor') is not None else ''})
            writes.append(pool.apply_async(_write_worker, (out, self.file_type, os.path.join(self.save_dir, f'{n}.{self.file_type}'))))
            v_timer.stop_timer()
            if self.verbose:
                stdout_information(msg=f'{i + 1}/{len(names)} {n} lifted ({v_timer.elapsed_time_str}s)')
        for w in writes:                                                       # writing overlaps the GPU lifting
            w.get()
        terminate_cpu_pool(pool=pool, force=False, source=self.__class__.__name__)
        summary = pd.DataFrame(rows)
        summary.to_csv(os.path.join(self.save_dir, '_summary.csv'), index=False)
        timer.stop_timer()
        stdout_success(msg=f'3D keypoints for {len(names)} video(s) ({self.method}) saved in {self.save_dir}', elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)
        return summary


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Overhead single-camera 2D -> 3D keypoint lifting for a SimBA project.')
    ap.add_argument('config_path')
    ap.add_argument('--method', default='lifter', choices=METHOD_OPTIONS)
    ap.add_argument('--calibration', default='camera_height', choices=CALIBRATION_OPTIONS)
    ap.add_argument('--camera_height_mm', type=float, default=None)
    ap.add_argument('--wall_height_mm', type=float, default=None)
    ap.add_argument('--box_floor_mm', type=float, default=156.0)
    ap.add_argument('--save_dir', default=None)
    ap.add_argument('--bp_map', default=None, help='JSON, e.g. \'{"Nose": "RIGHT_EAR"}\'')
    ap.add_argument('--no_gpu', action='store_true')
    ap.add_argument('--no_height_error', action='store_true', help='do not write the <animal>_<role>_z_err columns')
    a = ap.parse_args()
    Overhead3DLifter(config_path=a.config_path, save_dir=a.save_dir, method=a.method, calibration=a.calibration,
                     camera_height_mm=a.camera_height_mm, wall_height_mm=a.wall_height_mm, box_floor_mm=a.box_floor_mm,
                     bp_map=json.loads(a.bp_map) if a.bp_map else None, height_error=not a.no_height_error, gpu=not a.no_gpu).run()
