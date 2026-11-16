import functools
import json
import multiprocessing
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.sandbox.overhead_3d.overhead_3d_utils import (EDGES, ROLES,
                                                         backproject,
                                                         fit_arena_rect,
                                                         median_background,
                                                         project)
from simba.utils.checks import (check_float, check_if_dir_exists, check_int,
                                check_str, check_valid_boolean)
from simba.utils.data import get_cpu_pool, terminate_cpu_pool
from simba.utils.enums import Formats
from simba.utils.errors import NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data, read_df)

ROLE_BGR = [(60, 130, 230), (40, 90, 220), (40, 90, 220), (150, 160, 40), (150, 160, 40), (150, 160, 40), (120, 120, 120), (120, 120, 120), (120, 120, 120)]
ANIMAL_BGR = [(40, 40, 40), (60, 60, 210), (200, 120, 40), (40, 160, 60), (160, 60, 160)]
ELEV, DIST = np.radians(55), 560.0


def _view():
    """Virtual camera on the image-bottom side looking down at ELEV. The world is LEFT-handed (x right, y image-down,
    z up), so right = up x fwd and up = fwd x right (the right-handed forms mirror the view)."""
    d = np.array([0.0, np.cos(ELEV), np.sin(ELEV)])
    fwd = -d
    right = np.cross([0, 0, 1.0], fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    return np.stack([right, -up, fwd]), d * DIST


R_VIEW, CAM_OFF = _view()


def _to_screen(P, target, focal, panel):
    Q = (P - (target + CAM_OFF)) @ R_VIEW.T
    return np.stack([panel / 2 + focal * Q[..., 0] / Q[..., 2], panel * 0.55 + focal * Q[..., 1] / Q[..., 2]], -1)


def _line(img, a, b, clr, t=1):
    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), clr, t, cv2.LINE_AA)


def _render_chunk(frame_range: np.ndarray, group: int, video_path: str, P: dict, conf: dict, calib: dict, scene: dict, panel: int, save_dir: str):
    """Render frames [frame_range] of one video: original + 2D (re-projected) | 3D in the arena. Runs in a worker process."""
    meta = get_video_meta_data(video_path=video_path)
    writer = cv2.VideoWriter(os.path.join(save_dir, f'{group}.mp4'), cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value), meta['fps'], (2 * panel, panel))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_range[0]))
    sc = panel / max(meta['width'], meta['height'])
    floor, wall, target, focal = scene['floor_mm'], scene['wall_mm'], scene['target'], scene['focal']
    fl, rim = np.c_[floor, np.zeros(4)], np.c_[scene['rim_mm'], np.full(4, wall)]
    sf, sr = _to_screen(fl, target, focal, panel), _to_screen(rim, target, focal, panel)
    base = np.full((panel, panel, 3), 250, np.uint8)
    cv2.fillPoly(base, [np.int32(sf)], (238, 240, 236), cv2.LINE_AA)
    lo, hi = floor.min(0), floor.max(0)
    for x in np.arange(np.ceil(lo[0] / 20) * 20, hi[0], 20):
        s = _to_screen(np.array([[x, lo[1], 0], [x, hi[1], 0]]), target, focal, panel); _line(base, s[0], s[1], (222, 224, 220))
    for y in np.arange(np.ceil(lo[1] / 20) * 20, hi[1], 20):
        s = _to_screen(np.array([[lo[0], y, 0], [hi[0], y, 0]]), target, focal, panel); _line(base, s[0], s[1], (222, 224, 220))
    for i in range(4):
        j = (i + 1) % 4
        _line(base, sf[i], sf[j], (170, 175, 170)); _line(base, sr[i], sr[j], (170, 175, 170)); _line(base, sf[i], sr[i], (170, 175, 170))
    animals = list(P.keys())
    for t in frame_range:
        ok, frm = cap.read()
        if not ok:
            break
        left = np.full((panel, panel, 3), 255, np.uint8)
        h, w = int(frm.shape[0] * sc), int(frm.shape[1] * sc)
        y0, x0 = (panel - h) // 2, (panel - w) // 2
        left[y0:y0 + h, x0:x0 + w] = cv2.resize(frm, (w, h))
        right = base.copy()
        txt = []
        for ai, a in enumerate(animals):
            Pa, ca = P[a][t], conf[a][t]
            p2 = project(Pa, calib) * sc + [x0, y0]
            S = _to_screen(Pa, target, focal, panel)
            sh = Pa.copy(); sh[:, 2] = 0
            SS = _to_screen(sh, target, focal, panel)
            clr = ANIMAL_BGR[ai % len(ANIMAL_BGR)]
            for i, j in EDGES:
                if np.isfinite(Pa[[i, j]]).all():
                    _line(right, SS[i], SS[j], (205, 205, 205), 2)
                    _line(right, S[i], S[j], clr, 2)
                    if min(ca[i], ca[j]) >= 0.3:
                        _line(left, p2[i], p2[j], clr, 1)
            for k in (0, 1, 2, 5):
                if np.isfinite(Pa[k]).all():
                    _line(right, S[k], SS[k], (190, 190, 190))
            for k in range(len(ROLES)):
                if np.isfinite(Pa[k]).all():
                    cv2.circle(right, (int(S[k, 0]), int(S[k, 1])), 5, ROLE_BGR[k], -1, cv2.LINE_AA)
                    if ca[k] >= 0.3:
                        cv2.circle(left, (int(p2[k, 0]), int(p2[k, 1])), 3, ROLE_BGR[k], -1, cv2.LINE_AA)
            txt.append(f'{a} head {np.nanmean(Pa[:3, 2]):4.1f} mm')
        cv2.putText(left, f'{t / meta["fps"]:7.2f} s', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 160), 1, cv2.LINE_AA)
        for i, s in enumerate(txt if scene.get('annotate', True) else []):
            cv2.putText(right, s, (10, panel - 14 - 20 * (len(txt) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANIMAL_BGR[i % len(ANIMAL_BGR)], 1, cv2.LINE_AA)
        writer.write(np.hstack([left, right]))
    cap.release()
    writer.release()
    return group


# ------------------------------------------------------------------------------------------------------ POV mode
# Chosen after 4 design rounds (scratchpad pov_lab*: true eye-level views crop or blow up the partner because mice interact
# at 0-5 cm, and the 15 mm nose-ear vector jitters up to 40 deg/frame). Camera 55 mm behind and 30 mm above the head (it
# rises with the head when rearing), aimed 130 mm ahead, 90 deg view, heading = nose - centre (2-3x steadier), vector-smoothed
# 0.25 s: partner fully on screen ~70% of frames (pov_lab round 5; 95 mm up gave 80% but looked down from too high). The
# viewer itself is not drawn (its own body would fill the foreground).
POV_FOV_DEG, POV_NEAR = 90.0, 5.0
POV_BACK_MM, POV_UP_MM, POV_AHEAD_MM, POV_LOOK_H_MM, POV_SMOOTH_S = 20.0, 12.0, 150.0, 18.0, 0.25   # up = above the head
SKY_TOP, SKY_BOT = np.array([246, 238, 228]), np.array([226, 220, 214])
FLOOR_BGR, WALL_BGR, GRID_BGR = (176, 204, 222), (205, 196, 186), (160, 188, 206)
BODY_BGR = [(70, 70, 70), (90, 90, 225), (200, 130, 60), (60, 160, 70), (160, 70, 160)]


def _fill_nan(a: np.ndarray) -> np.ndarray:
    a, t = a.copy(), np.arange(len(a))
    for k in range(a.shape[1]):
        ok = np.isfinite(a[:, k])
        a[:, k] = np.interp(t, t[ok], a[ok, k]) if ok.any() else 0.0
    return a


def pov_frames(P: np.ndarray, fps: float, at: str = 'head'):
    """Per-frame camera (eye (T, 3), rotation (T, 3, 3)), horizontal heading (T, 2) and head position (T, 3) of one animal.
    World is LEFT-handed (x right, y image-down, z up): screen right = up x fwd, screen up = fwd x right."""
    from scipy.ndimage import gaussian_filter1d
    em = _fill_nan(np.nanmean(P[:, [1, 2]], 1))
    v = (_fill_nan(P[:, 0]) - _fill_nan(P[:, 5]))[:, :2]
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    v = gaussian_filter1d(v, max(1.0, POV_SMOOTH_S * fps), axis=0, mode='nearest')    # vector smoothing = circular mean
    h = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    head_z = gaussian_filter1d(em[:, 2], max(1.0, 0.2 * fps), mode='nearest')            # camera rides with the head (rears)
    if at == 'nose':                                                           # camera on the snout tip, 5 mm up
        nose = gaussian_filter1d(_fill_nan(P[:, 0]), max(1.0, 0.1 * fps), axis=0, mode='nearest')
        eye = np.c_[nose[:, :2], nose[:, 2] + 5.0]
    else:
        eye = np.c_[em[:, :2] - POV_BACK_MM * h, head_z + POV_UP_MM]
    look = np.c_[em[:, :2] + POV_AHEAD_MM * h, np.full(len(h), POV_LOOK_H_MM)]
    fwd = (look - eye) / np.linalg.norm(look - eye, axis=1, keepdims=True)
    r = np.cross(np.array([0, 0, 1.0]), fwd)
    r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-9
    return eye, np.stack([r, -np.cross(fwd, r), fwd], 1), h, em


def _cam(X, eye, R):
    return (np.asarray(X, float) - eye) @ R.T


def make_lens(fov_deg: float, panel: int) -> dict:
    """Projection for the POV panels. fov <= 120: pinhole, square panel. Wider (up to 180): cylindrical panorama
    (equirectangular: azimuth -> x, elevation -> y, same angular scale), panel 2:1 wide; straight-ahead is the centre,
    the left/right edges are +-fov/2 to the side. A pinhole camera cannot show 180 deg (the image would be infinitely wide)."""
    if fov_deg <= 120:
        return {'mode': 'pinhole', 'W': panel, 'H': panel, 'F': (panel / 2) / np.tan(np.radians(fov_deg / 2)), 'fov': fov_deg}
    W = 2 * panel
    return {'mode': 'pano', 'W': W, 'H': panel, 'k': W / np.radians(fov_deg), 'fov': fov_deg}


def _proj(Q, lens):
    Q = np.asarray(Q, float)
    if lens['mode'] == 'pinhole':
        return np.stack([lens['W'] / 2 + lens['F'] * Q[..., 0] / Q[..., 2], lens['H'] / 2 + lens['F'] * Q[..., 1] / Q[..., 2]], -1)
    az = np.arctan2(Q[..., 0], Q[..., 2])
    el = np.arctan2(Q[..., 1], np.hypot(Q[..., 0], Q[..., 2]))
    return np.stack([lens['W'] / 2 + lens['k'] * az, lens['H'] / 2 + lens['k'] * el], -1)


def _scale(depth, lens):
    """Pixels per mm at a given distance (for line widths / joint sizes)."""
    return (lens['F'] if lens['mode'] == 'pinhole' else lens['k']) / depth


def _densify(Q, lens, n=24):
    """Panorama: straight 3D edges are curves on screen, so sample them."""
    if lens['mode'] == 'pinhole' or len(Q) < 2:
        return Q
    out = []
    for i in range(len(Q)):
        a, b = Q[i], Q[(i + 1) % len(Q)]
        out.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
    return np.vstack(out)


def _clip_poly(Q):
    out, n = [], len(Q)
    for i in range(n):
        a, b = Q[i], Q[(i + 1) % n]
        ia, ib = a[2] >= POV_NEAR, b[2] >= POV_NEAR
        if ia:
            out.append(a)
        if ia != ib:
            out.append(a + (POV_NEAR - a[2]) / (b[2] - a[2]) * (b - a))
    return np.array(out)


def _fill(img, quad3d, eye, R, lens, clr):
    Q = _clip_poly(_cam(quad3d, eye, R))
    if len(Q) >= 3:
        s = _proj(_densify(Q, lens), lens)
        if np.isfinite(s).all() and np.abs(s).max() < 1e5:
            cv2.fillPoly(img, [np.int32(s)], clr, cv2.LINE_AA)


def _seg(img, a, b, eye, R, lens, clr, t=1):
    A, B = _cam([a, b], eye, R)
    if A[2] < POV_NEAR and B[2] < POV_NEAR:
        return
    if A[2] < POV_NEAR:
        A = A + (POV_NEAR - A[2]) / (B[2] - A[2]) * (B - A)
    if B[2] < POV_NEAR:
        B = B + (POV_NEAR - B[2]) / (A[2] - B[2]) * (A - B)
    pts = np.stack([A, B]) if lens['mode'] == 'pinhole' else A + np.linspace(0, 1, 16)[:, None] * (B - A)
    s = _proj(pts, lens)
    if np.isfinite(s).all() and np.abs(s).max() < 1e5:
        cv2.polylines(img, [np.int32(s)], False, clr, t, cv2.LINE_AA)


def _room(eye, R, lens, scene):
    g = np.linspace(0, 1, lens['H'])[:, None, None]
    img = np.broadcast_to(SKY_TOP * (1 - g) + SKY_BOT * g, (lens['H'], lens['W'], 3)).astype(np.uint8).copy()
    fl3, rim3 = np.c_[scene['floor_mm'], np.zeros(4)], np.c_[scene['rim_mm'], np.full(4, scene['wall_mm'])]
    _fill(img, fl3, eye, R, lens, FLOOR_BGR)
    lo, hi = scene['floor_mm'].min(0), scene['floor_mm'].max(0)
    for x in np.arange(np.ceil(lo[0] / 25) * 25, hi[0], 25):
        _seg(img, (x, lo[1], 0), (x, hi[1], 0), eye, R, lens, GRID_BGR)
    for y in np.arange(np.ceil(lo[1] / 25) * 25, hi[1], 25):
        _seg(img, (lo[0], y, 0), (hi[0], y, 0), eye, R, lens, GRID_BGR)
    ov = img.copy()
    walls = [np.array([fl3[i], fl3[(i + 1) % 4], rim3[(i + 1) % 4], rim3[i]]) for i in range(4)]
    for w in sorted(walls, key=lambda w: -np.mean(_cam(w, eye, R)[:, 2])):
        _fill(ov, w, eye, R, lens, WALL_BGR)
    img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
    for i in range(4):
        _seg(img, fl3[i], fl3[(i + 1) % 4], eye, R, lens, (120, 125, 130), 2)
        _seg(img, rim3[i], rim3[(i + 1) % 4], eye, R, lens, (150, 150, 155), 2)
        _seg(img, fl3[i], rim3[i], eye, R, lens, (150, 150, 155), 1)
    return img


def _body(img, Pb, eye, R, lens, clr, alpha=0.40):
    """Semi-transparent body: the solid spanned by the head/trunk keypoints (nose, ears, flanks, centre, tail base),
    which are tracked ON the animal's edge, and their projections on the floor (the keypoints are dorsal, the body
    reaches the floor). Not widened beyond the keypoints. Filled per triangle of the outline so concave poses stay honest."""
    idx = [k for k in (0, 1, 2, 3, 4, 5, 6) if np.isfinite(Pb[k]).all()]
    if len(idx) < 3:
        return img
    top = Pb[idx]
    pts = np.vstack([top, top * [1, 1, 0]])
    Q = _cam(pts, eye, R)
    Q = Q[Q[:, 2] > POV_NEAR]
    if len(Q) < 3:
        return img
    sp = _proj(_densify(Q, lens, 1) if lens['mode'] == 'pinhole' else Q, lens)
    if not np.isfinite(sp).all() or np.abs(sp).max() > 1e5:
        return img
    ov = img.copy()
    cv2.fillConvexPoly(ov, cv2.convexHull(np.int32(sp)), clr, cv2.LINE_AA)
    return cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)


def _skeleton(img, Pb, eye, R, lens, clr):
    """Partner skeleton. Every edge is drawn and clipped at the near plane / view edge (dropping edges with an endpoint
    behind the camera left joints floating when the animals touch); joints only where in front of the camera."""
    sh = Pb.copy()
    sh[:, 2] = 0.5
    for i, j in EDGES:                                                          # floor shadow
        if np.isfinite(sh[[i, j]]).all():
            _seg(img, sh[i], sh[j], eye, R, lens, (140, 160, 175), 3)
    Q = _cam(Pb, eye, R)
    fin = np.isfinite(Q).all(1)
    dist = np.linalg.norm(Q, axis=1)
    for i, j in EDGES:
        if fin[i] and fin[j]:
            th = int(np.clip(_scale(max(np.mean(dist[[i, j]]), 10.0), lens) * 3.0, 2, 14))
            _seg(img, Pb[i], Pb[j], eye, R, lens, (250, 250, 250), th + 3)
            _seg(img, Pb[i], Pb[j], eye, R, lens, clr, th)
    s = _proj(Q, lens)
    for k in np.argsort(-np.where(fin, dist, -1)):
        if fin[k] and Q[k, 2] > POV_NEAR and np.abs(s[k]).max() < 1e4:
            r = int(np.clip(_scale(max(dist[k], 10.0), lens) * 3.5, 3, 18))
            cv2.circle(img, tuple(np.int32(s[k])), r + 2, (250, 250, 250), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(np.int32(s[k])), r, ROLE_BGR[k], -1, cv2.LINE_AA)


def _render_chunk_pov(frame_range: np.ndarray, group: int, video_path: str, P: dict, conf: dict, calib: dict, scene: dict, panel: int,
                      save_dir: str, cams: dict, fov_deg: float = POV_FOV_DEG, body: bool = True):
    """Original + 2D | one POV panel per animal (see the POV mode notes above)."""
    meta = get_video_meta_data(video_path=video_path)
    animals = list(P.keys())
    lens = make_lens(fov_deg, panel)
    writer = cv2.VideoWriter(os.path.join(save_dir, f'{group}.mp4'), cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value), meta['fps'], (panel + len(animals) * lens['W'], panel))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_range[0]))
    sc = panel / max(meta['width'], meta['height'])
    for t in frame_range:
        ok, frm = cap.read()
        if not ok:
            break
        left = np.full((panel, panel, 3), 255, np.uint8)
        h, w = int(frm.shape[0] * sc), int(frm.shape[1] * sc)
        y0, x0 = (panel - h) // 2, (panel - w) // 2
        left[y0:y0 + h, x0:x0 + w] = cv2.resize(frm, (w, h))
        for ai, a in enumerate(animals):
            p2 = project(P[a][t], calib) * sc + [x0, y0]
            for i, j in EDGES:
                if np.isfinite(p2[[i, j]]).all() and min(conf[a][t, i], conf[a][t, j]) >= 0.3:
                    _line(left, p2[i], p2[j], ANIMAL_BGR[ai % len(ANIMAL_BGR)], 1)
        cv2.putText(left, f'{t / meta["fps"]:7.2f} s', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 160), 1, cv2.LINE_AA)
        tiles = [left]
        Pt = {b: P[b][t] for b in animals}
        for ai, a in enumerate(animals):
            eye, R = cams[a][0][t], cams[a][1][t]
            img = _room(eye, R, lens, scene)
            others = sorted([b for b in animals if b != a], key=lambda b: -np.nanmean(_cam(Pt[b], eye, R)[:, 2]))
            for b in others:
                if body:
                    img = _body(img, Pt[b], eye, R, lens, BODY_BGR[animals.index(b) % len(BODY_BGR)])
                _skeleton(img, Pt[b], eye, R, lens, ANIMAL_BGR[animals.index(b) % len(ANIMAL_BGR)])
            if lens['mode'] == 'pano':                                          # direction ticks: left / ahead / right
                for deg, lab in ((-90, 'left'), (-45, ''), (0, 'ahead'), (45, ''), (90, 'right')):
                    if abs(deg) <= fov_deg / 2:
                        x = int(np.clip(lens['W'] / 2 + lens['k'] * np.radians(deg), 2, lens['W'] - 3))
                        cv2.line(img, (x, lens['H'] - 30), (x, lens['H'] - 22), (110, 110, 110), 1, cv2.LINE_AA)
                        if lab:
                            cv2.putText(img, lab, (int(np.clip(x - 15, 6, lens['W'] - 48)), lens['H'] - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)
                cv2.line(img, (lens['W'] - 1, 0), (lens['W'] - 1, lens['H']), (200, 200, 200), 1)              # panel separator
            dist = np.nanmin([np.linalg.norm(Pt[b][0, :2] - Pt[a][0, :2]) for b in others]) if others else np.nan
            cv2.putText(img, f"{a}'s view", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ANIMAL_BGR[ai % len(ANIMAL_BGR)], 2, cv2.LINE_AA)
            cv2.putText(img, f'nose-nose {dist:4.0f} mm', (10, lens['H'] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
            tiles.append(img)
        writer.write(np.hstack(tiles))
    cap.release()
    writer.release()
    return group


class Overhead3DVisualizer(ConfigReader):
    """
    Side-by-side videos of :class:`overhead_3d_lifter.Overhead3DLifter` output: original video with the 2D keypoints
    (re-projected from the 3D, so they show exactly what was lifted) | the 3D keypoints of every animal in the arena
    from a fixed oblique virtual camera on the image-bottom side (same orientation as the video). Frames are rendered
    in parallel chunks and concatenated.

    :param Union[str, os.PathLike] config_path: Path to the SimBA project_config.ini.
    :param Optional[Union[str, os.PathLike]] data_dir: Directory with the lifter output (and its _summary.csv). Default: project ``csv/pose_3d``.
    :param Optional[Union[str, os.PathLike]] save_dir: Output directory. Default: project ``frames/output/pose_3d``.
    :param Optional[List[str]] video_names: Subset of videos to render. Default: all.
    :param int panel: Panel size (px); the output is 2 * panel wide. Default 500.
    :param Optional[float] sample_time: Render only the first N seconds. Default None (whole video).
    :param int core_cnt: Rendering processes. -1 = min(all cores, 8); every worker decodes the video, so more mostly costs RAM. Default -1.
    :param bool gpu: Use the GPU for concatenating the video chunks (if available). Default False.
    :param Optional[Tuple[float, float]] arena_mm: Real floor size (mm), e.g. (300, 174) for an Innovive mouse cage. Drawn centred on where the animals went (0.5-99.5th percentile), long side along the longer extent. None: the animals' extent + 15 mm.
    :param Optional[Tuple[float, float]] arena_top_mm: Interior size at the rim (mm) for tapered cages, e.g. (355, 215) for the Innovive cage; walls slope from arena_mm to this. None: vertical walls.
    :param Optional[float] wall_height_mm: Real wall height (mm) for drawing, e.g. 127. None: 100 mm.
    :param bool annotate: Overhead view: print per-animal head heights in the 3D panel. Default True.
    :param bool pov_body: Draw a semi-transparent body under each partner skeleton: the solid spanned by its head/trunk keypoints (tracked on the animal's edge) down to the floor, not widened beyond them. Default True.
    :param str pov_camera: ``head`` (default): 20 mm behind / 12 mm above the ear midpoint. ``nose``: on the snout tip (5 mm up); the partner is ~13 mm closer during fights but fully in view far less often (27% vs 83% of attack frames at 180 deg) because fighting mice are mostly beside the head.
    :param float pov_fov_deg: POV field of view. <= 120: normal (pinhole) square panels; up to 180: cylindrical panorama, 2:1 wide panels (a pinhole camera cannot show 180 deg). Default 90.
    :param str view: ``overhead`` (default): one oblique 3D panel with all animals. ``pov``: one panel per animal from just behind and above its head (20 mm back, 12 mm above the head and rising with it, aimed where the head points), the partner(s) as keypoint skeletons with floor shadows; the viewer itself is not drawn.

    :example:
    >>> Overhead3DVisualizer(config_path=r'D:/maplight_ri/project_folder/project_config.ini', video_names=['Trial_10_C24_D1_3']).run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 video_names: Optional[List[str]] = None,
                 panel: int = 500,
                 sample_time: Optional[float] = None,
                 core_cnt: int = -1,
                 gpu: bool = False,
                 view: str = 'overhead',
                 pov_fov_deg: float = 90.0,
                 pov_camera: str = 'head',
                 pov_body: bool = True,
                 annotate: bool = True,
                 arena_mm: Optional[Tuple[float, float]] = None,
                 arena_top_mm: Optional[Tuple[float, float]] = None,
                 wall_height_mm: Optional[float] = None,
                 verbose: bool = True):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        self.data_dir = os.path.join(self.project_path, 'csv', 'pose_3d') if data_dir is None else data_dir
        check_if_dir_exists(in_dir=self.data_dir)
        self.summary_path = os.path.join(self.data_dir, '_summary.csv')
        if not os.path.isfile(self.summary_path):
            raise NoFilesFoundError(msg=f'{self.summary_path} not found: run Overhead3DLifter first.', source=self.__class__.__name__)
        check_int(name=f'{self.__class__.__name__} panel', value=panel, min_value=200)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        check_valid_boolean(value=gpu, source=f'{self.__class__.__name__} gpu', raise_error=True)
        check_str(name=f'{self.__class__.__name__} view', value=view, options=['overhead', 'pov'])
        self.view, self.arena_mm, self.wall_height_mm, self.arena_top_mm = view, arena_mm, wall_height_mm, arena_top_mm
        check_float(name=f'{self.__class__.__name__} pov_fov_deg', value=pov_fov_deg, min_value=30.0, max_value=180.0)
        self.pov_fov_deg = float(pov_fov_deg)
        check_str(name=f'{self.__class__.__name__} pov_camera', value=pov_camera, options=['head', 'nose'])
        self.pov_camera = pov_camera
        check_valid_boolean(value=pov_body, source=f'{self.__class__.__name__} pov_body', raise_error=True)
        self.pov_body = pov_body
        check_valid_boolean(value=annotate, source=f'{self.__class__.__name__} annotate', raise_error=True)
        self.annotate = annotate
        self.save_dir = os.path.join(self.project_path, 'frames', 'output', 'pose_3d') if save_dir is None else save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.summary = pd.read_csv(self.summary_path)
        files = find_files_of_filetypes_in_directory(directory=self.data_dir, extensions=[f'.{self.file_type}'], raise_error=True)
        files = [f for f in files if not get_fn_ext(f)[1].startswith('_')]
        if video_names is not None:
            files = [f for f in files if get_fn_ext(f)[1] in video_names]
        if not files:
            raise NoFilesFoundError(msg=f'No lifted files for {video_names} in {self.data_dir}', source=self.__class__.__name__)
        self.files, self.panel, self.sample_time, self.gpu, self.verbose = files, panel, sample_time, gpu, verbose
        self.core_cnt = min(find_core_cnt()[0], 8) if core_cnt == -1 else core_cnt   # each worker decodes the video: RAM

    def _fit_floor(self, P: dict, calib: dict, video_path: Optional[str]) -> Optional[np.ndarray]:
        """Find the arena floor (known aspect, arena_mm) in the video's median background; accept it only if >= 90% of the
        animals' confident body centres lie inside, otherwise None (caller falls back)."""
        if video_path is None:
            return None
        ppm = calib['focal_px'] / calib['camera_height_mm']
        L, W = max(self.arena_mm), min(self.arena_mm)
        bg = median_background(video_path, 40)
        size = (L * ppm, W * ppm) if bg.shape[1] >= bg.shape[0] else (W * ppm, L * ppm)
        rect, s = fit_arena_rect(bg, size)
        c = np.concatenate([project(p[:, 5], calib) for p in P.values()])
        c = c[np.isfinite(c).all(1)]
        inside = np.mean((c[:, 0] >= rect[0, 0]) & (c[:, 0] <= rect[1, 0]) & (c[:, 1] >= rect[0, 1]) & (c[:, 1] <= rect[2, 1])) if len(c) else 0
        if self.verbose:
            stdout_information(msg=f'arena fit: scale {s:.2f} x arena_mm, {100 * inside:.0f}% of body centres inside' + ('' if inside >= 0.9 else ' -> rejected, placing by animal range'))
        return backproject(rect, np.zeros(4), calib)[:, :2] if inside >= 0.9 else None

    def _scene(self, P: dict, row: pd.Series, video_path: Optional[str] = None) -> dict:
        allxy = np.concatenate([p[..., :2].reshape(-1, 2) for p in P.values()])
        allxy = allxy[np.isfinite(allxy).all(1)]
        calib = {'centre': np.array([row['centre_x_px'], row['centre_y_px']]), 'focal_px': row['focal_px'], 'camera_height_mm': row['camera_height_mm']}
        if isinstance(row.get('floor_px'), str) and row['floor_px'].startswith('['):
            fpx = np.array(json.loads(row['floor_px']))
            floor = (fpx - calib['centre']) * (calib['camera_height_mm'] / calib['focal_px'])
            wall = float(row['wall_height_mm'])
        elif self.arena_mm is not None and (fitted := self._fit_floor(P, calib, video_path)) is not None:
            floor = fitted                                                     # arena located in the video itself
            wall = 100.0 if self.wall_height_mm is None else float(self.wall_height_mm)
        elif self.arena_mm is not None:                                        # fallback: known size, centred where the animals went
            lo, hi = np.percentile(allxy, 0.5, 0), np.percentile(allxy, 99.5, 0)
            c = (lo + hi) / 2
            L, W = max(self.arena_mm), min(self.arena_mm)
            half = np.array([L, W]) / 2 if (hi - lo)[0] >= (hi - lo)[1] else np.array([W, L]) / 2
            floor = np.array([c - half, [c[0] + half[0], c[1] - half[1]], c + half, [c[0] - half[0], c[1] + half[1]]])
            wall = 100.0 if self.wall_height_mm is None else float(self.wall_height_mm)
        else:                                                                  # no box: arena = where the animals went
            lo, hi = np.percentile(allxy, 1, 0) - 15, np.percentile(allxy, 99, 0) + 15
            floor = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
            wall = 100.0 if self.wall_height_mm is None else float(self.wall_height_mm)
        span = np.ptp(floor, 0).max()
        title = f'3D ({row["method"]}, camera {row["camera_height_mm"]:.0f} mm)'
        rim = floor
        if self.arena_top_mm is not None and self.arena_mm is not None:            # tapered cage: rim opening larger than the floor
            fc, fh = floor.mean(0), (floor.max(0) - floor.min(0)) / 2
            ratio = np.array([max(self.arena_top_mm) / max(self.arena_mm), min(self.arena_top_mm) / min(self.arena_mm)])
            th = fh * (ratio if fh[0] >= fh[1] else ratio[::-1])           # taper relative to the floor actually found
            rim = np.array([fc - th, [fc[0] + th[0], fc[1] - th[1]], fc + th, [fc[0] - th[0], fc[1] + th[1]]])
        return {'floor_mm': floor, 'rim_mm': rim, 'wall_mm': wall, 'target': np.r_[floor.mean(0), 35.0], 'focal': 900.0 * 165.0 / max(span, 120.0) * self.panel / 400, 'title': title}, calib

    def run(self):
        timer = SimbaTimer(start=True)
        pool = get_cpu_pool(core_cnt=self.core_cnt, source=self.__class__.__name__)
        for file in self.files:
            v_timer = SimbaTimer(start=True)
            name = get_fn_ext(file)[1]
            video_path = self.find_video_of_file(video_dir=self.video_dir, filename=name)
            meta = get_video_meta_data(video_path=video_path)
            df = read_df(file_path=file, file_type=self.file_type)
            rows = self.summary[self.summary['video'] == name]
            P, conf = {}, {}
            for a in rows['animal']:
                arr = np.full((len(df), len(ROLES), 3), np.nan)
                c = np.zeros((len(df), len(ROLES)))
                for k, role in enumerate(ROLES):
                    if f'{a}_{role}_x' in df.columns:
                        arr[:, k] = df[[f'{a}_{role}_x', f'{a}_{role}_y', f'{a}_{role}_z']].values
                        c[:, k] = df[f'{a}_{role}_p'].values
                P[a], conf[a] = arr, c
            scene, calib = self._scene(P, rows.iloc[0], video_path)
            scene['annotate'] = self.annotate
            n = min(len(df), meta['frame_count'])
            if self.sample_time is not None:
                n = min(n, int(self.sample_time * meta['fps']))
            chunks = np.array_split(np.arange(n), min(self.core_cnt, max(1, n // 100)))
            temp = os.path.join(self.save_dir, name, 'temp')
            os.makedirs(temp, exist_ok=True)
            if self.view == 'pov':
                cams = {a: pov_frames(P[a], meta['fps'], at=self.pov_camera) for a in P}
                fn = functools.partial(_render_chunk_pov, video_path=video_path, P=P, conf=conf, calib=calib, scene=scene, panel=self.panel, save_dir=temp, cams=cams, fov_deg=self.pov_fov_deg, body=self.pov_body)
            else:
                fn = functools.partial(_render_chunk, video_path=video_path, P=P, conf=conf, calib=calib, scene=scene, panel=self.panel, save_dir=temp)
            job = pool.starmap_async(fn, [(c, i) for i, c in enumerate(chunks)])
            try:                                                               # a worker that dies (e.g. out of memory) would
                job.get(timeout=120 + 0.5 * n / len(chunks))                   # otherwise hang the pool forever
            except multiprocessing.TimeoutError:
                terminate_cpu_pool(pool=pool, force=True, source=self.__class__.__name__)
                raise RuntimeError(f'{name}: rendering timed out (a worker process probably ran out of memory); retry with fewer cores (core_cnt).')
            save_path = os.path.join(self.save_dir, f'{name}{"_pov" if self.view == "pov" else ""}.mp4')
            concatenate_videos_in_folder(in_folder=temp, save_path=save_path, remove_splits=True, gpu=self.gpu, verbose=False)
            os.rmdir(os.path.dirname(temp)) if os.path.isdir(os.path.dirname(temp)) and not os.listdir(os.path.dirname(temp)) else None
            v_timer.stop_timer()
            if self.verbose:
                stdout_information(msg=f'{name}: {n} frames rendered on {len(chunks)} cores ({v_timer.elapsed_time_str}s)')
        terminate_cpu_pool(pool=pool, force=False, source=self.__class__.__name__)
        timer.stop_timer()
        stdout_success(msg=f'{len(self.files)} 2D-vs-3D video(s) saved in {self.save_dir}', elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)


def combine_overhead_pov(overhead_path: Union[str, os.PathLike], pov_path: Union[str, os.PathLike], save_path: Union[str, os.PathLike],
                         panel: int = 500, video_aspect: Optional[float] = None) -> str:
    """
    2 x 2 video from an ``overhead`` and a ``pov`` render of the same video (same panel size):
    top-left = original video + 2D keypoints (letterbox cropped, enlarged), top-right = overhead 3D (cropped to the arena, enlarged),
    bottom = the first two animals' POV panels. Cells are (2 * panel) x panel; the time stamp is redrawn top-left.

    :param video_aspect: width / height of the original video (to crop the letterbox). None: detected from the frame.
    """
    co, cp = cv2.VideoCapture(str(overhead_path)), cv2.VideoCapture(str(pov_path))
    fps = co.get(cv2.CAP_PROP_FPS)
    n = int(min(co.get(cv2.CAP_PROP_FRAME_COUNT), cp.get(cv2.CAP_PROP_FRAME_COUNT)))
    W, H = 2 * panel, panel
    tmp = str(save_path) + '.raw.mp4'
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value), fps, (2 * W, 2 * H))
    rows, box = None, None
    for t in range(n):
        ok1, fo = co.read()
        ok2, fp = cp.read()
        if not (ok1 and ok2):
            break
        orig, td = fo[:, :panel], fo[:, panel:2 * panel]
        if rows is None:                                                       # letterbox rows of the original tile
            if video_aspect is not None:
                hh = int(round(panel / video_aspect)); y0 = (panel - hh) // 2; rows = (y0, y0 + hh)
            else:
                nz = np.flatnonzero((orig < 245).any(axis=(1, 2)) & (np.arange(panel) > 30))
                rows = (int(nz.min()), int(nz.max()) + 1) if len(nz) else (0, panel)
        vid = orig[rows[0]:rows[1]]
        sc = min(W / vid.shape[1], H / vid.shape[0])
        vid = cv2.resize(vid, (int(vid.shape[1] * sc), int(vid.shape[0] * sc)))
        tl = np.full((H, W, 3), 255, np.uint8)
        y, x = (H - vid.shape[0]) // 2, (W - vid.shape[1]) // 2
        tl[y:y + vid.shape[0], x:x + vid.shape[1]] = vid
        cv2.putText(tl, f'{t / fps:7.2f} s', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 160), 2, cv2.LINE_AA)
        if box is None:                                                        # crop the 3D tile to the arena drawing
            ys, xs = np.nonzero((np.abs(td.astype(int) - 250) > 6).any(2))
            m = 12
            box = (max(ys.min() - m, 0), min(ys.max() + m, panel), max(xs.min() - m, 0), min(xs.max() + m, panel)) if len(ys) else (0, panel, 0, panel)
        crop = td[box[0]:box[1], box[2]:box[3]]
        sc3 = min(W / crop.shape[1], H / crop.shape[0])
        crop = cv2.resize(crop, (int(crop.shape[1] * sc3), int(crop.shape[0] * sc3)), interpolation=cv2.INTER_LINEAR)
        tr = np.full((H, W, 3), 250, np.uint8)
        y3, x3 = (H - crop.shape[0]) // 2, (W - crop.shape[1]) // 2
        tr[y3:y3 + crop.shape[0], x3:x3 + crop.shape[1]] = crop
        bl, br = fp[:, panel:panel + W], fp[:, panel + W:panel + 2 * W]
        if br.shape[1] < W:
            br = np.full((H, W, 3), 255, np.uint8)
        frame = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])
        cv2.line(frame, (0, H), (2 * W, H), (200, 200, 200), 1)
        cv2.line(frame, (W, 0), (W, 2 * H), (200, 200, 200), 1)
        vw.write(frame)
    co.release(); cp.release(); vw.release()
    import subprocess
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', '-movflags', '+faststart', str(save_path)], check=True)
    os.remove(tmp)
    return str(save_path)
