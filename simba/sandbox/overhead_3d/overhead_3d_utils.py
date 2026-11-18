"""
Helpers for lifting 2D pose-estimation from ONE overhead (zenith) camera to 3D keypoints (mm, z = height above floor).
Used by :class:`overhead_3d_lifter.Overhead3DLifter` and :class:`overhead_3d_visualizer.Overhead3DVisualizer`.

Camera model: pinhole looking straight down; a point at height z with floor-plane position P (mm from the optical axis,
image orientation: x right, y down, z up -> a LEFT-handed frame) images at ``px = centre + focal * P / (H - z)``.

Keypoint roles (the lifters' input order; project body-parts are mapped onto them):
    Nose, Left_Ear, Right_Ear, Left_Side, Right_Side, Center, Tail_Base, Tail_Center, Tail_End
Nose, ears, Center and Tail_Base are required; missing optional roles are passed with zero confidence.

Methods (z MAE, held-out synthetic benchmark / held-out real Keypoint-MoSeq 3D motion; see README):
    geometry  segment foreshortening + ear-gap magnification, nose up/down by Viterbi, RTS smoothing  6.4 / 12.9 mm
    lifter    learned residual MLP, DANNCE mouse-1 + Keypoint-MoSeq poses (default)                    4.6 /  4.9 mm
    fused     per-keypoint ridge stack of geometry + DANNCE-only lifter                                4.0 /  8.3 mm
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numba import jit
from scipy.ndimage import gaussian_filter1d, median_filter

ROLES = ['Nose', 'Left_Ear', 'Right_Ear', 'Left_Side', 'Right_Side', 'Center', 'Tail_Base', 'Tail_Center', 'Tail_End']
REQUIRED_ROLES = ['Nose', 'Left_Ear', 'Right_Ear', 'Center', 'Tail_Base']
NOSE, LEAR, REAR, LSIDE, RSIDE, CENTER, TB, TC, TE = range(9)
EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 5), (3, 6), (4, 6), (5, 6), (6, 7), (7, 8)]
TAIL_BASE_MM = 8.0
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = {'dannce': os.path.join(HERE, 'lifter_dannce.pt'),                 # DANNCE markerless_mouse_1 only (used by fused)
           'dannce_kpms': os.path.join(HERE, 'lifter_dannce_kpms.pt'),       # + Keypoint-MoSeq 3D, tail dropout (camera 230-360 mm)
           'dannce_kpms_wide': os.path.join(HERE, 'lifter_dannce_kpms_wide.pt')}   # same + camera 230 mm - 4 m, log(H) feature (lifter default)
REF_SEGMENT_MM = {'ne': 17.3, 'gap': 17.0, 'ec': 30.5, 'ctb': 30.7, 'fg': 36.0}   # Ressler SLEAP 2D medians = lifter domain
METHOD_OPTIONS = ['lifter', 'fused', 'geometry']

ROLE_PATTERNS = {'Nose': (('nose', 'snout'), ()), 'Left_Ear': (('ear',), ('left', '_l', 'l_', 'earl')),
                 'Right_Ear': (('ear',), ('right', '_r', 'r_', 'earr')), 'Left_Side': (('side', 'lat', 'flank'), ('left', '_l', 'l_')),
                 'Right_Side': (('side', 'lat', 'flank'), ('right', '_r', 'r_')), 'Center': (('center', 'centre', 'centroid', 'mid_body', 'body'), ()),
                 'Tail_Base': (('tail_base', 'tailbase', 'base_of_tail', 'tailstart', 'tail_start'), ()), 'Tail_Center': (('tail_center', 'tail_centre', 'tail_mid', 'tailcenter'), ()),
                 'Tail_End': (('tail_end', 'tail_tip', 'tailend', 'tailtip'), ())}


# --------------------------------------------------------------------------------------------------- numba kernels
@jit(nopython=True, cache=True)
def rts_smooth(y: np.ndarray, r: np.ndarray, q: float, edges: np.ndarray) -> np.ndarray:
    """
    Constant-velocity Kalman filter + Rauch-Tung-Striebel smoother on a 1D series, restarted at each segment.

    :param np.ndarray y: (T,) observations.
    :param np.ndarray r: (T,) observation variances.
    :param float q: process-noise scale.
    :param np.ndarray edges: (S + 1,) int64 segment boundaries (first 0, last T).
    :return: (T,) smoothed series.
    """
    out = np.empty_like(y)
    q00, q01, q11 = 0.25 * q, 0.5 * q, q
    for s in range(edges.shape[0] - 1):
        a, b = edges[s], edges[s + 1]
        n = b - a
        if n <= 0:
            continue
        xf0 = np.empty(n); xf1 = np.empty(n); pf = np.empty((n, 3))
        xp0 = np.empty(n); xp1 = np.empty(n); pp = np.empty((n, 3))
        x0, x1, p00, p01, p11 = y[a], 0.0, r[a], 0.0, 10.0
        for i in range(n):
            if i > 0:                                                         # predict: F = [[1, 1], [0, 1]]
                x0 = x0 + x1
                n00 = p00 + 2 * p01 + p11 + q00
                n01 = p01 + p11 + q01
                n11 = p11 + q11
                p00, p01, p11 = n00, n01, n11
            xp0[i], xp1[i] = x0, x1
            pp[i, 0], pp[i, 1], pp[i, 2] = p00, p01, p11
            k0, k1 = p00 / (p00 + r[a + i]), p01 / (p00 + r[a + i])            # update (observe x0)
            inn = y[a + i] - x0
            x0, x1 = x0 + k0 * inn, x1 + k1 * inn
            p00, p01, p11 = p00 - k0 * p00, p01 - k0 * p01, p11 - k1 * p01
            xf0[i], xf1[i] = x0, x1
            pf[i, 0], pf[i, 1], pf[i, 2] = p00, p01, p11
        s0, s1 = xf0[n - 1], xf1[n - 1]
        out[b - 1] = s0
        for i in range(n - 2, -1, -1):                                        # smooth: G = Pf F' inv(Pp[i+1])
            f00, f01, f11 = pf[i, 0], pf[i, 1], pf[i, 2]
            a00, a01, a10, a11 = f00 + f01, f01, f01 + f11, f11              # Pf F'
            q0, q1, q2 = pp[i + 1, 0], pp[i + 1, 1], pp[i + 1, 2]
            det = q0 * q2 - q1 * q1
            i00, i01, i11 = q2 / det, -q1 / det, q0 / det
            g00, g01 = a00 * i00 + a01 * i01, a00 * i01 + a01 * i11
            g10, g11 = a10 * i00 + a11 * i01, a10 * i01 + a11 * i11
            d0, d1 = s0 - xp0[i + 1], s1 - xp1[i + 1]
            s0, s1 = xf0[i] + g00 * d0 + g01 * d1, xf1[i] + g10 * d0 + g11 * d1
            out[a + i] = s0
    return out


@jit(nopython=True, cache=True)
def viterbi_two_state(unary: np.ndarray, z_hyp: np.ndarray, sigma: float, breaks: np.ndarray) -> np.ndarray:
    """
    Two-state (nose down / up) Viterbi decoding. Pairwise cost (dz / sigma)^2 / 2 between consecutive frames, zero across
    sequence breaks.

    :param np.ndarray unary: (T, 2) per-frame state costs.
    :param np.ndarray z_hyp: (T, 2) nose height under each state.
    :param float sigma: transition scale (mm).
    :param np.ndarray breaks: (T,) bool, True where the sequence restarts.
    :return: (T,) int64 decoded state.
    """
    T = unary.shape[0]
    back = np.zeros((T, 2), np.int64)
    c0, c1 = unary[0, 0], unary[0, 1]
    for t in range(1, T):
        n0, n1 = 0.0, 0.0
        for j in range(2):
            best, arg = 1e300, 0
            for i in range(2):
                pw = 0.0 if breaks[t] else 0.5 * ((z_hyp[t, j] - z_hyp[t - 1, i]) / sigma) ** 2
                v = (c0 if i == 0 else c1) + pw
                if v < best:
                    best, arg = v, i
            back[t, j] = arg
            if j == 0:
                n0 = best + unary[t, 0]
            else:
                n1 = best + unary[t, 1]
        c0, c1 = n0, n1
    s = np.zeros(T, np.int64)
    s[T - 1] = 0 if c0 <= c1 else 1
    for t in range(T - 1, 0, -1):
        s[t - 1] = back[t, s[t]]
    return s


# ------------------------------------------------------------------------------------------------------- camera
def backproject(px: np.ndarray, z: np.ndarray, calib: dict) -> np.ndarray:
    """(..., 2) pixels at heights z (...) -> (..., 3) world mm."""
    xy = (px - calib['centre']) * ((calib['camera_height_mm'] - z) / calib['focal_px'])[..., None]
    return np.concatenate([xy, z[..., None]], -1)


def project(P: np.ndarray, calib: dict) -> np.ndarray:
    """(..., 3) world mm -> (..., 2) pixels."""
    return calib['centre'] + calib['focal_px'] * P[..., :2] / (calib['camera_height_mm'] - P[..., 2:3])


def manual_calib(width: int, height: int, px_per_mm: float, camera_height_mm: float, centre: Optional[Tuple[float, float]] = None) -> dict:
    """Camera from a known camera height (mm) and the floor-plane pixels per mm; optical centre defaults to the image centre."""
    c = np.array([width / 2, height / 2], float) if centre is None else np.asarray(centre, float)
    return {'centre': c, 'focal_px': px_per_mm * camera_height_mm, 'camera_height_mm': float(camera_height_mm),
            'px_per_mm_floor': float(px_per_mm), 'floor': None, 'magnification': np.nan}


def median_background(video_path: Union[str, os.PathLike], n_frames: int = 60) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frm_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frms = []
    for idx in np.linspace(0, frm_cnt - 1, n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frm = cap.read()
        if ok:
            frms.append(frm)
    cap.release()
    return np.median(np.stack(frms), axis=0).astype(np.uint8)


def fit_arena_rect(bg: np.ndarray, size_px: Tuple[float, float], scales=np.linspace(0.85, 1.45, 31), band: int = 6, step: int = 2) -> Tuple[np.ndarray, float]:
    """
    Locate an axis-aligned arena floor of known shape (e.g. bedding in a home cage) in a median background image: the
    rectangle (aspect = size_px, size = size_px * scale) maximising [texture just inside its border] - [texture just
    outside], texture = local standard deviation (bedding is grainy, plastic walls are smooth). Box sums via an integral
    image, so the exhaustive position x scale search is fast. Returns ((4, 2) corners TL, TR, BR, BL in px, scale).
    """
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu = cv2.blur(g, (7, 7))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - mu * mu, 0))
    sd = np.minimum(sd, np.percentile(sd, 95))                                  # tame specular cage rims
    H, W = sd.shape
    I = np.zeros((H + 1, W + 1)); I[1:, 1:] = sd.cumsum(0).cumsum(1)

    def box(x0, y0, x1, y1):
        x0, x1, y0, y1 = np.clip(x0, 0, W), np.clip(x1, 0, W), np.clip(y0, 0, H), np.clip(y1, 0, H)
        return (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) / np.maximum((x1 - x0) * (y1 - y0), 1)

    best, par = -np.inf, None
    for s in scales:
        w, h = int(size_px[0] * s), int(size_px[1] * s)
        if w >= W or h >= H:
            continue
        X, Y = np.meshgrid(np.arange(0, W - w, step), np.arange(0, H - h, step))
        x1, y1 = X + w, Y + h
        ins = box(X, Y, x1, Y + band) + box(X, y1 - band, x1, y1) + box(X, Y, X + band, y1) + box(x1 - band, Y, x1, y1)
        out = box(X, Y - band, x1, Y) + box(X, y1, x1, y1 + band) + box(X - band, Y, X, y1) + box(x1, Y, x1 + band, y1)
        sc = ins - out
        k = np.unravel_index(np.argmax(sc), sc.shape)
        if sc[k] > best:
            best, par = sc[k], (X[k], Y[k], w, h, s)
    x, y, w, h, s = par
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float), float(s)


def _edge_profile(g: np.ndarray, theta: float, axis: str, span=(0.3, 0.7)) -> np.ndarray:
    h, w = g.shape[1:]
    t = np.tan(theta)
    gx, gy = g
    if axis == 'x':
        ys = np.arange(int(span[0] * h), int(span[1] * h))
        xs = np.arange(w, dtype=np.float64)[:, None] - t * (ys[None, :] - h / 2)
        img = (gx * np.cos(theta) - gy * np.sin(theta)).astype(np.float32)
        vals = cv2.remap(img, xs.astype(np.float32), np.broadcast_to(ys, xs.shape).astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    else:
        xs = np.arange(int(span[0] * w), int(span[1] * w))
        ys = np.arange(h, dtype=np.float64)[:, None] + t * (xs[None, :] - w / 2)
        img = (gy * np.cos(theta) - gx * np.sin(theta)).astype(np.float32)
        vals = cv2.remap(img, np.broadcast_to(xs, ys.shape).astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    return vals.mean(axis=1)


def detect_floor_quad(bg: np.ndarray, size_range=(0.5, 0.85), max_angle_deg: float = 5.0) -> np.ndarray:
    """Floor as a rotated square (walls brighter than floor): exhaustive search over angle, side and offsets of the
    signed brightness-edge profiles. Returns (4, 2) corners TL, TR, BR, BL in pixels."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float64), (5, 5), 0)
    g = np.stack([cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)])
    h, w = gray.shape
    sizes = np.arange(int(size_range[0] * min(h, w)), int(size_range[1] * min(h, w)))
    best, best_par = -np.inf, None
    for theta in np.deg2rad(np.arange(-max_angle_deg, max_angle_deg + 0.01, 0.25)):
        Px, Py = _edge_profile(g, theta, 'x'), _edge_profile(g, theta, 'y')
        for s in sizes:
            lx, ly = -Px[:w - s] + Px[s:], -Py[:h - s] + Py[s:]
            score = lx.max() + ly.max()
            if score > best:
                best, best_par = score, (theta, int(np.argmax(lx)), int(np.argmax(ly)), s)
    theta, l, t, s = best_par
    tn = np.tan(theta)

    def corner(bx, by):
        y = (by + tn * (bx + tn * h / 2 - w / 2)) / (1 + tn * tn)
        return np.array([bx - tn * (y - h / 2), y])

    return np.array([corner(l, t), corner(l + s, t), corner(l + s, t + s), corner(l, t + s)])


def calibrate_box(video_path: Union[str, os.PathLike], floor_size_mm: float = 156.0) -> Dict:
    """
    Calibrate from a square open-field box whose video is cropped at the rim (top of the walls): floor quad (auto) vs rim
    (= crop edges) -> optical centre and rim magnification m (fit rim = c + m (floor - c)). With wall height h:
    camera height H = h m / (m - 1). Picklable dict (runs in worker processes).
    """
    bg = median_background(video_path)
    h_px, w_px = bg.shape[:2]
    floor = detect_floor_quad(bg)
    rim = np.array([[0, 0], [w_px - 1, 0], [w_px - 1, h_px - 1], [0, h_px - 1]], dtype=np.float64)
    A = np.zeros((8, 3))
    A[0::2, 0], A[1::2, 0] = floor[:, 0], floor[:, 1]
    A[0::2, 1], A[1::2, 2] = 1, 1
    (m, tx, ty), *_ = np.linalg.lstsq(A, rim.ravel(), rcond=None)
    side_px = np.mean([np.linalg.norm(floor[i] - floor[(i + 1) % 4]) for i in range(4)])
    return {'floor': floor, 'magnification': float(m), 'centre': np.array([tx, ty]) / (1 - m), 'px_per_mm_floor': side_px / floor_size_mm}


def box_calib(box: Dict, wall_mm: float, magnification: Optional[float] = None, centre: Optional[np.ndarray] = None) -> dict:
    m = box['magnification'] if magnification is None else magnification
    H = wall_mm * m / (m - 1)
    return {'centre': box['centre'] if centre is None else centre, 'focal_px': box['px_per_mm_floor'] * H,
            'camera_height_mm': H, 'px_per_mm_floor': box['px_per_mm_floor'], 'floor': box['floor'], 'magnification': m}


def _fl(p, z, calib):
    return (p - calib['centre']) * ((calib['camera_height_mm'] - z) / calib['focal_px'])[:, None]


def ear_gap_heights(px: np.ndarray, conf: np.ndarray, box: Dict, H: float, smooth_frames: int = 5, ear_rest_mm: float = 15.0):
    """Ear height from ear-gap magnification z = H - f D / gap; true gap D set from long-body frames. -> (z (T,), valid (T,))."""
    focal = box['px_per_mm_floor'] * H
    le, re, tb = px[:, LEAR], px[:, REAR], px[:, TB]
    gap = np.linalg.norm(le - re, axis=1)
    frozen = np.r_[False, np.abs(np.diff(np.hstack([le, re]), axis=0)).max(axis=1) == 0]
    valid = (conf[:, [LEAR, REAR]].min(1) >= 0.5) & ~frozen
    body = np.linalg.norm(px[:, NOSE] - tb, axis=1)
    gap_ref = np.median(gap[valid & (body >= np.nanpercentile(body[valid], 70))])
    valid &= (gap > 0.6 * gap_ref) & (gap < 2.0 * gap_ref)
    t = np.arange(len(gap))
    gap_s = np.interp(t, t[valid], gap[valid]) if valid.sum() > 1 else np.full(len(gap), gap_ref)
    gap_s = median_filter(gap_s, size=smooth_frames, mode='nearest')
    D = gap_ref * (H - ear_rest_mm) / focal
    return H - focal * D / gap_s, valid


def estimate_wall_mm(px: np.ndarray, conf: np.ndarray, box: Dict, walls=np.arange(60, 201, 5), pct: float = 95) -> float:
    """Wall height (mm) at which, in the top 5% ear-height frames, the 3D tail-base -> ears distance equals the
    stretched-flat length. NaN if the video does not constrain it."""
    _, valid = ear_gap_heights(px, conf, box, H=500.0)
    valid = valid & (conf[:, TB] >= 0.5)
    if valid.sum() < 500:
        return np.nan
    ear, tb = px[:, [LEAR, REAR]].mean(1), px[:, TB]
    L = float(np.percentile((np.linalg.norm(ear - tb, axis=1) / box['px_per_mm_floor'])[valid], pct))
    ratios = []
    for w in walls:
        H = w * box['magnification'] / (box['magnification'] - 1)
        c = {'centre': box['centre'], 'focal_px': box['px_per_mm_floor'] * H, 'camera_height_mm': H}
        z, _ = ear_gap_heights(px, conf, box, H=H)
        Pe, Pt = _fl(ear, z, c), _fl(tb, np.full(len(tb), TAIL_BASE_MM), c)
        D = np.sqrt(((Pe - Pt) ** 2).sum(1) + (z - TAIL_BASE_MM) ** 2)
        r, zv = D[valid] / L, z[valid]
        ratios.append(float(np.median(r[zv >= np.percentile(zv, 95)])))
    p = np.array(ratios)
    if not (np.all(np.diff(p) > 0) and p.min() <= 1.0 <= p.max()):
        return np.nan
    return float(np.interp(1.0, p, walls))


# ---------------------------------------------------------------------------------------------- keypoint checks
def match_roles(body_parts: List[str], bp_map: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    """Map lifter roles to project body-part names (without animal prefix). Explicit bp_map entries win; the rest are
    matched by name (case-insensitive). Returns {role: body_part or None}."""
    bp_map = bp_map or {}
    low = {b: b.lower() for b in body_parts if b not in bp_map.values()}     # explicitly mapped parts are never re-used
    out = {}
    for role in ROLES:
        if role in bp_map:
            out[role] = bp_map[role]
            continue
        need, side = ROLE_PATTERNS[role]
        hits = [b for b, l in low.items() if any(n in l for n in need) and (not side or any(s in l for s in side))]
        if role in ('Left_Ear', 'Right_Ear', 'Left_Side', 'Right_Side'):
            other = ('right', '_r', 'r_', 'earr') if role.startswith('Left') else ('left', '_l', 'l_', 'earl')
            hits = [b for b in hits if not any(o in low[b] for o in other if len(o) > 2)]
        if role == 'Center':
            hits = [b for b in hits if 'tail' not in low[b]]
        if role == 'Tail_Base':
            hits = hits or [b for b, l in low.items() if 'tail' in l and 'base' in l] or [b for b, l in low.items() if l.rstrip('_0123456789') == 'tail']
        out[role] = hits[0] if len(hits) >= 1 else None
    return out


def keypoint_layout(px: np.ndarray, conf: np.ndarray, px_per_mm: float, min_p: float = 0.8) -> Dict[str, Tuple[float, float]]:
    """Median (along, lateral) position in mm of every role relative to the tail base, along nose -> tail base."""
    ok = conf[:, [NOSE, LEAR, REAR, CENTER, TB]].min(1) > min_p
    if ok.sum() < 50:
        return {}
    u = px[ok, NOSE] - px[ok, TB]
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
    v = np.stack([-u[:, 1], u[:, 0]], 1)
    out = {}
    for k, role in enumerate(ROLES):
        if conf[ok, k].mean() < 0.3:
            continue
        r = px[ok, k] - px[ok, TB]
        out[role] = (float(np.median((r * u).sum(1)) / px_per_mm), float(np.median((r * v).sum(1)) / px_per_mm))
    return out


def check_keypoint_layout(px: np.ndarray, conf: np.ndarray, px_per_mm: float) -> List[str]:
    """Anatomy sanity check: the nose must be the most anterior point, the centre near the midline, ears/flanks lateral.
    Returns warning strings (empty if the layout looks right). Catches body-part names that do not match the data."""
    lay = keypoint_layout(px, conf, px_per_mm)
    if not lay:
        return ['too few confident frames to check the keypoint layout']
    warn = []
    nose_along = lay['Nose'][0]
    for role, (al, lat) in lay.items():
        if role != 'Nose' and al > nose_along + 5.0:
            warn.append(f'{role} lies {al - nose_along:.0f} mm in front of the Nose: body-part names may not match the data')
    if 'Center' in lay and abs(lay['Center'][1]) > 8.0:
        warn.append(f'Center is {abs(lay["Center"][1]):.0f} mm off the midline: body-part names may not match the data')
    for pair in (('Left_Ear', 'Right_Ear'), ('Left_Side', 'Right_Side')):
        if all(p in lay for p in pair) and abs(lay[pair[0]][1] - lay[pair[1]][1]) < 2.0:
            warn.append(f'{pair[0]} and {pair[1]} are not on opposite sides of the body')
    return warn


def fix_handedness(px: np.ndarray, conf: np.ndarray, threshold: float = 0.2) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """The lifters expect Left_* on the image-left of the heading seen from above (cross(nose - tail base, left - right) < 0).
    Swap ears and/or flanks if, on confident frames, they are mostly on the wrong side. -> (px, conf, swapped pairs)."""
    px, conf = px.copy(), conf.copy()
    ok = conf[:, [NOSE, LEAR, REAR, LSIDE, RSIDE, TB]].min(1) > 0.8
    swapped = []
    if ok.sum() < 50:
        return px, conf, swapped
    h = px[ok, NOSE] - px[ok, TB]
    for l, r, name in ((LEAR, REAR, 'ears'), (LSIDE, RSIDE, 'flanks')):
        v = px[ok, l] - px[ok, r]
        if np.mean(np.sign(h[:, 0] * v[:, 1] - h[:, 1] * v[:, 0])) > threshold:
            px[:, [l, r]], conf[:, [l, r]] = px[:, [r, l]], conf[:, [r, l]]
            swapped.append(name)
    return px, conf, swapped


REF_BODY_MM = 82.0   # Ressler SLEAP nose -> tail-base 2D length, 95th percentile (the lifter's mouse)


def derive_center(px: np.ndarray, conf: np.ndarray, frac: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Projects without a Center keypoint: place it on the tail base -> ear-midpoint line (frac 0.5, the Ressler SLEAP
    Center sits 30.5 of ~60 mm). Confidence = min of the ears and tail base."""
    em = px[:, [LEAR, REAR]].mean(1)
    px = px.copy(); conf = conf.copy()
    px[:, CENTER] = px[:, TB] + frac * (em - px[:, TB])
    conf[:, CENTER] = conf[:, [LEAR, REAR, TB]].min(1)
    return px, conf


def body_scale(px: np.ndarray, conf: np.ndarray, px_per_mm: float, tol: float = 1.15) -> float:
    """Animal size relative to the lifter's mouse (95th-pct nose -> tail-base 2D length / REF_BODY_MM); 1.0 within tol
    (tol=1 returns the raw ratio). NaN if fewer than 100 confident frames."""
    ok = conf[:, [NOSE, TB]].min(1) >= 0.5
    if ok.sum() < 100:
        return np.nan
    s = float(np.percentile(np.linalg.norm(px[ok, NOSE] - px[ok, TB], axis=1) / px_per_mm, 95)) / REF_BODY_MM
    return 1.0 if 1 / tol < s < tol else s


def scaled_calib(calib: dict, s: float) -> dict:
    """The same image seen as a world scaled by 1/s (pixels unchanged: X/s, z/s and H/s with the same focal). Lifting
    with this calib and multiplying heights by s puts an animal s times the lifter's mouse into its training range."""
    c = dict(calib)
    c['camera_height_mm'] = calib['camera_height_mm'] / s
    c['px_per_mm_floor'] = calib['px_per_mm_floor'] * s
    return c


def normalise_proportions(px: np.ndarray, conf: np.ndarray, px_per_mm: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """Keypoint-definition normalisation for the lifter input (2D only, no 3D ground truth): rescale each segment so this
    animal's median 2D length matches the lifter's training domain (REF_SEGMENT_MM). Frame-to-frame foreshortening is
    kept (one constant ratio per segment). x, y of the output are always re-projected from the real points."""
    present = conf[:, :7].mean(0) > 0.3
    if not present[[NOSE, LEAR, REAR, CENTER, TB]].all():
        return px, {}
    ok = conf[:, [k for k in range(7) if present[k]]].min(1) > 0.8
    if ok.sum() < 50:
        return px, {}
    em, fm = px[:, [LEAR, REAR]].mean(1), px[:, [LSIDE, RSIDE]].mean(1)
    med = lambda v: np.median(np.linalg.norm(v[ok], axis=1)) / px_per_mm
    r = {'ctb': REF_SEGMENT_MM['ctb'] / med(px[:, CENTER] - px[:, TB]), 'ec': REF_SEGMENT_MM['ec'] / med(em - px[:, CENTER]),
         'ne': REF_SEGMENT_MM['ne'] / med(px[:, NOSE] - em), 'gap': REF_SEGMENT_MM['gap'] / med(px[:, LEAR] - px[:, REAR])}
    r['fg'] = REF_SEGMENT_MM['fg'] / med(px[:, LSIDE] - px[:, RSIDE]) if present[LSIDE] and present[RSIDE] else 1.0
    r = {k: float(np.clip(v, 0.5, 2.0)) for k, v in r.items()}
    P = px.copy()
    C = px[:, TB] + r['ctb'] * (px[:, CENTER] - px[:, TB])
    E = C + r['ec'] * (em - px[:, CENTER])
    P[:, CENTER] = C
    P[:, LEAR], P[:, REAR] = E + r['gap'] * (px[:, LEAR] - em), E + r['gap'] * (px[:, REAR] - em)
    P[:, NOSE] = E + r['ne'] * (px[:, NOSE] - em)
    F = fm - px[:, CENTER] + C
    P[:, LSIDE], P[:, RSIDE] = F + r['fg'] * (px[:, LSIDE] - fm), F + r['fg'] * (px[:, RSIDE] - fm)
    return P, r


# ------------------------------------------------------------------------------------------- method: geometry
GEO = dict(min_p=0.5, jump_mm=25.0, back_rise_flat_mm=12.0, len_pct=95, gap_weight=0.35, sigma_trans=8.0,
           up_delta=30.0, up_width=5.0, q_pitch=0.02, q_z=0.5)


def _interp_lowconf(px, conf, min_p):
    px = px.copy()
    t = np.arange(len(px))
    for k in range(px.shape[1]):
        ok = conf[:, k] >= min_p
        if ok.sum() < 2 or ok.all():
            continue
        for a in range(2):
            px[~ok, k, a] = np.interp(t[~ok], t[ok], px[ok, k, a])
    return px


def _rts(y, r, q, edges):
    y = np.ascontiguousarray(y, np.float64)
    return rts_smooth(y, np.ascontiguousarray(np.broadcast_to(np.asarray(r, np.float64), y.shape)), float(q), edges)


def lift_geometry(px: np.ndarray, conf: np.ndarray, calib: dict, fps: float = 30, return_lengths: bool = False):
    """Geometry + temporal decoding: centre from tail-base->centre foreshortening, ears from centre->ears foreshortening
    fused with self-calibrated ear-gap magnification, nose from skull foreshortening with up/down by Viterbi, RTS smoothing."""
    g = GEO
    H, f = calib['camera_height_mm'], calib['focal_px']
    n = len(px)
    px = _interp_lowconf(px, conf, g['min_p'])
    ok = conf[:, [NOSE, LEAR, REAR, CENTER, TB]].min(1) >= g['min_p']
    if ok.sum() < 10:
        ok = np.ones(n, bool)
    ear_px = px[:, [LEAR, REAR]].mean(1)
    length = lambda d, rise=0.0: float(np.hypot(np.percentile(d[ok], g['len_pct']), rise))
    c_mm = _fl(px[:, CENTER], np.full(n, 25.0), calib)
    breaks = np.r_[True, np.linalg.norm(np.diff(c_mm, axis=0), axis=1) > g['jump_mm']]
    edges = np.r_[0, np.flatnonzero(breaks[1:]) + 1, n].astype(np.int64)

    z_tb = np.full(n, TAIL_BASE_MM)
    z_c = z_tb + g['back_rise_flat_mm']
    L_tc = length(np.linalg.norm(_fl(px[:, CENTER], z_c, calib) - _fl(px[:, TB], z_tb, calib), axis=1), g['back_rise_flat_mm'])
    for _ in range(3):
        d = np.linalg.norm(_fl(px[:, CENTER], z_c, calib) - _fl(px[:, TB], z_tb, calib), axis=1)
        z_c = z_tb + np.maximum(g['back_rise_flat_mm'], np.sqrt(np.clip(L_tc ** 2 - d ** 2, 0, None)))
    z_c = _rts(z_c, 4.0, g['q_z'], edges)

    z_e = z_c + 10.0
    L_ce = length(np.linalg.norm(_fl(ear_px, z_e, calib) - _fl(px[:, CENTER], z_c, calib), axis=1))
    for _ in range(3):
        d = np.linalg.norm(_fl(ear_px, z_e, calib) - _fl(px[:, CENTER], z_c, calib), axis=1)
        z_e = z_c + np.sqrt(np.clip(L_ce ** 2 - d ** 2, 0, None))
    z_geo = z_e.copy()
    gap = np.linalg.norm(px[:, LEAR] - px[:, REAR], axis=1)
    okg = ok & (gap > 0.5 * np.median(gap[ok])) & (gap < 2.0 * np.median(gap[ok]))
    inv = np.where(okg, 1.0 / np.maximum(gap, 1e-3), np.nan)
    inv_s = _rts(np.where(np.isnan(inv), np.nanmedian(inv), inv), np.where(okg, 1.0, 1e6) * np.nanvar(inv), 1e-3 * np.nanvar(inv), edges)
    D = np.median((H - z_geo[okg]) / f / inv_s[okg])
    z_e = (1 - g['gap_weight']) * z_geo + g['gap_weight'] * (H - f * D * inv_s)
    z_e = np.maximum(_rts(z_e, 25.0, g['q_z'], edges), z_c)

    L_sk = length(np.linalg.norm(_fl(px[:, NOSE], z_e, calib) - _fl(ear_px, z_e, calib), axis=1))
    d = np.linalg.norm(_fl(px[:, NOSE], z_e, calib) - _fl(ear_px, z_e, calib), axis=1)
    r = np.sqrt(np.clip(L_sk ** 2 - d ** 2, 0, None))
    for _ in range(2):
        d = np.linalg.norm(_fl(px[:, NOSE], z_e - r, calib) - _fl(ear_px, z_e, calib), axis=1)
        r = np.sqrt(np.clip(L_sk ** 2 - d ** 2, 0, None))
    z_hyp = np.ascontiguousarray(np.stack([z_e - r, z_e + r], 1))
    p_up = 1.0 / (1.0 + np.exp(-(z_e - np.percentile(z_e[ok], 20) - g['up_delta']) / g['up_width']))
    p_up = np.clip(p_up, 1e-4, 1 - 1e-4)
    s = viterbi_two_state(np.ascontiguousarray(np.stack([-np.log(1 - p_up), -np.log(p_up)], 1)), z_hyp, float(g['sigma_trans']), breaks)
    pitch = _rts(np.arcsin(np.clip(np.where(s == 1, 1.0, -1.0) * r / L_sk, -1, 1)), 0.05, g['q_pitch'], edges)
    z_n = np.maximum(z_e + L_sk * np.sin(pitch), 2.0)

    z = np.stack([z_n, z_e, z_e, z_c - 3, z_c - 3, z_c, z_tb, np.full(n, 4.0), np.full(n, 2.0)], 1)
    out = backproject(px, np.clip(z, 0, None), calib)
    return (out, {'L_sk': L_sk, 'L_ce': L_ce, 'L_tc': L_tc}) if return_lengths else out


# ---------------------------------------------------------------------------------------------- method: lifter
LIFTER_CONF_MIN, LIFTER_MM = 0.3, 50.0
LIFTER_DEFAULT = 'dannce_kpms_wide'
H_FEAT_OPTIONS = ['lin', 'log']
_NETS, _WIN = {}, {}
# Expected |height error| (mm) per keypoint = Z_ERR_A[k] + Z_ERR_B * ensemble SD (both Gaussian-smoothed like z). Least-squares fit
# of the real error on the default lifter's ensemble SD, pooled over held-out KPMS test recordings (real 3D, ROLES 0,1,2,5,6) and
# DANNCE markerless_mouse_2 synthetic sequences (all 9), cameras 288-4000 mm. The raw SD is 3-7x smaller than the real error, hence
# the fit. Predicted vs actual MAE by quintile, KPMS: 1.9/2.4/3.1/4.6/7.8 vs 2.3/2.5/3.3/4.4/8.3 mm; DANNCE: ~1.5 mm too low.
Z_ERR_A = np.array([3.2, 1.23, 1.92, 1.89, 2.27, 0.9, 1.44, 1.05, 5.27])
Z_ERR_B = 1.573


def lifter_features(px: np.ndarray, conf: np.ndarray, calib: dict, h_feat: str = 'lin') -> np.ndarray:
    """Single-frame features: metric floor rays relative to Center rotated into the (5-frame smoothed) heading, masked
    below LIFTER_CONF_MIN, + mask channel + global features (Center offset from the optical axis, camera height).
    h_feat (from the checkpoint): 'lin' = (H - 300) / 50 (dannce, dannce_kpms), 'log' = log(H / 300) (dannce_kpms_wide)."""
    T = len(px)
    H = float(calib['camera_height_mm'])
    r = (px - calib['centre']) / calib['focal_px'] * H
    m = (conf >= LIFTER_CONF_MIN).astype(np.float64)

    def wmean(idx, fallback):
        w = m[:, idx]
        s = w.sum(1, keepdims=True)
        return np.where(s > 0, (r[:, idx] * w[..., None]).sum(1) / np.maximum(s, 1e-9), fallback)

    origin = np.where(m[:, 5:6] > 0, r[:, 5], wmean([3, 4, 6], r[:, 5]))
    post = np.where(m[:, 6:7] > 0, r[:, 6], origin)
    h = wmean([0, 1, 2], origin + (origin - post)) - post
    h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
    h = np.stack([np.convolve(np.pad(h[:, i], 2, mode='edge'), np.ones(5) / 5, 'valid') for i in range(2)], 1)
    a = np.arctan2(h[:, 1], h[:, 0])
    c, s = np.cos(-a), np.sin(-a)
    rel = r - origin[:, None]
    rel = np.stack([c[:, None] * rel[..., 0] - s[:, None] * rel[..., 1], s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1]], -1) * m[..., None]
    o = np.stack([c * origin[:, 0] - s * origin[:, 1], s * origin[:, 0] + c * origin[:, 1]], 1)
    h_glob = np.log(H / 300) if h_feat == 'log' else (H - 300) / 50
    glob = np.concatenate([o / 100, np.linalg.norm(origin, axis=1, keepdims=True) / 100, np.full((T, 1), h_glob)], 1)
    return np.concatenate([rel.reshape(T, -1) / LIFTER_MM, m, glob], 1).astype(np.float32)


def stack_frames(F: np.ndarray, half: int, stride: int, seq: Optional[np.ndarray] = None) -> np.ndarray:
    """Temporal lifter input: (T, d) per-frame features -> (T, (2 half + 1) d), the features at t + k stride for
    k = -half..half, clamped to the first / last frame of t's continuous segment (seq: (T,) segment id; None = one segment)."""
    if half == 0:
        return F
    T = len(F)
    t = np.arange(T)
    if seq is None:
        lo, hi = np.zeros(T, int), np.full(T, T - 1)
    else:
        b = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, T]
        idx = np.searchsorted(b, t, side='right') - 1
        lo, hi = b[idx], b[idx + 1] - 1
    cols = [F[np.clip(t + k * stride, lo, hi)] for k in range(-half, half + 1)]
    return np.concatenate(cols, 1)


def lifter_window(path: str, device: str = 'cpu') -> Tuple[int, int]:
    """(half, stride) of a lifter checkpoint: its temporal window in frames at 30 fps; (0, 1) = single-frame."""
    load_lifter(path, device)
    return _WIN[path]


def load_lifter(path: str, device: str = 'cpu'):
    """5-seed residual-MLP ensemble (Martinez et al. 2017 architecture), cached per (path, device).
    Returns (nets, h_feat): h_feat is the checkpoint's camera-height feature ('lin' when its 'feat' dict has none).
    The temporal window (half, stride) is available from lifter_window()."""
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, d, p):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(p))

        def forward(self, x):
            return x + self.f(x)

    class Lifter(nn.Module):
        def __init__(self, n_in, d=1024, n_blocks=2, p=0.25):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(p))
            self.blocks = nn.Sequential(*[Block(d, p) for _ in range(n_blocks)])
            self.out = nn.Linear(d, 9)

        def forward(self, x):
            return self.out(self.blocks(self.inp(x))) * LIFTER_MM + 25.0

    key = (path, device)
    if key not in _NETS:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        feat = dict(ck['feat'])
        h_feat = feat.pop('h_feat', 'lin')
        half, stride = int(feat.pop('half', 0)), int(feat.pop('stride', 1))
        if feat != {'scale_free': False} or h_feat not in H_FEAT_OPTIONS or half < 0 or stride < 1:
            raise ValueError(f'{path}: unsupported lifter features {ck["feat"]}')
        _WIN[path] = (half, stride)
        nets = []
        for st in ck['states']:
            net = Lifter(ck['n_in'], **ck['arch'])
            net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in st.items()})
            nets.append(net.eval().to(device))
        _NETS[key] = (nets, h_feat)
    return _NETS[key]


def lifter_device(gpu: bool = True) -> str:
    import torch
    return 'cuda' if gpu and torch.cuda.is_available() else 'cpu'


def predict_lifter_z(px: np.ndarray, conf: np.ndarray, calib: dict, weights: str = LIFTER_DEFAULT, gpu: bool = True, batch: int = 65536,
                     return_sd: bool = False, fps: float = 30.0) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """(T, 9) ensemble-mean heights (mm), clipped at 0; with return_sd also the (T, 9) ensemble standard deviation (mm).
    fps: video frame rate; a temporal lifter's window (trained at 30 fps) is rescaled to it."""
    import torch
    device = lifter_device(gpu)
    nets, h_feat = load_lifter(WEIGHTS[weights], device)
    half, stride = lifter_window(WEIGHTS[weights], device)
    X = torch.from_numpy(stack_frames(lifter_features(px, conf, calib, h_feat), half, max(1, int(round(stride * fps / 30)))))
    out, sds = [], []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda'):   # fp16: ~2x, mean change 0.02 mm
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(device, non_blocking=True)
            zb = torch.stack([n(xb) for n in nets]).float()
            out.append(zb.mean(0).cpu())
            if return_sd:
                sds.append(zb.std(0).cpu())
    z = np.clip(torch.cat(out).numpy().astype(np.float64), 0, None)
    return (z, torch.cat(sds).numpy().astype(np.float64)) if return_sd else z


def lift_lifter(px: np.ndarray, conf: np.ndarray, calib: dict, fps: float = 30, smooth_sigma: float = 2.0, weights: str = LIFTER_DEFAULT,
                gpu: bool = True, px_in: Optional[np.ndarray] = None, return_err: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Learned lifter heights, Gaussian-smoothed over time; x, y back-projected from px. px_in: optional (normalised)
    lifter input, the real px are still used for x, y. return_err: also the (T, 9) expected height error (mm) from the
    ensemble spread (Z_ERR_A, Z_ERR_B; calibrated for the default weights only)."""
    z, sd = predict_lifter_z(px if px_in is None else px_in, conf, calib, weights, gpu, return_sd=True, fps=fps) if return_err else \
            (predict_lifter_z(px if px_in is None else px_in, conf, calib, weights, gpu, fps=fps), None)
    if smooth_sigma:
        z = gaussian_filter1d(z, smooth_sigma * fps / 30, axis=0, mode='nearest')
        sd = None if sd is None else gaussian_filter1d(sd, smooth_sigma * fps / 30, axis=0, mode='nearest')
    P = backproject(px, z, calib)
    return (P, Z_ERR_A + Z_ERR_B * sd) if return_err else P


# ----------------------------------------------------------------------------------------------- method: fused
# Per-keypoint ridge stack (Wolpert 1992; Hoerl & Kennard 1970) fit on DANNCE mouse 1 out-of-fold lifters:
# z_k = zl_k + B[k] . [1, zg_k/30, zl_k/30, r_k, r_k (zg_k - zl_k)/30, |zg_k - zl_k|/30], r_k = projected / 3D length of the
# segment keypoint k hangs from; then Gaussian smoothing (sigma 4 frames at 30 fps). Uses the DANNCE-only lifter.
FUSED_B = np.array([[-14.5795, 16.298, -16.0916, 20.4219, -8.2666, -2.3829],
                    [-39.9717, 17.1787, -3.1008, 18.291, -4.308, 3.7784],
                    [-35.4127, 15.8646, -3.9962, 16.9483, -3.3353, 3.4988],
                    [-4.6452, -1.7034, 5.0942, 4.0195, 2.1604, 1.2518],
                    [-4.7541, -1.6735, 5.1547, 4.0445, 2.2236, 1.4309],
                    [-5.1348, -1.4343, 4.7203, 4.003, 2.2738, 1.8968],
                    [-3.0483, 0., 4.7091, 3.4032, -0.537, 0.8785],
                    [-2.979, 0., 1.5945, 6.474, 7.6199, 0.8225],
                    [-3.7636, 0., -0.0915, 12.0771, 16.3416, -0.1708]])
FUSED_SEG_OF = np.array([0, 1, 1, 2, 2, 2, 2, 2, 2])
FUSED_SMOOTH = 4.0


def lift_fused(px: np.ndarray, conf: np.ndarray, calib: dict, fps: float = 30, gpu: bool = True, px_in: Optional[np.ndarray] = None) -> np.ndarray:
    P, lens = lift_geometry(px, conf, calib, fps, return_lengths=True)
    zg = P[..., 2]
    zl = lift_lifter(px, conf, calib, fps, weights='dannce', gpu=gpu, px_in=px_in)[..., 2]
    ear = px[:, [LEAR, REAR]].mean(1)
    ze, zc = zg[:, LEAR], zg[:, CENTER]
    r = np.stack([np.linalg.norm(_fl(px[:, NOSE], ze, calib) - _fl(ear, ze, calib), axis=1) / lens['L_sk'],
                  np.linalg.norm(_fl(ear, ze, calib) - _fl(px[:, CENTER], zc, calib), axis=1) / lens['L_ce'],
                  np.linalg.norm(_fl(px[:, CENTER], zc, calib) - _fl(px[:, TB], zg[:, TB], calib), axis=1) / lens['L_tc']], 1)[:, FUSED_SEG_OF]
    X = np.stack([np.ones_like(zg), zg / 30, zl / 30, r, r * (zg - zl) / 30, np.abs(zg - zl) / 30], -1)
    z = gaussian_filter1d(zl + np.einsum('tkf,kf->tk', X, FUSED_B), FUSED_SMOOTH * fps / 30, axis=0, mode='nearest')
    return backproject(px, np.clip(z, 0, None), calib)


def lift(method: str, px: np.ndarray, conf: np.ndarray, calib: dict, fps: float = 30, gpu: bool = True, px_in: Optional[np.ndarray] = None,
         return_err: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Dispatch: (T, 9, 2) px + (T, 9) conf -> (T, 9, 3) world mm. return_err (method 'lifter' only): also the (T, 9)
    expected height error (mm)."""
    if return_err and method != 'lifter':
        raise ValueError("return_err is only available for method='lifter'")
    if method == 'geometry':
        return lift_geometry(px, conf, calib, fps)
    if method == 'lifter':
        return lift_lifter(px, conf, calib, fps, gpu=gpu, px_in=px_in, return_err=return_err)
    if method == 'fused':
        return lift_fused(px, conf, calib, fps, gpu=gpu, px_in=px_in)
    raise ValueError(f'method must be one of {METHOD_OPTIONS}')
