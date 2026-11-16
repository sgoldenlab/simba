"""
Retrain (and score) the overhead 2D -> 3D height lifter used by ``overhead_3d_utils.lift_lifter``.

One command reproduces the shipped default ``lifter_dannce_kpms_wide.pt`` (the "farcam" recipe):

    set KMP_DUPLICATE_LIB_OK=TRUE
    python simba/sandbox/overhead_3d/train_lifter.py --out C:/models/my_lifter.pt

    python simba/sandbox/overhead_3d/train_lifter.py --evaluate simba/sandbox/overhead_3d/lifter_dannce_kpms_wide.pt C:/models/my_lifter.pt

Writes ``<out>.pt`` (the checkpoint format ``overhead_3d_utils.load_lifter`` reads: n_in, arch, feat incl. h_feat, 5 fp16
state dicts, plus metadata) and ``<out>.json`` (every setting, sha256 of the data files, package versions, the training log
and the evaluation table).

Recipe (merged from ../zenith_height/bench/methods/lifter_mlp_train.py, kpms_lifter_train.py and farcam_lifter_train.py;
see ../zenith_height/bench/kpms_NOTES.md and farcam_NOTES.md for why each piece is there):
  - Training poses: DANNCE markerless_mouse_1 hand-labelled 3D poses, blended into smooth sequences (key-pose blending,
    head pitch / yaw, front-body pitch, mirroring), and real continuous 3D motion from the Keypoint-MoSeq open-field
    recordings (train split only; random Center definition, synthetic floor tail, real heading and shrunk trajectory).
    Sequences alternate between the two sources at ``kpms_fraction`` (default 0.5).
  - Per-animal morphology: size, body length, flank width / drop, ear gap, keypoint-definition offsets.
  - Virtual camera per sequence: height log-uniform in [camera_h_min_mm, camera_h_max_mm], floor px/mm uniform in
    [px_per_mm_min, px_per_mm_max] (focal = px/mm x H, so the animal's pixel size stays realistic and only the perspective
    changes), optical centre jitter, +-1.5 deg tilt, 3 % calibration height error.
  - Observation noise: 0.5-2.5 px jitter, tail-centre sliding along the tail, up to 6 % outliers with low confidence.
  - Keypoint-set dropout (optional): tail centre + tail end missing (p = 0.35), additionally both flanks (p = 0.15).
  - Network: residual MLP (Martinez et al. 2017; 1024 wide, 2 blocks, dropout 0.25) predicting the 9 keypoint heights from
    ``overhead_3d_utils.lifter_features`` (imported, so training and inference features cannot drift apart) with the
    log(H / 300) camera-height feature; smooth-L1 loss, AdamW + one-cycle, fresh synthetic data every round; an ensemble
    of ``n_seeds`` networks (seeds seed .. seed + n_seeds - 1).

Code dependency: only ``overhead_3d_utils`` (same folder). The research code in ../zenith_height/bench/ is NOT imported:
the needed functions are copied here with their random-number call order unchanged, so the defaults reproduce the
bench run (seed for seed, up to GPU non-determinism). The benchmark camera (Ressler box 4, from
``bench.common.get_calib``, which needs the raw video on I:) is stored as constants below.
Data dependency: the default data paths point at ../zenith_height/bench/ (DANNCE .mat files, Keypoint-MoSeq .h5).

Evaluation (``evaluate_lifter``), all through ``overhead_3d_utils.load_lifter`` + ``lifter_features``, ensemble mean in
fp32, z clipped at 0 and Gaussian-smoothed (sigma 2 frames), as ``lift_lifter`` does:
  - syn:       DANNCE mouse 2 (never trained on), 40 x 150-frame sequences in the Ressler box (= bench common.make_synthetic)
  - kpms_test: the 3 held-out Keypoint-MoSeq recordings, 7 keypoints (= bench kpms_common.make_kpms_test(SPLIT['test'], 'kp7'))
  - far:       both sets re-projected through cameras at H = 1, 2, 4 m (same floor px/mm; = bench farcam_eval.py)
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from typing import Dict, List, Optional, Union

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
from simba.sandbox.overhead_3d.overhead_3d_utils import (LIFTER_MM,
                                                         backproject,
                                                         lifter_features,
                                                         load_lifter, project)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_int, check_str, check_valid_boolean)
from simba.utils.errors import InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, '..', 'zenith_height', 'bench'))
DANNCE_TRAIN = os.path.join(BENCH, 'label3d_dannce.mat')          # DANNCE markerless_mouse_1 (MIT, github.com/spoonsso/dannce)
DANNCE_TEST = os.path.join(BENCH, 'label3d_dannce_m2.mat')        # markerless_mouse_2: evaluation only
DANNCE_SKELETON = os.path.join(BENCH, 'mouse22.mat')
KPMS_H5 = os.path.join(BENCH, 'data_kpms', 'open_field_3D.h5')    # Weinreb et al. 2024, zenodo 10636983, CC BY 4.0

# Benchmark camera: Ressler 2025-10-22 12-01-34_box4, bench.common.get_calib (box auto-calibration, wall 100 mm).
BENCH_CAMERA = {'centre': np.array([161.1346907692848, 186.8496666928773]), 'camera_height_mm': 287.9999999999999,
                'px_per_mm_floor': 1.5060517715626833, 'focal_px': 433.7429102100526,
                'floor': np.array([[57.58893472911431, 63.350801131601415], [292.4771007649826, 68.47607914702839],
                                   [287.3518227495556, 303.3642451828967], [52.46365671368733, 298.2389671674697]])}

# Keypoint-MoSeq recordings, split by recording day / animal tag (= bench kpms_common.SPLIT).
KPMS_SPLIT = {'test': ['21_12_10_def6a_3', '21_12_10_def6b_3', '22_27_04_cage4_mouse2_0'],
              'val': ['22_04_26_cage4_1_1'],
              'train': ['21_11_8_one_mouse', '21_12_10_def6a_1_1', '21_12_2_def6a_1', '21_12_2_def6b_2', '22_04_26_cage4_0', '22_04_26_cage4_0_2']}
KPMS_NAMES = ['spine1', 'spine2', 'spine3', 'tail base', 'forehead', 'left ear', 'right ear', 'nose', 'left hindpaw base',
              'left hindpaw tip', 'right hindpaw base', 'right hindpaw tip', 'left forepaw', 'right forepaw']
KPMS_CENTER_T, KPMS_POS_SCALE, KPMS_FLANK_HW = 0.12, 0.27, 17.0
SCORED = [0, 1, 2, 5, 6]                                           # keypoints with real 3D ground truth in KPMS
MIRROR = [0, 2, 1, 4, 3, 5, 6, 7, 8, 9]
P_DROP_TAIL, P_DROP_FLANK = 0.35, 0.15
ARCH = dict(d=1024, n_blocks=2, p=0.25)
FEAT = dict(half=0, stride=1, scale_free=False, h_feat='log')
EPOCHS, BATCH, N_SEQ, PER_SEQ, SEQ_LEN = 2, 1024, 1500, 50, 150
FAR_HEIGHTS_MM = (1000.0, 2000.0, 4000.0)


# ------------------------------------------------------------------------------------------------------------- data
def dannce_poses(mat_path: str, skeleton_path: str = DANNCE_SKELETON) -> np.ndarray:
    """(N, 10, 3) mm, z up: Nose, ears, flanks (synthesised 11 mm either side of SpineM, 4 mm lower), SpineM as Center,
    tail base / mid / end, + SpineF (10th point). Frames with any missing joint are dropped; z clipped at 0.5."""
    import scipy.io as sio
    names = [str(n) for n in sio.loadmat(skeleton_path, squeeze_me=True)['joint_names']]
    X = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)['labelData'][0].data_3d.reshape(-1, 22, 3)
    g = lambda n: X[:, names.index(n)]
    spine_f, spine_m = g('SpineF'), g('SpineM')
    axis = spine_f - g('Tail(base)'); axis[:, 2] = 0
    axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
    left = np.stack([-axis[:, 1], axis[:, 0], np.zeros(len(axis))], 1)
    flank = lambda s: spine_m + s * 11.0 * left - np.array([0, 0, 4.0])
    P = np.stack([g('Snout'), g('EarL'), g('EarR'), flank(+1), flank(-1), spine_m, g('Tail(base)'), g('Tail(mid)'), g('Tail(end)'), spine_f], 1)
    P = P[~np.isnan(P).any(axis=(1, 2))]
    P[..., 2] = np.maximum(P[..., 2], 0.5)
    return P


def canonical(P: np.ndarray) -> np.ndarray:
    """Centre on Center (xy) and rotate so the tail-base -> nose heading points along +x."""
    Q = P.copy()
    Q[:, :2] -= P[5, :2]
    h = P[0, :2] - P[6, :2]
    a = -np.arctan2(h[1], h[0])
    Q[:, :2] = Q[:, :2] @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
    return Q


def kpms_raw(h5_path: str, rec: str) -> np.ndarray:
    """(T, 14, 3) mm, floor = median hind-paw tip height, y negated (image orientation)."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        X = f['keypoint coordinates'][rec][:].astype(np.float64)
    X[..., 2] -= np.median(X[:, [9, 11], 2])
    X[..., 1] *= -1
    return X


def kpms_to_kp9(X: np.ndarray, center_t: float = KPMS_CENTER_T, flank_hw: float = KPMS_FLANK_HW, flank_dz: float = 4.0) -> np.ndarray:
    """(T, 14, 3) KPMS -> (T, 10, 3): the 9 lifter keypoints (tail centre / end = tail-base placeholders) + spine1."""
    g = lambda n: X[:, KPMS_NAMES.index(n)]
    center = g('spine2') + abs(center_t) * ((g('spine1') if center_t >= 0 else g('spine3')) - g('spine2'))
    axis = g('nose') - g('tail base'); axis[:, 2] = 0
    axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
    left = np.stack([axis[:, 1], -axis[:, 0], np.zeros(len(axis))], 1)
    flank = lambda s: center + s * flank_hw * left - np.array([0, 0, flank_dz])
    tb = g('tail base')
    P = np.stack([g('nose'), g('left ear'), g('right ear'), flank(+1), flank(-1), center, tb, tb, tb, g('spine1')], 1)
    P[..., 2] = np.maximum(P[..., 2], 0.5)
    return P


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ camera / projection
def camera_at(H: float, base: dict = BENCH_CAMERA) -> dict:
    """The benchmark camera moved to height H with the same floor px/mm (only the perspective changes)."""
    c = dict(base)
    c['camera_height_mm'], c['focal_px'] = float(H), base['px_per_mm_floor'] * float(H)
    return c


def rot_y(P, about, ang):
    Q = P - about
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * Q[..., 0] - s * Q[..., 2], Q[..., 1], s * Q[..., 0] + c * Q[..., 2]], -1) + about


def rot_z(P, about, ang):
    Q = P - about
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * Q[..., 0] - s * Q[..., 1], s * Q[..., 0] + c * Q[..., 1], Q[..., 2]], -1) + about


def project_cam(P, cam):
    """Virtual (possibly tilted) camera of a training sequence."""
    Hc = cam['H_true']
    if np.any(cam['tilt']):
        C = np.array([0, 0, Hc])
        P = rot_y(P, C, cam['tilt'][0])
        Q = P - C; c, s = np.cos(cam['tilt'][1]), np.sin(cam['tilt'][1])
        P = np.stack([Q[..., 0], c * Q[..., 1] - s * Q[..., 2], s * Q[..., 1] + c * Q[..., 2]], -1) + C
    return cam['centre'] + cam['focal_px'] * P[..., :2] / (Hc - P[..., 2:3])


def virtual_camera(rng, s):
    """Height log-uniform in [h_min, h_max]; p_near: fraction drawn from the old 0.8-1.25 x 288 mm range instead (0 = farcam).
    The rng.random() draw is kept even at p_near = 0 so the random stream matches the bench run."""
    if rng.random() < s['p_near']:
        H = BENCH_CAMERA['camera_height_mm'] * rng.uniform(0.8, 1.25)
    else:
        H = np.exp(rng.uniform(np.log(s['camera_h_min_mm']), np.log(s['camera_h_max_mm'])))
    ppm = rng.uniform(s['px_per_mm_min'], s['px_per_mm_max'])
    return dict(centre=np.array([rng.uniform(120, 190), rng.uniform(150, 200)]), focal_px=ppm * H, H_true=H,
                camera_height_mm=H * (1 + rng.normal(0, 0.03)), tilt=rng.normal(0, np.deg2rad(1.5), 2))


# --------------------------------------------------------------------------------------------- training sequences
def key_pose(poses, rng):
    p = poses[rng.integers(len(poses))].copy()
    if rng.random() < 0.5:                                                    # blend two poses
        q = poses[rng.integers(len(poses))]
        w = rng.uniform(0, 1)
        p = p * (1 - w) + q * w
    ear_mid = p[[1, 2]].mean(0)
    p[0] = rot_y(p[0], ear_mid, rng.normal(0, np.deg2rad(15)))                # head pitch
    p[[0, 1, 2]] = rot_z(p[[0, 1, 2]], p[5], rng.normal(0, np.deg2rad(8)))    # head yaw
    p[[0, 1, 2, 3, 4, 5, 9]] = rot_y(p[[0, 1, 2, 3, 4, 5, 9]], p[6], rng.normal(0, np.deg2rad(5)))   # front-body pitch
    p[..., 2] = np.maximum(p[..., 2], 0.5)
    if rng.random() < 0.5:
        p = p[MIRROR]; p[:, 1] *= -1
    return p


def morphology(body, rng):
    """Per-animal shape / keypoint-definition changes in the canonical body frame. (T, 10, 3) -> (T, 9, 3)."""
    b = body[:, :9].copy()
    b *= rng.uniform(0.9, 1.12)
    b[..., 0] *= rng.uniform(0.95, 1.18)
    hw, dz = rng.uniform(9, 20), rng.uniform(0, 8)
    b[:, 3] = b[:, 5] + np.array([0, hw, -dz]); b[:, 4] = b[:, 5] + np.array([0, -hw, -dz])
    em = b[:, [1, 2]].mean(1, keepdims=True)
    b[:, [1, 2]] = em + (b[:, [1, 2]] - em) * rng.uniform(0.85, 1.15)
    b += rng.normal(0, 1, (9, 3)) * np.array([2.5, 2.5, 1.5])
    b[..., 2] = np.maximum(b[..., 2], 0.5)
    return b


def place(body, yaw, pos):
    gt = body.copy()
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    gt[..., 0], gt[..., 1] = c * body[..., 0] - s * body[..., 1] + pos[:, None, 0], s * body[..., 0] + c * body[..., 1] + pos[:, None, 1]
    return gt


def observe(gt, cam, rng):
    """Pixel noise, tail-centre sliding, outliers with low confidence."""
    T = len(gt)
    px = project_cam(gt, cam)
    px += rng.normal(0, rng.uniform(0.5, 2.5), px.shape)
    conf = rng.uniform(0.6, 1.0, (T, 9))
    drift = np.clip(rng.uniform(-0.35, 0.25) + rng.normal(0, 0.15, T), -0.5, 0.4)
    bad = rng.random((T, 9)) < rng.uniform(0.0, 0.06)
    tb, te = project_cam(gt[:, 6], cam), project_cam(gt[:, 8], cam)
    px[:, 7] += drift[:, None] * np.where(drift[:, None] < 0, (px[:, 7] - tb), (te - px[:, 7]))
    px[bad] += rng.normal(0, rng.uniform(5, 25), (bad.sum(), 2)); conf[bad] = rng.uniform(0.0, 0.45, bad.sum())
    return px, conf


def dannce_sequence(poses, rng, cam):
    n_keys = rng.integers(3, 9)
    kt = np.concatenate([[0], np.cumsum(rng.uniform(8, 50, n_keys - 1))]); kt = kt / kt[-1] * (SEQ_LEN - 1)
    keys = np.stack([key_pose(poses, rng) for _ in range(n_keys)])
    t = np.arange(SEQ_LEN)
    u = np.clip(np.searchsorted(kt, t, side='right') - 1, 0, n_keys - 2)
    w = 0.5 - 0.5 * np.cos(np.pi * (t - kt[u]) / (kt[u + 1] - kt[u]))
    body = morphology(keys[u] * (1 - w)[:, None, None] + keys[u + 1] * w[:, None, None], rng)
    half_box = 78 - rng.uniform(15, 40)
    box_c = rng.normal(0, 12, 2)
    pos = np.cumsum(rng.normal(0, rng.uniform(0.4, 2.0), (SEQ_LEN, 2)), 0); pos -= pos.mean(0)
    pos = np.clip(pos + box_c + rng.uniform(-half_box, half_box, 2), box_c - half_box, box_c + half_box)
    yaw = np.cumsum(rng.normal(0, rng.uniform(0.01, 0.06), SEQ_LEN)) + rng.uniform(-np.pi, np.pi)
    gt = place(body, yaw, pos)
    return (*observe(gt, cam, rng), gt)


def canonical_seq(P):
    """(T, K, 3) -> per-frame body frame, heading angles (T,), Center xy (T, 2)."""
    o = P[:, 5, :2].copy()
    h = P[:, 0, :2] - P[:, 6, :2]
    a = np.arctan2(h[:, 1], h[:, 0])
    Q = P.copy()
    c, s = np.cos(-a)[:, None], np.sin(-a)[:, None]
    Q[..., 0], Q[..., 1] = c * (P[..., 0] - o[:, None, 0]) - s * (P[..., 1] - o[:, None, 1]), s * (P[..., 0] - o[:, None, 0]) + c * (P[..., 1] - o[:, None, 1])
    return Q, a, o


def synth_tail(body, rng):
    """KPMS has no tail centre / end: a floor tail behind the tail base along a slowly wandering angle."""
    T = len(body)
    ang = np.pi + np.cumsum(rng.normal(0, 0.03, T)) + rng.normal(0, 0.4)
    ang = np.pi + np.clip(ang - np.pi, -1.0, 1.0)
    d = np.stack([np.cos(ang), np.sin(ang)], 1)
    l1, l2 = rng.uniform(25, 35), rng.uniform(55, 75)
    bend = rng.normal(0, 0.3)
    d2 = np.stack([np.cos(ang + bend), np.sin(ang + bend)], 1)
    body[:, 7, :2] = body[:, 6, :2] + l1 * d
    body[:, 8, :2] = body[:, 7, :2] + (l2 - l1) * d2
    body[:, 7, 2] = np.clip(rng.uniform(1, 8) + np.cumsum(rng.normal(0, 0.05, T)), 0.5, 15)
    body[:, 8, 2] = np.clip(rng.uniform(1, 6) + np.cumsum(rng.normal(0, 0.05, T)), 0.5, 15)
    return body


def kpms_sequence(recs, rng, cam):
    X = recs[rng.integers(len(recs))]
    t0 = rng.integers(0, len(X) - SEQ_LEN)
    body, a, o = canonical_seq(kpms_to_kp9(X[t0:t0 + SEQ_LEN], center_t=rng.uniform(-0.75, 0.45)))
    body = morphology(synth_tail(body, rng), rng)
    if rng.random() < 0.5:
        body = body[:, MIRROR[:9]]; body[..., 1] *= -1; a = -a; o = o * np.array([1, -1])
    yaw = a + rng.uniform(-np.pi, np.pi)
    half_box = 78 - rng.uniform(15, 40)
    box_c = rng.normal(0, 12, 2)
    pos = (o - o.mean(0)) * rng.uniform(0.15, 0.45)
    c0, s0 = np.cos(yaw[0] - a[0]), np.sin(yaw[0] - a[0])
    pos = pos @ np.array([[c0, s0], [-s0, c0]])
    pos = np.clip(pos + box_c + rng.uniform(-half_box, half_box, 2), box_c - half_box, box_c + half_box)
    gt = place(body, yaw, pos)
    return (*observe(gt, cam, rng), gt)


def kpset_dropout(conf, rng):
    if rng.random() < P_DROP_TAIL:
        conf[:, [7, 8]] = 0.0
        if rng.random() < P_DROP_FLANK / P_DROP_TAIL:
            conf[:, [3, 4]] = 0.0
    return conf


def make_batch(src, rng, s, n_seq, per_seq):
    """n_seq fresh sequences; sequence i is KPMS when floor(i * f) != floor((i - 1) * f), f = kpms_fraction
    (f = 0.5: even i, exactly as the bench run)."""
    X, Y = [], []
    f = s['kpms_fraction']
    for i in range(n_seq):
        cam = virtual_camera(rng, s)
        if np.floor(i * f) != np.floor((i - 1) * f):
            px, conf, gt = kpms_sequence(src['kpms'], rng, cam)
        else:
            px, conf, gt = dannce_sequence(src['dannce'], rng, cam)
        if s['kp_dropout']:
            conf = kpset_dropout(conf, rng)
        sel = rng.choice(len(px), per_seq, replace=False)
        X.append(lifter_features(px, conf, cam, FEAT['h_feat'])[sel]); Y.append(gt[sel, :, 2])
    return np.concatenate(X), np.concatenate(Y).astype(np.float32)


# -------------------------------------------------------------------------------------------------------- network
def build_lifter(n_in: int):
    """Same module layout as overhead_3d_utils.load_lifter (state-dict keys must match)."""
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

    return Lifter(n_in, **ARCH)


def predict_z(nets, X: np.ndarray, device: str, smooth: float = 2.0) -> np.ndarray:
    """Ensemble-mean z (fp32), clipped at 0, Gaussian-smoothed over frames (as lift_lifter)."""
    import torch
    from scipy.ndimage import gaussian_filter1d
    modes = [n.training for n in nets]
    for n in nets:
        n.eval()
    Xt = torch.from_numpy(X)
    with torch.no_grad():
        z = torch.cat([torch.stack([n(Xt[i:i + 8192].to(device)) for n in nets]).mean(0).cpu() for i in range(0, len(Xt), 8192)]).numpy()
    for n, m in zip(nets, modes):
        n.train(m)
    z = np.clip(z.astype(np.float64), 0, None)
    return gaussian_filter1d(z, smooth, axis=0, mode='nearest') if smooth else z


# ------------------------------------------------------------------------------------------------ evaluation sets
def make_synthetic(poses9: np.ndarray, calib: dict, n_seq: int = 40, seed: int = 0, noise_px: float = 1.2) -> dict:
    """bench common.make_synthetic: smooth sequences through random (canonical) poses, moving in the box floor."""
    rng = np.random.default_rng(seed)
    floor_mm = backproject(calib['floor'], np.zeros(4), calib)[:, :2]
    lo, hi = floor_mm.min(0) + 35, floor_mm.max(0) - 35
    PX, CONF, GT, SEQ = [], [], [], []
    for s in range(n_seq):
        keys = poses9[rng.integers(0, len(poses9), size=6)]
        kt = np.linspace(0, SEQ_LEN - 1, len(keys))
        t = np.arange(SEQ_LEN)
        u = np.clip(np.searchsorted(kt, t, side='right') - 1, 0, len(keys) - 2)
        w = 0.5 - 0.5 * np.cos(np.pi * (t - kt[u]) / (kt[u + 1] - kt[u]))
        body = keys[u] * (1 - w)[:, None, None] + keys[u + 1] * w[:, None, None]
        pos = np.cumsum(rng.normal(0, 1.2, (SEQ_LEN, 2)), 0); pos -= pos.mean(0)
        pos = np.clip(pos + rng.uniform(lo, hi), lo, hi)
        yaw = np.cumsum(rng.normal(0, 0.03, SEQ_LEN)) + rng.uniform(-np.pi, np.pi)
        c, sn = np.cos(yaw), np.sin(yaw)
        xy = np.stack([c[:, None] * body[..., 0] - sn[:, None] * body[..., 1], sn[:, None] * body[..., 0] + c[:, None] * body[..., 1]], -1)
        gt = np.concatenate([xy + pos[:, None], body[..., 2:3]], -1)
        px = project(gt, calib) + rng.normal(0, noise_px, (SEQ_LEN, 9, 2))
        conf = np.full((SEQ_LEN, 9), 0.9)
        drift = rng.uniform(-0.35, 0.25, SEQ_LEN)
        tb, te = project(gt[:, 6], calib), project(gt[:, 8], calib)
        px[:, 7] += drift[:, None] * np.where(drift[:, None] < 0, (px[:, 7] - tb), (te - px[:, 7]))
        bad = rng.random((SEQ_LEN, 9)) < 0.02
        px[bad] += rng.normal(0, 12, (bad.sum(), 2)); conf[bad] = rng.uniform(0.05, 0.45, bad.sum())
        PX.append(px); CONF.append(conf); GT.append(gt); SEQ.append(np.full(SEQ_LEN, s))
    return {'px': np.concatenate(PX), 'conf': np.concatenate(CONF), 'gt': np.concatenate(GT), 'seq': np.concatenate(SEQ), 'calib': calib}


def make_kpms_set(raws: List[np.ndarray], calib: dict, seed: int = 0, noise_px: float = 1.2, max_frames: Optional[int] = None) -> dict:
    """bench kpms_common.make_kpms_test(..., 'kp7'): recordings projected through the camera, trajectory shrunk by 0.27 about
    the arena centre and centred on the box floor; tail centre / end missing (confidence 0)."""
    rng = np.random.default_rng(seed)
    box_c = backproject(calib['floor'], np.zeros(4), calib)[:, :2].mean(0)
    PX, CONF, GT, SEQ = [], [], [], []
    for s, X in enumerate(raws):
        X = X[:max_frames]
        P = kpms_to_kp9(X)
        c = X[:, KPMS_NAMES.index('tail base'), :2]
        ac = 0.5 * (np.percentile(c, 1, 0) + np.percentile(c, 99, 0))
        P[..., :2] += (box_c - ac - (1 - KPMS_POS_SCALE) * (P[:, 5, :2] - ac))[:, None]
        gt = P[:, :9]
        T = len(gt)
        px = project(gt, calib) + rng.normal(0, noise_px, (T, 9, 2))
        conf = np.zeros((T, 9)); conf[:, :7] = 0.9
        bad = (rng.random((T, 9)) < 0.02) & (conf > 0)
        px[bad] += rng.normal(0, 12, (bad.sum(), 2)); conf[bad] = rng.uniform(0.05, 0.45, bad.sum())
        PX.append(px); CONF.append(conf); GT.append(gt); SEQ.append(np.full(T, s))
    return {'px': np.concatenate(PX), 'conf': np.concatenate(CONF), 'gt': np.concatenate(GT), 'seq': np.concatenate(SEQ), 'calib': calib}


def height_metrics(pred: np.ndarray, d: dict) -> Dict[str, float]:
    """z MAE over all 9 keypoints (z_MAE9; DANNCE only, KPMS has no tail GT) and over the 5 with KPMS ground truth (z_MAE5),
    nose-up/down sign accuracy, and (rearing = GT ear midpoint > 55 mm) AUC and median head heights."""
    from sklearn.metrics import roc_auc_score
    gt = d['gt']
    dz = np.abs(pred[..., 2] - gt[..., 2])
    ears = lambda P: P[:, [1, 2], 2].mean(1)
    tilt_gt, tilt_pr = gt[:, 0, 2] - ears(gt), pred[:, 0, 2] - ears(pred)
    big = np.abs(tilt_gt) > 5
    head_pr = pred[:, [0, 1, 2], 2].mean(1)
    rear = ears(gt) > 55.0
    return {'z_MAE9': dz.mean(), 'z_MAE5': dz[:, SCORED].mean(), 'z_MAE_nose': dz[:, 0].mean(), 'z_MAE_ears': dz[:, [1, 2]].mean(),
            'nose_sign_acc': float(np.mean(np.sign(tilt_pr[big]) == np.sign(tilt_gt[big]))),
            'AUC_rear': roc_auc_score(rear, head_pr) if 0 < rear.mean() < 1 else np.nan,
            'head_flat_med': np.median(head_pr[~rear]), 'head_rear_med': np.median(head_pr[rear]) if rear.any() else np.nan}


def evaluate_lifter(checkpoints: List[Union[str, os.PathLike]],
                    dannce_test_path: Optional[Union[str, os.PathLike]] = DANNCE_TEST,
                    kpms_path: Optional[Union[str, os.PathLike]] = KPMS_H5,
                    far_heights_mm: tuple = FAR_HEIGHTS_MM,
                    device: Optional[str] = None,
                    verbose: bool = True):
    """
    Score lifter checkpoints on the held-out sets, through the production loader (``overhead_3d_utils.load_lifter``).

    :param List[Union[str, os.PathLike]] checkpoints: lifter .pt files (format of ``overhead_3d_utils.load_lifter``).
    :param Optional[Union[str, os.PathLike]] dannce_test_path: DANNCE markerless_mouse_2 label3d .mat (synthetic set 'syn'). None = skip.
    :param Optional[Union[str, os.PathLike]] kpms_path: Keypoint-MoSeq open_field_3D.h5; its 3 test recordings, 7 keypoints ('kpms_test'). None = skip.
    :param tuple far_heights_mm: extra camera heights (same floor px/mm) at which both sets are re-projected. The near set is always the 288 mm benchmark camera.
    :param Optional[str] device: 'cuda' / 'cpu'; None = cuda if available.
    :param bool verbose: print the result table.
    :return: pandas.DataFrame, one row per checkpoint x set x camera height (columns: checkpoint, set, H_mm, + ``height_metrics`` keys).

    :example:
    >>> df = evaluate_lifter(checkpoints=['simba/sandbox/overhead_3d/lifter_dannce_kpms_wide.pt'])
    """
    import pandas as pd
    import torch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    for c in checkpoints:
        check_file_exist_and_readable(file_path=c)
    sets = {}
    heights = [BENCH_CAMERA['camera_height_mm']] + [float(h) for h in far_heights_mm]
    poses9 = np.stack([canonical(p) for p in dannce_poses(dannce_test_path)[:, :9]]) if dannce_test_path else None
    raws = [kpms_raw(kpms_path, r) for r in KPMS_SPLIT['test']] if kpms_path else None
    for H in heights:
        calib = camera_at(H)
        if poses9 is not None:
            sets[('syn', H)] = make_synthetic(poses9, calib)
        if raws is not None:
            sets[('kpms_test', H)] = make_kpms_set(raws, calib)
    rows = []
    for c in checkpoints:
        nets, h_feat = load_lifter(str(c), device)
        for (name, H), d in sets.items():
            z = predict_z(nets, lifter_features(d['px'], d['conf'], d['calib'], h_feat), device)
            m = height_metrics(backproject(d['px'], z, d['calib']), d)
            if name == 'kpms_test':
                m['z_MAE9'] = np.nan
            rows.append({'checkpoint': os.path.basename(str(c)), 'set': name, 'H_mm': round(H), **m})
    df = pd.DataFrame(rows)
    if verbose:
        pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
        print(df.round(3).to_string(index=False), flush=True)
    return df


# ---------------------------------------------------------------------------------------------------------- train
def train_lifter(out_path: Union[str, os.PathLike],
                 dannce_path: Optional[Union[str, os.PathLike]] = DANNCE_TRAIN,
                 kpms_path: Optional[Union[str, os.PathLike]] = KPMS_H5,
                 kpms_fraction: float = 0.5,
                 camera_h_min_mm: float = 230.4,
                 camera_h_max_mm: float = 4032.0,
                 px_per_mm_min: float = 1.3,
                 px_per_mm_max: float = 1.75,
                 p_near: float = 0.0,
                 kp_dropout: bool = True,
                 n_seeds: int = 5,
                 rounds: int = 40,
                 seed: int = 0,
                 device: Optional[str] = None,
                 evaluate: bool = True) -> dict:
    """
    Train an ensemble height lifter for ``overhead_3d_utils`` from 3D mouse poses rendered through random virtual overhead cameras.
    The defaults reproduce the shipped ``lifter_dannce_kpms_wide.pt``.

    :param Union[str, os.PathLike] out_path: output checkpoint (.pt). A JSON sidecar with the same stem is written next to it.
    :param Optional[Union[str, os.PathLike]] dannce_path: DANNCE label3d .mat with hand-labelled 3D poses (default markerless_mouse_1). None = KPMS only.
    :param Optional[Union[str, os.PathLike]] kpms_path: Keypoint-MoSeq open_field_3D.h5 (train recordings used; 'val' only for the printed log). None = DANNCE only.
    :param float kpms_fraction: fraction of training sequences drawn from KPMS (rest DANNCE). Forced to 1 / 0 when one source is None.
    :param float camera_h_min_mm: lowest virtual camera height (mm). Heights are log-uniform between min and max.
    :param float camera_h_max_mm: highest virtual camera height (mm).
    :param float px_per_mm_min: lowest floor pixels/mm of the virtual camera (focal = px/mm x H).
    :param float px_per_mm_max: highest floor pixels/mm.
    :param float p_near: fraction of sequences whose camera height is drawn from the old 0.8-1.25 x 288 mm range instead (0 = default).
    :param bool kp_dropout: randomly drop the tail centre + tail end (p 0.35) and additionally the flanks (p 0.15) per sequence, so 7- and 5-keypoint projects stay in distribution.
    :param int n_seeds: ensemble size; seeds ``seed`` .. ``seed + n_seeds - 1``.
    :param int rounds: training rounds per network; each round draws 1500 fresh 150-frame sequences (50 frames each) and runs 2 epochs.
    :param int seed: first random seed.
    :param Optional[str] device: 'cuda' / 'cpu'; None = cuda if available.
    :param bool evaluate: score the ensemble with :func:`evaluate_lifter` after training and store the table in the sidecar.
    :return: the sidecar dict (also written to ``<out_path stem>.json``).

    :example:
    >>> train_lifter(out_path=r'C:/models/lifter_custom.pt', camera_h_max_mm=6000)
    """
    import torch
    import torch.nn as nn
    timer = SimbaTimer(start=True)
    check_str(name='out_path', value=str(out_path))
    if not str(out_path).endswith('.pt'):
        raise InvalidInputError(msg=f'out_path must end with .pt, got {out_path}', source=train_lifter.__name__)
    if dannce_path is None and kpms_path is None:
        raise InvalidInputError(msg='Give at least one of dannce_path, kpms_path', source=train_lifter.__name__)
    for p in (dannce_path, kpms_path):
        if p is not None:
            check_file_exist_and_readable(file_path=p)
    check_float(name='kpms_fraction', value=kpms_fraction, min_value=0.0, max_value=1.0)
    check_float(name='camera_h_min_mm', value=camera_h_min_mm, min_value=1.0)
    check_float(name='camera_h_max_mm', value=camera_h_max_mm, min_value=camera_h_min_mm)
    check_float(name='px_per_mm_min', value=px_per_mm_min, min_value=0.01)
    check_float(name='px_per_mm_max', value=px_per_mm_max, min_value=px_per_mm_min)
    check_float(name='p_near', value=p_near, min_value=0.0, max_value=1.0)
    check_valid_boolean(value=[kp_dropout, evaluate], source=train_lifter.__name__)
    check_int(name='n_seeds', value=n_seeds, min_value=1)
    check_int(name='rounds', value=rounds, min_value=1)
    check_int(name='seed', value=seed, min_value=0)
    kpms_fraction = 1.0 if dannce_path is None else 0.0 if kpms_path is None else float(kpms_fraction)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    s = dict(dannce_path=None if dannce_path is None else os.path.abspath(dannce_path), kpms_path=None if kpms_path is None else os.path.abspath(kpms_path),
             kpms_fraction=kpms_fraction, camera_h_min_mm=float(camera_h_min_mm), camera_h_max_mm=float(camera_h_max_mm),
             px_per_mm_min=float(px_per_mm_min), px_per_mm_max=float(px_per_mm_max), p_near=float(p_near), kp_dropout=kp_dropout,
             n_seeds=n_seeds, rounds=rounds, seed=seed, device=device, out_path=os.path.abspath(out_path))
    log = []

    def say(msg):
        log.append(msg); print(msg, flush=True)

    src, val = {}, None
    if kpms_path is not None:
        src['kpms'] = [kpms_raw(kpms_path, r) for r in KPMS_SPLIT['train']]
        val = make_kpms_set([kpms_raw(kpms_path, r) for r in KPMS_SPLIT['val']], BENCH_CAMERA, seed=1, max_frames=20000)
        X_val = lifter_features(val['px'], val['conf'], val['calib'], FEAT['h_feat'])
    if dannce_path is not None:
        src['dannce'] = np.stack([canonical(p) for p in dannce_poses(dannce_path)])
    say(f'training {n_seeds} x {rounds} rounds on {device}: ' + ', '.join(f'{k}={v}' for k, v in s.items() if k not in ('dannce_path', 'kpms_path', 'out_path')))

    nets = []
    for k in range(n_seeds):
        sd = seed + k
        torch.manual_seed(sd)
        rng = np.random.default_rng(sd)
        X0, _ = make_batch(src, rng, s, 2, 4)
        net = build_lifter(X0.shape[1]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=rounds * EPOCHS * (N_SEQ * PER_SEQ // BATCH), pct_start=0.05)
        seed_timer = SimbaTimer(start=True)
        for r in range(rounds):
            X, Y = make_batch(src, rng, s, N_SEQ, PER_SEQ)
            X, Y = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
            for _ in range(EPOCHS):
                p = torch.randperm(len(X), device=device)
                for i in range(0, len(X) - BATCH + 1, BATCH):
                    j = p[i:i + BATCH]
                    loss = nn.functional.smooth_l1_loss(net(X[j]), Y[j], beta=2.0)
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if r % 10 == 9 or r == rounds - 1:
                seed_timer.stop_timer()
                msg = f'seed {sd} round {r + 1}/{rounds} loss {loss.item():.2f} {seed_timer.elapsed_time:.0f}s'
                if val is not None:
                    m = height_metrics(backproject(val['px'], predict_z([net], X_val, device), val['calib']), val)
                    msg += f" | KPMS-val z_MAE5 {m['z_MAE5']:.2f} nose {m['z_MAE_nose']:.2f} ears {m['z_MAE_ears']:.2f} sign {m['nose_sign_acc']:.3f}"
                say(msg)
        nets.append(net)
    if val is not None:
        m = height_metrics(backproject(val['px'], predict_z(nets, X_val, device), val['calib']), val)
        say('ENSEMBLE KPMS-val ' + ' '.join(f'{k} {v:.3f}' for k, v in m.items()))

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    meta = {'settings': s, 'recipe': 'overhead_3d/train_lifter.py (farcam recipe: DANNCE m1 + KPMS, log-uniform camera height, log(H/300) feature)'}
    ck = {'states': [{k: (v.half() if v.is_floating_point() else v).cpu() for k, v in n.state_dict().items()} for n in nets],
          'n_in': int(X0.shape[1]), 'arch': dict(ARCH), 'feat': dict(FEAT),
          'cfg': [f'kpms_fraction={kpms_fraction}', 'kp_dropout' if kp_dropout else 'no_kp_dropout',
                  f'H loguniform({camera_h_min_mm:g}, {camera_h_max_mm:g}) mm p_near={p_near:g}', f'px/mm U({px_per_mm_min:g}, {px_per_mm_max:g})'],
          'kpms_split': KPMS_SPLIT, 'meta': meta}
    torch.save(ck, out_path)
    timer.stop_timer()
    import h5py
    import scipy
    import sklearn
    side = {**meta, 'checkpoint': os.path.abspath(out_path), 'checkpoint_sha256': file_sha256(out_path),
            'data_sha256': {os.path.basename(p): file_sha256(p) for p in (dannce_path, kpms_path) if p is not None},
            'kpms_split': KPMS_SPLIT, 'n_in': int(X0.shape[1]), 'arch': ARCH, 'feat': FEAT,
            'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__,
                            'torch': torch.__version__, 'cuda': torch.version.cuda, 'scipy': scipy.__version__, 'h5py': h5py.__version__,
                            'sklearn': sklearn.__version__, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None},
            'train_seconds': timer.elapsed_time, 'log': log}
    if evaluate:
        df = evaluate_lifter(checkpoints=[out_path], device=device, kpms_path=kpms_path or KPMS_H5)
        side['evaluation'] = json.loads(df.to_json(orient='records'))
    with open(os.path.splitext(out_path)[0] + '.json', 'w') as f:
        json.dump(side, f, indent=2, default=float)
    stdout_success(msg=f'Lifter saved to {out_path} (+ .json sidecar)', elapsed_time=timer.elapsed_time_str, source=train_lifter.__name__)
    return side


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train (default) or --evaluate an overhead_3d height lifter. Defaults = shipped lifter_dannce_kpms_wide.pt.')
    ap.add_argument('--out', help='output checkpoint .pt (a .json sidecar is written next to it)')
    ap.add_argument('--evaluate', nargs='+', metavar='CKPT', help='only score these checkpoints (no training)')
    ap.add_argument('--dannce', default=DANNCE_TRAIN, help="DANNCE label3d .mat training poses; 'none' to skip")
    ap.add_argument('--kpms', default=KPMS_H5, help="Keypoint-MoSeq open_field_3D.h5; 'none' to skip")
    ap.add_argument('--kpms_fraction', type=float, default=0.5, help='fraction of training sequences from KPMS')
    ap.add_argument('--h_min_mm', type=float, default=230.4, help='lowest virtual camera height (mm), log-uniform')
    ap.add_argument('--h_max_mm', type=float, default=4032.0, help='highest virtual camera height (mm)')
    ap.add_argument('--ppm_min', type=float, default=1.3, help='lowest floor px/mm')
    ap.add_argument('--ppm_max', type=float, default=1.75, help='highest floor px/mm')
    ap.add_argument('--p_near', type=float, default=0.0, help='fraction of cameras from the old 230-360 mm range')
    ap.add_argument('--no_kp_dropout', action='store_true', help='disable tail / flank keypoint-set dropout')
    ap.add_argument('--n_seeds', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default=None, help='cuda / cpu (default: cuda if available)')
    ap.add_argument('--no_eval', action='store_true', help='skip the evaluation after training')
    ap.add_argument('--far_heights', default='1000,2000,4000', help='--evaluate: far camera heights (mm)')
    a = ap.parse_args()
    none = lambda p: None if str(p).lower() == 'none' else p
    if a.evaluate:
        evaluate_lifter(checkpoints=a.evaluate, kpms_path=none(a.kpms), far_heights_mm=tuple(float(h) for h in a.far_heights.split(',') if h), device=a.device)
    elif a.out:
        train_lifter(out_path=a.out, dannce_path=none(a.dannce), kpms_path=none(a.kpms), kpms_fraction=a.kpms_fraction,
                     camera_h_min_mm=a.h_min_mm, camera_h_max_mm=a.h_max_mm, px_per_mm_min=a.ppm_min, px_per_mm_max=a.ppm_max,
                     p_near=a.p_near, kp_dropout=not a.no_kp_dropout, n_seeds=a.n_seeds, rounds=a.rounds, seed=a.seed,
                     device=a.device, evaluate=not a.no_eval)
    else:
        ap.error('give --out (train) or --evaluate CKPT [CKPT ...]')
