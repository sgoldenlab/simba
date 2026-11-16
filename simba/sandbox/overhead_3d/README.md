# overhead_3d: 3D keypoints from ONE overhead camera (SimBA projects)

This is the consolidated, optimised version of the work in `../zenith3d/` (handover, literature and references) and `../zenith_height/` (research, benchmark and training).
- Sandbox only. Note that `simba/sandbox/` is git-ignored.
- Never committed by Claude.

## Files

| file | what |
|---|---|
| `overhead_3d_utils.py` | camera model, calibration (box auto-calibration, or a known camera height), keypoint checks (anatomy layout, left/right handedness, proportion normalisation), numba kernels (`rts_smooth`, `viterbi_two_state`), and the methods `lift_geometry`, `lift_lifter`, `lift_fused` plus the `lift()` dispatcher |
| `overhead_3d_lifter.py` | `Overhead3DLifter(ConfigReader)`: SimBA project -> `csv/pose_3d/<video>.csv` + `_summary.csv`. Also has a CLI |
| `overhead_3d_visualizer.py` | `Overhead3DVisualizer(ConfigReader)`: original + re-projected 2D \| 3D, all animals, rendered in parallel. `view='overhead'` (oblique 3D) or `view='pov'`: one panel per animal from just behind and above its head (20 mm back, 12 mm above the head and rising with it, aimed where the head points; heading = nose - centre, vector-smoothed 0.25 s), with the partner as a keypoint skeleton. Use 180 deg with this close camera: during attacks the closest keypoints are ~14 mm apart, and a camera further back (55 mm) made fighting mice look ~9 cm apart. At 20/12 mm the partner stays in the 180-deg view in ~80% of attack frames. The viewer itself is not drawn (its snout as a reference was tried and rejected). `pov_camera='nose'` puts the camera on the snout tip; it was tried and is NOT the default, because the partner is fully in view in only 27% of attack frames (vs 83%), since fighting mice are mostly beside the head. `pov_body=True` (default) draws a semi-transparent body under each partner skeleton: the solid spanned by its head/trunk keypoints (they are tracked on the animal's edge) down to the floor. A version inflated by typical half-widths was rejected because it double-counted the width. in attacks the closest keypoints are ~14 mm apart but the body centres ~56 mm, so skeletons alone looked separated while the bodies touch. Partner edges are clipped at the view edge rather than dropped, so no joints float when the animals touch. `pov_fov_deg` (default 90): up to 120 gives normal square panels; up to 180 gives a cylindrical panorama with 2:1 panels and left/ahead/right ticks, since a pinhole camera cannot show 180 deg. This was chosen over eye-level first person after 4 design rounds (`../zenith_height/bench/pov_lab.py`): eye-level views crop or blow up the partner, because mice interact at 0-5 cm, and the nose-ear direction jitters by up to 40 deg/frame. The chosen view keeps the partner fully on screen ~70% of frames at a readable size (a 95 mm-high camera gave 80% but looked down from too high). With `arena_mm`, the floor is located in each video by `fit_arena_rect`: the known-aspect rectangle with the most bedding texture inside, accepted if >= 90% of body centres fall inside, else placed by the animals' range. `arena_top_mm` / `wall_height_mm` draw the tapered walls |
| `train_lifter.py` | reproducible retraining + evaluation of the lifter (one command; see "Retraining") |
| `run_maplight.py` | the exact maplight_ri run: lift all videos, render overhead / POV / 2 x 2 for three examples |
| `lifter_dannce_kpms_wide.pt` | **default** lifter (`LIFTER_DEFAULT`): DANNCE mouse 1 + Keypoint-MoSeq 3D, keypoint dropout, cameras 0.23-4 m, log camera-height feature (= `bench/methods/farcam_*`, rebuilt by `train_lifter.py`) |
| `lifter_dannce_kpms.pt` | previous default, near cameras only (250-410 mm) (= `bench/methods/kpms_lifter_b.pt`) |
| `lifter_dannce.pt` | DANNCE-only lifter, used by `fused` (= `zenith3d/lifter_mlp.pt`) |

## Usage

```python
from simba.sandbox.overhead_3d.overhead_3d_lifter import Overhead3DLifter
from simba.sandbox.overhead_3d.overhead_3d_visualizer import Overhead3DVisualizer

if __name__ == '__main__':   # REQUIRED on Windows: the classes use multiprocessing (spawn)
    # Ressler-type box: video cropped at the rim -> auto-calibration; wall height estimated from body reach
    Overhead3DLifter(config_path=r'I:/mitra/nick_ressler/project_folder/project_config.ini', calibration='box').run()
    # any other rig: give the camera height above the floor; px/mm, fps and resolution come from the project's video_info
    Overhead3DLifter(config_path=r'D:/troubleshooting/maplight_ri/project_folder/project_config.ini',
                     calibration='camera_height', camera_height_mm=1500,   # None = unknown (assumed, scaled by animal size)
                     bp_map={'Nose': 'RIGHT_EAR', 'Left_Ear': 'NOSE', 'Right_Ear': 'LEFT_EAR',
                             'Center': 'LEFT_SIDE', 'Left_Side': 'CENTER'}).run()
    # maplight Innovive cage: interior floor 522.6 cm2 (~300 x 174 mm at the bedding aspect), walls 127 mm
    for view in ('overhead', 'pov'):
        Overhead3DVisualizer(config_path=..., video_names=['Trial_10_C24_D1_3'], view=view, arena_mm=(300, 174), arena_top_mm=(355, 215), wall_height_mm=127).run()
```

**maplight_ri setup (2026-11-13):**
- **pixels/mm:** the cage dimensions confirm 1.4 px/mm to within about 4%.
- **Camera height:** the rim (14 cm up) appears only about 5% larger than the floor, so the camera is about 1-3 m up and the view is nearly orthographic. Hence `camera_height_mm=1500` with the wide lifter. Its heights are stable from 300 mm to 2.5 m.
- **The cage is tapered:** the floor is ~300 x 174 mm and the rim opening ~355 x 215 mm (top outside is 373 x 234). Noses reaching up the walls are legitimately outside the floor footprint, so draw it with `arena_top_mm`.
- **Automatic per-video px/mm correction from the animals' range was tried and REMOVED.** It flagged 16 videos, but on inspection most were tracking jumps or wall climbing, not zoom: the cage at 1.4 px/mm already fit the bedding. Only Trial_14 is genuinely larger (~15%). A per-video floor-corner click is the reliable fix.
- **Trial_14 is zoomed differently:** the cage looks larger in that video, so a single px/mm for all videos is not exact. Ask the lab whether the camera or zoom changed between sessions.
- **Outputs:** `D:\troubleshooting\maplight_ri\project_folder\pose_3d_preview\` (`pose_3d\` CSVs for 236 videos; `<video>.mp4` overhead and `<video>_pov.mp4` point-of-view videos for D1_1, D1_3 and D2_1).

- **Inputs** come from the SimBA project:
  - body-parts from `animal_bp_dict`;
  - pixels/mm, fps and resolution from `logs/video_info.csv`, via `read_video_info`, which raises if a video is missing;
  - pose from `csv/outlier_corrected_movement_location`;
  - videos from `videos/`.
- **Body-part mapping:** body-parts are matched to the roles Nose, Left_Ear, Right_Ear, Left_Side, Right_Side, Center, Tail_Base, Tail_Center and Tail_End by name; `bp_map` overrides. Nose, the ears, Center and Tail_Base are required. Missing optional roles get zero confidence; 7-keypoint projects lose almost nothing.
- **Outputs:**
  - Per frame and animal: `<animal>_<role>_x/_y/_z` in mm (x, y on the floor plane relative to the optical axis in image orientation; z is height above the floor), `<animal>_<role>_p`, and `<animal>_quality_ok` (nose, ears, centre and tail base confidence >= 0.5).
  - `_summary.csv`, one row per video and animal: calibration, swaps, quality and median head height. The visualizer reads its camera from this file.
- **Not in this folder:**
  - The benchmark, training code and the "fancy" body-volume prototype (`../zenith_height/bench/fancy_render.py`) stay in `../zenith_height/`.
  - The literature and references are in `../zenith3d/README.md`.

## Methods (height MAE)

| `method` | synthetic (DANNCE mouse 2, held out) | real 3D motion (Keypoint-MoSeq, held out) | notes |
|---|---|---|---|
| `lifter` (default, `lifter_dannce_kpms_wide.pt`) | 4.6 mm | **3.75 mm** | best on real data; robust to missing tail points; seeds agree best. Cameras 0.23-4 m: 4.2-5.4 mm at 1-4 m, where the previous lifter extrapolated to 11-61 mm |
| `lifter` with `weights='dannce_kpms'` (previous default) | 4.6 mm | 4.9 mm | near cameras only |
| `fused` | **4.0 mm** | 8.3 mm | geometry + DANNCE lifter ridge stack; the geometry over-reads hunched mice as head-up; this is what produced the delivered Ressler CSVs |
| `geometry` | 6.4 mm | 12.9 mm | numpy/numba only, no torch |

The 3D covers the trunk and head only: no limbs, the tail lies on the floor, and climbing cannot be represented. The lifters are trained on mice; rats are out of scope until rat 3D data (e.g. Rat7M) is added.

## Checks done before lifting (every animal, every video)

1. **`check_keypoint_layout`**: the nose must be the most anterior point, Center near the midline, and ears/flanks on opposite sides. It warns when the names don't match the data.
   - Example: maplight_ri's body-part names are SimBA's default order renamed by position, so the `RIGHT_EAR` column is the nose and `LEFT_SIDE` is the centre. That is what the `bp_map` above fixes.
   - The resident (white mouse) in maplight still triggers warnings in some videos because its ear tracking is poor. That is a tracking issue, not a naming one.
2. **`fix_handedness`**: swaps ears and/or flanks labelled the other way round from the lifters' convention: cross(nose - tail base, left - right) < 0, seen from above.
3. **`normalise_proportions`** (lifter input only): rescales each 2D segment to the lifter's training domain, using the Ressler SLEAP medians. x and y are always re-projected from the real points.

## Speed (RTX 4070, 32 cores; 20-min 30 fps video = 36,000 frames)

| step | before (zenith3d) | now | how |
|---|---|---|---|
| geometry | 3.3 s | 0.04 s (~1M fps) | numba RTS smoother + Viterbi (was 144k Python 2x2 inversions); identical to 1e-9 mm |
| lifter | 1.8 s (CPU) | 0.17 s | GPU, fp16 autocast (mean change 0.02 mm, max 0.4 mm) |
| fused | 5.4 s | 0.49 s | both of the above |
| whole Ressler project, box mode, 91 videos, 3.3M frames | ~10+ min | **79 s** | pooled calibration + streamed read -> GPU lift -> async write |
| whole maplight project, 236 videos x 2 animals, 4.2M animal-frames | (n/a) | **42 s** | as above |
| 2D-vs-3D video, 20 min | ~10 min | **66 s** (16 cores) | frame chunks in parallel, `concatenate_videos_in_folder` |

- The new code reproduces the old pipeline: the Ressler `fused` z matches the delivered CSVs to 0.04 mm, with `normalise=False` and `fix_left_right=False`.
- The worker pool defaults to min(cores, 8). With 32 workers each importing SimBA, the run hit a MemoryError; more workers mostly cost RAM.

## Gotchas
- Run from a script with an `if __name__ == '__main__':` guard on Windows; spawned workers re-import the main module.
- Scripts outside the repo import the *installed* simba. Add `sys.path.insert(0, 'C:/projects/simba/simba')`.
- `KMP_DUPLICATE_LIB_OK=TRUE` may be needed in simba_310 (torch OpenMP clash).
- For `calibration='camera_height'` the optical centre is the image centre. That is fine for centred crops; pass a better centre via `manual_calib` if needed.

## Retraining

`train_lifter.py` retrains the lifter in one command. The defaults reproduce the shipped default `lifter_dannce_kpms_wide.pt`.

```
set KMP_DUPLICATE_LIB_OK=TRUE
python simba/sandbox/overhead_3d/train_lifter.py --out C:/models/lifter_new.pt
python simba/sandbox/overhead_3d/train_lifter.py --evaluate simba/sandbox/overhead_3d/lifter_dannce_kpms_wide.pt C:/models/lifter_new.pt
```

- **Outputs:**
  - `lifter_new.pt`, in the format `load_lifter` reads (`n_in`, `arch`, `feat` incl. `h_feat`, 5 fp16 state dicts);
  - `lifter_new.json`: every setting, sha256 of the data files and the checkpoint, package versions, the training log and the evaluation table.
- **Python:** `train_lifter(...)` takes the same settings.
- **Code dependencies:** only `overhead_3d_utils` (`lifter_features`, `load_lifter`), so training and inference features cannot drift apart.
  - The recipe functions were copied from `../zenith_height/bench/methods/` (`lifter_mlp_train`, `kpms_lifter_train`, `farcam_lifter_train`). They keep the same random-number order, so one seed draws exactly the same training batches as the bench.
  - The benchmark camera (Ressler box 4) is stored as constants, so the raw video on `I:` is not needed.
- **Data:** the data files stay in `../zenith_height/bench/`, and the default paths point there.

| argument | default | what it is for |
|---|---|---|
| `--out` | (required) | checkpoint path; the `.json` sidecar is written next to it |
| `--dannce` | `bench/label3d_dannce.mat` | hand-labelled 3D poses (DANNCE mouse 1); `none` = KPMS only |
| `--kpms` | `bench/data_kpms/open_field_3D.h5` | real 3D motion (Keypoint-MoSeq; train recordings only, `val` for the printed log); `none` = DANNCE only |
| `--kpms_fraction` | 0.5 | share of training sequences from KPMS |
| `--h_min_mm`, `--h_max_mm` | 230.4, 4032 | virtual camera height range, log-uniform (= 288 mm x 0.8-14) |
| `--ppm_min`, `--ppm_max` | 1.3, 1.75 | floor px/mm of the virtual camera (focal = px/mm x H): the animal's apparent size in pixels |
| `--p_near` | 0 | share of cameras drawn from the old 230-360 mm range instead |
| `--no_kp_dropout` | off | turn off the random tail / flank dropout that keeps 7- and 5-keypoint projects in distribution |
| `--n_seeds` | 5 | ensemble size (seeds `seed` .. `seed + n - 1`) |
| `--rounds` | 40 | rounds per network: 1500 fresh sequences x 50 frames, 2 epochs |
| `--seed` | 0 | first seed |
| `--device` | cuda if available | `cuda` / `cpu` |
| `--no_eval` | off | skip the evaluation after training |
| `--evaluate CKPT ...` | | score checkpoints only: DANNCE mouse 2 synthetic (`syn`) and the 3 held-out KPMS recordings with 7 keypoints (`kpms_test`), at H = 288 mm and `--far_heights` (1, 2, 4 m) |

**When to retrain:**
- a new species (e.g. rats, which need rat 3D data);
- a different keypoint definition, e.g. a Center far from the mid-spine, or other body-parts;
- cameras outside 0.23-4 m.

A new cage or arena size does **not** need retraining: the inputs are metric (mm, relative to Center), and only the camera height and px/mm matter.

**Runtime (RTX 4070):** ~5 min of training for the defaults (5 seeds x ~1 min, synthetic data generated on the CPU), plus ~1 min of evaluation. On CPU it is much slower.

**Reproduction check (2026-11-13):** the defaults retrained from scratch (z MAE, mm):

| set (camera height) | shipped `lifter_dannce_kpms_wide.pt` | retrained |
|---|---|---|
| DANNCE mouse 2 synthetic, all 9 keypoints (288 mm) | 4.63 | 4.63 |
| KPMS test, 7 keypoints, 5 scored (288 mm) | 3.75 | 3.75 |
| DANNCE mouse 2, 5 keypoints (1 / 2 / 4 m) | 5.39 / 5.42 / 5.44 | 5.40 / 5.44 / 5.47 |
| KPMS test (1 / 2 / 4 m) | 4.24 / 4.33 / 4.43 | 4.26 / 4.36 / 4.47 |

- The training batches are bit-identical to the bench run. The weights differ only through GPU non-determinism.
- KPMS-val is 4.75 retrained vs 4.82 shipped. The seed spread (single seeds 4.2-5.7) is much larger than either gap.

**To make a new model the default:**
1. Copy the `.pt` into this folder.
2. Add a key to `WEIGHTS` in `overhead_3d_utils.py`.
3. Set `LIFTER_DEFAULT` to that key, which is what `Overhead3DLifter` uses. For one-off calls, pass `weights='<key>'` to `lift_lifter`.

`fused` keeps using `dannce`, because its ridge stack was fit on that lifter.

## Generalisation test across D:\troubleshooting projects (2026-11-13)

16 rodent projects were run with `camera_height_mm=None` (unknown) and 60-s previews were rendered. Outputs are in `D:\troubleshooting\overhead_3d_preview\`, with the overview in `_projects_summary.csv`. The batch script is in the session scratchpad (`run_all.py`); it uses patched config copies.

- **Most failures happened before 3D:**
  - configs still pointing at `C:\troubleshooting`;
  - config body-parts not matching the pose-file columns (lbl_frm_rng, pose_estimation, super_animal_import, sleap_two_animals, mouse_open_field);
  - videos missing from video_info;
  - two projects with no ears or tail base (hand tracking);
  - pose files that do not line up with their video (RAT_NOR, boris_test, multi_animal_dlc_two_c57): the keypoints are off the animal.
- **pixels/mm is often wrong** (implied body lengths 11-266 mm). Absolute mm cannot be trusted without a real calibration. Size normalisation (unknown camera height = 300 mm x apparent size) keeps the lifter in range, but the outputs are in the project's own (wrong) mm units.
- **Where the 2D is on the animal**, the lifter ran unchanged on every keypoint set, rig and animal count: cue_light, meberled, mouse_open_field, spontaneous_alternation, two_animals_sleap, sleap_two_animals and mitra. Postures look plausible.
- **Whole-arena previews are too small** to judge posture.
- **Fixes added during the test:**
  - body-part names taken from the file when the config disagrees;
  - skip (with a warning) files missing from video_info;
  - trackers without confidences;
  - derived Center;
  - more name variants;
  - unknown camera height;
  - visualizer capped at 8 workers plus a timeout (16 x 1080p workers ran out of RAM and hung the pool).
- **Next:**
  - a pose-to-video alignment check (keypoints on foreground?) so mismatched projects are flagged, not lifted;
  - a follow-camera / zoom in the visualizer;
  - a calibration helper (click the arena + enter its size and height) instead of trusting video_info px/mm.
