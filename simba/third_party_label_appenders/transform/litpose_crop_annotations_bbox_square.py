import glob
import os
import random
import shutil
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.utils.checks import (check_if_dir_exists, check_int,
                                check_valid_boolean)
from simba.utils.errors import InvalidInputError
from simba.utils.printing import (SimbaTimer, stdout_information,
                                  stdout_success, stdout_warning)
from simba.utils.read_write import get_fn_ext


class CropLPAnnotationsBboxSquare:
    """
    Creates a cropped Lightning Pose project where each labeled image is
    cropped to a square region around the keypoint bounding box, then resized
    to crop_size. This produces training images that match the inference
    pipeline's crop-and-resize behavior.

    Unlike :class:`~simba.third_party_label_appenders.transform.litpose_crop_annotations.CropLPAnnotations`,
    which takes a fixed-size crop centered on the keypoint centroid, this class
    computes a tight bounding box around the keypoints, pads it by a fraction,
    extends the shorter side to make a square, and resizes to ``crop_size``.

    .. seealso::
       :class:`~simba.third_party_label_appenders.transform.litpose_crop_annotations.CropLPAnnotations` Fixed-size center crop around keypoint centroid.

    :param str lp_project_dir:  Root of the source LP project.
    :param str save_dir:        Root of the new cropped LP project.
    :param Tuple[int, int] crop_size: Output size (width, height), e.g. (512, 512).
    :param float bbox_pad_frac: Fraction to pad the bbox on each side (default 0.15 = 15%).
    :param Optional[Union[bool, int]] visualize: Save annotated overlays for QC.
    :param bool verbose: If True, print per-frame progress (frame i/N within view j/M) as each image is cropped. Default False.
    """

    def __init__(self,
                 lp_project_dir: str,
                 save_dir: str,
                 crop_size: Tuple[int, int] = (512, 512),
                 bbox_pad_frac: float = 0.15,
                 visualize: Optional[Union[bool, int]] = None,
                 verbose: bool = False):

        check_if_dir_exists(in_dir=lp_project_dir)
        check_int(name="crop_size width", value=crop_size[0], min_value=1)
        check_int(name="crop_size height", value=crop_size[1], min_value=1)
        check_valid_boolean(value=verbose, source=f"{self.__class__.__name__} verbose")
        self.lp_project_dir = lp_project_dir
        self.save_dir = save_dir
        self.crop_size = crop_size
        self.bbox_pad_frac = bbox_pad_frac
        self.visualize = visualize
        self.verbose = verbose
        self.csv_paths = sorted([os.path.join(lp_project_dir, f) for f in os.listdir(lp_project_dir) if f.startswith("CollectedData_") and f.endswith(".csv")])
        if len(self.csv_paths) == 0:
            raise InvalidInputError(msg=f"No CollectedData_*.csv files found in {lp_project_dir}.")
        check_if_dir_exists(in_dir=os.path.join(lp_project_dir, "labeled-data"))

    @staticmethod
    def _get_bbox_for_image(img_rel: str, img_h: int, img_w: int,
                            xs: np.ndarray, ys: np.ndarray, valid: np.ndarray):
        """Get a bounding box for this image from keypoint coordinates."""
        if np.any(valid):
            x_valid, y_valid = xs[valid], ys[valid]
            x1 = int(np.floor(np.min(x_valid)))
            y1 = int(np.floor(np.min(y_valid)))
            x2 = int(np.ceil(np.max(x_valid)))
            y2 = int(np.ceil(np.max(y_valid)))
            return (x1, y1, x2, y2)

        cx, cy = img_w // 2, img_h // 2
        half = min(img_w, img_h) // 2
        return (cx - half, cy - half, cx + half, cy + half)

    def _bbox_to_square_crop(self, x1, y1, x2, y2, img_h, img_w):
        """Pad bbox by bbox_pad_frac, extend shorter side to make square, clamp to image bounds."""
        bw, bh = x2 - x1, y2 - y1
        px = int(bw * self.bbox_pad_frac)
        py = int(bh * self.bbox_pad_frac)
        x1p = x1 - px
        y1p = y1 - py
        x2p = x2 + px
        y2p = y2 + py

        pw, ph = x2p - x1p, y2p - y1p
        side = max(pw, ph)
        cx = (x1p + x2p) / 2.0
        cy = (y1p + y2p) / 2.0
        half = side / 2.0

        crop_x1 = int(np.floor(cx - half))
        crop_y1 = int(np.floor(cy - half))
        crop_x2 = crop_x1 + side
        crop_y2 = crop_y1 + side

        if crop_x1 < 0:
            crop_x1, crop_x2 = 0, side
        if crop_y1 < 0:
            crop_y1, crop_y2 = 0, side
        if crop_x2 > img_w:
            crop_x2, crop_x1 = img_w, img_w - side
        if crop_y2 > img_h:
            crop_y2, crop_y1 = img_h, img_h - side

        crop_x1 = max(0, int(crop_x1))
        crop_y1 = max(0, int(crop_y1))
        crop_x2 = min(img_w, int(crop_x2))
        crop_y2 = min(img_h, int(crop_y2))

        return crop_x1, crop_y1, crop_x2, crop_y2

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)
        timer = SimbaTimer(start=True)
        self._copy_project_files(lp_project_dir=self.lp_project_dir, save_dir=self.save_dir)
        self._update_config_paths(lp_project_dir=self.lp_project_dir, save_dir=self.save_dir)
        drop_positions = self._find_all_nan_positions(csv_paths=self.csv_paths)
        if drop_positions:
            stdout_warning(msg=f"Dropping {len(drop_positions)} row position(s) all-NaN in every view: {sorted(drop_positions)}")
        viz_candidates = []
        n_views = len(self.csv_paths)
        if self.verbose:
            stdout_information(msg=f"Cropping Lightning Pose annotations across {n_views} view(s)...",
                               source=self.__class__.__name__)
        for view_idx, csv_path in enumerate(self.csv_paths):
            self._process_csv(csv_path=csv_path,
                              lp_project_dir=self.lp_project_dir,
                              save_dir=self.save_dir,
                              crop_size=self.crop_size,
                              viz_candidates=viz_candidates,
                              drop_positions=drop_positions,
                              view_idx=view_idx,
                              n_views=n_views)
        if self.visualize and len(viz_candidates) > 0:
            viz_dir = os.path.join(self.save_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            if isinstance(self.visualize, int) and not isinstance(self.visualize, bool):
                sample = random.sample(viz_candidates, min(self.visualize, len(viz_candidates)))
            else:
                sample = viz_candidates
            for cropped_path, new_xs, new_ys, bp_names, viz_fn in sample:
                viz_img = cv2.imread(cropped_path)
                if viz_img is None:
                    continue
                for bp_idx in range(len(bp_names)):
                    bx, by = new_xs[bp_idx], new_ys[bp_idx]
                    if np.isnan(bx) or np.isnan(by):
                        continue
                    pt = (int(round(bx)), int(round(by)))
                    cv2.circle(viz_img, pt, 4, (0, 0, 255), -1)
                    cv2.putText(viz_img, bp_names[bp_idx], (pt[0] + 6, pt[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(viz_dir, viz_fn), viz_img)
            stdout_success(msg=f"Saved {len(sample)} visualizations in {viz_dir}")
        timer.stop_timer()
        stdout_success(msg=f"Cropped LP annotations saved in {self.save_dir}", elapsed_time=timer.elapsed_time_str)

    def _process_csv(self, csv_path, lp_project_dir, save_dir, crop_size, viz_candidates, drop_positions,
                     view_idx=0, n_views=1):
        _, csv_fn, csv_ext = get_fn_ext(filepath=csv_path)
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        bp_names = [df.columns[i][1] for i in range(0, len(df.columns), 2)]
        n_frames = len(df)
        out_rows = []
        for row_pos in range(n_frames):
            if self.verbose:
                stdout_information(msg=f"View {view_idx + 1}/{n_views} ({csv_fn}): processing frame {row_pos + 1}/{n_frames}...",
                                   source=self.__class__.__name__)
            if row_pos in drop_positions:
                continue
            idx = df.index[row_pos]
            coords = df.loc[idx].values.astype(float)
            xs = coords[0::2]
            ys = coords[1::2]
            valid = ~np.isnan(xs) & ~np.isnan(ys)

            img_rel = str(idx)
            img_path = os.path.join(lp_project_dir, img_rel.replace("/", os.sep))
            if not os.path.isfile(img_path):
                stdout_warning(msg=f"{csv_fn}: skipped row_pos={row_pos} (image not found)")
                continue
            img = cv2.imread(img_path)
            if img is None:
                stdout_warning(msg=f"{csv_fn}: skipped row_pos={row_pos} (cv2.imread returned None)")
                continue
            h, w = img.shape[:2]

            bbox = self._get_bbox_for_image(img_rel=img_rel, img_h=h, img_w=w, xs=xs, ys=ys, valid=valid)
            crop_x1, crop_y1, crop_x2, crop_y2 = self._bbox_to_square_crop(
                bbox[0], bbox[1], bbox[2], bbox[3], h, w
            )

            cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_h, crop_w = cropped.shape[:2]

            scale_x = crop_size[0] / crop_w
            scale_y = crop_size[1] / crop_h
            resized = cv2.resize(cropped, crop_size, interpolation=cv2.INTER_LINEAR)

            out_img_path = os.path.join(save_dir, img_rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
            cv2.imwrite(out_img_path, resized)

            new_coords = coords.copy()
            new_xs = np.where(np.isnan(xs), np.nan, (xs - crop_x1) * scale_x)
            new_ys = np.where(np.isnan(ys), np.nan, (ys - crop_y1) * scale_y)
            new_coords[0::2] = new_xs
            new_coords[1::2] = new_ys
            out_rows.append((idx, new_coords))

            if self.visualize:
                parts = img_rel.replace("/", os.sep).split(os.sep)
                viz_fn = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else parts[-1]
                viz_candidates.append((out_img_path, new_xs.copy(), new_ys.copy(), list(bp_names), viz_fn))

        if len(out_rows) == 0:
            stdout_warning(msg=f"No valid rows in {csv_fn}{csv_ext}.")
            return

        indices, data = zip(*out_rows)
        out_df = pd.DataFrame(np.array(data), index=list(indices), columns=df.columns)
        out_df.index.name = df.index.name
        out_csv_path = os.path.join(save_dir, f"{csv_fn}{csv_ext}")
        out_df.to_csv(out_csv_path)
        stdout_success(msg=f"Saved {out_csv_path} ({len(out_df)} rows).")

    @staticmethod
    def _copy_project_files(lp_project_dir: str, save_dir: str):
        COPY_DIRS = ("configs", "calibrations", "models")
        COPY_EXTS = (".yaml", ".yml", ".sh", ".json", ".jsonl", ".zip", ".txt")
        for d in COPY_DIRS:
            src = os.path.join(lp_project_dir, d)
            dst = os.path.join(save_dir, d)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                stdout_success(msg=f"Copied directory {d}/", source="CropLPAnnotations")
        for entry in os.listdir(lp_project_dir):
            src_path = os.path.join(lp_project_dir, entry)
            if not os.path.isfile(src_path):
                continue
            _, ext = os.path.splitext(entry)
            if ext.lower() in COPY_EXTS:
                shutil.copy2(src_path, os.path.join(save_dir, entry))
                stdout_success(msg=f"Copied {entry}", source="CropLPAnnotations")

    @staticmethod
    def _to_posix_path(path: str) -> str:
        p = path.replace("\\", "/")
        if len(p) >= 2 and p[1] == ":":
            p = p[2:]
        return p

    def _update_config_paths(self, lp_project_dir: str, save_dir: str):
        yaml_files = glob.glob(os.path.join(save_dir, "**", "*.yaml"), recursive=True)
        yaml_files += glob.glob(os.path.join(save_dir, "**", "*.yml"), recursive=True)
        if len(yaml_files) == 0:
            return
        old_posix = self._to_posix_path(lp_project_dir)
        new_posix = self._to_posix_path(save_dir)
        VIDEO_KEYS = ("video_dir", "test_videos_directory")
        n_updated = 0
        for yaml_path in yaml_files:
            with open(yaml_path, "r") as f:
                lines = f.readlines()
            changed = False
            for i, line in enumerate(lines):
                if old_posix not in line:
                    continue
                key = line.split(":")[0].strip() if ":" in line else ""
                if key in VIDEO_KEYS:
                    continue
                lines[i] = line.replace(old_posix, new_posix)
                changed = True
            if changed:
                with open(yaml_path, "w") as f:
                    f.writelines(lines)
                n_updated += 1
        if n_updated > 0:
            stdout_success(msg=f"Updated paths in {n_updated} config file(s)", source="CropLPAnnotations")

    @staticmethod
    def _row_all_nan(coords: np.ndarray) -> bool:
        xs, ys = coords[0::2], coords[1::2]
        return not np.any(~np.isnan(xs) & ~np.isnan(ys))

    @staticmethod
    def _find_all_nan_positions(csv_paths):
        per_csv_nan = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
            nan_set = set()
            for row_pos in range(len(df)):
                coords = df.iloc[row_pos].values.astype(float)
                if CropLPAnnotationsBboxSquare._row_all_nan(coords):
                    nan_set.add(row_pos)
            per_csv_nan.append(nan_set)
        if not per_csv_nan:
            return set()
        common = per_csv_nan[0]
        for s in per_csv_nan[1:]:
            common = common & s
        return common


#if __name__ == "__main__":
# cropper = CropLPAnnotationsBboxSquare(lp_project_dir=r"H:\sina\project_0609_5cam_0626\project_0609_5cam_0626",
#                                               save_dir=r"H:\sina\project_0609_5cam_0626_cropped",
#                                               bbox_pad_frac=0.25,
#                                               visualize=100,
#                                               verbose=True)
# cropper.run()
