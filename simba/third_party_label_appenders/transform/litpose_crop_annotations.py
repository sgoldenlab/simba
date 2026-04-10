import glob
import os
import shutil
import random
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.utils.checks import (check_if_dir_exists, check_int)
from simba.utils.errors import InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success, stdout_warning
from simba.utils.read_write import get_fn_ext


class CropLPAnnotations:
    """
    Creates a new, self-contained Lightning Pose project from an existing one where
    every labeled image is cropped to a fixed size around the annotated animal. The
    output project is ready for training/inference: configs, calibrations, models,
    scripts, and ``project.yaml`` are copied, config paths are updated to the new
    location, and all ``CollectedData_*.csv`` keypoint coordinates are shifted to
    match the cropped frames. Rows that are invalid in any camera view are dropped
    so that all per-view CSVs stay aligned.

    :param str lp_project_dir:  Root of the source LP project (e.g. ``Z:/home/simon/lp_300126``).
    :param str save_dir:        Root of the new cropped LP project.
    :param Tuple[int, int] crop_size: Output crop ``(width, height)`` in pixels (e.g. ``(512, 512)``).
                                Each crop is centered on the keypoint centroid per frame.
    :param Optional[Union[bool, int]] visualize: If ``True``, save annotated overlay images for every
                                cropped frame to ``save_dir/visualizations/``. If ``int``, save that many
                                randomly sampled overlays. ``None`` / ``False`` disables visualization.
    """

    def __init__(self,
                 lp_project_dir: str,
                 save_dir: str,
                 crop_size: Tuple[int, int] = (512, 512),
                 visualize: Optional[Union[bool, int]] = None):

        check_if_dir_exists(in_dir=lp_project_dir)
        check_int(name="CropLPAnnotations crop_size width", value=crop_size[0], min_value=1)
        check_int(name="CropLPAnnotations crop_size height", value=crop_size[1], min_value=1)
        if isinstance(visualize, int) and not isinstance(visualize, bool):
            check_int(name="CropLPAnnotations visualize", value=visualize, min_value=1)
        self.lp_project_dir = lp_project_dir
        self.save_dir = save_dir
        self.crop_size = crop_size
        self.visualize = visualize
        self.csv_paths = sorted([os.path.join(lp_project_dir, f) for f in os.listdir(lp_project_dir)
                                 if f.startswith("CollectedData_") and f.endswith(".csv")])
        if len(self.csv_paths) == 0:
            raise InvalidInputError(msg=f"No CollectedData_*.csv files found in {lp_project_dir}.")
        check_if_dir_exists(in_dir=os.path.join(lp_project_dir, "labeled-data"))

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)
        timer = SimbaTimer(start=True)
        self._copy_project_files(lp_project_dir=self.lp_project_dir, save_dir=self.save_dir)
        self._update_config_paths(lp_project_dir=self.lp_project_dir, save_dir=self.save_dir)
        valid_indices = self._find_common_valid_indices(csv_paths=self.csv_paths,
                                                        lp_project_dir=self.lp_project_dir,
                                                        crop_size=self.crop_size)
        stdout_success(msg=f"Found {len(valid_indices)} rows valid across all {len(self.csv_paths)} CSV files.")
        viz_candidates = []
        for csv_path in self.csv_paths:
            self._process_csv(csv_path=csv_path,
                              lp_project_dir=self.lp_project_dir,
                              save_dir=self.save_dir,
                              crop_size=self.crop_size,
                              viz_candidates=viz_candidates,
                              valid_indices=valid_indices)
        if self.visualize and len(viz_candidates) > 0:
            viz_dir = os.path.join(self.save_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            if isinstance(self.visualize, int) and not isinstance(self.visualize, bool):
                sample = random.sample(viz_candidates, min(self.visualize, len(viz_candidates)))
            else:
                sample = viz_candidates
            for cropped_path, new_xs, new_ys, bp_names, viz_fn in sample:
                viz_img = cv2.imread(cropped_path)
                for bp_idx in range(len(bp_names)):
                    bx, by = new_xs[bp_idx], new_ys[bp_idx]
                    if np.isnan(bx) or np.isnan(by):
                        continue
                    pt = (int(round(bx)), int(round(by)))
                    cv2.circle(viz_img, pt, 4, (0, 0, 255), -1)
                    cv2.putText(viz_img, bp_names[bp_idx], (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(viz_dir, viz_fn), viz_img)
            stdout_success(msg=f"Saved {len(sample)} visualizations in {viz_dir}")
        timer.stop_timer()
        stdout_success(msg=f"Cropped LP annotations saved in {self.save_dir}", elapsed_time=timer.elapsed_time_str)

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
        """Convert a path to POSIX format, stripping Windows drive letters (e.g. ``Z:/home/...`` -> ``/home/...``)."""
        p = path.replace("\\", "/")
        if len(p) >= 2 and p[1] == ":":
            p = p[2:]
        return p

    def _update_config_paths(self, lp_project_dir: str, save_dir: str):
        import re
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
            stdout_success(msg=f"Updated paths in {n_updated} config file(s) to {new_posix}", source="CropLPAnnotations")

    @staticmethod
    def _row_is_valid(idx, df, lp_project_dir, crop_size):
        """Check if a row has valid keypoints, a readable image, and the image is large enough for the crop."""
        coords = df.loc[idx].values.astype(float)
        xs, ys = coords[0::2], coords[1::2]
        if not np.any(~np.isnan(xs) & ~np.isnan(ys)):
            return False
        img_rel = str(idx)
        img_path = os.path.join(lp_project_dir, img_rel.replace("/", os.sep))
        if not os.path.isfile(img_path):
            return False
        img = cv2.imread(img_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        if w < crop_size[0] or h < crop_size[1]:
            return False
        return True

    def _find_common_valid_indices(self, csv_paths, lp_project_dir, crop_size):
        """Return the row-position indices (ints) that are valid across ALL CSVs."""
        per_csv_valid = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
            valid = set()
            for row_pos in range(len(df)):
                idx = df.index[row_pos]
                if self._row_is_valid(idx, df, lp_project_dir, crop_size):
                    valid.add(row_pos)
            per_csv_valid.append(valid)
        common = per_csv_valid[0]
        for s in per_csv_valid[1:]:
            common = common & s
        return common

    def _process_csv(self,
                     csv_path: str,
                     lp_project_dir: str,
                     save_dir: str,
                     crop_size: Tuple[int, int],
                     viz_candidates: list,
                     valid_indices: set):

        _, csv_fn, csv_ext = get_fn_ext(filepath=csv_path)
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        bp_names = [df.columns[i][1] for i in range(0, len(df.columns), 2)]
        out_rows = []
        for row_pos in range(len(df)):
            if row_pos not in valid_indices:
                continue
            idx = df.index[row_pos]
            coords = df.loc[idx].values.astype(float)
            xs = coords[0::2]
            ys = coords[1::2]
            valid = ~np.isnan(xs) & ~np.isnan(ys)

            img_rel = str(idx)
            img_path = os.path.join(lp_project_dir, img_rel.replace("/", os.sep))
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            x_valid, y_valid = xs[valid], ys[valid]
            x_min, x_max = float(np.min(x_valid)), float(np.max(x_valid))
            y_min, y_max = float(np.min(y_valid)), float(np.max(y_valid))

            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            half_w = crop_size[0] / 2.0
            half_h = crop_size[1] / 2.0
            crop_x1 = int(np.floor(cx - half_w))
            crop_y1 = int(np.floor(cy - half_h))
            crop_x2 = crop_x1 + crop_size[0]
            crop_y2 = crop_y1 + crop_size[1]
            if crop_x1 < 0:
                crop_x1, crop_x2 = 0, crop_size[0]
            if crop_y1 < 0:
                crop_y1, crop_y2 = 0, crop_size[1]
            if crop_x2 > w:
                crop_x2, crop_x1 = w, w - crop_size[0]
            if crop_y2 > h:
                crop_y2, crop_y1 = h, h - crop_size[1]

            cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
            out_img_path = os.path.join(save_dir, img_rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
            cv2.imwrite(out_img_path, cropped)

            new_coords = coords.copy()
            new_xs = np.where(np.isnan(xs), np.nan, xs - crop_x1)
            new_ys = np.where(np.isnan(ys), np.nan, ys - crop_y1)
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


# if __name__ == "__main__":
#     cropper = CropLPAnnotations(lp_project_dir=r"F:\netholabs\litpose_projects\projects_lp_compressed.8.4.2026",
#                                 save_dir=r"F:\netholabs\litpose_projects\projects_lp_compressed.8.4.2026_cropped",
#                                 crop_size=(512, 512),
#                                 visualize=40)
#     cropper.run()
