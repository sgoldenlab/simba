import datetime
import glob
import os
import shutil
from typing import Dict, List, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2
import pandas as pd
import yaml

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".flv", ".m4v", ".webm", ".h264"}
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpeg", ".jpg", ".webp"}
PROJECT_YAML = 'project.yaml'
LABELED_DATA = 'labeled-data'
KEYPOINT_NAMES_KEY = 'keypoint_names'
VIEW_NAMES_KEY = 'view_names'

# master_dir / other_dirs accept os.PathLike, so every helper that receives one of
# them has to accept it too -- annotating these `str` makes type checkers flag each
# internal call site
ProjectPath = Union[str, os.PathLike]


def _log(msg: str, level: str = 'INFO'):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {level}: {msg}')


class LitPoseMergeProjects:
    """
    Merge one or more LitPose/DLC-style projects into a master project.

    Merges videos, labeled-data images, ``CollectedData_*.csv`` annotation files, and camera ``calibrations`` files. Annotations sharing the same camera suffix (e.g. ``CollectedData_cam1``) are concatenated row-wise; duplicate image rows are handled according to ``duplicate_method``. Any files inside each project's ``calibrations`` sub-folder are copied into the master ``calibrations`` sub-folder (recursively, preserving nested sub-folders); duplicate calibration files are handled according to ``duplicate_method``.

    :param Union[str, os.PathLike] master_dir:              Root of the master LitPose project.
    :param List[Union[str, os.PathLike]] other_dirs:        Roots of projects to merge in.
    :param Literal['skip', 'raise'] duplicate_method:       How to handle duplicate videos, images, or annotation rows. ``'skip'`` silently ignores them; ``'raise'`` raises an error.
    :param bool skip_videos:                                If True, skip copying video files during merge. Default True.
    :param bool normalize_layout:                           If True, and the master's own ``CollectedData_*.csv`` paths resolve under a nested copy of the project rather than the project root, lift that nested ``labeled-data`` up into the root before merging. Without it the merged CSVs keep rows whose images are not where the paths say. Default True.
    :param bool verbose:                                    Print per-item progress.

    :example:

    >>> merger = LitPoseMergeProjects(master_dir=r'Z:\\home\\simon\\lp_300126', other_dirs=[r'F:\\netholabs\\projects_lp_compressed.8.4.2026\\projects_lp_compressed.8.4.2026'], duplicate_method='skip', verbose=True)
    >>> merger.run()
    """

    def __init__(self,
                 master_dir: ProjectPath,
                 other_dirs: List[ProjectPath],
                 duplicate_method: Literal['skip', 'raise'] = 'skip',
                 skip_videos: bool = True,
                 normalize_layout: bool = True,
                 verbose: bool = True):

        if not os.path.isdir(master_dir):
            raise NotADirectoryError(f'Master directory not found: {master_dir}')
        for d in other_dirs:
            if not os.path.isdir(d):
                raise NotADirectoryError(f'Other directory not found: {d}')
        if duplicate_method not in ('skip', 'raise'):
            raise ValueError(f'duplicate_method must be "skip" or "raise", got {duplicate_method}')
        self.master_dir = master_dir
        self.other_dirs = other_dirs
        self.duplicate_method = duplicate_method
        self.skip_videos = skip_videos
        self.normalize_layout = normalize_layout
        self.verbose = verbose
        self._img_root = {d: self._resolve_img_root(d) for d in [master_dir] + list(other_dirs)}
        for d, root in self._img_root.items():
            if root != d and verbose:
                _log(f'Image paths in {d} resolve under the nested copy {root}, not the project root.',
                     level='WARNING')
        self._validate_schemas()

    @staticmethod
    def _read_project_yaml(project_dir: ProjectPath) -> Dict:
        yaml_path = os.path.join(project_dir, PROJECT_YAML)
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f'project.yaml not found: {yaml_path}')
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _get_csv_bodyparts(csv_path: str) -> List[str]:
        df = pd.read_csv(csv_path, header=[0, 1, 2], nrows=0)
        body_parts = []
        for col in df.columns[1:]:
            bp = col[1]
            if 'unnamed:' not in bp.lower() and bp not in body_parts:
                body_parts.append(bp)
        return body_parts

    @staticmethod
    def _get_csv_scorer(csv_path: str) -> str:
        df = pd.read_csv(csv_path, header=[0, 1, 2], nrows=0)
        scorers = set()
        for col in df.columns[1:]:
            if 'unnamed:' not in col[0].lower():
                scorers.add(col[0])
        if len(scorers) == 1:
            return scorers.pop()
        return ', '.join(sorted(scorers))

    @staticmethod
    def _get_csv_image_paths(csv_path: str) -> List[str]:
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        return df.iloc[:, 0].values.tolist()

    @staticmethod
    def _check_images_exist(project_dir: ProjectPath, img_paths: List[str]) -> List[str]:
        missing = []
        for img_path in img_paths:
            full_path = os.path.join(project_dir, img_path)
            if not os.path.isfile(full_path):
                missing.append(img_path)
        return missing

    @staticmethod
    def _get_image_resolution(img_path: str) -> Tuple[int, int]:
        img = cv2.imread(img_path)
        if img is None:
            return (-1, -1)
        return (img.shape[1], img.shape[0])

    @staticmethod
    def _find_collected_data_csvs(directory: ProjectPath) -> Dict[str, str]:
        """Return dict keyed by camera suffix -> file path. E.g. 'cam1' -> '/.../CollectedData_cam1.csv'.

        Only looks at the project root — copies inside ``models/`` or ``outputs/`` are ignored
        so they don't shadow the canonical root file.
        """
        paths = glob.glob(os.path.join(directory, 'CollectedData*.csv'))
        result = {}
        for p in paths:
            fn = os.path.splitext(os.path.basename(p))[0]
            suffix = fn.replace('CollectedData_', '').replace('CollectedData', '')
            result[suffix] = p
        return result

    @classmethod
    def _resolve_img_root(cls, project_dir: ProjectPath, probe: int = 40) -> ProjectPath:
        """Return the directory the ``CollectedData_*.csv`` image paths are relative to.

        Projects sometimes arrive nested one level down -- an extract that kept the
        archive's own folder -- so ``labeled-data/x/img.jpg`` resolves under
        ``<project>/<project>/`` while the CSVs still sit in ``<project>/``. Score the
        project root and every immediate sub-folder that holds a ``labeled-data`` against
        a sample of the referenced paths, and take whichever resolves the most.
        """
        csvs = cls._find_collected_data_csvs(project_dir)
        if not csvs:
            return project_dir
        sample = cls._get_csv_image_paths(sorted(csvs.values())[0])[:probe]
        if not sample:
            return project_dir

        def score(root):
            return sum(os.path.isfile(os.path.join(root, p.replace('/', os.sep))) for p in sample)

        best, best_n = project_dir, score(project_dir)
        if best_n == len(sample):
            return project_dir
        for entry in sorted(os.listdir(project_dir)):
            cand = os.path.join(project_dir, entry)
            if not os.path.isdir(cand) or not os.path.isdir(os.path.join(cand, LABELED_DATA)):
                continue
            n = score(cand)
            if n > best_n:
                best, best_n = cand, n
        return best

    def _normalise_master_layout(self):
        """Lift a nested master ``labeled-data`` up into the master project root.

        The merged CSVs keep the master's own rows verbatim, so unless its images sit
        where those rows say they do, the merge produces a project full of dangling
        references. Sessions already present at the root are merged file by file under
        ``duplicate_method`` rather than replaced.
        """
        root = self._img_root[self.master_dir]
        if root == self.master_dir:
            return
        src_dir = os.path.join(root, LABELED_DATA)
        dst_dir = os.path.join(self.master_dir, LABELED_DATA)
        if not os.path.isdir(src_dir):
            return
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        moved_sessions, moved_imgs = 0, 0
        for session_name in sorted(os.listdir(src_dir)):
            src_session = os.path.join(src_dir, session_name)
            if not os.path.isdir(src_session):
                continue
            dst_session = os.path.join(dst_dir, session_name)
            if not os.path.isdir(dst_session):
                os.makedirs(dst_session)
            existing = set(os.listdir(dst_session))
            for fname in os.listdir(src_session):
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                    continue
                if fname in existing:
                    self._handle_duplicate('image', f'{session_name}/{fname}')
                    continue
                shutil.move(os.path.join(src_session, fname), os.path.join(dst_session, fname))
                existing.add(fname)
                moved_imgs += 1
            moved_sessions += 1
        self._img_root[self.master_dir] = self.master_dir
        if self.verbose:
            _log(f'Lifted {moved_imgs} image(s) across {moved_sessions} session(s) from {src_dir} '
                 f'into {dst_dir}', level='COMPLETE')

    def _report_dangling(self):
        """Warn about merged rows whose image is not on disk where the path says."""
        for suffix, csv_path in sorted(self._find_collected_data_csvs(self.master_dir).items()):
            paths = self._get_csv_image_paths(csv_path)
            missing = self._check_images_exist(self.master_dir, paths)
            if missing:
                _log(f'{os.path.basename(csv_path)}: {len(missing)}/{len(paths)} merged row(s) point at '
                     f'images that are not on disk. First: {missing[0]}', level='WARNING')
            elif self.verbose:
                _log(f'{os.path.basename(csv_path)}: all {len(paths)} merged row(s) resolve on disk')

    def _validate_schemas(self):
        master_yaml = self._read_project_yaml(self.master_dir)
        master_keypoints = master_yaml.get(KEYPOINT_NAMES_KEY, [])
        master_views = master_yaml.get(VIEW_NAMES_KEY, [])
        all_dirs = [self.master_dir] + list(self.other_dirs)

        for other_dir in self.other_dirs:
            other_yaml = self._read_project_yaml(other_dir)
            other_keypoints = other_yaml.get(KEYPOINT_NAMES_KEY, [])
            other_views = other_yaml.get(VIEW_NAMES_KEY, [])
            if master_keypoints != other_keypoints:
                raise ValueError(f'Keypoint mismatch between master and {other_dir}. '
                                 f'Master: {master_keypoints}, Other: {other_keypoints}')
            if master_views != other_views:
                raise ValueError(f'View/camera mismatch between master and {other_dir}. '
                                 f'Master: {master_views}, Other: {other_views}')

        master_scorer = None
        view_resolutions = {}
        for project_dir in all_dirs:
            project_csvs = self._find_collected_data_csvs(project_dir)
            project_label = 'master' if project_dir == self.master_dir else project_dir

            csv_suffixes = set(project_csvs.keys())
            missing_views = set(master_views) - csv_suffixes
            if missing_views:
                raise ValueError(f'{project_label} is missing CollectedData CSVs for views: {sorted(missing_views)}. '
                                 f'Expected one CSV per view: {master_views}')

            for suffix, csv_path in project_csvs.items():
                bp = self._get_csv_bodyparts(csv_path)
                if bp != master_keypoints:
                    raise ValueError(f'CSV bodyparts in {os.path.basename(csv_path)} from {project_label} '
                                     f'do not match project.yaml keypoints. '
                                     f'CSV: {bp}, Expected: {master_keypoints}')

                scorer = self._get_csv_scorer(csv_path)
                if master_scorer is None:
                    master_scorer = scorer
                elif scorer != master_scorer:
                    raise ValueError(f'Scorer mismatch in {os.path.basename(csv_path)} from {project_label}. '
                                     f'Expected: {master_scorer}, Found: {scorer}')

                if suffix not in master_views:
                    raise ValueError(f'CollectedData_{suffix}.csv in {project_label} has no matching view '
                                     f'in project views: {master_views}')

                img_root = self._img_root[project_dir]
                img_paths = self._get_csv_image_paths(csv_path)
                missing_imgs = self._check_images_exist(img_root, img_paths)
                if missing_imgs:
                    if self.verbose:
                        # master rows are carried through the merge verbatim, so for the master
                        # these are not skipped -- they survive as dangling references
                        fate = ('kept as-is, leaving dangling image references'
                                if project_dir == self.master_dir else 'skipped during merge')
                        _log(f'{len(missing_imgs)} image(s) referenced in {os.path.basename(csv_path)} from {project_label} '
                             f'not found on disk and will be {fate}. First missing: {missing_imgs[0]}',
                             level='WARNING')
                missing_set = set(missing_imgs)
                present_paths = [p for p in img_paths if p not in missing_set]

                resolutions = set()
                for img_path in present_paths:
                    res = self._get_image_resolution(os.path.join(img_root, img_path))
                    resolutions.add(res)
                if len(resolutions) > 1:
                    raise ValueError(f'Mixed image resolutions in {os.path.basename(csv_path)} from {project_label}: {sorted(resolutions)}. '
                                     f'All images for a given camera must share the same resolution.')
                if len(resolutions) == 1:
                    view_res = resolutions.pop()
                    if suffix in view_resolutions:
                        if view_res != view_resolutions[suffix]:
                            raise ValueError(f'Resolution mismatch for {suffix} across projects. '
                                             f'Expected {view_resolutions[suffix]}, '
                                             f'got {view_res} in {project_label}')
                    else:
                        view_resolutions[suffix] = view_res

        if self.verbose:
            _log(f'Schema validation passed: {len(master_keypoints)} keypoints, {len(master_views)} views, '
                 f'scorer "{master_scorer}", consistent resolutions across all projects', level='COMPLETE')

    def _handle_duplicate(self, item_type: str, name: str):
        msg = f'Duplicate {item_type}: {name}'
        if self.duplicate_method == 'raise':
            raise ValueError(msg)
        else:
            if self.verbose:
                _log(f'{msg}. Skipping...', level='WARNING')

    def _merge_videos(self, other_dir: ProjectPath):
        master_videos_dir = os.path.join(self.master_dir, 'videos')
        other_videos_dir = os.path.join(other_dir, 'videos')
        if not os.path.isdir(other_videos_dir):
            if self.verbose:
                _log(f'No videos directory found in {other_dir}. Skipping video merge for this project.', level='WARNING')
            return
        if not os.path.isdir(master_videos_dir):
            os.makedirs(master_videos_dir)
        existing_entries = set(os.listdir(master_videos_dir))
        copied_dirs, copied_files = 0, 0
        for entry in os.listdir(other_videos_dir):
            src = os.path.join(other_videos_dir, entry)
            dst = os.path.join(master_videos_dir, entry)
            if os.path.isdir(src):
                if entry in existing_entries:
                    self._handle_duplicate('video directory', entry)
                    continue
                shutil.copytree(src, dst)
                existing_entries.add(entry)
                copied_dirs += 1
                if self.verbose:
                    _log(f'Copied video directory: {entry}')
            elif os.path.splitext(entry)[1].lower() in VIDEO_EXTENSIONS:
                if entry in existing_entries:
                    self._handle_duplicate('video', entry)
                    continue
                shutil.copy2(src, dst)
                existing_entries.add(entry)
                copied_files += 1
                if self.verbose:
                    _log(f'Copied video file: {entry}')
        if self.verbose:
            _log(f'Copied {copied_dirs} video directory(s) and {copied_files} video file(s) from {other_dir}')

    def _merge_labeled_data(self, other_dir: ProjectPath):
        master_labeled_dir = os.path.join(self.master_dir, LABELED_DATA)
        other_labeled_dir = os.path.join(self._img_root[other_dir], LABELED_DATA)
        if not os.path.isdir(other_labeled_dir):
            if self.verbose:
                _log(f'No labeled-data directory found in {other_dir}. Skipping image merge for this project.', level='WARNING')
            return
        if not os.path.isdir(master_labeled_dir):
            os.makedirs(master_labeled_dir)
        copied = 0
        for session_name in os.listdir(other_labeled_dir):
            other_session_dir = os.path.join(other_labeled_dir, session_name)
            if not os.path.isdir(other_session_dir):
                continue
            master_session_dir = os.path.join(master_labeled_dir, session_name)
            if not os.path.isdir(master_session_dir):
                os.makedirs(master_session_dir)
            existing_imgs = set(os.listdir(master_session_dir))
            for fname in os.listdir(other_session_dir):
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                    continue
                if fname in existing_imgs:
                    self._handle_duplicate('image', f'{session_name}/{fname}')
                    continue
                shutil.copy2(os.path.join(other_session_dir, fname), os.path.join(master_session_dir, fname))
                existing_imgs.add(fname)
                copied += 1
        if self.verbose:
            _log(f'Copied {copied} labeled image(s) from {other_dir}')

    def _merge_annotations(self, other_dir: ProjectPath):
        master_csvs = self._find_collected_data_csvs(self.master_dir)
        other_csvs = self._find_collected_data_csvs(other_dir)
        if not other_csvs:
            if self.verbose:
                _log(f'No CollectedData CSV files found in {other_dir}. Skipping annotation merge.', level='WARNING')
            return
        for suffix, other_csv_path in other_csvs.items():
            other_df = pd.read_csv(other_csv_path, header=[0, 1, 2])
            missing_imgs = self._check_images_exist(self._img_root[other_dir], other_df.iloc[:, 0].values.tolist())
            if missing_imgs:
                if self.verbose:
                    _log(f'Skipping {len(missing_imgs)} row(s) in {os.path.basename(other_csv_path)} with missing image files. First: {missing_imgs[0]}', level='WARNING')
                other_df = other_df[~other_df.iloc[:, 0].isin(set(missing_imgs))]
            if len(other_df) == 0:
                if self.verbose:
                    _log(f'No rows with existing images for {os.path.basename(other_csv_path)}; nothing to merge.')
                continue
            other_img_paths = set(other_df.iloc[:, 0].values)
            if suffix in master_csvs:
                master_csv_path = master_csvs[suffix]
                master_df = pd.read_csv(master_csv_path, header=[0, 1, 2])
                existing_img_paths = set(master_df.iloc[:, 0].values)
                duplicates = other_img_paths & existing_img_paths
                if duplicates:
                    for dup in duplicates:
                        self._handle_duplicate('annotation row', f'{os.path.basename(other_csv_path)}: {dup}')
                    other_df = other_df[~other_df.iloc[:, 0].isin(existing_img_paths)]
                if len(other_df) == 0:
                    if self.verbose:
                        _log(f'No new annotation rows to merge for CollectedData_{suffix}')
                    continue
                merged_df = pd.concat([master_df, other_df], axis=0, ignore_index=True)
                merged_df.to_csv(master_csv_path, index=False)
                verify_df = pd.read_csv(master_csv_path, header=[0, 1, 2])
                if len(verify_df) != len(merged_df):
                    _log(f'WRITE MISMATCH for {os.path.basename(master_csv_path)}: in-memory {len(merged_df)} rows, on-disk {len(verify_df)} rows after re-read.', level='WARNING')
                if self.verbose:
                    _log(f'Appended {len(other_df)} rows to {os.path.basename(master_csv_path)} (in-memory total: {len(merged_df)}, on-disk re-read total: {len(verify_df)})')
            else:
                new_csv_path = os.path.join(self.master_dir, os.path.basename(other_csv_path))
                shutil.copy2(other_csv_path, new_csv_path)
                master_csvs[suffix] = new_csv_path
                if self.verbose:
                    _log(f'Copied new annotation file: {os.path.basename(other_csv_path)} ({len(other_df)} rows)')

    def _merge_calibrations(self, other_dir: ProjectPath):
        master_calib_dir = os.path.join(self.master_dir, 'calibrations')
        other_calib_dir = os.path.join(other_dir, 'calibrations')
        if not os.path.isdir(other_calib_dir):
            if self.verbose:
                _log(f'No calibrations directory found in {other_dir}. Skipping calibration merge for this project.', level='WARNING')
            return
        if not os.path.isdir(master_calib_dir):
            os.makedirs(master_calib_dir)
        copied = 0
        for root, _, files in os.walk(other_calib_dir):
            rel_root = os.path.relpath(root, other_calib_dir)
            dst_root = master_calib_dir if rel_root == '.' else os.path.join(master_calib_dir, rel_root)
            for fname in files:
                if ':' in fname:                                    # skip NTFS alternate-data-stream artifacts (e.g. Zone.Identifier)
                    continue
                rel_name = fname if rel_root == '.' else os.path.join(rel_root, fname)
                dst = os.path.join(dst_root, fname)
                if os.path.isfile(dst):
                    self._handle_duplicate('calibration file', rel_name)
                    continue
                if not os.path.isdir(dst_root):
                    os.makedirs(dst_root)
                shutil.copy2(os.path.join(root, fname), dst)
                copied += 1
                if self.verbose:
                    _log(f'Copied calibration file: {rel_name}')
        if self.verbose:
            _log(f'Copied {copied} calibration file(s) from {other_dir}')

    def run(self):
        import time
        start = time.time()
        if self.normalize_layout:
            self._normalise_master_layout()
        for project_cnt, other_dir in enumerate(self.other_dirs):
            project_start = time.time()
            if self.verbose:
                _log(f'Merging project {project_cnt + 1}/{len(self.other_dirs)}: {other_dir}')
            if not self.skip_videos:
                self._merge_videos(other_dir)
            elif self.verbose:
                _log('Skipping video merge (skip_videos=True)')
            self._merge_labeled_data(other_dir)
            self._merge_annotations(other_dir)
            self._merge_calibrations(other_dir)
            if self.verbose:
                _log(f'Project {project_cnt + 1}/{len(self.other_dirs)} merged ({time.time() - project_start:.1f}s)', level='COMPLETE')
        self._report_dangling()
        _log(f'{len(self.other_dirs)} project(s) merged into {self.master_dir} ({time.time() - start:.1f}s)', level='COMPLETE')



# merger = LitPoseMergeProjects(master_dir=r"I:\sina\project_5cam_cage21_22_0911",
#                               other_dirs=[r"I:\sina\project_0609_5cam_0823"],
#                               duplicate_method='skip',
#                               verbose=True,
#                               skip_videos=False)
# merger.run()