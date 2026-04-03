"""
Merge independent YOLO-format projects (detection, segmentation, or pose) into one dataset.

Each input project must supply a ``map.yaml`` (or equivalent Ultralytics YAML) pointing at
``images/train``, ``images/val``, and matching ``labels/`` trees. Class names and task type
must match across inputs. Typical sources include
:class:`~simba.third_party_label_appenders.transform.sam3_to_yolo_seg.SAM3ToYoloSeg` and
:class:`~simba.third_party_label_appenders.transform.sam3_to_yolo_bbox.SAM3ToYoloBBox` outputs.
"""

import os
import random
from typing import Dict, List, Optional, Tuple, Union

import yaml

from simba.third_party_label_appenders.converters import create_yolo_yaml
from simba.utils.checks import (check_file_exist_and_readable, check_float, check_if_dir_exists, check_int, check_valid_boolean, check_valid_lst)
from simba.utils.enums import Options
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success, stdout_information
import shutil

from simba.utils.read_write import (create_directory, find_files_of_filetypes_in_directory)
from simba.utils.warnings import DuplicateNamesWarning


BBOX_FIELD_CNT = 5
SEG_MIN_FIELD_CNT = 7
KPT_FIELD_MODULO = 3
TRAIN = 'train'
VAL = 'val'
IMAGES = 'images'
LABELS = 'labels'

class MergeYoloProjects:
    """
    Merge multiple YOLO projects into a single YOLO project.

    Reads each project's YAML, validates that all projects share the same task type (bounding-box detection, segmentation, or keypoint pose) and class names, then copies all images and labels into a single output project with train/val splits.

    .. seealso::

       * :class:`~simba.third_party_label_appenders.transform.sam3_to_yolo_seg.SAM3ToYoloSeg`
       * :class:`~simba.third_party_label_appenders.transform.sam3_to_yolo_bbox.SAM3ToYoloBBox`

    :param List[Union[str, os.PathLike]] yaml_paths: List of paths to YOLO project YAML files.
    :param Union[str, os.PathLike] save_dir: Root output directory for the merged project.
    :param Optional[float] train_val_split: If provided, reshuffle all samples and split at this ratio (0.1-0.9). If None, preserve each project's existing train/val assignments. Default None.
    :param Optional[int] seed: Random seed for reproducible splitting. Only used when ``train_val_split`` is not None.
    :param bool verbose: If True, print progress. Default True.

    :example:
    >>> merger = MergeYoloProjects(yaml_paths=[r'/project_a/map.yaml', r'/project_b/map.yaml'], save_dir=r'/merged_project', train_val_split=0.8)
    >>> merger.run()
    """

    def __init__(self,
                 yaml_paths: List[Union[str, os.PathLike]],
                 save_dir: Union[str, os.PathLike],
                 train_val_split: Optional[float] = None,
                 seed: Optional[int] = None,
                 verbose: bool = True):

        check_valid_lst(data=yaml_paths, source=f'{self.__class__.__name__} yaml_paths', min_len=2, valid_dtypes=(str,))
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        if train_val_split is not None:
            check_float(name=f'{self.__class__.__name__} train_val_split', value=train_val_split, min_value=0.1, max_value=0.9)
        if seed is not None:
            check_int(name=f'{self.__class__.__name__} seed', value=seed)
        for p in yaml_paths:
            check_file_exist_and_readable(file_path=p)

        self.yaml_paths, self.save_dir = yaml_paths, save_dir
        self.train_val_split, self.seed, self.verbose = train_val_split, seed, verbose

    def run(self):
        timer = SimbaTimer(start=True)
        if self.seed is not None: random.seed(self.seed)

        projects = self._parse_yamls()
        self._validate_projects(projects)

        img_train_dir = os.path.join(self.save_dir, IMAGES, TRAIN)
        img_val_dir = os.path.join(self.save_dir, IMAGES, VAL)
        lbl_train_dir = os.path.join(self.save_dir, LABELS, TRAIN)
        lbl_val_dir = os.path.join(self.save_dir, LABELS, VAL)
        create_directory(paths=[img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir], overwrite=False)

        if self.train_val_split is not None:
            all_pairs = []
            for project in projects:
                for split in (TRAIN, VAL):
                    all_pairs.extend(project['pairs'][split])
            random.shuffle(all_pairs)
            split_idx = int(len(all_pairs) * self.train_val_split)
            train_pairs = all_pairs[:split_idx]
            val_pairs = all_pairs[split_idx:]
        else:
            train_pairs, val_pairs = [], []
            for project in projects:
                train_pairs.extend(project['pairs'][TRAIN])
                val_pairs.extend(project['pairs'][VAL])

        train_cnt = self._copy_pairs(train_pairs, img_train_dir, lbl_train_dir, TRAIN)
        val_cnt = self._copy_pairs(val_pairs, img_val_dir, lbl_val_dir, VAL)

        name_map = projects[0]['names']
        map_path = os.path.join(self.save_dir, 'map.yaml')
        create_yolo_yaml(path=self.save_dir, train_path=img_train_dir, val_path=img_val_dir, names=name_map, save_path=map_path)

        timer.stop_timer()
        stdout_success(msg=f'Merged YOLO project created at {self.save_dir}. {train_cnt} train, {val_cnt} val samples from {len(self.yaml_paths)} projects.', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)

    def _parse_yamls(self) -> List[Dict]:
        projects = []
        for yaml_path in self.yaml_paths:
            with open(yaml_path, 'r') as f:
                cfg = yaml.safe_load(f)

            root = cfg.get('path', os.path.dirname(yaml_path))
            train_img_dir = os.path.join(root, cfg.get('train', 'images/train'))
            val_img_dir = os.path.join(root, cfg.get('val', 'images/val'))
            train_lbl_dir = train_img_dir.replace(os.sep + IMAGES + os.sep, os.sep + LABELS + os.sep)
            val_lbl_dir = val_img_dir.replace(os.sep + IMAGES + os.sep, os.sep + LABELS + os.sep)

            names_raw = cfg.get('names', {})
            if isinstance(names_raw, list):
                names = {name: idx for idx, name in enumerate(names_raw)}
            elif isinstance(names_raw, dict):
                if all(isinstance(k, int) for k in names_raw.keys()):
                    names = {v: k for k, v in names_raw.items()}
                else:
                    names = names_raw
            else:
                raise InvalidInputError(msg=f'Unexpected names format in {yaml_path}: {type(names_raw)}', source=self.__class__.__name__)

            pairs = {TRAIN: [], VAL: []}
            for split, i_dir, l_dir in [(TRAIN, train_img_dir, train_lbl_dir), (VAL, val_img_dir, val_lbl_dir)]:
                if not os.path.isdir(i_dir):
                    if self.verbose:
                        stdout_information(msg=f'{yaml_path}: {split} image dir not found ({i_dir}), skipping split...')
                    continue
                img_files = find_files_of_filetypes_in_directory(directory=i_dir, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, as_dict=True, raise_warning=False)
                if isinstance(img_files, list) and len(img_files) == 0:
                    continue
                for stem, img_path in img_files.items():
                    lbl_path = os.path.join(l_dir, f'{stem}.txt')
                    if os.path.isfile(lbl_path):
                        pairs[split].append((img_path, lbl_path, stem))

            task_type = self._detect_task_type(pairs, yaml_path)
            project = {'yaml_path': yaml_path, 'names': names, 'task_type': task_type, 'pairs': pairs}
            if self.verbose:
                stdout_information(msg=f'Project {yaml_path}: {len(pairs[TRAIN])} train, {len(pairs[VAL])} val pairs, task={task_type}, classes={list(names.keys())}')
            projects.append(project)
        return projects

    def _detect_task_type(self, pairs: Dict[str, List[Tuple]], yaml_path: str) -> str:
        all_pairs = pairs[TRAIN] + pairs[VAL]
        if len(all_pairs) == 0:
            raise NoFilesFoundError(msg=f'No matched image/label pairs found in {yaml_path}.', source=self.__class__.__name__)
        sample_paths = [p[1] for p in all_pairs[:20]]
        field_counts = set()
        for lbl_path in sample_paths:
            with open(lbl_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        field_counts.add(len(line.split()))
                        break

        if not field_counts:
            raise InvalidInputError(msg=f'All sampled label files in {yaml_path} are empty.', source=self.__class__.__name__)

        if field_counts == {BBOX_FIELD_CNT}:
            return 'bbox'
        elif all(c >= SEG_MIN_FIELD_CNT and c % 2 == 1 for c in field_counts):
            return 'segment'
        elif all((c - 1) % KPT_FIELD_MODULO == 0 or (c - BBOX_FIELD_CNT) % KPT_FIELD_MODULO == 0 for c in field_counts):
            return 'keypoint'
        else:
            unique_counts = sorted(field_counts)
            raise InvalidInputError(msg=f'Cannot determine task type for {yaml_path}. Label field counts: {unique_counts}. Expected 5 for bbox, odd >5 for segmentation, or multiples of 3 (+bbox) for keypoint.', source=self.__class__.__name__)

    def _validate_projects(self, projects: List[Dict]):
        task_types = set(p['task_type'] for p in projects)
        if len(task_types) > 1:
            details = {p['yaml_path']: p['task_type'] for p in projects}
            raise InvalidInputError(msg=f'All projects must be the same task type, but found: {details}', source=self.__class__.__name__)

        ref_names = projects[0]['names']
        for p in projects[1:]:
            if p['names'] != ref_names:
                raise InvalidInputError(msg=f'Class name mismatch. {projects[0]["yaml_path"]} has {ref_names}, but {p["yaml_path"]} has {p["names"]}', source=self.__class__.__name__)

        stem_to_projects = {}
        for p in projects:
            for split in (TRAIN, VAL):
                for pair in p['pairs'][split]:
                    stem = pair[2]
                    if stem not in stem_to_projects:
                        stem_to_projects[stem] = []
                    stem_to_projects[stem].append(p['yaml_path'])
        duplicates = {s: srcs for s, srcs in stem_to_projects.items() if len(srcs) > 1}
        if len(duplicates) > 0:
            example_dupes = dict(list(duplicates.items())[:10])
            DuplicateNamesWarning(msg=f'{len(duplicates)} duplicate filenames found across projects. Only the first occurrence will be kept. Examples: {example_dupes}', source=self.__class__.__name__)

    def _copy_pairs(self, pairs: List[Tuple], img_dir: str, lbl_dir: str, split_name: str) -> int:
        display_name = 'validation' if split_name == VAL else split_name
        unique_pairs = {}
        for img_path, lbl_path, stem in pairs:
            if stem not in unique_pairs:
                unique_pairs[stem] = (img_path, lbl_path)
        if self.verbose:
            stdout_information(msg=f'Copying {len(unique_pairs)} {display_name} image/label pairs...')
        for cnt, (stem, (img_path, lbl_path)) in enumerate(unique_pairs.items()):
            shutil.copy(img_path, os.path.join(img_dir, os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(lbl_dir, os.path.basename(lbl_path)))
            if self.verbose and (cnt + 1) % 10 == 0:
                stdout_information(msg=f'{display_name}: copied {cnt + 1}/{len(unique_pairs)}...')
        if self.verbose:
            stdout_information(msg=f'{display_name}: copied {len(unique_pairs)}/{len(unique_pairs)} complete.')
        return len(unique_pairs)

# merger = MergeYoloProjects(yaml_paths=[r'F:\netholabs\moira\original\moira_litpose\map.yaml', r"F:\netholabs\sam3_to_bbox\map.yaml"], save_dir=r'F:\netholabs\moira_lp_sam', train_val_split=0.75)
# merger.run()