import os
import random
import shutil
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from simba.third_party_label_appenders.transform.utils import (
    create_yolo_keypoint_yaml, get_yolo_keypoint_flip_idx)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_tuple)
from simba.utils.data import get_cpu_pool, terminate_cpu_pool
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (create_directory, find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_img)
from simba.utils.yolo import (create_yolo_sample_visualizations, keypoint_array_to_yolo_annotation_str)


def _litpose_to_yolo_worker(task):
    """
    Module-level worker: write the YOLO label file for one labeled image, and the image itself.

    Without greyscale or CLAHE the pixels are unchanged, so the original file is copied byte-for-byte (its size is read from the
    header without decoding). With greyscale or CLAHE the image is decoded, transformed, and written as PNG.

    Must stay at module level (not a bound method) so it is picklable for multiprocessing under the Windows 'spawn' start method.
    """
    img_path, img_save_path, lbl_save_path, keypoint_values, bp_id_idx, greyscale, clahe, padding = task
    img_lbl = ''
    check_file_exist_and_readable(img_path, True)
    copy_img = not greyscale and not clahe
    if copy_img:
        with Image.open(img_path) as pil_img:
            img_w, img_h = pil_img.size
    else:
        img = read_img(img_path=img_path, greyscale=greyscale, clahe=clahe)
        img_h, img_w = img.shape[0], img.shape[1]
    keypoints_with_id = {}
    for k, bp_idx in enumerate(bp_id_idx):
        keypoints_with_id[k] = np.nan_to_num(keypoint_values.reshape(-1, 2)[bp_idx], nan=0.0, posinf=0.0, neginf=0.0)
    for cls_id, keypoints in keypoints_with_id.items():
        if np.all(np.isnan(keypoints)) or np.all(keypoints == 0.0) or np.all(np.isnan(keypoints) | (keypoints == 0.0)):
            continue
        visibility_col = np.full((keypoints.shape[0], 1), fill_value=2).flatten()
        keypoints = np.insert(keypoints, 2, visibility_col, axis=1)
        both_zero = (keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)
        has_nan_or_inf = ~np.isfinite(keypoints[:, 0]) | ~np.isfinite(keypoints[:, 1])
        mask = both_zero | has_nan_or_inf
        keypoints[mask, 2] = 0
        keypoints[~np.isfinite(keypoints)] = 0
        instance_str = f'{cls_id} '
        instance_str += keypoint_array_to_yolo_annotation_str(x=keypoints, img_w=img_w, img_h=img_h, padding=padding)
        img_lbl += instance_str
    with open(lbl_save_path, mode='wt', encoding='utf-8') as f:
        f.write(img_lbl)
    if copy_img:
        shutil.copyfile(img_path, img_save_path)
    else:
        cv2.imwrite(img_save_path, img)
    return img_save_path


class LitPose2YOLO:
    """
    Convert LitPose keypoint annotations into a YOLO keypoint dataset.

    :param Union[str, os.PathLike] litpose_dir: Path to LitPose directory containing annotation CSV files and the ``labeled-data`` image folder. Only ``CollectedData*.csv`` files in the top level are read; copies LitPose stores under ``models/`` are ignored.
    :param Union[str, os.PathLike] save_dir: Output directory where YOLO-formatted ``images`` and ``labels`` subdirectories are created.
    :param float train_size: Fraction of samples assigned to the training split. Default 0.7.
    :param bool verbose: If True, print per-image progress during conversion.
    :param float padding: Extra padding factor used when computing normalized YOLO boxes from keypoints.
    :param Optional[int] sample_n: Optional cap on the number of sampled frames before split. If None, all frames are used.
    :param Optional[Tuple[int, ...]] flip_idx: Optional keypoint flip index order for YOLO pose augmentation. If None, inferred from body-part names.
    :param Tuple[str, ...] names: Class names in YOLO index order.
    :param bool greyscale: If True, load and save images in grayscale (written as PNG).
    :param bool clahe: If True, apply CLAHE preprocessing when reading images (written as PNG). If both ``greyscale`` and ``clahe`` are False, the original image files are copied unchanged, keeping their format.
    :param int core_cnt: Number of worker processes used to read, convert and write images. 1 (default) runs serially. -1 uses all available cores.
    :param Optional[Union[bool, int]] visualize: If ``True``, save a keypoint + bounding-box overlay of every converted image to ``save_dir/visualizations/`` for sanity checking. If ``int``, save that many randomly sampled overlays. ``None`` / ``False`` (default) disables visualization. Overlays are drawn from the image and label files read back from ``save_dir``, so they verify what is on disk.

    References
    ----------
    .. [1] Lightning Pose documentation: https://lightning-pose.readthedocs.io/en/latest/
    .. [2] Biderman et al., Lightning Pose: improved animal pose estimation via semi-supervised learning, Bayesian ensembling and cloud-native open-source tools, *Nature Methods* (2024), doi: https://doi.org/10.1038/s41592-024-02319-1

    :example:

    >>> runner = LitPose2YOLO(litpose_dir=r'Z:\home\simon\lp_300126', save_dir=r'E:\litpose_yolo\bbox', verbose=True, clahe=False, greyscale=False, sample_n=1000, padding=0.15, core_cnt=4)
    >>> runner.run()
    """

    def __init__(self,
                 litpose_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 train_size: float = 0.7,
                 verbose: bool = False,
                 padding: float = 0.00,
                 sample_n: Optional[int] = None,
                 flip_idx: Optional[Tuple[int, ...]] = None,
                 names: Tuple[str, ...] = ('mouse',),
                 greyscale: bool = False,
                 clahe: bool = False,
                 core_cnt: int = 1,
                 visualize: Optional[Union[bool, int]] = None) -> None:

        check_if_dir_exists(in_dir=litpose_dir, source=f'{self.__class__.__name__} litpose_dir')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale')
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe')
        check_float(name=f'{self.__class__.__name__} padding', value=padding, max_value=1.0, min_value=0.0, raise_error=True)
        check_float(name=f'{self.__class__.__name__} train_size', value=train_size, max_value=0.99, min_value=0.1)
        check_valid_tuple(x=names, source=f'{self.__class__.__name__} names', minimum_length=1, valid_dtypes=(str,))
        if sample_n is not None:
            check_int(name=f'{self.__class__.__name__} sample', value=sample_n, min_value=1)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        max_cores = find_core_cnt()[0]
        self.core_cnt = max_cores if core_cnt == -1 else min(core_cnt, max_cores)
        if isinstance(visualize, bool):
            check_valid_boolean(value=visualize, source=f'{self.__class__.__name__} visualize')
        elif visualize is not None:
            check_int(name=f'{self.__class__.__name__} visualize', value=visualize, min_value=1)
        self.visualize = visualize
        check_if_dir_exists(in_dir=save_dir)
        csv_paths = find_files_of_filetypes_in_directory(directory=litpose_dir, extensions=['.csv'], raise_warning=False, sort_alphabetically=True)
        self.annotation_paths = [x for x in csv_paths if 'collecteddata' in get_fn_ext(filepath=x)[1].lower()]
        if len(self.annotation_paths) == 0:
            raise NoFilesFoundError(msg=f'No CollectedData*.csv annotation files found in the top level of {litpose_dir}', source=self.__class__.__name__)
        self.labeled_imgs_dir, self.litpose_dir = os.path.join(litpose_dir, 'labeled-data'), litpose_dir
        check_if_dir_exists(in_dir=self.labeled_imgs_dir)
        if flip_idx is not None:
            check_valid_tuple(x=flip_idx, source=f'{self.__class__.__name__} flip_idx', valid_dtypes=(int,), minimum_length=1)
        self.img_dir, self.lbl_dir = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(self.img_dir, 'train'), os.path.join(self.img_dir, 'val')
        self.lbl_train_dir, self.lb_val_dir = os.path.join(self.lbl_dir, 'train'), os.path.join(self.lbl_dir, 'val')
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lb_val_dir], overwrite=False)
        self.names = {k: v for k, v in enumerate(names)}
        self.map_path = os.path.join(save_dir, 'map.yaml')
        self.verbose, self.greyscale, self.train_size, self.clahe = verbose, greyscale, train_size, clahe
        self.padding, self.flip_idx, self.save_dir, self.sample_n = padding, flip_idx, save_dir, sample_n

    def run(self):
        annotations, timer, body_part_headers, first_body_part_headers = [], SimbaTimer(start=True), [], None
        for _, annotation_path in enumerate(self.annotation_paths):
            annotation_filename = get_fn_ext(filepath=annotation_path)[1]
            annotation_data = pd.read_csv(annotation_path, header=[0, 1, 2])
            img_paths = annotation_data.pop(annotation_data.columns[0]).reset_index(drop=True).values
            body_parts, body_part_headers = [], []
            for i in annotation_data.columns[1:]:
                if 'unnamed:' not in i[1].lower() and i[1] not in body_parts:
                    body_parts.append(i[1])
            for i in body_parts:
                body_part_headers.extend((f'{i}_x', f'{i}_y'))
            if first_body_part_headers is None:
                first_body_part_headers = body_part_headers
            elif body_part_headers != first_body_part_headers:
                raise InvalidInputError(msg=f'Body-parts in {annotation_path} ({body_part_headers}) do not match body-parts in {self.annotation_paths[0]} ({first_body_part_headers}). All LitPose annotation files must share the same body-parts in the same order.', source=self.__class__.__name__)
            annotation_data.columns = body_part_headers
            check_valid_dataframe(df=annotation_data, source=self.__class__.__name__, valid_dtypes=Formats.NUMERIC_DTYPES.value)
            annotation_data = annotation_data.reset_index(drop=True)
            img_names = [get_fn_ext(os.path.basename(x))[1] for x in img_paths]
            save_names = [f'{annotation_filename}_{os.path.basename(os.path.dirname(p))}_{get_fn_ext(p)[1]}' for p in img_paths]
            annotation_data['img_name'] = img_names
            annotation_data['img_path'] = img_paths
            annotation_data['save_name'] = save_names
            annotations.append(annotation_data)

        if self.flip_idx is None:
            self.flip_idx = get_yolo_keypoint_flip_idx(x=list(dict.fromkeys([x[:-2] for x in body_part_headers])))

        annotations = pd.concat(annotations, axis=0).reset_index(drop=True)
        duplicated_names = annotations['save_name'][annotations['save_name'].duplicated()].unique()
        if len(duplicated_names) > 0:
            raise InvalidInputError(msg=f'{len(duplicated_names)} annotated images map to the same output file name and would overwrite each other (e.g., {list(duplicated_names[:5])}). Check the annotation files for images listed more than once.', source=self.__class__.__name__)
        if self.sample_n is not None:
            annotations = annotations.sample(n=min(self.sample_n, len(annotations))).reset_index(drop=True)
        train_idx = set(random.sample(list(range(0, len(annotations))), int(len(annotations) * self.train_size)))
        bp_id_idx = np.array_split(np.array(range(0, int(len(body_part_headers) / 2))), len(self.names.keys()))
        bp_id_idx = [list(x) for x in bp_id_idx]
        keypoint_values = annotations[body_part_headers].values.astype(float)
        tasks = []
        for idx, (save_name, img_path) in enumerate(zip(annotations['save_name'], annotations['img_path'])):
            img_ext = '.png' if self.greyscale or self.clahe else os.path.splitext(img_path)[1]
            if idx in train_idx:
                img_save_path, lbl_save_path = os.path.join(self.img_train_dir, f'{save_name}{img_ext}'), os.path.join(self.lbl_train_dir, f'{save_name}.txt')
            else:
                img_save_path, lbl_save_path = os.path.join(self.img_val_dir, f'{save_name}{img_ext}'), os.path.join(self.lb_val_dir, f'{save_name}.txt')
            tasks.append((os.path.join(self.litpose_dir, img_path), img_save_path, lbl_save_path, keypoint_values[idx], bp_id_idx, self.greyscale, self.clahe, self.padding))
        if self.verbose:
            stdout_information(msg=f'Converting {len(tasks)} images on {self.core_cnt} core(s)...', source=self.__class__.__name__)
        if self.core_cnt == 1:
            for cnt, _ in enumerate(map(_litpose_to_yolo_worker, tasks)):
                if self.verbose:
                    stdout_information(msg=f'Processed image {cnt + 1}/{len(tasks)}...')
        else:
            pool = get_cpu_pool(core_cnt=self.core_cnt, verbose=self.verbose, source=self.__class__.__name__)
            try:
                for cnt, _ in enumerate(pool.imap_unordered(_litpose_to_yolo_worker, tasks, chunksize=16)):
                    if self.verbose:
                        stdout_information(msg=f'Processed image {cnt + 1}/{len(tasks)}...')
            finally:
                terminate_cpu_pool(pool=pool, force=False, verbose=self.verbose, source=self.__class__.__name__)
        create_yolo_keypoint_yaml(path=self.save_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=self.names, save_path=self.map_path, kpt_shape=(len(self.flip_idx), 3), flip_idx=self.flip_idx)
        if self.visualize:
            self._visualize(tasks=tasks)
        timer.stop_timer()
        stdout_success(msg=f'YOLO formated data saved in {self.save_dir} directory', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)

    def _visualize(self, tasks: list, batch_size: int = 50) -> None:
        """Draw keypoint + box overlays from the written image/label files, in batches so memory stays bounded when visualizing every image."""
        viz_tasks = tasks if self.visualize is True else random.sample(tasks, min(int(self.visualize), len(tasks)))
        viz_dir = os.path.join(self.save_dir, 'visualizations')
        for batch_start in range(0, len(viz_tasks), batch_size):
            samples = []
            for _, img_save_path, lbl_save_path, *_ in viz_tasks[batch_start:batch_start + batch_size]:
                with open(lbl_save_path, mode='r', encoding='utf-8') as f:
                    samples.append((get_fn_ext(filepath=img_save_path)[1], cv2.imread(img_save_path), f.read()))
            create_yolo_sample_visualizations(samples=samples, save_dir=viz_dir, names=tuple(self.names.values()), draw_labels=False, verbose=self.verbose, source=self.__class__.__name__, kpt_shape=(len(self.flip_idx), 3))


# if __name__ == "__main__":
#     runner = LitPose2YOLO(litpose_dir=r'I:\sina\project_5cam_cage21_22_0911_cropped',
#                           save_dir=r'I:\sina\yolo_project_5cam_cage21_22_0911_cropped',
#                           verbose=True,
#                           clahe=False,
#                           greyscale=False,
#                           visualize=150,
#                           sample_n=None,
#                           padding=0.15,
#                           core_cnt=4)
#     runner.run()