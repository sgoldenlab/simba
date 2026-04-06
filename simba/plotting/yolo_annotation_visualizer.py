import os
import random
from typing import Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2
import numpy as np
import yaml

from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_boolean)
from simba.utils.data import create_color_palette
from simba.utils.enums import Options
from simba.utils.errors import InvalidInputError, NoDataError
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext)
from simba.utils.yolo import detect_yolo_project_type

BBOX_VALUE_CNT = 4
KPT_DIM = 3

class YOLOAnnotationVisualizer(object):
    """
    Visualize YOLO annotation label files overlaid on their source images.

    .. seealso::
       For visualizing YOLO bounding-box inference results on video, see :func:`simba.plotting.yolo_visualize.YOLOVisualizer`.
       For visualizing YOLO keypoint pose-estimation results on video, see :func:`simba.plotting.yolo_pose_visualizer.YOLOPoseVisualizer`.
       For visualizing YOLO segmentation polygon results on video, see :func:`simba.plotting.yolo_seg_visualizer.YOLOSegmentationVisualizer`.
       For auto-detecting the YOLO project type from a label file, see :func:`simba.utils.yolo.detect_yolo_project_type`.

    :param Union[str, os.PathLike] map_yaml_path: Path to the YOLO project ``map.yaml`` file.
    :param Union[str, os.PathLike] save_dir: Directory where annotated images are saved.
    :param Optional[str] split: Which split to visualize: ``'train'``, ``'val'``, or ``'all'``. Default ``'all'``.
    :param Optional[int] n: Number of images to visualize. If ``None``, visualize every image. Default ``None``.
    :param Optional[int] circle_size: Radius of keypoint circles. If ``None``, computed from image dimensions.
    :param Optional[int] thickness: Line thickness for bounding boxes / polygon edges. If ``None``, computed from image dimensions.
    :param str palette: Color palette name (e.g. ``'Set1'``). Default ``'Set1'``.
    :param str img_format: Output image format extension. Default ``'.png'``.
    :param float seg_opacity: Opacity of filled segmentation polygons (0.0–1.0). Default ``0.5``.
    :param bool verbose: Print progress messages. Default ``True``.

    :example:
    >>> viz = YOLOAnnotationVisualizer(map_yaml_path=r'F:\netholabs\moira_lp_sam\map.yaml', save_dir=r'F:\netholabs\annotation_visualizations', n=400)
    >>> viz.run()
    >>> viz = YOLOAnnotationVisualizer(map_yaml_path=r'/path/to/map.yaml', save_dir=r'/path/to/output')
    >>> viz.run()
    >>> viz = YOLOAnnotationVisualizer(map_yaml_path=r'/path/to/map.yaml', save_dir=r'/path/to/output', n=50, circle_size=5, thickness=2, img_format='.jpeg')
    >>> viz.run()
    """

    def __init__(self,
                 map_yaml_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 split: Optional[Literal['train', 'val', 'all']] = 'all',
                 n: Optional[int] = None,
                 circle_size: Optional[int] = None,
                 thickness: Optional[int] = None,
                 palette: str = 'Set1',
                 img_format: str = '.png',
                 seg_opacity: float = 0.5,
                 verbose: bool = True):

        check_file_exist_and_readable(file_path=map_yaml_path)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        check_str(name=f'{self.__class__.__name__} split', value=split, options=('train', 'val', 'all'))
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose', raise_error=True)
        if n is not None:
            check_int(name=f'{self.__class__.__name__} n', value=n, min_value=1)
        if circle_size is not None:
            check_int(name=f'{self.__class__.__name__} circle_size', value=circle_size, min_value=1)
        if thickness is not None:
            check_int(name=f'{self.__class__.__name__} thickness', value=thickness, min_value=1)
        if not img_format.startswith('.'):
            img_format = f'.{img_format}'
        check_str(name=f'{self.__class__.__name__} img_format', value=img_format.lower(), options=('.png', '.jpeg', '.jpg', '.bmp', '.webp'))

        with open(map_yaml_path, 'r') as f:
            self.yolo_map = yaml.safe_load(f)

        required_keys = ['path', 'names']
        missing = [k for k in required_keys if k not in self.yolo_map]
        if len(missing) > 0:
            raise InvalidInputError(msg=f'map.yaml missing required keys: {missing}', source=self.__class__.__name__)

        self.project_path = self.yolo_map['path']
        self.names = self.yolo_map['names']
        self.kpt_shape = self.yolo_map.get('kpt_shape', None)
        self.save_dir = save_dir
        self.split = split
        self.n = n
        self.circle_size = circle_size
        self.thickness = thickness
        self.palette = palette
        self.img_format = img_format.lower()
        self.seg_opacity = seg_opacity
        self.verbose = verbose

    def _find_image_label_pairs(self) -> List[Tuple[str, str]]:
        splits = []
        if self.split == 'all':
            for key in ('train', 'val', 'test'):
                if key in self.yolo_map:
                    splits.append(key)
        else:
            if self.split not in self.yolo_map:
                raise InvalidInputError(msg=f'Split "{self.split}" not found in map.yaml', source=self.__class__.__name__)
            splits.append(self.split)

        pairs = []
        for s in splits:
            img_dir = os.path.join(self.project_path, self.yolo_map[s])
            if not os.path.isabs(img_dir):
                img_dir = os.path.normpath(os.path.join(self.project_path, self.yolo_map[s]))

            lbl_dir_candidate = img_dir.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)
            if lbl_dir_candidate == img_dir:
                lbl_dir_candidate = os.path.join(self.project_path, 'labels', s)

            if not os.path.isdir(img_dir):
                raise InvalidInputError(msg=f'Image directory not found: {img_dir}', source=self.__class__.__name__)
            if not os.path.isdir(lbl_dir_candidate):
                raise InvalidInputError(msg=f'Label directory not found: {lbl_dir_candidate}', source=self.__class__.__name__)

            img_files = find_files_of_filetypes_in_directory(directory=img_dir, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, as_dict=True, raise_error=True)
            lbl_files = find_files_of_filetypes_in_directory(directory=lbl_dir_candidate, extensions=['.txt'], as_dict=True, raise_error=False)
            if lbl_files is None:
                lbl_files = {}

            for img_name, img_path in img_files.items():
                if img_name in lbl_files:
                    pairs.append((img_path, lbl_files[img_name]))

        if len(pairs) == 0:
            raise NoDataError(msg='No matched image/label pairs found.', source=self.__class__.__name__)
        return pairs

    @staticmethod
    def _parse_bbox_line(parts: List[str], img_w: int, img_h: int) -> Tuple[int, np.ndarray]:
        class_id = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        xc_px, yc_px = xc * img_w, yc * img_h
        w_px, h_px = w * img_w, h * img_h
        x1 = int(xc_px - w_px / 2)
        y1 = int(yc_px - h_px / 2)
        x2 = int(xc_px + w_px / 2)
        y2 = int(yc_px + h_px / 2)
        return class_id, np.array([x1, y1, x2, y2], dtype=np.int32)

    @staticmethod
    def _parse_keypoint_line(parts: List[str], img_w: int, img_h: int) -> Tuple[int, np.ndarray, np.ndarray]:
        class_id = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        xc_px, yc_px = xc * img_w, yc * img_h
        w_px, h_px = w * img_w, h * img_h
        x1 = int(xc_px - w_px / 2)
        y1 = int(yc_px - h_px / 2)
        x2 = int(xc_px + w_px / 2)
        y2 = int(yc_px + h_px / 2)
        bbox = np.array([x1, y1, x2, y2], dtype=np.int32)
        kp_values = [float(v) for v in parts[5:]]
        kps = []
        for i in range(0, len(kp_values), KPT_DIM):
            kx = int(kp_values[i] * img_w)
            ky = int(kp_values[i + 1] * img_h)
            vis = int(kp_values[i + 2])
            kps.append((kx, ky, vis))
        return class_id, bbox, np.array(kps, dtype=np.int32)

    @staticmethod
    def _parse_seg_line(parts: List[str], img_w: int, img_h: int) -> Tuple[int, np.ndarray]:
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        points = []
        for i in range(0, len(coords), 2):
            px = int(coords[i] * img_w)
            py = int(coords[i + 1] * img_h)
            points.append([px, py])
        return class_id, np.array(points, dtype=np.int32)

    def _draw_bbox(self, img: np.ndarray, class_id: int, bbox: np.ndarray, color: tuple, thickness: int) -> np.ndarray:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness, lineType=cv2.LINE_AA)
        label = self.names.get(class_id, str(class_id))
        cv2.putText(img, label, (bbox[0], max(bbox[1] - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, max(1, thickness // 2), cv2.LINE_AA)
        return img

    def _draw_keypoints(self, img: np.ndarray, class_id: int, bbox: np.ndarray, kps: np.ndarray, colors: list, circle_size: int, thickness: int) -> np.ndarray:
        color = tuple(int(c) for c in colors[0])
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness, lineType=cv2.LINE_AA)
        label = self.names.get(class_id, str(class_id))
        cv2.putText(img, label, (bbox[0], max(bbox[1] - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, max(1, thickness // 2), cv2.LINE_AA)
        for kp_idx, kp in enumerate(kps):
            if kp[2] > 0:
                clr_idx = min(kp_idx + 1, len(colors) - 1)
                clr = tuple(int(c) for c in colors[clr_idx])
                cv2.circle(img, (int(kp[0]), int(kp[1])), circle_size, clr, -1)
        return img

    def _draw_segmentation(self, img: np.ndarray, class_id: int, polygon: np.ndarray, color: tuple, thickness: int) -> np.ndarray:
        overlay = img.copy()
        pts = polygon.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.fillPoly(overlay, [pts], color=color)
        cv2.addWeighted(overlay, self.seg_opacity, img, 1 - self.seg_opacity, 0, img)
        label = self.names.get(class_id, str(class_id))
        cx, cy = int(polygon[:, 0].mean()), int(polygon[:, 1].mean())
        cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, max(1, thickness // 2), cv2.LINE_AA)
        return img

    def run(self):
        timer = SimbaTimer(start=True)
        pairs = self._find_image_label_pairs()

        first_lbl_path = pairs[0][1]
        project_type = detect_yolo_project_type(label_path=first_lbl_path)
        if self.verbose:
            stdout_information(msg=f'Detected YOLO project type: {project_type} ({len(pairs)} image/label pairs found)', source=self.__class__.__name__)

        if self.n is not None:
            sample_n = min(self.n, len(pairs))
            pairs = random.sample(pairs, sample_n)

        n_classes = len(self.names)
        kp_count = 0
        if project_type == 'keypoint' and self.kpt_shape is not None:
            kp_count = self.kpt_shape[0]
        palette_size = max(n_classes, kp_count + 1, 10)
        class_colors = create_color_palette(pallete_name=self.palette, increments=palette_size)

        for img_cnt, (img_path, lbl_path) in enumerate(pairs):
            img = cv2.imread(img_path)
            if img is None:
                if self.verbose:
                    stdout_information(msg=f'Could not read image: {img_path}, skipping...', source=self.__class__.__name__)
                continue
            img_h, img_w = img.shape[:2]

            if self.circle_size is None:
                circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(img_w, img_h), circle_frame_ratio=80)
            else:
                circle_size = self.circle_size
            if self.thickness is None:
                thickness = max(1, circle_size)
            else:
                thickness = self.thickness

            with open(lbl_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                if project_type == 'bbox':
                    class_id, bbox = self._parse_bbox_line(parts, img_w, img_h)
                    color = tuple(int(c) for c in class_colors[class_id % len(class_colors)])
                    img = self._draw_bbox(img, class_id, bbox, color, thickness)

                elif project_type == 'keypoint':
                    class_id, bbox, kps = self._parse_keypoint_line(parts, img_w, img_h)
                    colors = class_colors[:max(kps.shape[0] + 1, 1)]
                    img = self._draw_keypoints(img, class_id, bbox, kps, colors, circle_size, thickness)

                elif project_type == 'segmentation':
                    class_id, polygon = self._parse_seg_line(parts, img_w, img_h)
                    color = tuple(int(c) for c in class_colors[class_id % len(class_colors)])
                    img = self._draw_segmentation(img, class_id, polygon, color, thickness)

            _, img_name, _ = get_fn_ext(filepath=img_path)
            save_path = os.path.join(self.save_dir, f'{img_name}{self.img_format}')
            cv2.imwrite(save_path, img)
            if self.verbose:
                stdout_information(msg=f'Annotated image {img_cnt + 1}/{len(pairs)} saved ({img_name})', source=self.__class__.__name__)

        timer.stop_timer()
        stdout_success(msg=f'{len(pairs)} annotated images saved in {self.save_dir}', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)


#if __name__ == '__main__':
# viz = YOLOAnnotationVisualizer(map_yaml_path=r"E:\open_video\open_field_2\yolo_bbox_project\map.yaml",
#                                    save_dir=r"E:\open_video\open_field_2\yolo_bbox_project\annotations_imgs",
#                                    n=150)
# viz.run()
