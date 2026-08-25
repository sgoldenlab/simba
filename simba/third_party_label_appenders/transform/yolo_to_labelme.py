import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import yaml

from simba.third_party_label_appenders.transform.utils import arr_to_b64
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_instance,
                                check_str, check_valid_boolean)
from simba.utils.enums import Options
from simba.utils.errors import InvalidInputError, NoDataError
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (create_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_img)
from simba.utils.warnings import (CorruptedFileWarning, DuplicateNamesWarning,
                                  InvalidValueWarning)
from simba.utils.yolo import detect_yolo_project_type

BBOX_VALUE_CNT = 4
LABELME_VERSION = '5.3.1'
SPLITS = ('train', 'val', 'test')


class Yolo2Labelme():
    r"""
    Convert a YOLO bounding-box project into a flat labelme directory so that the annotations can be edited by hand.

    All selected ``train``/``val``/``test`` splits are flattened into a single output directory holding one ``.json``
    file per image alongside a copy of that image, which is the layout labelme expects when opening a directory. Each
    YOLO ``class_id x_center y_center width height`` line becomes a labelme ``rectangle`` shape holding the two
    de-normalized corner points, labelled with the class name read from ``map.yaml``.

    .. seealso::
       For the reverse conversion - edited labelme rectangles back into a YOLO bounding-box project - see :class:`simba.third_party_label_appenders.transform.labelme_to_yolo.LabelmeBoundingBoxes2YoloBoundingBoxes`.
       To only inspect YOLO annotations without editing them, see :class:`simba.third_party_label_appenders.transform.yolo_to_imgs.Yolo2Imgs` and :class:`simba.plotting.yolo_annotation_visualizer.YOLOAnnotationVisualizer`.
       For auto-detecting the YOLO project type from a label file, see :func:`simba.utils.yolo.detect_yolo_project_type`.

    .. note::
       Flattening the splits discards the train/val membership of each image. If a file name occurs in more than one split, the later copy is saved with its split appended to the name (e.g. ``frm_1.json`` and ``frm_1_val.json``)
       so that it is not overwritten. Images that have no - or an empty - label file are saved with an empty ``shapes`` list so that annotations can be added to them.

    .. important::
       For YOLO bounding-box projects only - keypoint and segmentation projects raise an error rather than being silently reduced to bounding boxes.
       ``imageData`` is written as a JPEG-encoded base64 string, so passing the edited directory back through ``LabelmeBoundingBoxes2YoloBoundingBoxes`` - which rebuilds its images from
       ``imageData`` - re-encodes the images. Pass ``img_data=False`` to keep the json files small and have labelme read the copied images off disk instead, but note that the reverse converter then cannot be used.

    :param Union[str, os.PathLike] map_yaml_path: Path to the YOLO project ``map.yaml`` file. Requires the ``path`` and ``names`` keys.
    :param Union[str, os.PathLike] save_dir: Directory where the labelme ``.json`` files and the copied images are saved. Created if it does not exist.
    :param Optional[str] split: Which split to convert: ``'train'``, ``'val'``, ``'test'``, or ``'all'``. Default ``'all'``.
    :param bool img_data: If True, embeds each image in its ``.json`` file as a base64 string. Default ``True``.
    :param Optional[str] labelme_version: Version number encoded in the json files. Default ``'5.3.1'``.
    :param Optional[Dict[Any, Any]] flags: Flags included in the json files. Default None, which writes an empty dict.
    :param bool verbose: If True, prints progress. Default ``True``.
    :return: None

    :example:

    >>> runner = Yolo2Labelme(map_yaml_path=r"G:\netholabs\batch_0731\pellet_test\map.yaml", save_dir=r"G:\netholabs\batch_0731\pellet_test_labelme")
    >>> runner.run()
    """

    def __init__(self,
                 map_yaml_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 split: Optional[Literal['train', 'val', 'test', 'all']] = 'all',
                 img_data: bool = True,
                 labelme_version: Optional[str] = LABELME_VERSION,
                 flags: Optional[Dict[Any, Any]] = None,
                 verbose: bool = True) -> None:

        check_file_exist_and_readable(file_path=map_yaml_path)
        check_if_dir_exists(in_dir=os.path.dirname(save_dir), source=f'{self.__class__.__name__} save_dir', raise_error=True)
        create_directory(paths=save_dir, overwrite=False)
        check_str(name=f'{self.__class__.__name__} split', value=split, options=SPLITS + ('all',))
        check_str(name=f'{self.__class__.__name__} labelme_version', value=labelme_version)
        check_valid_boolean(value=[img_data, verbose], source=f'{self.__class__.__name__} img_data and verbose', raise_error=True)
        if flags is not None:
            check_instance(source=f'{self.__class__.__name__} flags', instance=flags, accepted_types=(dict,))

        with open(map_yaml_path, 'r') as f:
            self.yolo_map = yaml.safe_load(f)
        missing = [k for k in ('path', 'names') if k not in self.yolo_map]
        if len(missing) > 0:
            raise InvalidInputError(msg=f'The map.yaml file {map_yaml_path} is missing required keys: {missing}', source=self.__class__.__name__)

        names = self.yolo_map['names']
        if isinstance(names, dict):
            self.names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            self.names = {k: str(v) for k, v in enumerate(names)}
        else:
            raise InvalidInputError(msg=f'The names entry in the map.yaml file {map_yaml_path} has to be a dict or a list, got {type(names).__name__}', source=self.__class__.__name__)

        self.project_path, self.map_yaml_path = self.yolo_map['path'], map_yaml_path
        self.save_dir, self.split, self.img_data = save_dir, split, img_data
        self.labelme_version, self.verbose = labelme_version, verbose
        self.flags = {} if flags is None else flags

    def _find_image_label_pairs(self) -> List[Tuple[str, Union[str, None], str]]:
        """Helper returning one ``(image_path, label_path_or_None, split_name)`` tuple per image in the selected split(s)."""

        if self.split == 'all':
            splits = [s for s in SPLITS if s in self.yolo_map]
            if len(splits) == 0:
                raise InvalidInputError(msg=f'None of the splits {SPLITS} are present in the map.yaml file {self.map_yaml_path}', source=self.__class__.__name__)
        else:
            if self.split not in self.yolo_map:
                raise InvalidInputError(msg=f'The split {self.split} is not present in the map.yaml file {self.map_yaml_path}', source=self.__class__.__name__)
            splits = [self.split]

        pairs = []
        for split in splits:
            split_sub_path = str(self.yolo_map[split]).replace('\\', os.sep).replace('/', os.sep)
            img_dir = split_sub_path if os.path.isabs(split_sub_path) else os.path.normpath(os.path.join(self.project_path, split_sub_path))
            lbl_dir = img_dir.replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}')
            if lbl_dir == img_dir:
                lbl_dir = os.path.join(self.project_path, 'labels', split)
            if not os.path.isdir(img_dir):
                raise InvalidInputError(msg=f'The image directory of split {split} could not be found: {img_dir}', source=self.__class__.__name__)
            if not os.path.isdir(lbl_dir):
                raise InvalidInputError(msg=f'The label directory of split {split} could not be found: {lbl_dir}', source=self.__class__.__name__)
            img_files = find_files_of_filetypes_in_directory(directory=img_dir, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, as_dict=True, raise_error=True)
            lbl_files = find_files_of_filetypes_in_directory(directory=lbl_dir, extensions=['.txt'], as_dict=True, raise_error=False)
            lbl_files = {} if lbl_files is None else lbl_files
            for img_name, img_path in img_files.items():
                pairs.append((img_path, lbl_files.get(img_name, None), split))

        if len(pairs) == 0:
            raise NoDataError(msg=f'No images found in the YOLO project of the map.yaml file {self.map_yaml_path}', source=self.__class__.__name__)
        return pairs

    def _bbox_line_to_points(self, line: str, img_h: int, img_w: int) -> Union[List[List[float]], None]:
        """Helper converting one normalized YOLO bounding-box line into two labelme corner points clipped to the image."""

        parts = line.split()
        if len(parts) != BBOX_VALUE_CNT + 1:
            InvalidValueWarning(msg=f'A label line holds {len(parts) - 1} value(s) following the class id, expected {BBOX_VALUE_CNT}. Skipping annotation...', source=self.__class__.__name__)
            return None
        x_c, y_c, w, h = [float(v) for v in parts[1:]]
        x_min, x_max = max(0.0, (x_c - w / 2) * img_w), min(float(img_w), (x_c + w / 2) * img_w)
        y_min, y_max = max(0.0, (y_c - h / 2) * img_h), min(float(img_h), (y_c + h / 2) * img_h)
        return [[x_min, y_min], [x_max, y_max]]

    def run(self):
        timer = SimbaTimer(start=True)
        pairs = self._find_image_label_pairs()
        annotated_lbl_paths = [x[1] for x in pairs if x[1] is not None and os.path.getsize(x[1]) > 0]
        if len(annotated_lbl_paths) == 0:
            raise NoDataError(msg=f'All label files of the YOLO project of the map.yaml file {self.map_yaml_path} are missing or empty.', source=self.__class__.__name__)
        project_type = detect_yolo_project_type(label_path=annotated_lbl_paths[0])
        if project_type != 'bbox':
            raise InvalidInputError(msg=f'The YOLO project of the map.yaml file {self.map_yaml_path} is a {project_type} project. Only bbox projects can be converted to labelme rectangles.', source=self.__class__.__name__)

        saved_names, shape_cnt, empty_cnt, corrupt_cnt = {}, 0, 0, 0
        for file_cnt, (img_path, lbl_path, split) in enumerate(pairs):
            if self.verbose:
                print(f'Converting YOLO annotations to labelme, image {file_cnt + 1}/{len(pairs)} (split {split})...')
            try:
                img = read_img(img_path=img_path)
            except Exception:
                CorruptedFileWarning(msg=f'The image {img_path} could not be read and is skipped, together with its annotations.', source=self.__class__.__name__)
                corrupt_cnt += 1
                continue
            img_h, img_w = img.shape[:2]
            _, img_name, img_ext = get_fn_ext(filepath=img_path)
            if img_name in saved_names.keys():
                DuplicateNamesWarning(msg=f'The image name {img_name} is present in both the {saved_names[img_name]} and the {split} split. Saving the {split} copy as {img_name}_{split}...', source=self.__class__.__name__)
                img_name = f'{img_name}_{split}'
            saved_names[img_name] = split
            shapes = []
            if lbl_path is not None:
                with open(lbl_path, 'r') as f:
                    lbl_lines = [x.strip() for x in f.readlines()]
                for line in lbl_lines:
                    if len(line) == 0:
                        continue
                    class_id = int(float(line.split()[0]))
                    if class_id not in self.names.keys():
                        InvalidValueWarning(msg=f'The class id {class_id} in label file {lbl_path} is not present in the map.yaml names {list(self.names.keys())}. Skipping annotation...', source=self.__class__.__name__)
                        continue
                    points = self._bbox_line_to_points(line=line, img_h=img_h, img_w=img_w)
                    if points is None:
                        continue
                    shapes.append({'label': self.names[class_id],
                                   'points': points,
                                   'group_id': None,
                                   'description': "",
                                   'shape_type': 'rectangle',
                                   'flags': {}})
            if len(shapes) == 0:
                empty_cnt += 1
            shape_cnt += len(shapes)
            out = {'version': self.labelme_version,
                   'flags': self.flags,
                   'shapes': shapes,
                   'imagePath': f'{img_name}{img_ext}',
                   'imageData': arr_to_b64(img) if self.img_data else None,
                   'imageHeight': img_h,
                   'imageWidth': img_w}
            with open(os.path.join(self.save_dir, f'{img_name}.json'), 'w') as f:
                json.dump(out, f)
            shutil.copyfile(img_path, os.path.join(self.save_dir, f'{img_name}{img_ext}'))

        timer.stop_timer()
        if empty_cnt > 0:
            stdout_information(msg=f'{empty_cnt} of {len(pairs)} image(s) hold no annotations and are saved with an empty shapes list.', source=self.__class__.__name__)
        if corrupt_cnt > 0:
            stdout_information(msg=f'{corrupt_cnt} of {len(pairs)} image(s) could not be read and are excluded.', source=self.__class__.__name__)
        if self.verbose:
            stdout_success(msg=f'YOLO to labelme conversion complete. {shape_cnt} annotation(s) for {len(saved_names)} image(s) saved in directory {self.save_dir}.', elapsed_time=timer.elapsed_time_str)


# runner = Yolo2Labelme(map_yaml_path=r"G:\netholabs\batch_0731\food_pellet_3\map.yaml", save_dir=r"G:\netholabs\batch_0731\pellet_labelme_0825")
# runner.run()
