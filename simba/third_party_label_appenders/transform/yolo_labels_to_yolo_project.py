import os
import random
from typing import Tuple, Union

from simba.third_party_label_appenders.converters import create_yolo_yaml
from simba.utils.checks import (check_float, check_if_dir_exists,
                                check_valid_boolean, check_valid_tuple)
from simba.utils.enums import Options
from simba.utils.errors import InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (copy_files_to_directory, create_directory,
                                    find_files_of_filetypes_in_directory,
                                    recursive_file_search)
from simba.utils.warnings import MissingFileWarning


class YoloLabels2YoloProject:
    """
    Build a YOLO detection project from separate label and image directories.

    .. note::
       Train/val assignment is random each run unless you fix the random seed before calling :meth:`run`.

    .. seealso::
       For COCO or other sources into YOLO layout, see other classes under :mod:`simba.third_party_label_appenders.transform`.

    :param Union[str, os.PathLike] lbl_dir: Directory of YOLO-format ``.txt`` annotation files (one per image stem).
    :param Union[str, os.PathLike] img_dir: Directory of images. Extensions must be among ``Options.ALL_IMAGE_FORMAT_OPTIONS``.
    :param Union[str, os.PathLike] project_dir: Root output folder for the YOLO project (created if missing where supported by helpers).
    :param Tuple[str, ...] names: Class names in index order (first name → class id 0). Default ``('mouse',)``.
    :param float train_val_split: Fraction of matched samples for training; must be between 0.1 and 0.9. Default 0.7.
    :param bool recursive: If True, search ``img_dir`` and ``lbl_dir`` recursively for files in subdirectories. Default False.
    :return: None (writes files on disk). :meth:`run` also prints success via :func:`simba.utils.printing.stdout_success`.

    :example:
    >>> project = YoloLabels2YoloProject(lbl_dir=r"/path/to/labels", img_dir=r"/path/to/images", project_dir=r"/path/to/yolo_project", names=("mouse",), train_val_split=0.7)
    >>> project.run()
    """

    def __init__(self,
                 lbl_dir: Union[str, os.PathLike],
                 img_dir: Union[str, os.PathLike],
                 project_dir: Union[str, os.PathLike],
                 names: Tuple[str, ...] = ('mouse',),
                 train_val_split: float = 0.7,
                 recursive: bool = False):

        check_if_dir_exists(in_dir=lbl_dir, source=f'{self.__class__.__name__} lbl_dir')
        check_if_dir_exists(in_dir=img_dir, source=f'{self.__class__.__name__} img_dir')
        check_if_dir_exists(in_dir=project_dir, source=f'{self.__class__.__name__} project_dir')
        check_valid_tuple(x=names, source=f'{self.__class__.__name__} names', minimum_length=1, valid_dtypes=(str,))
        check_float(name=f'{self.__class__.__name__} train_val_split', value=train_val_split, min_value=0.1, max_value=0.9)
        check_valid_boolean(value=recursive, source=f'{self.__class__.__name__} recursive')

        if recursive:
            self.lbl_paths = recursive_file_search(directory=lbl_dir, extensions=['.txt'], as_dict=True, raise_error=True)
            self.img_paths = recursive_file_search(directory=img_dir, extensions=list(Options.ALL_IMAGE_FORMAT_OPTIONS.value), as_dict=True, raise_error=True)
        else:
            self.lbl_paths = find_files_of_filetypes_in_directory(directory=lbl_dir, extensions=['.txt'], as_dict=True, raise_error=True, sort_alphabetically=True)
            self.img_paths = find_files_of_filetypes_in_directory(directory=img_dir, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, as_dict=True, raise_error=True, sort_alphabetically=True)
        self.train_val_split, self.project_dir = train_val_split, project_dir
        self.names = {name: idx for idx, name in enumerate(names)}

        lbl_keys, img_keys = set(self.lbl_paths.keys()), set(self.img_paths.keys())
        missing_imgs = sorted(lbl_keys.difference(img_keys))
        missing_lbls = sorted(img_keys.difference(lbl_keys))
        self.sample_keys = sorted(lbl_keys.intersection(img_keys))
        if len(missing_imgs) > 0:
            MissingFileWarning(msg=f'Missing images for {len(missing_imgs)} annotations. These images will be skipped', source=self.__class__.__name__)
        if len(missing_lbls) > 0:
            MissingFileWarning(msg=f'Missing labels for {len(missing_lbls)} images. These images will be skipped', source=self.__class__.__name__)
        if len(self.sample_keys) == 0:
            raise InvalidInputError(msg='No matched image/label pairs found. Cannot create YOLO project.', source=self.__class__.__name__)

    def run(self):
        timer = SimbaTimer(start=True)
        self.img_dir, self.lbl_dir = os.path.join(self.project_dir, 'images'), os.path.join(self.project_dir, 'labels')
        self.img_train_dir, self.img_val_dir = os.path.join(self.img_dir, 'train'), os.path.join(self.img_dir, 'val')
        self.lbl_train_dir, self.lb_val_dir = os.path.join(self.lbl_dir, 'train'), os.path.join(self.lbl_dir, 'val')
        create_directory(paths=[self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lb_val_dir], overwrite=False)
        self.map_path = os.path.join(self.project_dir, 'map.yaml')

        train_keys = set(random.sample(self.sample_keys, int(len(self.sample_keys) * self.train_val_split)))
        val_keys = [x for x in self.sample_keys if x not in train_keys]

        train_imgs = [self.img_paths[k] for k in self.sample_keys if k in train_keys]
        val_imgs = [self.img_paths[k] for k in val_keys]
        train_lbls = [self.lbl_paths[k] for k in self.sample_keys if k in train_keys]
        val_lbls = [self.lbl_paths[k] for k in val_keys]

        copy_files_to_directory(file_paths=train_imgs, dir=self.img_train_dir, verbose=True)
        copy_files_to_directory(file_paths=val_imgs, dir=self.img_val_dir, verbose=True)

        copy_files_to_directory(file_paths=train_lbls, dir=self.lbl_train_dir, verbose=True, check_validity=False)
        copy_files_to_directory(file_paths=val_lbls, dir=self.lb_val_dir, verbose=True, check_validity=False)

        create_yolo_yaml(path=self.project_dir, train_path=self.img_train_dir, val_path=self.img_val_dir, names=self.names, save_path=self.map_path)
        timer.stop_timer()
        stdout_success(msg=f'YOLO bbox project created at {self.project_dir}.', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)

#
# x = YoloLabels2YoloProject(lbl_dir=r'F:\netholabs\moira\original\moira_lbls',
#                            img_dir=r'F:\netholabs\moira\original\images',
#                            project_dir=r"F:\netholabs\moira\original\moira_yolo_project",
#                            recursive=True)
# x.run()




