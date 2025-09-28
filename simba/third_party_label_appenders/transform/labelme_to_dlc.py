import itertools
import json
import os
from datetime import datetime
from typing import Optional, Union

import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2

from simba.mixins.image_mixin import ImageMixin
from simba.third_party_label_appenders.transform.utils import b64_to_arr
from simba.utils.checks import (check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_str,
                                check_valid_boolean)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    img_array_to_clahe, read_json)


class Labelme2DLC:

    """
    Convert labels from labelme format to DLC annotation format.

    .. note::
        See `labelme GitHub repo <https://github.com/wkentaro/labelme>`__.

    .. seealso::
       For DLC -> Labelme annotation conversion, see :func:`simba.third_party_label_appenders.transform.dlc_to_labelme.DLC2Labelme`

    :param Union[str, os.PathLike] labelme_dir: Directory with labelme json files.
    :param Optional[str] scorer: Name of the scorer (anticipated by DLC as header)
    :param bool greyscale: If True, convert images to grayscale.
    :param bool clahe: If True, apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    :param bool verbose: If True, prints progress.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to save the DLC annotations. If None, then same directory as labelme_dir with `_dlc_annotations` suffix.
    :return: None


    :example:
    >>> labelme_dir = r"D:\platea\ts_annotations"
    >>> runner = Labelme2DLC(labelme_dir=labelme_dir)
    >>> runner.run()
    """

    def __init__(self,
                 labelme_dir: Union[str, os.PathLike],
                 scorer: str = 'SN',
                 greyscale: bool = False,
                 clahe: bool = False,
                 verbose: bool = True,
                 save_dir: Optional[Union[str, os.PathLike]] = None) -> None:

        check_if_dir_exists(in_dir=labelme_dir)
        check_str(name=f'{self.__class__.__name__} scorer', value=scorer)
        self.annotation_paths = find_files_of_filetypes_in_directory(directory=labelme_dir, extensions=['.json'], raise_error=True, as_dict=True)
        if save_dir is None:
            self.save_dir = os.path.join(os.path.dirname(labelme_dir), os.path.basename(labelme_dir) + f'_dlc_annotations_{datetime.now().strftime("%Y%m%d%H%M%S")}')
            if not os.path.isdir(self.save_dir): os.makedirs(self.save_dir)
        else:
            check_if_dir_exists(in_dir=save_dir)
            self.save_dir = save_dir
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale', raise_error=True)
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe', raise_error=True)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        self.clahe, self.greyscale, self.scorer, self.labelme_dir, self.verbose = clahe, greyscale, scorer, labelme_dir, verbose
        self.file_cnt = len(list(self.annotation_paths.keys()))


    def run(self):
        timer = SimbaTimer(start=True)
        results_dict, images = {}, {}
        for file_cnt, (file_name, annot_path) in enumerate(self.annotation_paths.items()):
            if self.verbose:
                print(f'Reading labelme file {file_cnt+1}/{self.file_cnt}...')
            annot_data = read_json(x=annot_path)
            check_if_keys_exist_in_dict(data=annot_data, key=['shapes', 'imageData', 'imagePath'], name=annot_path)
            img_name = os.path.basename(annot_data['imagePath'])
            images[img_name] = b64_to_arr(annot_data['imageData'])
            if self.greyscale:
                images[img_name] = ImageMixin.img_to_greyscale(img=images[img_name])
            if self.clahe:
                images[img_name] = img_array_to_clahe(img=images[img_name])
            for bp_data in annot_data['shapes']:
                check_if_keys_exist_in_dict(data=bp_data, key=['label', 'points'], name=annot_path)
                point_x, point_y = bp_data['points'][0][0], bp_data['points'][0][1]
                lbl = bp_data['label']
                id = os.path.join('labeled-data', os.path.basename(self.labelme_dir), img_name)
                if id not in results_dict.keys():
                    results_dict[id] = {f'{lbl}': {'x': point_x, 'y': point_y}}
                else:
                    results_dict[id].update({f'{lbl}': {'x': point_x, 'y': point_y}})

        bp_names = set()
        for img, bp in results_dict.items(): bp_names.update(set(bp.keys()))
        col_names = list(itertools.product(*[[self.scorer], bp_names, ['x', 'y']]))
        columns = pd.MultiIndex.from_tuples(col_names)
        results = pd.DataFrame(columns=columns)
        results.columns.names = ['scorer', 'bodyparts', 'coords']
        for img, bp_data in results_dict.items():
            for bp_name, bp_cords in bp_data.items():
                results.at[img, (self.scorer, bp_name, 'x')] = bp_cords['x']
                results.at[img, (self.scorer, bp_name, 'y')] = bp_cords['y']

        for img_cnt, (img_name, img) in enumerate(images.items()):
            if self.verbose:
                print(f'Saving DLC file {img_cnt+1}/{self.file_cnt}...')
            img_save_path = os.path.join(self.save_dir, img_name)
            cv2.imwrite(img_save_path, img)
        save_path = os.path.join(self.save_dir, f'CollectedData_{self.scorer}.csv')
        results.to_csv(save_path)
        timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f'DLC annotations for {self.file_cnt} images saved in directory {self.save_dir}', elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)


# labelme_dir = r"D:\platea\ts_annotations"
# runner = Labelme2DLC(labelme_dir=labelme_dir)
# runner.run()