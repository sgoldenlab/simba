import json
import os
from datetime import datetime
from typing import Optional, Tuple, Union

import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np

from simba.mixins.image_mixin import ImageMixin
from simba.third_party_label_appenders.transform.utils import (
    arr_to_b64, b64_to_arr, normalize_img_dict, scale_pose_img_sizes)
from simba.utils.checks import (check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_str,
                                check_valid_boolean, check_valid_tuple)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    img_array_to_clahe)


class LabelMe2DataFrame:
    """
    Convert a directory of labelme .json files into a pandas dataframe.

    .. note::
       The images are stores as a 64-bit bytestring under the ``image`` header of the output dataframe.

    :param Union[str, os.PathLike] labelme_dir: Directory with labelme json files.
    :param Optional[bool] greyscale: If True, converts the labelme images to greyscale if in rgb format. Default: False.
    :param Optional[bool] pad: If True, checks if all images are the same size and if not; pads the images with black border so all images are the same size.
    :param Union[Literal['min', 'max'], Tuple[int, int]] size: The size of the output images. Can be the smallesgt (min) the largest (max) or a tuple with the width and height of the images. Automatically corrects the labels to account for the image size.
    :param Optional[bool] normalize: If true, normalizes the images. Default: False.
    :param Optional[Union[str, os.PathLike]] save_path: The location where to store the dataframe. If None, then returns the dataframe. Default: None.

    :rtype: Union[None, pd.DataFrame]

    :example I:
    >>> LABELME_DIR = r'C:\troubleshooting\coco_data\labels\test_2'
    >>> runner = LabelMe2DataFrame(labelme_dir=LABELME_DIR)
    >>> runner.run()

    :example II:
    >>> LABELME_DIR = r'C:\troubleshooting\coco_data\labels\test_2'
    >>> runner = LabelMe2DataFrame(labelme_dir=LABELME_DIR, greyscale=True, pad=True, normalize=True, size='min')
    >>> runner.run()

    """

    def __init__(self,
                 labelme_dir: Union[str, os.PathLike],
                 greyscale: Optional[bool] = False,
                 clahe: bool = False,
                 pad: Optional[bool] = False,
                 size: Union[Literal['min', 'max'], Tuple[int, int]] = None,
                 normalize: bool = False,
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 verbose: bool = True):

        if save_path is not None:
            if os.path.isdir(save_path):
                save_path = os.path.join(save_path, f'labelme_data_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
        save_path = save_path
        if size is not None:
            if isinstance(size, tuple):
                check_valid_tuple(source=f'{self.__class__.__name__} size', accepted_lengths=(2,), valid_dtypes=(int,), x=size)
            elif isinstance(size, str):
                check_str(name=f'{self.__class__.__name__} size', value=size, options=('min', 'max',), raise_error=True)
        check_if_dir_exists(in_dir=labelme_dir)
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale', raise_error=True)
        check_valid_boolean(value=pad, source=f'{self.__class__.__name__} pad', raise_error=True)
        check_valid_boolean(value=normalize, source=f'{self.__class__.__name__} normalize', raise_error=True)
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe', raise_error=True)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        self.annotation_paths = find_files_of_filetypes_in_directory(directory=labelme_dir, extensions=['.json'], raise_error=True)
        self.greyscale, self.pad, self.size = greyscale, pad, size
        self.normalize, self.verbose, self.clahe = normalize, verbose, clahe
        self.labelme_dir, self.save_path = labelme_dir, save_path


    def run(self):
        timer = SimbaTimer(start=True)
        images, annotations = {}, []
        for cnt, annot_path in enumerate(self.annotation_paths):
            if self.verbose:
                print(f'Reading annotation file {cnt + 1}/{len(self.annotation_paths)}...')
            with open(annot_path) as f:
                annot_data = json.load(f)
            check_if_keys_exist_in_dict(data=annot_data, key=['shapes', 'imageData'], name=annot_path)
            img_name = annot_data['imagePath']
            images[img_name] = b64_to_arr(annot_data['imageData'])
            if self.greyscale:
                images[img_name] = ImageMixin.img_to_greyscale(img=images[img_name])
            if self.clahe:
                images[img_name] = img_array_to_clahe(img=images[img_name])
            img_data = {}
            for bp_data in annot_data['shapes']:
                check_if_keys_exist_in_dict(data=bp_data, key=['label', 'points'], name=annot_path)
                point_x, point_y = bp_data['points'][0], bp_data['points'][1]
                lbl = bp_data['label']
                img_data[f'{lbl}_x'], img_data[f'{lbl}_y'] = point_x, point_y
            img_data['image_name'] = img_name
            annotations.append(pd.DataFrame.from_dict(img_data, orient='index').T)
        if self.pad:
            if self.verbose: print('Padding images...')
            images = ImageMixin.pad_img_stack(image_dict=images)
        if self.normalize:
            if self.verbose: print('Normalizing images...')
            images = normalize_img_dict(img_dict=images)
        img_lst = []
        for k, v in images.items():
            img_lst.append(arr_to_b64(v))
        out = pd.concat(annotations).reset_index(drop=True)
        out['image'] = img_lst
        if self.size is not None:
            if self.verbose: print('Resizing images...')
            pose_data = out.drop(['image', 'image_name'], axis=1)
            pose_data_arr = pose_data.values.reshape(-1, int(pose_data.shape[1] / 2), 2).astype(np.float32)
            new_pose, out['image'] = scale_pose_img_sizes(pose_data=pose_data_arr, imgs=list(out['image']), size=self.size)
            new_pose = new_pose.reshape(pose_data.shape[0], pose_data.shape[1])
            out.iloc[:, : new_pose.shape[1]] = new_pose
        if self.save_path is None:
            return out
        else:
            if self.verbose: print('Saving CSV file...')
            out.to_csv(self.save_path)
            timer.stop_timer()
            if self.verbose:
                stdout_success(msg=f'Labelme CSV file saved at {self.save_path}', elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)


# LABELME_DIR = r'C:\troubleshooting\coco_data\labels\test_2'
# runner = LabelMe2DataFrame(labelme_dir=LABELME_DIR)
#
#
# #runner.run()
#
# LABELME_DIR = r'C:\troubleshooting\coco_data\labels\test_2'
# runner = LabelMe2DataFrame(labelme_dir=LABELME_DIR, greyscale=True, pad=True, normalize=True, size='max', save_path=r'D:\imgs')
# runner.run()
#labelme_to_df(labelme_dir=r'C:\troubleshooting\coco_data\labels\test_2').run()
#    >>> labelme_to_df(labelme_dir=r'C:\troubleshooting\coco_data\labels\test_read', greyscale=False, pad=False, normalize=False, size='min').run()