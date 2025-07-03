import json
import os
from typing import Any, Dict, Optional, Union

import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.third_party_label_appenders.transform.utils import arr_to_b64
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_instance,
                                check_valid_boolean)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_img, recursive_file_search


class DLC2Labelme():

    """
    Convert a folder of DLC annotations into labelme json format.

    .. seealso::
       For Labelme -> DLC annotation conversion, see :func:`simba.third_party_label_appenders.transform.labelme_to_dlc.Labelme2DLC`

    :param Union[str, os.PathLike] dlc_dir: Folder with DLC annotations. I.e., directory inside
    :param Union[str, os.PathLike] save_dir: Directory to where to save the labelme json files.
    :param Optional[str] labelme_version: Version number encoded in the json files.
    :param Optional[Dict[Any, Any] flags: Flags included in the json files.
    :param Optional[bool] verbose: If True, prints progress.
    :return: None

    :example:
    >>> DLC2Labelme(dlc_dir="D:\TS_DLC\labeled-data\ts_annotations", save_dir="C:\troubleshooting\coco_data\labels\test").run()
    >>> DLC2Labelme(dlc_dir=r'D:\rat_resident_intruder\dlc_data\WIN_20190816081353', save_dir=r'D:\rat_resident_intruder\labelme').run()
    """

    def __init__(self,
                 dlc_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 labelme_version: Optional[str] = '5.3.1',
                 flags: Optional[Dict[Any, Any]] = None,
                 verbose: bool = True,
                 greyscale: bool = False,
                 clahe: bool = False) -> None:

        check_if_dir_exists(dlc_dir, source=f'{self.__class__.__name__} dlc_dir')
        check_if_dir_exists(save_dir, source=f'{self.__class__.__name__} save_dir')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale')
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe')
        self.data_paths = recursive_file_search(directory=dlc_dir, substrings='CollectedData', extensions='csv', raise_error=True)
        self.version, self.verbose, self.save_dir = labelme_version, verbose, save_dir
        if flags is not None:
            check_instance(source=f'{self.__class__.__name__} flags', instance=flags, accepted_types=(dict,))
        self.flags = {} if flags is None else {}
        self.clahe, self.greyscale = clahe, greyscale

    def run(self):
        timer = SimbaTimer(start=True)
        self.body_parts_per_file, self.filecnt = {}, 0
        for file_cnt, file_path in enumerate(self.data_paths):
            file_dir = os.path.dirname(file_path)
            video_name = os.path.basename(os.path.dirname(file_path))
            body_part_headers = ['image']
            annotation_data = pd.read_csv(file_path, header=[0, 1, 2])
            body_parts = []
            for i in annotation_data.columns[1:]:
                if 'unnamed:' not in i[1].lower() and i[1] not in body_parts:
                    body_parts.append(i[1])
            for i in body_parts:
                body_part_headers.append(f'{i}_x'); body_part_headers.append(f'{i}_y')
            self.body_parts_per_file[file_path] = body_part_headers
            print(annotation_data)
            annotation_data.columns = body_part_headers
            for cnt, (idx, idx_data) in enumerate(annotation_data.iterrows()):
                if self.verbose:
                    print(f'Processing image {cnt + 1}/{len(annotation_data)}... (video {file_cnt + 1}/{len(self.data_paths)} ({video_name}))')
                _, img_name, ext = get_fn_ext(filepath=idx_data['image'])
                video_img_name = f'{video_name}.{img_name}'
                img_path = os.path.join(file_dir, os.path.join(f'{img_name}{ext}'))
                check_file_exist_and_readable(file_path=img_path)
                img = read_img(img_path=img_path, clahe=self.clahe, greyscale=self.greyscale)
                idx_data = idx_data.to_dict()
                shapes = []
                for bp_name in body_parts:
                    img_shapes = {'label': bp_name,
                                  'points': [idx_data[f'{bp_name}_x'], idx_data[f'{bp_name}_y']],
                                  'group_id': None,
                                  'description': "",
                                  'shape_type': 'point',
                                  'flags': {}}
                    shapes.append(img_shapes)
                out = {"version": self.version,
                       'flags': self.flags,
                       'shapes': shapes,
                       'imagePath': img_path,
                       'imageData': arr_to_b64(img),
                       'imageHeight': img.shape[0],
                       'imageWidth': img.shape[1]}
                save_path = os.path.join(self.save_dir, f'{video_img_name}.json')
                with open(save_path, "w") as f:
                    json.dump(out, f)
                self.filecnt += 1
        timer.stop_timer()
        if self.verbose:
            stdout_success(f'Labelme data for {self.filecnt} image(s) saved in {self.save_dir} directory', elapsed_time=timer.elapsed_time_str)


#DLC2Labelme(dlc_dir=r'D:\rat_resident_intruder\dlc_data\WIN_20190816081353', save_dir=r'D:\rat_resident_intruder\labelme').run()

# DLC_DIR = r"D:\TS_DLC\labeled-data\ts_annotations"
# SAVE_DIR = r"C:\troubleshooting\coco_data\labels\test"
#
# runner = DLC2Labelme(dlc_dir=DLC_DIR, save_dir=SAVE_DIR)
# runner.run()
#


    #>>> DLC2Labelme(dlc_dir=r'D:\rat_resident_intruder\dlc_data\WIN_20190816081353', save_dir=r'D:\rat_resident_intruder\labelme').run()