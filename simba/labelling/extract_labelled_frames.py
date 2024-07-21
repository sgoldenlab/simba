__author__ = "Simon Nilsson"

import os.path
from typing import Dict, List, Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_str, check_that_column_exist,
                                check_valid_boolean, check_valid_lst)
from simba.utils.enums import Dtypes
from simba.utils.errors import FrameRangeError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df,
                                    read_frm_of_video)


class AnnotationFrameExtractor(ConfigReader):
    """
    Extracts all human annotated frames where behavior is annotated as present into images within a SimBA project.

    :param str config_path: path to SimBA configparser.ConfigParser project_config.in
    :param list clfs: Names of classifiers to extract behavior-present images from.
    :param List[Union[str, os.PathLike]] data_paths: List of files with annotations to extract images from.
    :param int downsample: How much to downsample each image resolution with. If None or 1, then no Downsample.
    :param bool greyscale: If True, saves the images as greyscale.

    :example:
    >>> extractor = AnnotationFrameExtractor(config_path='project_folder/project_config.ini', clfs=['Sniffing', 'Attack'], settings={'downsample': 2})
    >>> extractor.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_paths: List[Union[str, os.PathLike]],
                 clfs: List[str],
                 downsample: Optional[Union[float, int]] = None,
                 img_format: Optional[Literal['png', 'webp', 'jpg']] = 'png',
                 greyscale: Optional[bool] = False):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        check_valid_lst(data=clfs, valid_dtypes=(str,), valid_values=self.clf_names, min_len=1, source='classifiers')
        check_valid_lst(data=data_paths, valid_dtypes=(str,), min_len=1, source='data_paths')
        for i in data_paths: check_file_exist_and_readable(file_path=i)
        if downsample is not None:
            check_float(name='downsample', value=downsample, min_value=1)
        check_str(name='img_format', value=img_format, options=('png', 'webp', 'jpg'))
        check_valid_boolean(source='greyscale', value=greyscale)
        self.clfs = clfs
        self.downsample = downsample
        self.img_format = img_format
        self.greyscale = greyscale
        self.data_paths = data_paths

    def run(self):
        if not os.path.exists(self.annotated_frm_dir):
            os.makedirs(self.annotated_frm_dir)
        for file_cnt, file_path in enumerate(self.data_paths):
            _, video_name, _ = get_fn_ext(filepath=file_path)
            video_path = find_video_of_file(video_dir=self.video_dir, filename=video_name, raise_error=True)
            df = read_df(file_path=file_path, file_type=self.file_type)
            for clf in self.clfs:
                check_that_column_exist(df=df, column_name=clf, file_name=file_path)
            cap = cv2.VideoCapture(video_path)
            video_meta_data = get_video_meta_data(video_path=video_path)
            if isinstance(self.downsample, (int, float)) and self.downsample > 1:
                size = (int(video_meta_data['width'] / self.downsample), int(video_meta_data['height'] / self.downsample))
            else:
                size = None
            for clf in self.clfs:
                save_dir = os.path.join(self.annotated_frm_dir, video_name, clf)
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                annot_idx = list(df.index[df[clf] == 1])
                for frm_cnt, frm in enumerate(annot_idx):
                    img = read_frm_of_video(video_path=cap, frame_index=frm, size=size, greyscale=self.greyscale)
                    cv2.imwrite(os.path.join(save_dir, f"{str(frm)}.{self.img_format}"), img)
                    print(f"Saved {clf} annotated img ({str(frm_cnt)}/{str(len(annot_idx))}), Video: {video_name} ...")

        self.timer.stop_timer()
        stdout_success(msg=f"Annotated frames saved in {self.annotated_frm_dir} directory", elapsed_time=self.timer.elapsed_time_str)


# test = AnnotationFrameExtractor(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 clfs=['Attack'],
#                                 downsample=4,
#                                 data_paths=['/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/targets_inserted/Together_1.csv'],
#                                 img_format='jpg',
#                                 greyscale=True)
# test.run()
