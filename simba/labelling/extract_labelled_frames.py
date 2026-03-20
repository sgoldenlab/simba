__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os.path
from typing import Dict, List, Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_file_exist_and_readable, check_float, check_str, check_that_column_exist, check_valid_lst, check_valid_boolean, check_if_df_field_is_boolean)
from simba.utils.printing import stdout_success, stdout_information
from simba.utils.read_write import (find_video_of_file, get_fn_ext, get_video_meta_data, read_df, read_frm_of_video, create_directory, check_valid_dataframe)
from simba.utils.warnings import FrameRangeWarning
from simba.utils.data import detect_bouts
from simba.mixins.plotting_mixin import PlottingMixin

IMG_FORMATS = ('png', 'webp', 'jpg')
VIDEO_FORMATS = ('mp4', 'avi', 'webm')

class AnnotationFrameExtractor(ConfigReader):
    """
    Extract frames annotated as behavior-present and save them as image files.

    :param Union[str, os.PathLike] config_path: Path to the SimBA ``project_config.ini`` file.
    :param List[Union[str, os.PathLike]] data_paths: Annotation file paths to read labels from.
    :param List[str] clfs: Names of classifiers to extract behavior-present images for.
    :param Optional[Union[float, int]] img_downsample_factor: Optional image downsampling factor. If ``None`` or ``1``, no downsampling is applied.
    :param Optional[Literal['png', 'webp', 'jpg']] img_format: Output image format.
    :param Optional[bool] img_greyscale: If ``True``, save images in grayscale.

    :example:
    >>> extractor = AnnotationFrameExtractor(
    ...     config_path='project_folder/project_config.ini',
    ...     data_paths=['project_folder/csv/targets_inserted/video_1.csv'],
    ...     clfs=['Sniffing', 'Attack'],
    ...     img_downsample_factor=2,
    ...     img_format='png',
    ...     img_greyscale=False
    ... )
    >>> extractor.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_paths: List[Union[str, os.PathLike]],
                 clfs: List[str],
                 img_downsample_factor: Optional[Union[float, int]] = None,
                 img_format: Optional[Literal['png', 'webp', 'jpg']] = None,
                 img_greyscale: Optional[bool] = False):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        check_valid_lst(data=clfs, valid_dtypes=(str,), valid_values=self.clf_names, min_len=1, source='classifiers')
        check_valid_lst(data=data_paths, valid_dtypes=(str,), min_len=1, source='data_paths')
        for i in data_paths: check_file_exist_and_readable(file_path=i)
        if img_downsample_factor is not None: check_float(name=f'{self.__class__.__name__} img_downsample_factor', value=img_downsample_factor, min_value=1.0)
        if img_format is not None: check_str(name=f'{self.__class__.__name__} img_format', value=img_format, options=IMG_FORMATS)
        if img_greyscale is not None: check_valid_boolean(value=img_greyscale, source=f'{self.__class__.__name__} img_greyscale', raise_error=True)
        self.img_downsample_factor, self.img_format, self.video_format = img_downsample_factor, img_format
        self.img_greyscale, self.data_paths, self.clfs = img_greyscale, data_paths, clfs

        self.video_lk = {}
        for file_cnt, file_path in enumerate(self.data_paths):
            _, video_name, _ = get_fn_ext(filepath=file_path)
            video_path = find_video_of_file(video_dir=self.video_dir, filename=video_name, raise_error=True)
            self.video_lk[video_name] = video_path

    def run(self):
        create_directory(paths=os.path.join(self.annotated_frm_dir, 'images'), overwrite=False, verbose=False)
        for file_cnt, data_path in enumerate(self.data_paths):
            video_name = get_fn_ext(filepath=data_path)[1]
            data_df = read_df(file_path=data_path, file_type=self.file_type)
            check_valid_dataframe(df=data_df, source=f'{self.__class__.__name__} {data_path}', required_fields= self.clfs)
            for field in self.clfs: check_if_df_field_is_boolean(df=data_df, field=field, df_name=data_path)
            video_path = self.video_lk[video_name]
            _ = get_video_meta_data(video_path=video_path)
            cap, size = cv2.VideoCapture(video_path), None
            for clf in self.clfs:
                clf_annot_idx = list(data_df.index[data_df[clf] == 1])
                if len(clf_annot_idx) == 0:
                    FrameRangeWarning(msg=f'No annotations found for classifier {clf} in video {video_name}. Skipping the classifier from file {data_path} ....', source=self.__class__.__name__)
                    continue
                for frm_cnt, frm in enumerate(clf_annot_idx):
                    img = read_frm_of_video(video_path=cap, frame_index=frm, size=size, greyscale=self.img_greyscale)
                    img_save_path = os.path.join(self.annotated_frm_dir, 'imgs', f'{video_name}_{clf}_{frm_cnt}.{self.img_format}')
                    cv2.imwrite(img_save_path, img)
                    stdout_information(f"Saved {clf} annotated image for classifier {clf} ({str(frm_cnt)}/{str(len(clf_annot_idx))}), video: {file_cnt+1}/{len(self.data_paths)}, ({video_name}) ...")
        self.timer.stop_timer()
        stdout_success(msg=f"Annotated frames saved in {self.annotated_frm_dir} directory", elapsed_time=self.timer.elapsed_time_str)



# test = AnnotationFrameExtractor(config_path=r'E:\troubleshooting\mitra\project_folder\project_config.ini',
#                                 clfs=['grooming'],
#                                 img_downsample_factor=4,
#                                 data_paths=[r"E:\troubleshooting\mitra\project_folder\csv\targets_inserted\grooming\502_MA141_Gi_Saline_0517.csv"],
#                                 img_format=None,
#                                 img_greyscale=True)
# test.run()
