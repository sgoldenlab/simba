__author__ = "Simon Nilsson"

import os.path
from typing import Dict, List

import cv2

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_that_column_exist
from simba.utils.enums import Dtypes
from simba.utils.errors import FrameRangeError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df)


class AnnotationFrameExtractor(ConfigReader):
    """
    Extracts all human annotated frames where behavior is annotated as present into .pngs within a SimBA project.

    :param list clfs: Names of classifiers to extract behavior-present images from.
    :param dict settings: User-defined settings. E.g., how much to downsample the png images.
    :param str config_path: path to SimBA configparser.ConfigParser project_config.in

    .. note::
        Use settings={'downsample': 2} to downsample all .pngs to 0.5 original size for disk space savings.

    :example:
    >>> extractor = AnnotationFrameExtractor(config_path='project_folder/project_config.ini', clfs=['Sniffing', 'Attack'], settings={'downsample': 2})
    >>> extractor.run()
    """

    def __init__(self, clfs: List[str], settings: Dict[str, int], config_path: str):
        self.clfs = clfs
        self.settings = settings
        super().__init__(config_path=config_path)

    def run(self):
        if not os.path.exists(self.annotated_frm_dir):
            os.makedirs(self.annotated_frm_dir)
        for file_path in self.target_file_paths:
            _, video_name, _ = get_fn_ext(filepath=file_path)
            video_path = find_video_of_file(
                video_dir=self.video_dir, filename=video_name
            )
            df = read_df(file_path=file_path, file_type=self.file_type)
            for clf in self.clfs:
                check_that_column_exist(df=df, column_name=clf, file_name=file_path)
            cap = cv2.VideoCapture(video_path)
            video_meta_data = get_video_meta_data(video_path=video_path)
            for clf in self.clfs:
                save_dir = os.path.join(self.annotated_frm_dir, video_name, clf)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                annot_idx = list(df.index[df[clf] == 1])
                for frm_cnt, frm in enumerate(annot_idx):
                    cap.set(1, frm)
                    ret, img = cap.read()
                    if not ret:
                        raise FrameRangeError(
                            msg=f'Frame {str(frm+1)} is annotated as {clf} present. But frame {str(frm+1)} does not exist in video file {video_path}. The video file contains {video_meta_data["frame_count"]} frames.'
                        )
                    if self.settings["downsample"] != Dtypes.NONE.value:
                        img = cv2.resize(
                            img,
                            (
                                int(img.shape[1] / self.settings["downsample"]),
                                int((img.shape[0] / self.settings["downsample"])),
                            ),
                            cv2.INTER_NEAREST,
                        )
                    cv2.imwrite(os.path.join(save_dir, f"{str(frm)}.png"), img)
                    print(
                        f"Saved {clf} annotated img ({str(frm_cnt)}/{str(len(annot_idx))}), Video: {video_name}"
                    )
        self.timer.stop_timer()
        stdout_success(
            msg=f"Annotated frames saved in {self.annotated_frm_dir} directory",
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = AnnotationFrameExtractor(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 clfs=['Sniffing', 'Attack'],
#                                 settings={'downsample': 2})
# test.run()
