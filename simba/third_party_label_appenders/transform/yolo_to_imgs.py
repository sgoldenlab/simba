import os
from typing import Optional, Union

import cv2
import numpy as np

from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import check_float, check_if_dir_exists, check_str
from simba.utils.data import create_color_palettes
from simba.utils.enums import Formats, Options
from simba.utils.read_write import read_img, recursive_file_search


class Yolo2Imgs():
    """
    Overlay YOLO-format annotations (bounding boxes and, where present, pose keypoints) onto their source
    images for visual inspection.

    .. seealso::
       To assemble a YOLO project (train/val split + ``map.yaml``) from separate label and image
       directories, see :class:`simba.third_party_label_appenders.transform.yolo_labels_to_yolo_project.YoloLabels2YoloProject`.

    :param Union[str, os.PathLike] yolo_dir: Directory holding the YOLO ``.txt`` label files and their images (searched recursively).
    :param Union[str, os.PathLike] save_dir: Directory where the annotated images are written.
    :param Optional[str] palette: Named SimBA color palette used for the annotation overlays. Default ``'Set1'``.
    :param Optional[Union[float, int]] circle_size: Radius, in pixels, of drawn keypoint circles. If None, a size is derived automatically from each image's dimensions.

    :example:

    >>> runner = Yolo2Imgs(yolo_dir=r"/path/to/yolo_labels", save_dir=r"/path/to/annotated_imgs")
    >>> runner.run()
    """

    def __init__(self,
                 yolo_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 palette: Optional[str] = None,
                 circle_size: Optional[Union[float, int]] = None):

        check_if_dir_exists(in_dir=yolo_dir, source=f'{self.__class__.__name__} yolo_dir', raise_error=True)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir', raise_error=True)
        pose_palettes = Options.PALETTE_OPTIONS_CATEGORICAL.value + Options.PALETTE_OPTIONS.value
        palette = 'Set1' if palette is None else palette
        check_str(name=f'{self.__class__.__name__} palette', value=palette, options=pose_palettes)
        if circle_size is not None: check_float(name=f'{self.__class__.__name__} circle_size', value=circle_size, min_value=0.1)
        self.yolo_dir, self.save_dir, self.palette = yolo_dir, save_dir, palette
        self.circle_size = circle_size
    def run(self):
        lbl_paths = recursive_file_search(directory=self.yolo_dir, extensions=['.txt'], as_dict=True, raise_error=True)
        img_paths = recursive_file_search(directory=self.yolo_dir, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, as_dict=True, raise_error=True)

        for lbl_cnt, (lbl_name, lbl_path) in enumerate(lbl_paths.items()):
            img = read_img(img_path=img_paths[lbl_name])
            h, w = img.shape[:2]
            img_circle = self.circle_size if self.circle_size is not None else PlottingMixin().get_optimal_circle_size(frame_size=(w, h), circle_frame_ratio=100)
            with open(lbl_path, "r") as file: lbls = file.read().strip().split('\n')
            lbls = [x.strip().split() for x in lbls if x.strip()]
            for obs in lbls:
                if len(obs) < 5:
                    continue
                obs_lbl = [x for x in obs if x != '']
                obs_bbox = [float(x) for x in obs_lbl[1:5]]
                obs_kp = [float(x) for x in obs_lbl[5:]]
                if len(obs_kp) % 3 != 0:
                    continue
                obs_kp = [obs_kp[i:i + 3] for i in range(0, len(obs_kp), 3)]

                # Convert bbox from normalized to pixel
                box_cx, box_cy = int(obs_bbox[0] * w), int(obs_bbox[1] * h)
                box_w, box_h = int(obs_bbox[2] * w), int(obs_bbox[3] * h)
                tl = (int(box_cx - box_w / 2), int(box_cy - box_h / 2))
                br = (int(box_cx + box_w / 2), int(box_cy + box_h / 2))
                color_lst = create_color_palettes(1, len(obs_kp) + 1)[0]

                img = cv2.rectangle(img, tl, br, tuple(color_lst[0]), 2, lineType=-1)
                # for kp_cnt, kp in enumerate(obs_kp):
                #     if kp[2] != 0:
                #         x, y = int(kp[0] * w), int(kp[1] * h)
                #         img = cv2.circle(img, center=(x, y), radius=img_circle, color=color_lst[kp_cnt+1], thickness=-1)
                cv2.imshow('asdasdasd', img)
                cv2.waitKey(300,)
                print(tl, br, img.shape, lbl_path)

            #break




                #

                #
                #
                #
                # obs_bbox = [int(obs_bbox[2] * w), int(obs_bbox[3] * h), int(obs_bbox[0] * w), int(obs_bbox[1] * h), ]

                #print(tl, br, img.shape)


                # obs_kp = [[int(x[0] * w), int(x[1] * w), int(x[2])] for x in obs_kp]
                # tl = (int(obs_bbox[0] - obs_bbox[2]), int(obs_bbox[1] - obs_bbox[3]))
                # br = (int(obs_bbox[0] + obs_bbox[2]), int(obs_bbox[1] + obs_bbox[3]))
                #

                #bbox = np.vstack([tl, tr, br, bl])

                #break
            #break
                #
                #





            #break





# runner = Yolo2Imgs(yolo_dir=r"D:\cvat_annotations\yolo_07032025\bbox_test", save_dir=r'D:\cvat_annotations\yolo_07032025\imgs')
# runner.run()