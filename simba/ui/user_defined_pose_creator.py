__author__ = "Simon Nilsson"

import glob
import os
from typing import List, Union

import cv2
import imutils
import numpy as np

import simba
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.data import create_color_palettes
from simba.utils.enums import Paths, TextOptions
from simba.utils.errors import InvalidFileTypeError


class PoseConfigCreator(PlottingMixin):
    """

    Class for creating user-defined pose-estimation pipeline in SimBA through a GUI interface.

    ..seealso::
       `GitHub tutorial/documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Pose_config.md>`__.

    :param str pose_name: Name of the user-defined pose-estimation setting.
    :param str no_animals: Number of animals in the user-defined pose-estimation setting.
    :param str img_path: Path to image representation of user-defined pose-estimation setting
    :param List[str] bp_list: Body-parts in the user-defined pose-estimation setting.
    :param List[int] animal_id_int_list: Integers representing the animal ID which the body-parts belong to.

    :examples:
    >>> pose_config_creator = PoseConfigCreator(pose_name="My_test_config", no_animals=2, img_path='simba/splash_050122.png', bp_list=['Ear', 'Nose', 'Left_ear', 'Ear', 'Nose', 'Left_ear'], animal_id_int_list= [1, 1, 1, 2, 2, 2])
    >>> pose_config_creator.launch()
    """

    def __init__(self,
                 pose_name: str,
                 no_animals: int,
                 img_path: Union[str, os.PathLike],
                 bp_list: List[str],
                 animal_id_int_list: List[str]):


        PlottingMixin.__init__(self)
        self.pose_name, self.img_path = pose_name, img_path
        self.bp_list, self.animal_id_int_list = bp_list, animal_id_int_list
        check_file_exist_and_readable(img_path)
        self.no_animals, self.font = no_animals, cv2.FONT_HERSHEY_DUPLEX
        self.img = cv2.imread(img_path)
        if not isinstance(self.img, (list, tuple, np.ndarray)):
            raise InvalidFileTypeError(msg=f"The chosen image file could not be read as an image ({img_path})")
        self.img_height, self.img_width = int(self.img.shape[0]), int(self.img.shape[1])
        if self.img_width < 300:
            self.img = imutils.resize(self.img, width=800)
            self.img_height, self.img_width = int(self.img.shape[0]), int(self.img.shape[1])
            self.img = np.uint8(self.img)
        self.circle_scale = self.get_optimal_circle_size(frame_size=(self.img_height, self.img_width), circle_frame_ratio=50)
        self.font_size, self.x_scale, self.y_scale = self.get_optimal_font_scales(text='Left click on body part XXXXXXXXXX',accepted_px_width=self.img_width, accepted_px_height=int(self.img_height/10))
        cv2.namedWindow("Define pose", cv2.WINDOW_NORMAL)
        self.overlay = self.img.copy()

        if self.no_animals > 1:
            for cnt, (bp_name, animal_number_id) in enumerate(zip(self.bp_list, self.animal_id_int_list)):
                self.bp_list[cnt] = "{}_{}".format(bp_name, animal_number_id)
        self.color_lst = create_color_palettes(1, len(self.bp_list))[0]

    def launch(self):
        def draw_circle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(
                    self.overlay,
                    (x, int(y - self.side_img.shape[0])),
                    self.circle_scale,
                    self.color_lst[self.bp_cnt],
                    -1,
                )
                cv2.putText(
                    self.overlay,
                    str(self.bp_cnt + 1),
                    (x + 4, int(y - self.side_img.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size,
                    self.color_lst[self.bp_cnt],
                    TextOptions.TEXT_THICKNESS.value,
                )
                self.cord_written = True

        for bp_cnt, bp_name in enumerate(self.bp_list):
            self.cord_written = False
            self.bp_cnt = bp_cnt
            self.side_img = np.zeros(
                (int(self.img_height / 4), self.img_width, 3), np.uint8
            )
            cv2.putText(
                self.side_img,
                "Left click on body part {}.".format(bp_name),
                (10, 50),
                self.font,
                self.font_size,
                self.color_lst[bp_cnt],
                TextOptions.TEXT_THICKNESS.value,
            )
            img_concat = cv2.vconcat([self.side_img, self.overlay])
            cv2.namedWindow("Define pose", cv2.WINDOW_NORMAL)
            cv2.imshow("Define pose", img_concat)
            while not self.cord_written:
                cv2.setMouseCallback("Define pose", draw_circle)
                img_concat = cv2.vconcat([self.side_img, self.overlay])
                cv2.namedWindow("Define pose", cv2.WINDOW_NORMAL)
                cv2.imshow("Define pose", img_concat)
                cv2.waitKey(1)
            cv2.destroyWindow("Define pose")
        self.save()

    def save(self):
        overlay = cv2.resize(self.overlay, (250, 300))
        simba_cw = os.path.dirname(simba.__file__)
        img_dir = os.path.join(simba_cw, Paths.SCHEMATICS.value)
        pose_name_path = os.path.join(simba_cw, Paths.PROJECT_POSE_CONFIG_NAMES.value)
        bp_path = os.path.join(simba_cw, Paths.SIMBA_BP_CONFIG_PATH.value)
        no_animals_path = os.path.join(simba_cw, Paths.SIMBA_NO_ANIMALS_PATH.value)
        for path in [pose_name_path, bp_path, no_animals_path]:
            check_file_exist_and_readable(file_path=path)
        prior_imgs = len(glob.glob(img_dir + "/*.png"))
        new_img_name = f"{str(prior_imgs+1)}.png"
        new_img_path = os.path.join(img_dir, new_img_name)
        self.bp_list = ",".join(self.bp_list)
        with open(pose_name_path, "a") as f:
            f.write(self.pose_name + "\n")
        with open(bp_path, "a") as fd:
            fd.write(self.bp_list + "\n")
        with open(no_animals_path, "a") as fd:
            fd.write(str(self.no_animals) + "\n")
        cv2.imwrite(new_img_path, overlay)


# pose_config_creator = PoseConfigCreator(pose_name="My_test_config", no_animals=1, img_path='/Users/simon/Desktop/crop_circles_3.png', bp_list=['Ear', 'Nose', 'Left_ear'], animal_id_int_list= [1, 1, 1, 2, 2, 2])
# pose_config_creator.launch()
