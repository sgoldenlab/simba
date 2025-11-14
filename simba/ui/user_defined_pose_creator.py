__author__ = "Simon Nilsson"

import os
from typing import List, Union, Optional

import cv2
import imutils
import numpy as np
from copy import deepcopy

import simba
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_img_path, check_valid_lst)
from simba.utils.data import create_color_palettes
from simba.utils.enums import Paths, TextOptions
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import (find_files_of_filetypes_in_directory, read_img)

WINDOW_NAME = "DEFINE POSE ESTIMATED BODY-PARTS"

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
                 animal_cnt: int,
                 img_path: Union[str, os.PathLike],
                 bp_list: List[str],
                 animal_id_int_list: List[int],
                 circle_scale: Optional[int] = None):

        check_str(name="POSE CONFIG NAME", value=pose_name, allow_blank=False, raise_error=True, invalid_substrs=(',',))
        check_int(name="NUMBER OF ANIMALS", value=animal_cnt, min_value=1, raise_error=True)
        if circle_scale is not None: check_int(name="circle_size", value=circle_scale, min_value=1, raise_error=True)
        check_valid_img_path(path=img_path, raise_error=True)
        check_valid_lst(data=bp_list, source=f'{self.__class__.__name__} bp_list', valid_dtypes=(str,), min_len=1, raise_error=True)
        for bp_name in bp_list:
            if "," in bp_name: raise InvalidInputError(msg=f'Commas are not allowed in body-part names. A comma was found in body-part name: {bp_name}.', source=self.__class__.__name__)
        bp_list = [x.strip() for x in bp_list]
        if animal_cnt > 1:
            check_valid_lst(data=animal_id_int_list, source=f'{self.__class__.__name__} animal_id_int_list', valid_dtypes=(int,), min_len=len(bp_list), raise_error=True)
            if animal_cnt != len(list(set(animal_id_int_list))):
                raise InvalidInputError(msg=f'The number of animals (no_animals) is set to {animal_cnt}, but the number of unique IDs (animal_id_int_list) is set to {len(list(set(animal_id_int_list)))}.', source=self.__class__.__name__)
        self.pose_name, self.img_path, self.animal_cnt = pose_name, img_path, animal_cnt
        self.bp_list, self.animal_id_int_list = bp_list, animal_id_int_list

        PlottingMixin.__init__(self)
        self.img = read_img(img_path=img_path)
        self.img_h, self.img_w= int(self.img.shape[0]), int(self.img.shape[1])
        if self.img_w < 800:
            self.img = imutils.resize(self.img, width=800).astype(np.uint8)
            self.img_h, self.img_w = int(self.img.shape[0]), int(self.img.shape[1])
        self.side_img_size = (int(self.img_h / 4), self.img_w)
        if circle_scale is None:
            self.circle_scale = self.get_optimal_circle_size(frame_size=(self.img_h, self.img_w), circle_frame_ratio=50)
        else:
            self.circle_scale = deepcopy(self.circle_scale)
        self.font_size, self.x_scale, self.y_scale = self.get_optimal_font_scales(text='Left click on body part XXXXXXXXXXWWW',accepted_px_width=self.side_img_size[1], accepted_px_height=int(self.side_img_size[0]/2), text_thickness=4)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.overlay = self.img.copy()

        if self.animal_cnt > 1:
            for cnt, (bp_name, animal_number_id) in enumerate(zip(self.bp_list, self.animal_id_int_list)):
                self.bp_list[cnt] = f"{bp_name}_{animal_number_id}"
        self.color_lst = create_color_palettes(1, len(self.bp_list))[0]

    def launch(self):
        def draw_circle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clr = tuple([int(x) for x in self.color_lst[self.bp_cnt]])
                cv2.circle(self.overlay, (x, int(y - self.side_img.shape[0])), self.circle_scale, self.color_lst[self.bp_cnt], -1)
                self.overlay = PlottingMixin().put_text(img=self.overlay, text=str(self.bp_cnt + 1), pos=(x + 4, int(y - self.side_img.shape[0])), font_size=self.font_size, font_thickness=4, font=TextOptions.FONT.value, text_color=clr, text_bg_alpha=0.8)
                #cv2.putText(self.overlay, str(self.bp_cnt + 1), (x + 4, int(y - self.side_img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.color_lst[self.bp_cnt], 4)
                self.cord_written = True

        for bp_cnt, bp_name in enumerate(self.bp_list):
            self.cord_written = False
            self.bp_cnt = bp_cnt
            self.side_img = np.zeros((int(self.img_h / 4), self.img_w, 3), np.uint8)
            if self.overlay.ndim != 3:
                self.side_img = cv2.cvtColor(self.side_img, cv2.COLOR_BGR2GRAY)
            clr = tuple([int(x) for x in self.color_lst[self.bp_cnt]])
            self.side_img = PlottingMixin().put_text(img=self.side_img, text=f"LEFT CLICK ON BODY-PART {bp_name}.", pos=(int(self.side_img_size[0]/10), int(self.side_img_size[0]/2)), font_size=self.font_size, font_thickness=6, font=TextOptions.FONT.value, text_color=clr, text_bg_alpha=1.0)
            img_concat = cv2.vconcat([self.side_img, self.overlay])
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, img_concat)
            while not self.cord_written:
                cv2.setMouseCallback(WINDOW_NAME, draw_circle)
                img_concat = cv2.vconcat([self.side_img, self.overlay])
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, img_concat)
                cv2.waitKey(1)
            cv2.destroyWindow(WINDOW_NAME)
        self.save()

    def save(self):
        overlay = cv2.resize(self.overlay, (250, 300))

        simba_cw = os.path.dirname(simba.__file__)
        img_dir = os.path.join(simba_cw, Paths.SCHEMATICS.value)
        check_if_dir_exists(in_dir=img_dir, source=self.__class__.__name__, create_if_not_exist=True)

        pose_name_path = os.path.join(simba_cw, Paths.PROJECT_POSE_CONFIG_NAMES.value)
        bp_path = os.path.join(simba_cw, Paths.SIMBA_BP_CONFIG_PATH.value)
        no_animals_path = os.path.join(simba_cw, Paths.SIMBA_NO_ANIMALS_PATH.value)
        for path in [pose_name_path, bp_path, no_animals_path]:
            check_file_exist_and_readable(file_path=path)
        prior_img_cnt = len(find_files_of_filetypes_in_directory(directory=img_dir, extensions=['.png'], raise_warning=False, raise_error=False))
        new_img_name = f"{prior_img_cnt + 1}.png"
        new_img_path = os.path.join(img_dir, new_img_name)
        joined_bp_lst = ",".join(self.bp_list)
        with open(pose_name_path, "a") as f:
            f.write(self.pose_name + "\n")
        with open(bp_path, "a") as fd:
            fd.write(joined_bp_lst + "\n")
        with open(no_animals_path, "a") as fd:
            fd.write(str(self.animal_cnt) + "\n")
        cv2.imwrite(new_img_path, overlay)

#
# pose_config_creator = PoseConfigCreator(pose_name="My_test_config",
#                                         animal_cnt=2,
#                                         img_path=r"C:\Users\sroni\OneDrive\Desktop\desktop\ATTACK_0_feature_importance_bar_graph.png",
#                                         bp_list=['Ear', 'Nose', 'Left_ear', 'Ear', 'Nose', 'Left_ear'],
#                                         animal_id_int_list= [1, 1, 1, 2, 2, 2])
# pose_config_creator.launch()
#
#





# pose_config_creator = PoseConfigCreator(pose_name="My_test_config",
#                                         animal_cnt=2,
#                                         img_path=r"C:\troubleshooting\two_animals_16_bp_JAG\project_folder\videos\Together_1\0.png",
#                                         bp_list=['Ear', 'Nose', 'Left_ear', 'Ear', 'Nose', 'Left_ear'],
#                                         animal_id_int_list= [1, 1, 1, 2, 2, 2])
# pose_config_creator.launch()
