__author__ = "Simon Nilsson", "JJ Choong"

import cv2
import numpy as np
import imutils
from simba.read_config_unit_tests import check_file_exist_and_readable
from simba.drop_bp_cords import createColorListofList
import os, glob


class PoseConfigCreator(object):
    def __init__(self,
                 pose_name: str,
                 no_animals: int,
                 no_bps: int,
                 img_path: str,
                 bp_list: list,
                 animal_cnt: int,
                 animal_cnt_list: list):

        def draw_circle(self, event, x, y, flags, param):
            if (event == cv2.EVENT_LBUTTONDBLCLK):
                if self.centre_cord_status == False:
                    cv2.circle(overlay, (x, y - self.side_img.shape[0]), 10, self.color_lst[self.bp_cnt], -1)
                    cv2.putText(overlay, str(self.bp_cnt + 1), (x + 4, y - self.side_img.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_lst[self.bp_cnt], 2)
                    cv2.imshow('Define pose', overlay)
                    self.centre_cord_status = True

        check_file_exist_and_readable(img_path)
        self.no_animals, self.font = no_animals, cv2.FONT_HERSHEY_SIMPLEX
        self.img = cv2.imread(img_path)
        self.img_height, self.img_width = int(self.img.shape[0]), int(self.img.shape[1])
        if self.img_width < 300:
            self.img = imutils.resize(self.img, width=800)
            self.img_height, self.img_width = int(self.img.shape[0]), int(self.img.shape[1])
            self.img = np.uint8(self.img)
        self.font_scale = max(self.img_width, self.img_height) / (max(self.img_width, self.img_height) * 1.2)
        cv2.namedWindow('Define pose', cv2.WINDOW_NORMAL)
        overlay = self.img.copy()
        self.color_lst = createColorListofList(self.no_animals, len(bp_list))

        for bp_cnt, bp_name in enumerate(bp_list):
            self.bp_cnt = bp_cnt
            self.side_img = np.zeros((100, self.img_width, 3), np.uint8)
            cv2.putText(self.side_img, 'Double left click {}. Press ESC to continue.'.format(bp_name), (10, 50), self.font, self.font_scale, self.color_lst[bp_cnt], 2)
            img_concat = cv2.vconcat([self.side_img, overlay])
            cv2.imshow('Define pose', img_concat)
            self.centre_cord_status = False
            while True:
                cv2.setMouseCallback('Define pose', draw_circle)
                cv2.imshow('Define pose', img_concat)
                k = cv2.waitKey(20) & 0xFF
                if k == 27:
                    cv2.destroyWindow('Define pose')
                    break

        overlay = cv2.resize(overlay, (250, 300))
        simba_dir = os.path.dirname(__file__)
        img_dir = os.path.join(simba_dir, 'pose_configurations', 'schematics')
        pose_name_path = os.path.join(simba_dir, 'pose_configurations', 'configuration_names', 'pose_config_names.csv')
        bp_path = os.path.join(simba_dir, 'pose_configurations', 'bp_names', 'bp_names.csv')
        no_animals_path = os.path.join(simba_dir, 'pose_configurations', 'no_animals', 'no_animals.csv')
        prior_imgs = len(glob.glob(img_dir + '/*.png'))
        new_img_name = 'Picture' + str(prior_imgs + 1) + '.png'
        new_img_path = os.path.join(img_dir, new_img_name)
        bp_list = ",".join(bp_list)

        with open(pose_name_path, 'a') as fd:
            fd.write(pose_name + '\n')
        with open(bp_path, 'a') as fd:
            fd.write(bp_list + '\n')
        with open(no_animals_path, 'a') as fd:
            fd.write(str(animal_cnt) + '\n')
        cv2.imwrite(new_img_path, overlay)










