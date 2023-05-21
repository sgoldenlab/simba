import cv2
import numpy as np


def zoom_in(self, zoom_val):
    self.not_done = True

    def get_x_y_callback(event, x, y, flags, param):
        if event == 1:
            self.click_loc = (int(x), int(y))
            self.not_done = False

    def initiate_zoom_in_callback():
        while True:
            cv2.setMouseCallback("Define shape", get_x_y_callback)
            cv2.waitKey(20)
            if self.not_done == False:
                break

    def get_new_image_spec():
        px_avail_right, px_avail_bottom = (
            self.frame_width - self.click_loc[0],
            self.frame_height - self.click_loc[1],
        )
        px_avail_left, px_avail_top = (
            self.frame_width - px_avail_right,
            self.frame_height - px_avail_bottom,
        )

        if (px_avail_left < (self.new_img_size[0] / 2)) or (
            px_avail_right < (self.new_img_size[0] / 2)
        ):
            top_left_x = 0
            bottom_right_x = self.new_img_size[0]
        else:
            top_left_x = int(self.click_loc[0] - (self.new_img_size[0] / 2))
            bottom_right_x = int(self.click_loc[0] + (self.new_img_size[0] / 2))
        if (px_avail_top < (self.new_img_size[1] / 2)) or (
            px_avail_bottom < (self.new_img_size[1] / 2)
        ):
            top_left_y = 0
            bottom_right_y = self.new_img_size[1]
        else:
            top_left_y = int(self.click_loc[1] - (self.new_img_size[1] / 2))
            bottom_right_y = int(self.click_loc[1] + (self.new_img_size[1] / 2))

        self.working_frame = self.working_frame[
            top_left_y:bottom_right_y, top_left_x:bottom_right_x
        ]
        self.working_frame = cv2.resize(
            self.working_frame,
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_AREA,
        )
        self.insert_all_ROIs_into_image(show_zoomed_img=True)

    self.zoom_pct = int(zoom_val) / 100
    self.current_zoom = self.current_zoom + zoom_val
    self.new_img_size = (
        int(self.frame_width - (self.frame_width * self.zoom_pct)),
        int(self.frame_height - (self.frame_height * self.zoom_pct)),
    )
    initiate_zoom_in_callback()
    get_new_image_spec()


# self.interact_method = interact_method
# print(self.interact_method)
#
# def zoom_callback(event, x, y, flags, param):
#     if event == 1:
#         self.zoom_center_X, self.zoom_center_Y = int(x), int(y)
#         self.new_cords_dict = {}
#         if self.interact_method is 'zoom_in':
#             self.top_left_x_new = int(self.zoom_center_X - (self.new_size[0] / 2))
#             if self.top_left_x_new < 0: self.top_left_x_new = 0
#             self.top_left_y_new = int(self.zoom_center_Y - (self.new_size[1] / 2))
#             if self.top_left_y_new < 0: self.top_left_y_new = 0
#             self.bottom_right_x_new = int(self.zoom_center_X + (self.new_size[0] / 2))
#             self.bottom_right_y_new = int(self.zoom_center_Y + (self.new_size[1] / 2))
#             self.cropped_img = self.zoomed_img[self.top_left_y_new:self.bottom_right_y_new, self.top_left_x_new:self.bottom_right_x_new]
#             self.cropped_img = cv2.resize(self.cropped_img, (self.frame_width, self.frame_height), interpolation = cv2.INTER_AREA)
#             self.zoomed_img = self.cropped_img
#             cv2.imshow('Define shape', self.cropped_img)
#
# if self.interact_method == 'zoom_in' or 'zoom_out':
#     self.zoom_pct = int(pct) / 100
#     if interact_method == 'zoom_in':
#         self.new_size = (self.frame_width - (self.frame_width * self.zoom_pct), self.frame_height - (self.frame_height * self.zoom_pct))
#         cv2.setMouseCallback('Define shape', zoom_callback)
#
