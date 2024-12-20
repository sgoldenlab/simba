from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.data import savgol_smoother
from simba.utils.read_write import read_df, read_frm_of_video
import numpy as np
import os
import cv2

FRAME_IDX = -1
VIDEO_NAME = '2022-06-20_NOB_DOT_4'
CONFIG_PATH = r'/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini'

config = ConfigReader(config_path=CONFIG_PATH, read_video_info=False)
config.read_roi_data()
shapes, colors = GeometryMixin.simba_roi_to_geometries(rectangles_df=config.rectangles_df, circles_df=config.circles_df, polygons_df=config.polygon_df, color=True)
video_roi_shapes = list(shapes[VIDEO_NAME].values())


roi_shapes = GeometryMixin.view_shapes(shapes=video_roi_shapes, size=750, thickness=12, bg_clr= (0, 0, 0), color_palette='Pastel1')
# cv2.imshow('sasd', roi_shapes)
# cv2.waitKey(10000)

frm = read_frm_of_video(os.path.join(config.video_dir, VIDEO_NAME +'.mp4'), frame_index=FRAME_IDX)
roi_shape_with_bg = GeometryMixin.view_shapes(shapes=video_roi_shapes, size=750, thickness=12, bg_img=frm, color_palette='Pastel1')
# cv2.imshow('sasd', roi_shapes)
# cv2.waitKey(10000)


data_path = os.path.join(config.outlier_corrected_dir, VIDEO_NAME + f'.{config.file_type}')
df = read_df(data_path, file_type=config.file_type)
animal_data = df[['Tail_base_x', 'Tail_base_y', 'Nose_x', 'Nose_y']].values
animal_data = animal_data.reshape(-1, 2, 2)
animal_lines = GeometryMixin().multiframe_bodyparts_to_line(data=animal_data)

FRAME_IDX = 120
frm = read_frm_of_video(os.path.join(config.video_dir, VIDEO_NAME +'.mp4'), frame_index=FRAME_IDX)
animal_line_img = GeometryMixin.view_shapes(shapes=[animal_lines[FRAME_IDX], video_roi_shapes[3]], size=750, thickness=12, bg_img=frm, color_palette='Pastel1')
cv2.imshow('sasd', animal_line_img)
cv2.waitKey(10000)

to_the_left = GeometryMixin().static_point_lineside(lines=np.array(animal_lines[FRAME_IDX].coords.xy).reshape(-1, 2, 2).astype(np.float32), point=np.array(video_roi_shapes[3].centroid))









#view_image(roi_shapes)
