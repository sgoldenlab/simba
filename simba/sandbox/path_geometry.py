from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.data import savgol_smoother
from simba.utils.read_write import read_df, read_frm_of_video
import numpy as np
import os
import cv2

FRAME_IDX = -1
BODY_PART = 'Nose'
VIDEO_NAME = 'SI_DAY3_308_CD1_PRESENT'
CONFIG_PATH = r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/project_config.ini'

config = ConfigReader(config_path=CONFIG_PATH, read_video_info=False)
config.read_roi_data()
frm = read_frm_of_video(os.path.join(config.video_dir, VIDEO_NAME +'.mp4'), frame_index=FRAME_IDX)
shapes, colors = GeometryMixin.simba_roi_to_geometries(rectangles_df=config.rectangles_df, circles_df=config.circles_df, polygons_df=config.polygon_df, color=True)
video_roi_shapes = list(shapes[VIDEO_NAME].values())

roi_shapes = GeometryMixin.view_shapes(shapes=video_roi_shapes, size=500, thickness=12, color_palette='Pastel1')
# cv2.imshow('sasd', roi_shapes)
# cv2.waitKey(10000)

roi_shapes_w_bg = GeometryMixin.view_shapes(shapes=video_roi_shapes, size=500, bg_img=frm, thickness=12, color_palette='Pastel1')
# cv2.imshow('sasd', roi_shapes_w_bg)
# cv2.waitKey(10000)

data_path = os.path.join(config.outlier_corrected_dir, VIDEO_NAME + f'.{config.file_type}')
df = read_df(data_path, file_type=config.file_type)
animal_data = df[['Nose_x', 'Nose_y']].values
animal_path = GeometryMixin.to_linestring(data=animal_data)

nose_path = GeometryMixin.view_shapes(shapes=video_roi_shapes + [animal_path], size=500, bg_img=frm, thickness=12, color_palette='Pastel1')
# cv2.imshow('sasd', nose_path)
# cv2.waitKey(5000)

animal_data = savgol_smoother(data=animal_data, fps=15, time_window=1000)
animal_path = GeometryMixin.to_linestring(data=animal_data)








length = GeometryMixin.length(shape=animal_path, pixels_per_mm=1.5, unit='m')

dist = GeometryMixin.locate_line_point(path=animal_path, px_per_mm=1.5, fps=15, geometry=shapes[VIDEO_NAME]['Top_left'])
distances = dist['raw_distances']

dist = GeometryMixin.locate_line_point(path=animal_path, px_per_mm=1.5, fps=15, geometry=shapes[VIDEO_NAME]['Bottom_left'])
distances = dist['raw_distances']


#TIME STAMPS WHEN ANIMAL IS 1CM OR LESS FROM THE CAGE ROI
cage_dist = GeometryMixin.locate_line_point(path=animal_path, px_per_mm=1.5, fps=15, geometry=shapes[VIDEO_NAME]['Cage'])
less_than_1cm_timepoints = np.argwhere(dist['raw_distances']  < 10).flatten() / 15


frm = read_frm_of_video(os.path.join(config.video_dir, VIDEO_NAME +'.mp4'), frame_index=-1)
buffered_path = GeometryMixin.buffer_shape(shape=animal_path, size_mm=10, pixels_per_mm=1.5)
buffered_path_img = GeometryMixin.view_shapes(shapes=[buffered_path], size=500, bg_img=frm, thickness=12, color_palette='Pastel1')
# cv2.imshow('sasd', buffered_path_img)
# cv2.waitKey(5000)


#CHECK IF PATH IS INSIDE A SPECIFIC POLYGON

w = GeometryMixin.is_touching(shapes=[buffered_path, shapes[VIDEO_NAME]['Cage']])

#GeometryMixin.compute_pct_shape_overlap
frm = read_frm_of_video(os.path.join(config.video_dir, VIDEO_NAME +'.mp4'), frame_index=-1)
not_crossed = GeometryMixin.difference(shapes=[shapes[VIDEO_NAME]['Top_left'], buffered_path])
not_crossed_img = GeometryMixin.view_shapes(shapes=[not_crossed], size=500, bg_img=frm, thickness=12, color_palette='Set1')
# cv2.imshow('sasd', not_crossed_img)
# cv2.waitKey(5000)


frm = read_frm_of_video(os.path.join(config.video_dir, VIDEO_NAME +'.mp4'), frame_index=-1)
shifted_geos = GeometryMixin.adjust_geometry_locations(geometries=video_roi_shapes, shift=(100, 5))
shifted_geos_img = GeometryMixin.view_shapes(shapes=shifted_geos, size=500, bg_img=frm, thickness=12, color_palette='Set1')
cv2.imshow('sasd', shifted_geos_img)
cv2.waitKey(5000)















