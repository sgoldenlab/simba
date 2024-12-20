import cv2
import numpy as np
import pandas as pd
from simba.utils.read_write import read_frm_of_video
from simba.mixins.geometry_mixin import GeometryMixin
from simba.plotting.geometry_plotter import GeometryPlotter
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.circular_statistics import CircularStatisticsMixin
from simba.mixins.config_reader import ConfigReader


CONFIG_PATH = '/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini'
config = ConfigReader(config_path=CONFIG_PATH, create_logger=False)
config.read_roi_data()

response_windows = GeometryMixin.simba_roi_to_geometries(rectangles_df=config.rectangles_df)[0]['3A_Mouse_5-choice_MouseTouchBasic_s9_a6_grayscale']

HEADERS = ['NOSE_X', 'NOSE_Y', 'NOSE_P', 'EAR_LEFT_X', 'EAR_LEFT_Y', 'EAR_LEFT_P', 'EAR_RIGHT_X', 'EAR_RIGHT_Y', 'EAR_RIGHT_P', 'LAT_LEFT_X', 'LAT_LEFT_Y', 'LAT_LEFT_P', 'LAT_RIGHT_X', 'LAT_RIGHT_Y', 'LAT_RIGHT_P', 'CENTER_X', 'CENTER_Y', 'CENTER_P', 'TAIL_BASE_X', 'TAIL_BASE_Y', 'TAIL_BASE_P']
X_HEADERS = [x for x in HEADERS if x.endswith('_X')]
Y_HEADERS = [x for x in HEADERS if x.endswith('_Y')]
P_HEADERS = [x for x in HEADERS if x.endswith('_P')]
BP_HEADERS = [x for x in HEADERS if x not in P_HEADERS]


VIDEO_PATH = '/Users/simon/Desktop/envs/platea_data/3A_Mouse_5-choice_MouseTouchBasic_s9_a6_grayscale.mp4'
DATA_PATH = '/Users/simon/Desktop/envs/platea_data/3A_Mouse_5-choice_MouseTouchBasic_s9_a6_grayscale_new.mp4.npy'
data = pd.DataFrame(np.load(DATA_PATH).reshape(37420, 21), columns=HEADERS).head(200)

bp_df = data[BP_HEADERS]
bp_df.to_csv('/Users/simon/Desktop/envs/simba/troubleshooting/platea/project_folder/csv/outlier_corrected_movement_location/Video_1.csv')
animal_polygons = GeometryMixin().multiframe_bodyparts_to_polygon(data=bp_df.values.reshape(-1, 7, 2), parallel_offset=10)

GRID_SIZE = (5, 5)
grid, aspect_ratio = GeometryMixin().bucket_img_into_grid_square(img_size=(1280, 720), bucket_grid_size_mm=10, px_per_mm=10)

grid_lsts = []
for k, v in grid.items():
    x = []
    for i in range(200):
        x.append(v)
    grid_lsts.append(x)

nose_arr = data[['NOSE_X', 'NOSE_Y']].values.astype(np.int64)
left_ear_arr = data[['EAR_LEFT_X', 'EAR_LEFT_Y']].values.astype(np.int64)
right_ear_arr = data[['EAR_RIGHT_X', 'EAR_RIGHT_Y']].values.astype(np.int64)
center_arr = data[['CENTER_X', 'CENTER_Y']].values.astype(np.int64)
midpoints = FeatureExtractionMixin.find_midpoints(bp_1=left_ear_arr, bp_2=right_ear_arr, percentile=0.5)

circles = GeometryMixin().multiframe_bodyparts_to_circle(data=center_arr.reshape(-1, 1, 2), parallel_offset=200, pixels_per_mm=1)
circles = [x for xs in circles for x in xs]

response_windows_list = []
for i in response_windows.values():
    x = []
    for j in range(200):
        x.append(i)
    response_windows_list.append(x)



geometries = []
# for i in range(len(grid_lsts)):
#     geometries.append(grid_lsts[i])

for i in range(len(response_windows_list)):
    geometries.append(response_windows_list[i])

geometries.append(circles)
geometries.append(animal_polygons)

geometry_plotter = GeometryPlotter(config_path=CONFIG_PATH,
                                   geometries=geometries,
                                   video_name='3A_Mouse_5-choice_MouseTouchBasic_s9_a6_grayscale',
                                   thickness=10,
                                   palette='spring')
geometry_plotter.run()

#
#
# img = read_frm_of_video(video_path=VIDEO_PATH, frame_index=100)
# img_shapes = grid + response_windows_lst[100] + circles[100]
# img_shapes.append(animal_polygons[100])
# img = GeometryMixin().view_shapes(shapes=img_shapes, bg_img=img, color_palette='set')
#








# lines_df = {}
# for k, v in response_windows.items():
#     target = np.array(v.centroid).astype(np.int64)
#     line = FeatureExtractionMixin().jitted_line_crosses_to_static_targets(left_ear_array=left_ear_arr,
#                                                                    right_ear_array=right_ear_arr,
#                                                                    nose_array=nose_arr,
#                                                                    target_array=target)
#     target_x = np.full((line.shape[0], 1), target[0])
#     target_y = np.full((line.shape[0], 1), target[1])
#     line = np.hstack([line, target_x, target_y])
#     lines_df[k] = line

# lines_lst = []
# for i in range(lines_df['Window_1'].shape[0]):
#     lines_lst.append([])
#
# for k, v in lines_df.items():
#     for i in range(v.shape[0]):
#         if v[i][1] != -1.0:
#             line = np.array([[v[i][1], v[i][2]], [v[i][-2], v[i][-1]]])
#             line_str = GeometryMixin.to_linestring(data=line)
#             lines_lst[i].append(line_str)
#         else:
#             line = np.array([[0, 0], [0, 0]])
#             line_str = GeometryMixin.to_linestring(data=line)
#             lines_lst[i].append(line_str)

#direction = CircularStatisticsMixin.direction_two_bps(anterior_loc=nose_arr, posterior_loc=midpoints)



# cv2.imshow('sdfsdf', img)
# cv2.waitKey(5000)
