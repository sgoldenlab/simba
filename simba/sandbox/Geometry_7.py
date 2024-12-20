import os
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.image_mixin import ImageMixin
from simba.utils.read_write import read_df
from simba.plotting.geometry_plotter import GeometryPlotter

CONFIG_PATH = '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini'
VIDEO_NAME = '2022-06-20_NOB_DOT_4'

cfg = ConfigReader(config_path=CONFIG_PATH, read_video_info=False)
data = read_df(os.path.join(cfg.outlier_corrected_dir, VIDEO_NAME + f'.{cfg.file_type}'), file_type=cfg.file_type)
video_path = os.path.join(cfg.video_dir, VIDEO_NAME + '.mp4')
animal_df = data[[x for x in data.columns if x in cfg.animal_bp_dict['Animal_1']['X_bps'] + cfg.animal_bp_dict['Animal_1']['Y_bps']]]
animal_data = animal_df.drop(['Tail_end_x', 'Tail_end_y'], axis=1).values.reshape(-1, 7, 2).astype(np.int)

animal_polygons = GeometryMixin.bodyparts_to_polygon(data=animal_data)[:300]
geometry_plotter = GeometryPlotter(config_path=CONFIG_PATH,
                                   geometries=[animal_polygons],
                                   video_name=VIDEO_NAME,
                                   thickness=10,
                                   bg_opacity=0.2)
geometry_plotter.run()

#
# animal_rectangles = GeometryMixin().multiframe_minimum_rotated_rectangle(shapes=animal_polygons)
# geometry_plotter = GeometryPlotter(config_path=CONFIG_PATH,
#                                    geometries=[animal_rectangles],
#                                    video_name=VIDEO_NAME,
#                                    thickness=10)
# geometry_plotter.run()
#
# animal_polygons_buffered = GeometryMixin.bodyparts_to_polygon(data=animal_data[:300], parallel_offset=100, pixels_per_mm=1.88)
# geometry_plotter = GeometryPlotter(config_path=CONFIG_PATH,
#                                    geometries=[animal_polygons],
#                                    video_name=VIDEO_NAME,
#                                    thickness=10,
#                                    bg_opacity=1.0)
# geometry_plotter.run()

imgs = ImageMixin().read_img_batch_from_video(video_path=video_path, start_frm=0, end_frm=299)
imgs = np.stack(list(imgs.values()))
# animal_polygons_buffered = np.array(animal_polygons_buffered).T.reshape(-1, 1)
# sliced_images = ImageMixin().slice_shapes_in_imgs(imgs=imgs, shapes=animal_polygons_buffered)

# ImageMixin.img_stack_to_video(imgs=sliced_images,
#                               save_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/frames/output/geometry_visualization/sliced.mp4',
#                               fps=30,
#                               verbose=True)

# animal_polygons_tighter = np.array(animal_polygons).T.reshape(-1, 1)
# sliced_images = ImageMixin().slice_shapes_in_imgs(imgs=imgs, shapes=animal_polygons_tighter)
# ImageMixin.img_stack_to_video(imgs=sliced_images,
#                               save_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/frames/output/geometry_visualization/sliced_tighter.mp4',
#                               fps=30,
#                               verbose=True)

animal_head = animal_df[['Nose_x', 'Nose_y', 'Ear_left_x', 'Ear_left_y', 'Ear_right_x', 'Ear_right_y']].values.reshape(-1, 3, 2).astype(np.int)
#animal_head_polygons = GeometryMixin.bodyparts_to_polygon(data=animal_head, parallel_offset=100, pixels_per_mm=1.88)[:300]
# animal_head_polygons = np.array(animal_head_polygons).T.reshape(-1, 1)
# sliced_images = ImageMixin().slice_shapes_in_imgs(imgs=imgs, shapes=animal_head_polygons)
# ImageMixin.img_stack_to_video(imgs=sliced_images,
#                               save_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/frames/output/geometry_visualization/sliced_head.mp4',
#                               fps=30,
#                               verbose=True)

# animal_head_polygons = GeometryMixin.bodyparts_to_polygon(data=animal_head, parallel_offset=25, pixels_per_mm=1.88)[:300]
# animal_head_polygons = np.array(animal_head_polygons).T.reshape(-1, 1)
# sliced_images = ImageMixin().slice_shapes_in_imgs(imgs=imgs, shapes=animal_head_polygons)
# ImageMixin.img_stack_to_video(imgs=sliced_images,
#                               save_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/frames/output/geometry_visualization/sliced_head_tighter_even.mp4',
#                               fps=30,
#                               verbose=True)



# animal_head_polygons = GeometryMixin.bodyparts_to_polygon(data=animal_head)[:300]
# head_centers = GeometryMixin.get_center(shape=animal_head_polygons)
# head_circles = GeometryMixin.bodyparts_to_circle(data=head_centers, parallel_offset=100, pixels_per_mm=1.88)
# head_circles = np.array(head_circles).T.reshape(-1, 1)
# sliced_images = ImageMixin().slice_shapes_in_imgs(imgs=imgs, shapes=head_circles)
# ImageMixin.img_stack_to_video(imgs=sliced_images,
#                               save_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/frames/output/geometry_visualization/sliced_head_circles.mp4',
#                               fps=30,
#                               verbose=True)


# head_circles = GeometryMixin.bodyparts_to_circle(data=head_centers, parallel_offset=50, pixels_per_mm=1.88)
# head_circles = np.array(head_circles).T.reshape(-1, 1)
# sliced_images = ImageMixin().slice_shapes_in_imgs(imgs=imgs, shapes=head_circles)
# ImageMixin.img_stack_to_video(imgs=sliced_images,
#                               save_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/frames/output/geometry_visualization/sliced_head_circles_tighter.mp4',
#                               fps=30,
#                               verbose=True)


#video_bg_substraction_mp(video_path=video_path, save_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/frames/output/geometry_visualization/bg_removed.mp4')
