import numpy as np
import os
from simba.third_party_label_appenders.converters import geometries_to_yolo
from simba.utils.read_write import find_files_of_filetypes_in_directory, read_df, get_fn_ext, find_video_of_file
from simba.mixins.geometry_mixin import GeometryMixin
from simba.bounding_box_tools.yolo.model import fit_yolo, inference_yolo
from simba.bounding_box_tools.yolo.visualize import YOLOVisualizer
from simba.third_party_label_appenders.converters import simba_rois_to_yolo, split_yolo_train_test_val

TRAIN_DATA_DIR = r'/mnt/c/troubleshooting/mitra/project_folder/csv/yolo_train'
YOLO_SAVE_DIR = r"/mnt/c/troubleshooting/yolo_animal"
VIDEO_DIR = r'/mnt/c/troubleshooting/mitra/project_folder/videos'
BODY_PARTS_HULL = ['Nose_x', 'Nose_y', 'Tail_base_x', 'Tail_base_y', 'Left_side_x', 'Left_side_y', 'Right_side_x', 'Right_side_y']

# #########################
#
# file_paths = find_files_of_filetypes_in_directory(directory=TRAIN_DATA_DIR, extensions=['.csv'])
# video_polygons = {}
# for file_path in file_paths:
#     print(file_path)
#     animal_data = read_df(file_path=file_path, file_type='csv', usecols=BODY_PARTS_HULL).values.reshape(-1, 4, 2).astype(np.int32)x
#     animal_polygons = GeometryMixin().multiframe_bodyparts_to_polygon(data=animal_data)
#     animal_polygons = GeometryMixin().multiframe_minimum_rotated_rectangle(shapes=animal_polygons)
#     video_polygons[get_fn_ext(file_path)[1]] = GeometryMixin().geometries_to_exterior_keypoints(geometries=animal_polygons)
#
# #######################
#
# for video_name, polygons in video_polygons.items():
#     polygons = {0: polygons}
#     video_path = find_video_of_file(video_dir=VIDEO_DIR, filename=video_name)
#     geometries_to_yolo(geometries=polygons, video_path=video_path, save_dir=YOLO_SAVE_DIR, sample=500, obb=True)
#
#
# #######################
# YOLO_TRAINING_DIR = r"/mnt/c/troubleshooting/yolo_data_split_animal" # DIRECTORY WHERE WE SHOULD STORE THE SLIP DATA.
# split_yolo_train_test_val(data_dir=YOLO_SAVE_DIR, save_dir=YOLO_TRAINING_DIR, verbose=True)
#
# #
# # #NEXT, WE TRAIN A YOLO MODEL BASED ON THE SPLIT DATA
# INITIAL_WEIGHTS_PATH = r'/mnt/c/Users/sroni/Downloads/yolov8n.pt' #SOME INITIAL WEIGHTS TO START WITH, THEY CAN BE DOWNLOADED AT https://huggingface.co/Ultralytics
MODEL_SAVE_DIR = "/mnt/c/troubleshooting/yolo_mdl" #DIRECTORY WHERE TO SAVE THE TRAINED MODEL AND PERFORMANCE STATISTICS
# fit_yolo(initial_weights=INITIAL_WEIGHTS_PATH, model_yaml=os.path.join(YOLO_TRAINING_DIR, 'map.yaml'), save_path=MODEL_SAVE_DIR, epochs=25, batch=16, plots=True)

#NEXT, WE USE THE TRAINED MODEL TO FIND THE ROIS IN A NEW VIDEO
INFERENCE_RESULTS = "/mnt/c/troubleshooting/yolo_results" #DIRECTORY WHERE TO STORE THE RESULTS IN CSV FORMAT
VIDEO_PATH = r'/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_Saline_0513.mp4' #PATH TO VIDEO TO ANALYZE
inference_yolo(weights_path=os.path.join(MODEL_SAVE_DIR, 'train', 'weights', 'best.pt'), video_path=VIDEO_PATH, verbose=True, save_dir=INFERENCE_RESULTS, gpu=True, batch_size=16, interpolate=True)


# #FINALLY, WE VISUALIZE THE RESULTS TO CHECK THAT THE PREDICTIONS ARE ACCURATE
# DATA_PATH = r"/mnt/c/troubleshooting/yolo_results/501_MA142_Gi_CNO_0514_clipped.csv" #PATH TO ONE OF THE CSV FILES CREATED IN THE PRIOR STEP.
# VIDEO_PATH = r'/mnt/c/troubleshooting/mitra/project_folder/videos/clipped/501_MA142_Gi_CNO_0514_clipped.mp4' # PATH TO THE VIDEO REPRESENTING THE ``DATA_PATH`` FILE.
# SAVE_DIR = r"/mnt/c/troubleshooting/yolo_videos" # DIRECTORY WHERE TO SAVE THE YOLO VIDEO
#
# yolo_visualizer = YOLOVisualizer(data_path=DATA_PATH, video_path=VIDEO_PATH, save_dir=SAVE_DIR, palette='Accent', thickness=20, core_cnt=-1, verbose=False)
# yolo_visualizer.run()
#
#



#geometries_to_yolo(geometries=animal_polygons, video_path=r'C:\troubleshooting\mitra\project_folder\videos\501_MA142_Gi_CNO_0514.mp4', save_dir=r"C:\troubleshooting\coco_data", sample=500, obb=True)










