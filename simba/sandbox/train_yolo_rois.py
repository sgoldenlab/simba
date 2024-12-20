import os

from simba.bounding_box_tools.yolo.model import fit_yolo, inference_yolo
from simba.bounding_box_tools.yolo.visualize import YOLOVisualizer
from simba.third_party_label_appenders.converters import (
    simba_rois_to_yolo, split_yolo_train_test_val)

# FIRST WE CREATE SOME YOLO FORMATED DATA LABELS AND EXTRACT THEIR ASSOCIATED IMAGES FROM THE VIDEOS AND SIMBA ROI DEFINITIONS
ROI_PATH = r"/mnt/c/troubleshooting/RAT_NOR/project_folder/logs/measures/ROI_definitions.h5"
VIDEO_DIR = r'/mnt/c/troubleshooting/RAT_NOR/project_folder/videos' #DIRECTORY HOLDING THE VIDEOS WITH DEFINED ROI'S
YOLO_SAVE_DIR = r"/mnt/c/troubleshooting/yolo_data" # DIRECTORY WHICH SHOULD STORE THE YOLO FORMATED DATA
VIDEO_FRAME_COUNT = 200 # THE NUMBER OF LABELS WE SHOULD CREATE FROM EACH VIDEO
GREYSCALE = False #IF THE VIDEOS ARE IN COLOR, CONVERT  THE YOLO IMAGES TO GREYSCALE OR NOT

#WE CREATE THE YOLO DATASET BASED ON THE ABOVE DEFINITIONS
simba_rois_to_yolo(roi_path=ROI_PATH, video_dir=VIDEO_DIR, save_dir=YOLO_SAVE_DIR, roi_frm_cnt=VIDEO_FRAME_COUNT, greyscale=GREYSCALE, verbose=True, obb=True)


#NEXT, WE SPLIT THE DATA CREATED IN THE PRIOR CELL INTO DATA FOR TRAINING, TESTING AND VALIDATION.
YOLO_TRAINING_DIR = r"/mnt/c/troubleshooting/yolo_data_split" # DIRECTORY WHERE WE SHOULD STORE THE SLIP DATA.
split_yolo_train_test_val(data_dir=YOLO_SAVE_DIR, save_dir=YOLO_TRAINING_DIR, verbose=True)


#NEXT, WE TRAIN A YOLO MODEL BASED ON THE SPLIT DATA
INITIAL_WEIGHTS_PATH = r'/mnt/c/troubleshooting/coco_data/weights/yolov8n-obb.pt' #SOME INITIAL WEIGHTS TO START WITH, THEY CAN BE DOWNLOADED AT https://huggingface.co/Ultralytics
MODEL_SAVE_DIR = "/mnt/c/troubleshooting/yolo_mdl" #DIRECTORY WHERE TO SAVE THE TRAINED MODEL AND PERFORMANCE STATISTICS
fit_yolo(initial_weights=INITIAL_WEIGHTS_PATH, model_yaml=os.path.join(YOLO_TRAINING_DIR, 'map.yaml'), save_path=MODEL_SAVE_DIR, epochs=25, batch=16, plots=True)


#NEXT, WE USE THE TRAINED MODEL TO FIND THE ROIS IN A NEW VIDEO
INFERENCE_RESULTS = "/mnt/c/troubleshooting/yolo_results" #DIRECTORY WHERE TO STORE THE RESULTS IN CSV FORMAT
VIDEO_PATH = r'/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/clipped/03152021_NOB_IOT_8_clipped.mp4' #PATH TO VIDEO TO ANALYZE
inference_yolo(weights_path=os.path.join(MODEL_SAVE_DIR, 'train', 'weights', 'best.pt'), video_path=VIDEO_PATH, verbose=True, save_dir=INFERENCE_RESULTS, gpu=False, batch_size=16)

#FINALLY, WE VISUALIZE THE RESULTS TO CHECK THAT THE PREDICTIONS ARE ACCURATE
DATA_PATH = r"/mnt/c/troubleshooting/yolo_results/03152021_NOB_IOT_8_clipped.csv" #PATH TO ONE OF THE CSV FILES CREATED IN THE PRIOR STEP.
VIDEO_PATH = r'/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/clipped/03152021_NOB_IOT_8_clipped.mp4' # PATH TO THE VIDEO REPRESENTING THE ``DATA_PATH`` FILE.
SAVE_DIR = r"/mnt/c/troubleshooting/yolo_videos" # DIRECTORY WHERE TO SAVE THE YOLO VIDEO

yolo_visualizer = YOLOVisualizer(data_path=DATA_PATH, video_path=VIDEO_PATH, save_dir=SAVE_DIR, palette='Accent', thickness=20, core_cnt=-1, verbose=False)
yolo_visualizer.run()