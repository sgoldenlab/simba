import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from typing import Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.third_party_label_appenders.converters import \
    yolo_obb_data_to_bounding_box
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_array, check_valid_boolean,
                                check_valid_lst, check_valid_tuple, get_fn_ext)
from simba.utils.data import df_smoother, savgol_smoother
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError, SimBAGPUError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import find_core_cnt, get_video_meta_data


def inference_yolo_pose_tracks(weights_path: Union[str, os.PathLike],
                               video_path: Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]],
                               keypoint_names: Tuple[str, ...],
                               botsort_config_path: Union[str, os.PathLike],
                               verbose: Optional[bool] = False,
                               save_dir: Optional[Union[str, os.PathLike]] = None,
                               device: Union[Literal['cpu'], int] = 0,
                               format: Optional[str] = None,
                               batch_size: Optional[int] = 4,
                               torch_threads: int = 8,
                               half_precision: bool = True,
                               stream: bool = False,
                               threshold: float = 0.7,
                               max_tracks: Optional[int] = 2):

    OUT_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
    if isinstance(video_path, list):
        check_valid_lst(data=video_path, source=f'{inference_yolo.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
    elif isinstance(video_path, str):
        check_file_exist_and_readable(file_path=video_path)
        video_path = [video_path]
    for i in video_path:
        _ = get_video_meta_data(video_path=i)
    check_file_exist_and_readable(file_path=weights_path)
    check_file_exist_and_readable(file_path=botsort_config_path)
    check_valid_boolean(value=verbose, source=inference_yolo.__name__)
    check_int(name=f'{inference_yolo.__name__} batch_size', value=batch_size, min_value=1)
    check_float(name=f'{inference_yolo.__name__} threshold', value=threshold, min_value=10e-6, max_value=1.0)
    check_valid_tuple(x=keypoint_names, source=f'{inference_yolo.__name__} keypoint_names', min_integer=1, valid_dtypes=(str,))
    if max_tracks is not None:
        check_int(name=f'{inference_yolo.__name__} max_tracks', value=max_tracks, min_value=1)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=f'{inference_yolo.__name__} save_dir')
    keypoint_col_names = [f'{i}_{s}'.upper() for i in keypoint_names for s in ['x', 'y', 'p']]
    OUT_COLS.extend(keypoint_col_names)
    torch.set_num_threads(torch_threads)
    model = load_yolo_model(weights_path=weights_path, device=device, format=format)
    results = {}
    class_dict = model.names
    timer = SimbaTimer(start=True)
    for path in video_path:
        _, video_name, _ = get_fn_ext(filepath=path)
        _ = get_video_meta_data(video_path=path)
        video_out = []
        predictions = model.track(source=path, stream=False, tracker=botsort_config_path)
        print(predictions)
        #video_predictions = model.predict(source=path, half=half_precision, batch=batch_size, stream=stream)




#fit_yolo(initial_weights=r"/mnt/d/yolo_weights/yolo11n-pose.pt", model_yaml=r"/mnt/d/netholabs/yolo_data_1/map.yaml", save_path=r"/mnt/d/netholabs/yolo_mdls_1", batch=32, epochs=100)

# VIDEO_PATH = "/mnt/d/netholabs/yolo_videos/2025-05-28_19-46-56.mp4"
# BOTSORT_PATH = "/mnt/c/projects/simba/simba/simba/assets/botsort.yml"
# inference_yolo_pose_tracks(weights_path=r"/mnt/d/netholabs/yolo_mdls_1/train/weights/best.pt",
#                            video_path=VIDEO_PATH,
#                            save_dir=r"/mnt/d/netholabs/yolo_test/results",
#                            verbose=True,
#                            device=0,
#                            format='onnx',
#                            keypoint_names=('nose', 'ear_left', 'ear_right', 'lateral_left', 'center', 'lateral_right', 'tail_base'),
#                            batch_size=32,
#                            botsort_config_path=BOTSORT_PATH)


#
#
#
# #fit_yolo(initial_weights=r"/mnt/d/yolo_weights/yolo11n-pose.pt", model_yaml=r"/mnt/d/netholabs/yolo_data_1/map.yaml", save_path=r"/mnt/d/netholabs/yolo_mdls_1", batch=32, epochs=100)
#
# #video_path = "/mnt/d/netholabs/videos/2025-04-17_17-09-28.h264"
# video_path = "/mnt/d/netholabs/videos_/2025-05-27_20-59-48.mp4"
# video_path = "/mnt/d/netholabs/yolo_videos/input/mp4_20250606083508/2025-05-28_19-50-23.mp4"
# #video_path = "/mnt/c/Users/sroni/Downloads/2025-05-27_00-32-26.mp4"
#
#
# #
# inference_yolo_pose(weights_path=r"/mnt/d/netholabs/yolo_mdls_1/train/weights/best.pt",
#                     video_path=video_path,
#                     save_dir=r"/mnt/d/netholabs/yolo_test/results",
#                     verbose=True,
#                     device=0,
#                     format='onnx',
#                     keypoint_names=('nose', 'ear_left', 'ear_right', 'lateral_left', 'center', 'lateral_right', 'tail_base'),
#                     batch_size=32)
#
#
# #



# fit_yolo(initial_weights=r"/mnt/d/yolo_weights/yolov8n.pt",
#          model_yaml=r"/mnt/d/netholabs/imgs/yolo_train_test_val/map.yaml", save_path=r"/mnt/d/netholabs/imgs/yolo_mdl", batch=32, epochs=100)

# fit_yolo(initial_weights=r"/mnt/c/troubleshooting/coco_data/weights/yolov8n-obb.pt",
#          model_yaml=r"/mnt/c/troubleshooting/coco_data/model.yaml",
#          project_path=r"/mnt/c/troubleshooting/coco_data/mdl",
#          batch=16, epochs=100)



# fit_yolo(initial_weights=r"/mnt/c/troubleshooting/coco_data/weights/yolov8n.pt",
#          model_yaml=r"/mnt/d/netholabs/yolo_test/yolo_train/map.yaml",
#          save_path=r"/mnt/d/netholabs/yolo_test/yolo_mdl", epochs=150, batch=24)



# inference_yolo(weights_path=r"/mnt/d/netholabs/yolo_test/yolo_mdl/train10/weights/best.pt",
#                video_path=r"/mnt/d/netholabs/out/2025-04-17_17-05-07.mp4",
#                save_dir=r"/mnt/d/netholabs/yolo_test/results",
#                verbose=True,
#                gpu=False)

# inference_yolo(weights=r"/mnt/c/troubleshooting/coco_data/mdl/train8/weights/best.pt",
#                video_path=r"/mnt/c/troubleshooting/mitra/project_folder/videos/FRR_gq_Saline_0624.mp4",
#                save_dir=r"/mnt/c/troubleshooting/coco_data/mdl/results",
#                verbose=True,
#                gpu=True)
#
#


# r = inference_yolo(weights=r"C:\troubleshooting\coco_data\mdl\train\weights\best.pt",
#                video_path=r"C:\troubleshooting\mitra\project_folder\videos\clipped\501_MA142_Gi_CNO_0514_clipped.mp4",
#                batch=100)


#fit_yolo(initial_weights=r"C:\troubleshooting\coco_data\weights\yolov8n-obb.pt", data=r"C:\troubleshooting\coco_data\model.yaml", project_path=r"C:\troubleshooting\coco_data\mdl", batch=16)


# """
# initial_weights=r"C:\Users\sroni\Downloads\yolov8n.pt", data=r"C:\troubleshooting\coco_data\model.yaml", epochs=30, project_path=r"C:\troubleshooting\coco_data\mdl", batch=16)
# """
#
#
#



