import os
from typing import List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings

import numpy as np
import pandas as pd
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int,
                                check_valid_boolean, check_valid_lst,
                                check_valid_tuple, get_fn_ext)
from simba.utils.enums import Options
from simba.utils.errors import (CountError, InvalidFilepathError,
                                InvalidFileTypeError, SimBAGPUError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_pkg_version, get_video_meta_data)
from simba.utils.yolo import (_get_undetected_obs, filter_yolo_keypoint_data,
                              load_yolo_model)

OUT_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'TRACK', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
COORD_COLS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
NEAREST, CLASS_ID, CONFIDENCE, FRAME  = 'nearest', 'CLASS_ID', 'CONFIDENCE', 'FRAME'

class YOLOPoseTrackInference():
    def __init__(self,
                 weights_path: Union[str, os.PathLike],
                 video_path: Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]],
                 keypoint_names: Tuple[str, ...],
                 config_path: Union[str, os.PathLike],
                 verbose: Optional[bool] = False,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 device: Union[Literal['cpu'], int] = 0,
                 format: Optional[str] = None,
                 batch_size: Optional[int] = 4,
                 torch_threads: int = 8,
                 half_precision: bool = True,
                 stream: bool = False,
                 interpolate: bool = False,
                 threshold: float = 0.7,
                 max_tracks: Optional[int] = 2,
                 smoothing: Optional[int] = None,
                 imgsz: int = 320,
                 iou: float = 0.5):

        _ = get_pkg_version(pkg='ultralytics', raise_error=True)
        gpu_available, gpus = _is_cuda_available()
        if not gpu_available:
            raise SimBAGPUError(msg='No GPU detected.', source=self.__class__.__name__)
        else:
            print(f'GPUS AVAILABLE: {gpus}')
        if isinstance(video_path, list):
            check_valid_lst(data=video_path, source=f'{self.__class__.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
        elif os.path.isfile(video_path):
            check_file_exist_and_readable(file_path=video_path)
            video_path = [video_path]
        elif os.path.isdir(video_path):
            video_path = find_files_of_filetypes_in_directory(directory=video_path, extensions=list(Options.ALL_VIDEO_FORMAT_OPTIONS.value), as_dict=False)
        for i in video_path:
            _ = get_video_meta_data(video_path=i)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir', raise_error=True)
        check_file_exist_and_readable(file_path=weights_path)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=interpolate, source=f'{self.__class__.__name__} interpolate')
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        check_int(name=f'{self.__class__.__name__} imgsz', value=imgsz, min_value=1)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=10e-6, max_value=1.0)
        check_float(name=f'{self.__class__.__name__} iou', value=iou, min_value=10e-6, max_value=1.0)
        check_valid_tuple(x=keypoint_names, source=f'{self.__class__.__name__} keypoint_names', min_integer=1, valid_dtypes=(str,))
        check_file_exist_and_readable(file_path=config_path)
        if smoothing is not None:
            check_int(name=f'{self.__class__.__name__} smoothing', value=smoothing, min_value=1, raise_error=True)
        self.keypoint_col_names = [f'{i}_{s}'.upper() for i in keypoint_names for s in ['x', 'y', 'p']]
        self.keypoint_cord_col_names = [f'{i}_{s}'.upper() for i in keypoint_names for s in ['x', 'y']]
        OUT_COLS.extend(self.keypoint_col_names)
        COORD_COLS.extend(self.keypoint_cord_col_names)
        torch.set_num_threads(torch_threads)
        self.model = load_yolo_model(weights_path=weights_path, device=device, format=format)
        self.half_precision, self.stream, self.video_path, self.config_path = half_precision, stream, video_path, config_path
        self.batch_size, self.threshold, self.max_tracks, self.iou = batch_size, threshold, max_tracks, iou
        self.verbose, self.save_dir, self.imgsz, self.interpolate, self.device = verbose, save_dir, imgsz, interpolate, device
        if self.model.model.task != 'pose':
            raise InvalidFileTypeError(msg=f'The model {weights_path} is not a pose model. It is a {self.model.model.task} model', source=self.__class__.__name__)
        if self.model.model.kpt_shape[0] != len(keypoint_names):
            raise CountError(msg=f'The YOLO model expects {self.model.model.model.head.kpt_shape[0]} keypoints but you passed {len(keypoint_names)}: {keypoint_names}', source=self.__class__.__name__)
        self.class_ids, self.smoothing = self.model.names, smoothing


    def run(self):
        self.results = {}
        timer = SimbaTimer(start=True)
        for video_cnt, video_path in enumerate(self.video_path):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=video_path)
            video_meta = get_video_meta_data(video_path=video_path)
            video_out = []
            video_predictions = self.model.track(source=video_path,
                                                 stream=self.stream,
                                                 tracker=self.config_path,
                                                 conf=self.threshold,
                                                 half=self.half_precision,
                                                 imgsz=self.imgsz,
                                                 persist=False,
                                                 iou=self.iou,
                                                 device=self.device,
                                                 max_det=self.max_tracks)

            for frm_cnt, video_prediction in enumerate(video_predictions):
                boxes = video_prediction.obb.data if video_prediction.obb is not None else video_prediction.boxes.data
                boxes = boxes.cpu().numpy().astype(np.float32)
                keypoints = video_prediction.keypoints.data.cpu().numpy().astype(np.float32)
                detected_classes = np.unique(boxes[:, -1]).astype(int) if boxes.size > 0 else []
                for class_id, class_name in self.class_ids.items():
                    if class_id not in detected_classes:
                        video_out.append(_get_undetected_obs(frm_id=frm_cnt, class_id=class_id, class_name=class_name, value_cnt=(10 + (len(self.keypoint_col_names)))))
                        continue
                    if boxes.shape[1] != 7: boxes = np.insert(boxes, 4, -1, axis=1)
                    cls_boxes, cls_keypoints = filter_yolo_keypoint_data(bbox_data=boxes, keypoint_data=keypoints, class_id=class_id, confidence=None, class_idx=-1, confidence_idx=None)
                    for i in range(cls_boxes.shape[0]):
                        frm_results = np.array([frm_cnt, boxes[i][-1], self.class_ids[boxes[i][-1]], boxes[i][-2], boxes[i][-3]])
                        box = np.array([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][1], boxes[i][2], boxes[i][3], boxes[i][0], boxes[i][3]]).astype(np.int32)
                        frm_results = np.append(frm_results, box)
                        frm_results = np.append(frm_results, keypoints[i].flatten())
                        video_out.append(frm_results)


            self.results[video_name] = pd.DataFrame(video_out, columns=OUT_COLS)
            self.results[video_name]['FRAME'] = self.results[video_name]['FRAME'].astype(np.int64)
            self.results[video_name].loc[:, CLASS_ID] = (pd.to_numeric(self.results[video_name][CLASS_ID], errors='coerce').fillna(0).astype(np.int32))

            if self.interpolate:
                for class_id in self.class_ids.keys():
                    class_df = self.results[video_name][self.results[video_name][CLASS_ID] == int(class_id)]
                    for cord_col in COORD_COLS:
                        class_df[cord_col] = class_df[cord_col].astype(np.float32).replace([-1, 0], np.nan)
                        class_df[cord_col] = class_df[cord_col].interpolate(method=NEAREST, axis=0).ffill().bfill()
                    class_df[CONFIDENCE] = class_df[CONFIDENCE].astype(np.float32).replace([-1, 0], np.nan)
                    class_df[CONFIDENCE] = class_df[CONFIDENCE].interpolate(method=NEAREST, axis=0).ffill().bfill()
                    self.results[video_name].update(class_df)
            if self.smoothing:
                frms_in_smoothing_window = int(self.smoothing / (1000 / video_meta['fps']))
                if frms_in_smoothing_window > 1:
                    for class_id in self.class_ids.keys():
                        class_df = self.results[video_name][self.results[video_name][CLASS_ID] == int(class_id)]
                        for cord_col in COORD_COLS:
                            class_df[cord_col] = class_df[cord_col].rolling(window=frms_in_smoothing_window, win_type='gaussian', center=True).mean(std=5).fillna(self.results[video_name][cord_col]).abs()
                        self.results[video_name].update(class_df)

            self.results[video_name] = self.results[video_name].replace([-1, -1.0, '-1'], 0).reset_index(drop=True)
            if self.save_dir:
                save_path = os.path.join(self.save_dir, f'{video_name}.csv')
                try:
                    self.results[video_name].to_csv(save_path)
                except PermissionError:
                    raise InvalidFilepathError(msg=f'Permission error: Cannot save file {save_path}. Is the file open in another program?', source=self.__class__.__name__)
                del self.results[video_name]
            video_timer.stop_timer()
            if self.verbose:
                print(f'Video {video_name} complete (elapsed time: {video_timer.elapsed_time_str}s, video {video_cnt+1}/{len(self.video_path)})')

        timer.stop_timer()
        if not self.save_dir:
            if self.verbose:
                print(f'YOLO results created for {len(self.video_path)} videos', timer.elapsed_time_str)
            return self.results
        else:
            if self.verbose:
                print(f'YOLO results for {len(self.video_path)} videos saved in {self.save_dir} directory', timer.elapsed_time_str)
            return None

# VIDEO_PATH = "/mnt/d/netholabs/yolo_videos/input/mp4_20250606083508/2025-05-28_19-50-23.mp4"
# #VIDEO_PATH = "/mnt/d/netholabs/yolo_videos/2025-05-28_19-46-56.mp4"
# VIDEO_PATH = "/mnt/d/netholabs/yolo_videos/2025-05-28_19-46-56.mp4"
# BOTSORT_PATH = "/mnt/c/projects/simba/simba/simba/assets/bytetrack.yml"
#BOTSORT_PATH = "/mnt/c/projects/simba/simba/simba/assets/botsort.yml"


# VIDEO_PATH = r"/mnt/d/ares/data/termite_2/videos/termite.mp4"
# WEIGHTS_PASS = r"/mnt/d/ares/data/termite_2/yolo/mdl/train13/weights/best.pt"
# SAVE_DIR = "/mnt/d/ares/data/termite_2/yolo/results"

# VIDEO_PATH = r"/mnt/d/ares/data/ant/sleap_video/ant.mp4"
# WEIGHTS_PASS = r"/mnt/d/ares/data/ant/yolo/mdl/train6/weights/best.pt"
# SAVE_DIR = "/mnt/d/ares/data/ant/yolo/results"


# from simba.utils.read_write import find_files_of_filetypes_in_directory
#
# VIDEO_PATH = find_files_of_filetypes_in_directory(directory=r'E:\netholabs_videos\mosaics\subset', extensions=['.avi'])
#
#
# #VIDEO_PATH = r"D:\cvat_annotations\videos\mp4_20250624155703\s16-Chasing.mp4"
# WEIGHTS_PASS = r"E:\netholabs_videos\mosaics\yolo_mdl_wo_tail\mdl\train2\weights\best.pt"
# SAVE_DIR = r"E:\netholabs_videos\mosaics\yolo_mdl_wo_tail\results_tracks"
# BOTSORT_PATH = r"C:\projects\simba\simba\simba\assets\bytetrack.yml"
#
# KEYPOINT_NAMES = ('Nose', 'Left_ear', 'Right_ear', 'Left_side', 'Center', 'Right_side', 'Tail_base')
#
# i = YOLOPoseTrackInference(weights_path=WEIGHTS_PASS,
#                            video_path=VIDEO_PATH,
#                            save_dir=SAVE_DIR,
#                            verbose=False,
#                            device=0,
#                            format=None,
#                            keypoint_names=KEYPOINT_NAMES,
#                            batch_size=32,
#                            threshold=0.5,
#                            config_path=BOTSORT_PATH,
#                            interpolate=False,
#                            imgsz=640,
#                            max_tracks=3,
#                            stream=True,
#                            iou=0.2)
# i.run()


# VIDEO_PATH = r"E:\netholabs_videos\two_tracks\videos"
# WEIGHTS_PASS = r"D:\netholabs\yolo_mosaic_data_102315\mdl\train\weights\best.pt"
# SAVE_DIR = r"E:\netholabs_videos\two_tracks\csv_track_025"
# BOTSORT_PATH = r"C:\projects\simba\simba\simba\assets\bytetrack.yml"
#
# KEYPOINT_NAMES = ('Nose', 'Left_ear', 'Right_ear', 'Left_side', 'Center', 'Right_side', 'Tail_base', 'Tail_mid', 'Tail_end')
#
# i = YOLOPoseTrackInference(weights_path=WEIGHTS_PASS,
#                            video_path=VIDEO_PATH,
#                            save_dir=SAVE_DIR,
#                            verbose=True,
#                            device=0,
#                            format=None,
#                            keypoint_names=KEYPOINT_NAMES,
#                            batch_size=500,
#                            threshold=0.25,
#                            config_path=BOTSORT_PATH,
#                            interpolate=False,
#                            imgsz=640,
#                            max_tracks=3,
#                            stream=True,
#                            iou=0.2)
# i.run()
