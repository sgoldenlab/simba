import os
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from simba.bounding_box_tools.yolo.utils import (_get_undetected_obs,
                                                 filter_yolo_keypoint_data,
                                                 load_yolo_model)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_array, check_valid_boolean,
                                check_valid_lst, check_valid_tuple, get_fn_ext)
from simba.utils.errors import CountError, InvalidFileTypeError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_video_meta_data

OUT_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'TRACK', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
COORD_COLS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
NEAREST = 'nearest'

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
                 imgsz: int = 320):

        if isinstance(video_path, list):
            check_valid_lst(data=video_path, source=f'{self.__class__.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
        elif isinstance(video_path, str):
            check_file_exist_and_readable(file_path=video_path)
            video_path = [video_path]
        for i in video_path: _ = get_video_meta_data(video_path=i)
        check_file_exist_and_readable(file_path=weights_path)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=interpolate, source=f'{self.__class__.__name__} interpolate')
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        check_int(name=f'{self.__class__.__name__} imgsz', value=imgsz, min_value=1)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=10e-6, max_value=1.0)
        check_valid_tuple(x=keypoint_names, source=f'{self.__class__.__name__} keypoint_names', min_integer=1, valid_dtypes=(str,))
        check_file_exist_and_readable(file_path=config_path)
        self.keypoint_col_names = [f'{i}_{s}'.upper() for i in keypoint_names for s in ['x', 'y', 'p']]
        self.keypoint_cord_col_names = [f'{i}_{s}'.upper() for i in keypoint_names for s in ['x', 'y']]
        OUT_COLS.extend(self.keypoint_col_names)
        COORD_COLS.extend(self.keypoint_cord_col_names)
        torch.set_num_threads(torch_threads)
        self.model = load_yolo_model(weights_path=weights_path, device=device, format=format)
        self.half_precision, self.stream, self.video_path, self.config_path = half_precision, stream, video_path, config_path
        self.batch_size, self.threshold, self.max_tracks = batch_size, threshold, max_tracks
        self.verbose, self.save_dir, self.imgsz, self.interpolate, self.device = verbose, save_dir, imgsz, interpolate, device
        if self.model.model.task != 'pose':
            raise InvalidFileTypeError(msg=f'The model {weights_path} is not a pose model. It is a {self.model.model.task} model', source=self.__class__.__name__)
        if self.model.model.kpt_shape[0] != len(keypoint_names):
            raise CountError(msg=f'The YOLO model expects {self.model.model.model.head.kpt_shape[0]} keypoints but you passed {len(keypoint_names)}: {keypoint_names}', source=self.__class__.__name__)
        self.class_ids = self.model.names


    def run(self):
        self.results = {}
        timer = SimbaTimer(start=True)
        for video_cnt, video_path in enumerate(self.video_path):
            _, video_name, _ = get_fn_ext(filepath=video_path)
            _ = get_video_meta_data(video_path=video_path)
            video_out = []
            video_predictions = self.model.track(source=video_path, stream=self.stream, tracker=self.config_path, conf=self.threshold, half=self.half_precision, imgsz=self.imgsz, persist=False, device=self.device, max_det=self.max_tracks)
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
            self.results[video_name][COORD_COLS] = self.results[video_name][COORD_COLS].astype(float).astype(int)
            if self.interpolate:
                for cord_col in COORD_COLS:
                    self.results[video_name][cord_col] = self.results[video_name][cord_col].astype(np.int32).replace(to_replace=-1, value=np.nan)
                    self.results[video_name][cord_col] = self.results[video_name][cord_col].interpolate(method=NEAREST, axis=0).ffill().bfill()
        timer.stop_timer()
        if not self.save_dir:
            if self.verbose:
                print(f'YOLO results created', timer.elapsed_time_str)
            return self.results
        else:
            for k, v in self.results.items():
                save_path = os.path.join(self.save_dir, f'{k}.csv')
                v.to_csv(save_path)
                if self.verbose:
                    print(f'YOLO results saved in {self.save_dir} directory', timer.elapsed_time_str)
        stdout_success(msg=f'{len(self.video_path)} poe-estimation data files saved in {self.save_dir}', elapsed_time=timer.elapsed_time_str)
        return None


# VIDEO_PATH = "/mnt/d/netholabs/yolo_videos/input/mp4_20250606083508/2025-05-28_19-50-23.mp4"
# #VIDEO_PATH = "/mnt/d/netholabs/yolo_videos/2025-05-28_19-46-56.mp4"
# VIDEO_PATH = "/mnt/d/netholabs/yolo_videos/2025-05-28_19-46-56.mp4"
BOTSORT_PATH = "/mnt/c/projects/simba/simba/simba/assets/bytetrack.yml"
#BOTSORT_PATH = "/mnt/c/projects/simba/simba/simba/assets/botsort.yml"


VIDEO_PATH = r"/mnt/d/ares/data/termite_2/videos/termite.mp4"
WEIGHTS_PASS = r"/mnt/d/ares/data/termite_2/yolo/mdl/train13/weights/best.pt"
SAVE_DIR = "/mnt/d/ares/data/termite_2/yolo/results"

VIDEO_PATH = r"/mnt/d/ares/data/ant/sleap_video/ant.mp4"
WEIGHTS_PASS = r"/mnt/d/ares/data/ant/yolo/mdl/train6/weights/best.pt"
SAVE_DIR = "/mnt/d/ares/data/ant/yolo/results"

i = YOLOPoseTrackInference(weights_path=WEIGHTS_PASS,
                           video_path=VIDEO_PATH,
                           save_dir=SAVE_DIR,
                           verbose=True,
                           device=0,
                           format='onnx',
                           keypoint_names=('head', 'thorax', 'abdomen'),
                           batch_size=32,
                           threshold=0.01,
                           config_path=BOTSORT_PATH,
                           interpolate=False,
                           imgsz=640,
                           max_tracks=5,
                           stream=True)
i.run()
