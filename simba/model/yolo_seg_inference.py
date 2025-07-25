import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
try:
    from typing import Literal
except:
    from typing_extensions import Literal

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int,
                                check_valid_boolean, check_valid_lst,
                                get_fn_ext)
from simba.utils.data import resample_geometry_vertices
from simba.utils.enums import Options
from simba.utils.errors import InvalidFileTypeError
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_video_meta_data)
from simba.utils.yolo import load_yolo_model, yolo_predict

TASK = 'segment'

class YOLOSegmentationInference():

    """
     Run inference on video(s) using a trained YOLO segmentation model.

    :param Union[str, os.Pathlike] weights_path: Path to the trained YOLO `.pt` weights file.
    :param Union[str, os.Pathlike]  video_path: Path to a single video or a list of video paths to run inference on.
    :param bool verbose: Whether to print progress information. Default is True.
    :param Union[str, os.Pathlike]  save_dir: Directory where output videos and data will be saved.
    :param Union[str, int] device: Device to run inference on; use 'cpu' or an integer GPU index (e.g., 0).
    :param str format: Optional export format for the model. Supported values: "onnx", "engine", "torchscript", "onnxsimplify", "coreml", "openvino", "pb", "tf", "tflite". Defaults to None.
    :param Optional[int] batch_size: Number of frames to process at once. Increase for faster performance with sufficient memory.
    :param int torch_threads: Number of CPU threads to use (when on CPU).
    :param bool half_precision: Whether to use half-precision (FP16) for inference on GPU. Default is True.
    :param bool stream: Whether to stream video processing (less memory, suitable for long videos).
    :param float threshold: Confidence threshold for object/segmentation detection.
    :param int max_tracks: Optional maximum number of objects to track. If None, tracking is disabled.
    :param bool interpolate: Whether to interpolate results (useful for smoothing or low-FPS videos).
    :param int imgsz: Inference image size (width/height in pixels); must be multiple of 32.
    :param float iou: IoU threshold for non-max suppression (NMS).
    :param bool retina_msk: Whether to use high-resolution Retina-style masks.
    :param int vertice_cnt: Number of vertices used to approximate the segmentation mask polygon.

    .. note::
       To **create** YOLO segmentation dataset for fitting, use :func:`simba.third_party_label_appenders.transform.labelme_to_yolo_seg.LabelmeKeypoints2YoloSeg`.
       To fit YOLO model, see `:func:~simba.bounding_box_tools.yolo.yolo_fit.FitYolo`.
       To visualize the segmentation results, see :func:`simba.bounding_box_tools.yolo.yolo_seg_visualizer.YOLOSegmentationVisualizer`

    :example:
    >>> weights_path = r"D:\platea\yolo_071525\mdl\train3\weights\best.pt"
    >>> video_path = r"D:\platea\platea_videos\videos\clipped\10B_Mouse_5-choice_MustTouchTrainingNEWFINAL_a7.mp4"
    >>> save_dir=r"D:\platea\platea_videos\videos\yolo_results"
    >>> runner = YOLOSegmentationInference(weights_path=weights_path, video_path=video_path, save_dir=save_dir, verbose=True, device=0, format=None, stream=True, batch_size=10, imgsz=320, interpolate=True, threshold=0.8, retina_msk=True)
    >>> runner.run()

    """

    def __init__(self,
                 weights_path: Union[str, os.PathLike],
                 video_path: Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]],
                 verbose: Optional[bool] = True,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 device: Union[Literal['cpu'], int] = 0,
                 format: Optional[str] = None,
                 batch_size: Optional[int] = 4,
                 torch_threads: int = 8,
                 half_precision: bool = True,
                 stream: bool = False,
                 threshold: float = 0.5,
                 max_tracks: Optional[int] = None,
                 interpolate: bool = False,
                 imgsz: int = 640,
                 iou: float = 0.5,
                 retina_msk: Optional[bool] = False,
                 vertice_cnt: int = 30):

        if isinstance(video_path, list):
            check_valid_lst(data=video_path, source=f'{self.__class__.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
        elif os.path.isfile(video_path):
            video_path = [video_path]
        elif os.path.isdir(video_path):
            video_path = find_files_of_filetypes_in_directory(directory=video_path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, as_dict=False)
        for i in video_path:
            _ = get_video_meta_data(video_path=i)
        check_file_exist_and_readable(file_path=weights_path)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=interpolate, source=f'{self.__class__.__name__} interpolate')
        check_valid_boolean(value=retina_msk, source=f'{self.__class__.__name__} retina_msk')
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        check_int(name=f'{self.__class__.__name__} imgsz', value=imgsz, min_value=1)
        check_int(name=f'{self.__class__.__name__} vertice_cnt', value=vertice_cnt, min_value=3)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=10e-6, max_value=1.0)
        check_float(name=f'{self.__class__.__name__} iou', value=iou, min_value=10e-6, max_value=1.0)
        if max_tracks is not None:
            check_int(name=f'{self.__class__.__name__} max_tracks', value=max_tracks, min_value=1)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        torch.set_num_threads(torch_threads)
        self.model = load_yolo_model(weights_path=weights_path, device=device, format=format)
        self.half_precision, self.stream, self.video_path, self.retina_msk = half_precision, stream, video_path, retina_msk
        self.device, self.batch_size, self.threshold, self.max_tracks, self.iou = device, batch_size, threshold, max_tracks, iou
        self.verbose, self.save_dir, self.imgsz, self.interpolate = verbose, save_dir, imgsz, interpolate
        self.vertice_cnt = vertice_cnt
        if self.model.model.task != TASK:
            raise InvalidFileTypeError(msg=f'The model {weights_path} is not a segmentation model. It is a {self.model.model.task} model', source=self.__class__.__name__)
        self.vertice_col_names = ['FRAME', 'ID']
        for i in range(self.vertice_cnt):
            self.vertice_col_names.append(f"VERTICE_{i}_X"); self.vertice_col_names.append(f"VERTICE_{i}_Y")
    def run(self):
        results = {}
        class_dict = self.model.names
        timer = SimbaTimer(start=True)
        for path in self.video_path:
            _, video_name, _ = get_fn_ext(filepath=path)
            video_meta_data = get_video_meta_data(video_path=path)
            video_results = []
            video_predictions = yolo_predict(model=self.model,
                                             source=path,
                                             half=self.half_precision,
                                             batch_size=self.batch_size,
                                             stream=self.stream,
                                             imgsz=self.imgsz,
                                             device=self.device,
                                             threshold=self.threshold,
                                             max_detections=self.max_tracks,
                                             verbose=self.verbose,
                                             iou=self.iou,
                                             retina_msk=self.retina_msk)
            for frm_cnt, video_prediction in enumerate(video_predictions):
                if video_prediction.masks is not None:
                    boxes = video_prediction.obb.data if video_prediction.obb is not None else video_prediction.boxes.data
                    boxes = boxes.cpu().numpy().astype(np.float32)
                    detected_classes = boxes[:, -1].astype(int) if boxes.size > 0 else []
                    detected_masks = video_prediction.masks.xy
                    for detection_cnt, detected_class in enumerate(detected_classes):
                        mask = detected_masks[detection_cnt].reshape(-1, detected_masks[detection_cnt].shape[0], 2)
                        vertices = resample_geometry_vertices(vertices=mask, vertice_cnt=self.vertice_cnt).flatten()
                        vertices = np.insert(vertices, 0, detected_class)
                        vertices = np.insert(vertices, 0, frm_cnt)
                        video_results.append(vertices)

            vertices = pd.DataFrame(video_results, columns=self.vertice_col_names)
            if self.save_dir:
                save_path = os.path.join(self.save_dir, f'{video_name}.csv')
                vertices.to_csv(save_path)
            else:
                results[video_name] = vertices

        timer.stop_timer()
        if not self.save_dir:
            if self.verbose: print(f'YOLO results created', timer.elapsed_time_str)
            return results
        else:
            if self.verbose:
                print(f'YOLO results saved in {self.save_dir} directory', timer.elapsed_time_str)
            return None

# weights_path = r"D:\platea\yolo_071525\mdl\train3\weights\best.pt"
# video_path = r"D:\platea\platea_videos\videos\clipped\10B_Mouse_5-choice_MustTouchTrainingNEWFINAL_a7.mp4"
# save_dir=r"D:\platea\platea_videos\videos\yolo_results"
# i = YOLOSegmentationInference(weights_path=weights_path,
#                              video_path=video_path,
#                              save_dir=save_dir,
#                              verbose=True,
#                              device=0,
#                              format=None,
#                              stream=True,
#                              batch_size=10,
#                              imgsz=320,
#                              interpolate=True,
#                              threshold=0.8,
#                              retina_msk=True)
# i.run()