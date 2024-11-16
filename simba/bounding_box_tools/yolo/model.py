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

from simba.third_party_label_appenders.converters import \
    yolo_obb_data_to_bounding_box
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_boolean, check_valid_lst,
                                get_fn_ext)
from simba.utils.data import df_smoother, savgol_smoother
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_video_meta_data


def fit_yolo(initial_weights: Union[str, os.PathLike],
             model_yaml: Union[str, os.PathLike],
             save_path: Union[str, os.PathLike],
             epochs: Optional[int] = 25,
             batch: Optional[Union[int, float]] = 16,
             plots: Optional[bool] = True):
    """
    Trains a YOLO model using specified initial weights and a configuration YAML file.

    .. note::
       `Download initial weights <https://huggingface.co/Ultralytics>`__.
       `Example model_yaml <https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model.yaml>`__.

    :param initial_weights: Path to the pre-trained YOLO model weights (usually a `.pt` file). Example weights can be found [here](https://huggingface.co/Ultralytics).
    :param model_yaml: YAML file containing paths to the training, validation, and testing datasets and the object class mappings. Example YAML file can be found [here](https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model.yaml).
    :param save_path: Directory path where the trained model, logs, and results will be saved.
    :param epochs: Number of epochs to train the model. Default is 5.
    :param batch: Batch size for training. Default is 16.
    :return: None. The trained model and associated training logs are saved in the specified `project_path`.

    :example:
    >>> fit_yolo(initial_weights=r"C:\troubleshooting\coco_data\weights\yolov8n-obb.pt", data=r"C:\troubleshooting\coco_data\model.yaml", save_path=r"C:\troubleshooting\coco_data\mdl", batch=16)

    """
    if not torch.cuda.is_available():
        raise ModuleNotFoundError('No GPU detected.')
    check_file_exist_and_readable(file_path=initial_weights)
    check_file_exist_and_readable(file_path=model_yaml)
    check_if_dir_exists(in_dir=save_path)
    check_int(name='epochs', value=epochs, min_value=1)
    model = YOLO(initial_weights)
    model.train(data=model_yaml, epochs=epochs, project=save_path, batch=batch, plots=plots)

def inference_yolo(weights_path: Union[str, os.PathLike],
                   video_path: Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]],
                   verbose: Optional[bool] = False,
                   save_dir: Optional[Union[str, os.PathLike]] = None,
                   gpu: Optional[bool] = False,
                   half_precision: Optional[bool] = True,
                   batch_size: Optional[int] = 4,
                   smoothing_method: Optional[Literal['savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential']] = None,
                   smoothing_time_window: Optional[int] = None,
                   interpolate: Optional[bool] = False,
                   stream: Optional[bool] = True) -> Union[None, Dict[str, pd.DataFrame]]:

    """
    Performs object detection inference on a video using a YOLO model.

    This function runs YOLO-based object detection on a given video file, optionally utilizing GPU acceleration for
    inference, and either returns the results or saves them to a specified directory. The function outputs bounding box
    coordinates and class confidence scores for detected objects in each frame of the video.

    :param Union[str, os.PathLike] weights_path: Path to the YOLO model weights file.
    :param Union[str, os.PathLike] video_path: Path to the input video file for performing inference.
    :param Optional[bool] verbose: If True, outputs progress information and timing. Defaults to False.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the inference results as CSV files. If not provided, results are returned as a dictionary. Defaults to None.
    :param Optional[bool] gpu: If True, performs inference on the GPU. Defaults to False.
    :param Optional[bool] stream: If True, iterate over a generater. Default to True. Recommended on longer videos.
    :param Optional[bool] batch_size: Number of frames to process in parallel. Default to 4.
    :param Optional[bool] interpolate: If True, interpolates missing bounding boxes using ``nearest`` method.

    :example:
    >>> inference_yolo(weights_path=r"/mnt/c/troubleshooting/coco_data/mdl/train8/weights/best.pt", video_path=r"/mnt/c/troubleshooting/mitra/project_folder/videos/FRR_gq_Saline_0624.mp4", save_dir=r"/mnt/c/troubleshooting/coco_data/mdl/results", verbose=True, gpu=True)
    """
    COORD_COLS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
    OUT_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
    SMOOTHING_METHODS = ('savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential')
    NEAREST = 'nearest'

    if isinstance(video_path, list):
        check_valid_lst(data=video_path, source=f'{inference_yolo.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
    elif isinstance(video_path, str):
        check_file_exist_and_readable(file_path=video_path)
        video_path = [video_path]
    for i in video_path:
        _ = get_video_meta_data(video_path=i)
    check_file_exist_and_readable(file_path=weights_path)
    check_valid_boolean(value=[half_precision, gpu, verbose, stream, interpolate], source=inference_yolo.__name__)
    check_int(name=f'{inference_yolo.__name__} batch_size', value=batch_size, min_value=1)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=f'{inference_yolo.__name__} save_dir')
    if smoothing_method is not None:
        check_str(name=f'{inference_yolo.__name__} smoothing', value=smoothing_method, options=SMOOTHING_METHODS)
        check_float(name=f'{inference_yolo.__name__} smoothing_time_window', value=smoothing_time_window, min_value=10e-6)
    timer = SimbaTimer(start=True)
    torch.set_num_threads(8)
    model = YOLO(weights_path, verbose=verbose)
    if gpu:
        model.export(format='engine')
        model.to('cuda')
    class_dict = model.names
    results = {}
    for path in video_path:
        _, video_name, _ = get_fn_ext(filepath=path)
        video_meta_data = get_video_meta_data(video_path=path)
        video_out = []
        video_predictions = model.predict(source=path, half=half_precision, batch=batch_size, stream=stream)
        for frm_cnt, video_prediction in enumerate(video_predictions):
            if video_prediction.obb is not None:
                boxes = np.array(video_prediction.obb.data.cpu()).astype(np.float32)
            else:
                boxes = np.array(video_prediction.boxes.data.cpu()).astype(np.float32)
            for c in list(class_dict.keys()):
                cls_data = boxes[np.argwhere(boxes[:, -1] == c)]
                if cls_data.shape[0] == 0:
                    video_out.append(np.array([frm_cnt, c, class_dict[c], -1, -1, -1, -1, -1, -1, -1, -1, -1]))
                else:
                    cls_data = cls_data.reshape(-1, boxes.shape[1])[np.argmax(boxes[:, -2].flatten())]
                    if video_prediction.obb is not None:
                        box = yolo_obb_data_to_bounding_box(center_x=cls_data[0], center_y=cls_data[1], width=cls_data[2], height=cls_data[3], angle=cls_data[4]).flatten()
                    else:
                        box = np.array([cls_data[0], cls_data[1], cls_data[2], cls_data[1], cls_data[2], cls_data[3], cls_data[0], cls_data[3]]).astype(np.int32)
                    video_out.append([frm_cnt, cls_data[-1], class_dict[cls_data[-1]], cls_data[-2]] + list(box))
        results[video_name] = pd.DataFrame(video_out, columns=OUT_COLS)
        if interpolate:
            for cord_col in COORD_COLS:
                results[video_name][cord_col] = results[video_name][cord_col].astype(np.int32).replace(to_replace=-1, value=np.nan)
                results[video_name][cord_col] = results[video_name][cord_col].interpolate(method=NEAREST, axis=0).ffill().bfill()
        if smoothing_method is not None:
            if smoothing_method != 'savitzky-golay':
                smoothened = df_smoother(data=results[video_name][COORD_COLS], fps=video_meta_data['fps'], time_window=smoothing_time_window, source=inference_yolo.__name__, method=smoothing_method)
            else:
                smoothened = savgol_smoother(data=results[video_name][COORD_COLS], fps=video_meta_data['fps'], time_window=smoothing_time_window, source=inference_yolo.__name__)
            results[video_name].update(smoothened)
    timer.stop_timer()
    if not save_dir:
        stdout_success(f'YOLO results created', elapsed_time=timer.elapsed_time_str)
        return results
    else:
        for k, v in results.items():
            save_path = os.path.join(save_dir, f'{k}.csv')
            v.to_csv(save_path)
        if verbose:
            stdout_success(f'YOLO results saved in {save_dir} directory', elapsed_time=timer.elapsed_time_str)


# fit_yolo(initial_weights=r"/mnt/c/troubleshooting/coco_data/weights/yolov8n-obb.pt",
#          model_yaml=r"/mnt/c/troubleshooting/coco_data/model.yaml",
#          project_path=r"/mnt/c/troubleshooting/coco_data/mdl",
#          batch=16, epochs=100)


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



