import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_int, get_fn_ext)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_video_meta_data


def fit_yolo(initial_weights: Union[str, os.PathLike],
             model_yaml: Union[str, os.PathLike],
             project_path: Union[str, os.PathLike],
             epochs: Optional[int] = 5,
             batch: Optional[Union[int, float]] = 16):
    """
    Trains a YOLO model using specified initial weights and a configuration YAML file.

    .. note::
       `Download initial weights <https://docs.ultralytics.com/tasks/obb/#export>`__.
       `Example model_yaml <https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model.yaml>`__.

    :param initial_weights: Path to the pre-trained YOLO model weights (usually a `.pt` file). Example weights can be found [here](https://docs.ultralytics.com/tasks/obb/#export).
    :param model_yaml: YAML file containing paths to the training, validation, and testing datasets and the object class mappings. Example YAML file can be found [here](https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model.yaml).
    :param project_path: irectory path where the trained model, logs, and results will be saved.
    :param epochs: Number of epochs to train the model. Default is 5.
    :param batch: Batch size for training. Default is 16.
    :return: None. The trained model and associated training logs are saved in the specified `project_path`.

    :example:
    >>> fit_yolo(initial_weights=r"C:\troubleshooting\coco_data\weights\yolov8n-obb.pt", data=r"C:\troubleshooting\coco_data\model.yaml", project_path=r"C:\troubleshooting\coco_data\mdl", batch=16)

    """
    if not torch.cuda.is_available():
        raise ModuleNotFoundError('No GPU detected.')
    check_file_exist_and_readable(file_path=initial_weights)
    check_file_exist_and_readable(file_path=model_yaml)
    check_if_dir_exists(in_dir=project_path)
    check_int(name='epochs', value=epochs, min_value=1)
    model = YOLO(initial_weights)
    model.train(data=model_yaml, epochs=epochs, project=project_path, batch=batch)

def inference_yolo(weights: Union[str, os.PathLike],
                   video_path: Union[str, os.PathLike],
                   verbose: Optional[bool] = False,
                   save_dir: Optional[Union[str, os.PathLike]] = None,
                   gpu: Optional[bool] = False) -> Union[None, Dict[str, pd.DataFrame]]:
    """
    Performs object detection inference on a video using a YOLO model.

    This function runs YOLO-based object detection on a given video file, optionally utilizing GPU acceleration for
    inference, and either returns the results or saves them to a specified directory. The function outputs bounding box
    coordinates and class confidence scores for detected objects in each frame of the video.

    :param Union[str, os.PathLike] weights: Path to the YOLO model weights file.
    :param Union[str, os.PathLike] video_path: Path to the input video file for performing inference.
    :param Optional[bool] verbose: If True, outputs progress information and timing. Defaults to False.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the inference results as CSV files. If not provided, results are returned as a dictionary. Defaults to None.
    :param Optional[bool] gpu: If True, performs inference on the GPU. Defaults to False.

    :example:
    >>> inference_yolo(weights=r"/mnt/c/troubleshooting/coco_data/mdl/train8/weights/best.pt", video_path=r"/mnt/c/troubleshooting/mitra/project_folder/videos/FRR_gq_Saline_0624.mp4", save_dir=r"/mnt/c/troubleshooting/coco_data/mdl/results", verbose=True, gpu=True)
    """

    timer = SimbaTimer(start=True)
    torch.set_num_threads(8)
    model = YOLO(weights, verbose=verbose)
    if gpu:
        model.export(format='engine')
        model.to('cuda')
    results = {}
    out_cols = ['FRAME', 'CLASS', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=inference_yolo.__name__)
    if os.path.isfile(video_path):
        _ = get_video_meta_data(video_path=video_path)
        _, video_name, _ = get_fn_ext(filepath=video_path)
        video_out = []
        video_results = model(video_path)
        for frm_cnt, frm in enumerate(video_results):
            if frm.obb is not None:
                data = np.array(frm.obb.data.cpu()).astype(np.float32)
            else:
                data = np.array(frm.boxes.data.cpu()).astype(np.float32)
            classes = np.unique(data[:, -1])
            for c in classes:
                cls_data = data[np.argwhere(data[:, -1] == c)].reshape(-1, data.shape[1])
                cls_data = cls_data[np.argmax(data[:, -2].flatten())]
                cord_data = np.array([cls_data[0], cls_data[1], cls_data[0], cls_data[3], cls_data[1], cls_data[3], cls_data[2], cls_data[1]]).astype(np.int32)
                video_out.append([frm_cnt, cls_data[-1], cls_data[-2]] + list(cord_data))
        results[video_name] = pd.DataFrame(video_out, columns=out_cols)

    if not save_dir:
        return results
    else:
        for k, v in results.items():
            save_path = os.path.join(save_dir, f'{k}.csv')
            v.to_csv(save_path)
        if verbose:
            timer.stop_timer()
            stdout_success(f'Results saved in {save_dir} directory', elapsed_time=timer.elapsed_time_str)






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



