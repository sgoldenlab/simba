import functools
import multiprocessing
import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from simba.utils.enums import Defaults
from simba.utils.read_write import (find_core_cnt, get_video_meta_data,
                                    read_img_batch_from_video_gpu)


def fit_yolo(initial_weights: Union[str, os.PathLike],
             data: Union[str, os.PathLike],
             project_path: Union[str, os.PathLike],
             epochs: Optional[int] = 5,
             batch: Optional[Union[int, float]] = 16):
    """

    :param initial_weights:
    :param data:
    :param project_path:
    :param epochs:
    :param batch:
    :return:

    :example:
    >>> fit_yolo(initial_weights=r"C:\troubleshooting\coco_data\weights\yolov8n-obb.pt", data=r"C:\troubleshooting\coco_data\model.yaml", project_path=r"C:\troubleshooting\coco_data\mdl", batch=16)

    """

    if not torch.cuda.is_available():
        raise ModuleNotFoundError('No GPU detected.')
    model = YOLO(initial_weights)
    model.train(data=data, epochs=epochs, project=project_path, batch=batch)

def inference_yolo(weights: Union[str, os.PathLike],
                   video_path: Union[str, os.PathLike],
                   batch: Optional[Union[int, float]] = 100,
                   verbose: Optional[bool] = False,
                   save_dir: Optional[Union[str, os.PathLike]] = None):

    torch.set_num_threads(8)
    model = YOLO(weights, verbose=verbose)
    # model.export(format='engine')
    # model.to('cuda')
    results = []
    out_cols = ['FRAME', 'CLASS', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
    if os.path.isfile(video_path):
        _ = get_video_meta_data(video_path=video_path)
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
                results.append([frm_cnt, cls_data[-1], cls_data[-2]] + list(cord_data))
        results = pd.DataFrame(results, columns=out_cols)

    if not save_dir:
        return results


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



