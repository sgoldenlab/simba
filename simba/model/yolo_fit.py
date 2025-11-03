import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from typing import Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_boolean, check_valid_device)
from simba.utils.enums import Options
from simba.utils.errors import SimBAGPUError, SimBAPAckageVersionError
from simba.utils.read_write import find_core_cnt
from simba.utils.yolo import load_yolo_model


class FitYolo():

    """
    Trains a YOLO model using specified weights and a configuration YAML file.

    .. note::
       -  Can fit whatever model (bbox, kpts, segmentation).
       - `Download initial weights <https://huggingface.co/Ultralytics>`__.
       - `Example model_yaml bounding-boxes <https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model.yaml>`__.
       - `Example model_yaml keypoints <https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model_keypoints.yaml>`__.

    .. seealso::
       :func:`simba.bounding_box_tools.yolo.utils.fit_yolo`
       :func:`simba.bounding_box_tools.yolo.utils.load_yolo_model`

    :param weights_path: Path to the pre-trained YOLO model weights (e.g., 'yolov8.pt').
    :param model_yaml: Path to the dataset configuration YAML file containing train/val/test paths and class labels.
    :param save_path: Directory where the trained model, logs, and results will be saved.
    :param epochs: Number of training epochs. Must be ≥ 1. Default is 25.
    :param batch: Batch size used for training. Default is 16.
    :param plots: Whether to generate training plots (e.g., loss curves). Default is True.
    :param imgsz: Image size (height/width in pixels) used for training. Must be ≥ 1. Default is 640.
    :param format: Format of the YOLO model weights (e.g., 'pt', 'onnx'). Must be one of the supported formats or None.
    :param device: Device to train on. Can be 'cpu' or a GPU index (e.g., 0). Default is 0.
    :param verbose: If True, prints detailed logs during training. Default is True.
    :param workers: Number of worker threads for data loading. Use -1 to use all available cores. Default is 8.
    :return: None. Trained model and logs are saved to the specified save_path.

    :references:
       .. [1] Augustine, Farhan, Shawn M. Doss, Ryan M. Lee, and Harvey S. Singer. “YOLOv11-Based Quantification and Temporal Analysis of Repetitive Behaviors in Deer Mice.” Neuroscience 577 (June 2025): 343–56. https://doi.org/10.1016/j.neuroscience.2025.05.017.

    :example:
    >>> fitter = FitYolo(weights_path=r"D:\yolo_weights\yolo11n-pose.pt",  model_yaml=r"D:\cvat_annotations\yolo_07032025\map.yaml",  save_path=r"D:\cvat_annotations\yolo_07032025\mdl",  epochs=1500,  batch=32,  format='onnx',  device=0,  imgsz=640)
    >>> fitter.run()

    """

    def __init__(self,
                 weights_path: Union[str, os.PathLike],
                 model_yaml: Union[str, os.PathLike],
                 save_path: Union[str, os.PathLike],
                 epochs: int = 200,
                 batch: Union[int, float] = 16,
                 plots: bool = True,
                 imgsz: int = 640,
                 format: Optional[str] = None,
                 device:  Union[Literal['cpu'], int] = 0,
                 verbose: bool = True,
                 workers: int = 8,
                 patience: int = 100):

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        if not _is_cuda_available()[0]:
            raise SimBAGPUError(msg='No GPU detected.', source=self.__class__.__name__)
        if YOLO is None:
            raise SimBAPAckageVersionError(msg='Ultralytics package not detected.', source=self.__class__.__name__)
        check_file_exist_and_readable(file_path=weights_path)
        check_file_exist_and_readable(file_path=model_yaml)
        check_valid_boolean(value=verbose, source=f'{__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=plots, source=f'{__class__.__name__} plots', raise_error=True)
        check_if_dir_exists(in_dir=save_path)
        if format is not None: check_str(name=f'{__class__.__name__} format', value=format.lower(), options=Options.VALID_YOLO_FORMATS.value, raise_error=True)
        check_int(name=f'{__class__.__name__} epochs', value=epochs, min_value=1)
        check_int(name=f'{__class__.__name__} imgsz', value=imgsz, min_value=1)
        check_int(name=f'{__class__.__name__} workers', value=workers, min_value=-1, unaccepted_vals=[0], max_value=find_core_cnt()[0])
        check_int(name=f'{__class__.__name__} patience', value=patience, min_value=1)
        if workers == -1: workers = find_core_cnt()[0]
        check_valid_device(device=device)
        self.model_yaml, self.epochs, self.batch  = model_yaml, epochs, batch
        self.imgsz, self.device, self.workers, self.format = imgsz, device, workers, format
        self.plots, self.save_path, self.verbose, self.weights_path, self.patience = plots, save_path, verbose, weights_path, patience


    def run(self):
        model = load_yolo_model(weights_path=self.weights_path,
                                verbose=self.verbose,
                                format=self.format,
                                device=self.device)

        model.train(data=self.model_yaml,
                    epochs=self.epochs,
                    project=self.save_path,
                    batch=self.batch,
                    plots=self.plots,
                    imgsz=self.imgsz,
                    workers=self.workers,
                    device=self.device,
                    patience=self.patience)


if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    parser = argparse.ArgumentParser(description="Fit YOLO model using ultralytics package.")
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the trained YOLO model weights (e.g., yolo11n-pose.pt)')
    parser.add_argument('--model_yaml', type=str, required=True, help='Path to map.yaml (model structure and label definitions)')
    parser.add_argument('--save_path', type=str, required=True, help='Directory where trained model and logs will be saved')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train the model. Default is 25')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training. Default is 16')
    parser.add_argument('--plots', type=lambda x: str(x).lower() == 'true', default=True, help='Whether to plot training results. Use "True" or "False". Default is True')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training. Default is 640')
    parser.add_argument('--format', type=str, default=None,  help=f'Format of the YOLO model. Must be one of: {", ".join(Options.VALID_YOLO_FORMATS.value)}')
    parser.add_argument('--device', type=str, default='0', help='Device to train on. Use "cpu" or GPU index (e.g., "0"). Default is "0"')
    parser.add_argument('--verbose', type=lambda x: str(x).lower() == 'true', default=True, help='Print verbose messages. Use "True" or "False". Default is True')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loader workers. Default is 8. Use -1 for max cores')
    parser.add_argument('--patience', type=int, default=100, help='Number of epochs to wait without improvement in validation metrics before early stopping the training. Default is 100')


    args = parser.parse_args()

    yolo_fitter = FitYolo(weights_path=args.weights_path,
                          model_yaml=args.model_yaml,
                          save_path=args.save_path,
                          epochs=args.epochs,
                          batch=args.batch,
                          plots=args.plots,
                          imgsz=args.imgsz,
                          format=args.format,
                          device=int(args.device) if args.device != 'cpu' else 'cpu',
                          verbose=args.verbose,
                          workers=args.workers)
    yolo_fitter.run()



# fitter = FitYolo(weights_path=r"E:\yolo_resident_intruder\mdl\train3\weights\best.pt",
#                  model_yaml=r"E:\maplight_videos\yolo_mdl\map.yaml",
#                  save_path=r"E:\maplight_videos\yolo_mdl\mdl",
#                  epochs=1500,
#                  batch=22,
#                  format=None,
#                  device=0,
#                  imgsz=640)
# fitter.run()
#

# fitter = FitYolo(weights_path=r"E:\netholabs_videos\3d\yolo_mdl\mdl\train5\weights\best.pt",
#                  model_yaml=r"E:\netholabs_videos\3d\yolo_mdl\map.yaml",
#                  save_path=r"E:\netholabs_videos\3d\yolo_mdl\mdl",
#                  epochs=1500,
#                  batch=8,
#                  format=None,
#                  device=0,
#                  imgsz=820)
# fitter.run()



# fitter = FitYolo(weights_path=r"D:\yolo_weights\yolo11m-pose.pt",
#                  model_yaml=r"D:\cvat_annotations\frames\yolo_072125\map.yaml",
#                  save_path=r"D:\cvat_annotations\frames\yolo_072125\mdl",
#                  epochs=1000,
#                  batch=24,
#                  format=None,
#                  device=0,
#                  imgsz=640)
# fitter.run()
# # #

#
# fitter = FitYolo(weights_path=r"D:\yolo_weights\yolo11m-seg.pt",
#                  model_yaml=r"D:\troubleshooting\mitra\mitra_yolo_seg\map.yaml",
#                  save_path=r"D:\troubleshooting\mitra\mitra_yolo_seg\mdl",
#                  epochs=1500,
#                  batch=16,
#                  format=None,
#                  device=0,
#                  imgsz=640)
# fitter.run()
#

