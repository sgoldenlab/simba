import os
import sys
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

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
                                check_valid_boolean, check_valid_device,
                                check_valid_url)
from simba.utils.enums import Options
from simba.utils.errors import SimBAGPUError, SimBAPAckageVersionError
from simba.utils.printing import stdout_information
from simba.utils.read_write import find_core_cnt, get_current_time
from simba.utils.yolo import load_yolo_model

#YOLO_X_PATH = "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11x-pose.pt"

YOLO_M_PATH = "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11m-pose.pt"



class FitYolo():

    """
    Fit an Ultralytics YOLO model (detection, pose, or segmentation) from SimBA projects with parameter validation.

    .. note::
       - Works with any Ultralytics model flavour (bbox, pose, segmentation).
       - Download starter weights from `HuggingFace <https://huggingface.co/Ultralytics>`__.
       - Example dataset YAMLs: `bbox <https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model.yaml>`__, `pose <https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model_keypoints.yaml>`__.

    .. seealso::
       :func:`simba.bounding_box_tools.yolo.utils.fit_yolo` for the functional API.
       :func:`simba.bounding_box_tools.yolo.utils.load_yolo_model` to load trained weights.
       For  instructions, see `YOLO Pose Estimation Training Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/yolo_train.md>`_.

    :param Union[str, os.PathLike] weights_path: Path to base weights (e.g., ``yolo11n.pt`` or ``.onnx`` export).
    :param Union[str, os.PathLike] model_yaml: Dataset configuration YAML describing dataset folders and class labels.
    :param Union[str, os.PathLike] save_path: Directory where training outputs (weights, metrics, plots) are written.
    :param int epochs: Training epochs to run. Must be ≥ 1. Default ``200``.
    :param Union[int, float] batch: Batch size per step. Default ``16``.
    :param bool plots: If ``True``, Ultralytics saves training curves. Default ``True``.
    :param int imgsz: Square image resolution used during training. Default ``640``.
    :param Optional[str] format: Optional weights format override. Must belong to :class:`simba.utils.enums.Options.VALID_YOLO_FORMATS`. Default ``None``.
    :param Union[Literal['cpu'], int] device: Compute device string or CUDA index. Default ``0``.
    :param bool verbose: Emit detailed progress information. Default ``True``.
    :param int workers: Data-loader worker processes. Use ``-1`` for all cores. Default ``8``.
    :param int patience: Early-stopping patience (epochs without improvement). Default ``100``.
    :raises SimBAGPUError: If no CUDA-capable GPU is detected.
    :raises SimBAPAckageVersionError: If ``ultralytics`` is unavailable in the environment.
    :raises FileNotFoundError: If ``weights_path`` or ``model_yaml`` do not exist.
    :raises ValueError: If provided arguments fail SimBA validation checks.

    :example:
       >>> fitter = FitYolo(
       ...     weights_path=r"D:\\yolo_weights\\yolo11n-pose.pt",
       ...     model_yaml=r"D:\\datasets\\pose_project\\map.yaml",
       ...     save_path=r"D:\\datasets\\pose_project\\mdl",
       ...     epochs=300,
       ...     batch=24,
       ...     device=0,
       ...     imgsz=640,
       ... )
       >>> fitter.run()

    """

    def __init__(self,
                 model_yaml: Union[str, os.PathLike],
                 save_path: Union[str, os.PathLike],
                 weights_path: Optional[Union[str, os.PathLike]] = None,
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
        if weights_path is not None:
            check_file_exist_and_readable(file_path=weights_path)
            self.weights_path = weights_path
        else:
            self._download_start_weights()
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
        self.plots, self.save_path, self.verbose, self.patience = plots, save_path, verbose, patience

    def _download_start_weights(self, url: str = YOLO_M_PATH, save_path: Union[str, os.PathLike] = "yolo11m-pose.pt"):
        print(f'No start weights provided, downloading {save_path} from {url}...')
        check_valid_url(url=url, raise_error=True, source=self.__class__.__name__)
        if not os.path.isfile(save_path):
            urllib.request.urlretrieve(url, save_path)
            stdout_information(msg=f'Downloaded initial weights from {url}', source=self.__class__.__name__)
        self.weights_path = save_path
        print(self.weights_path)


    def run(self):
        stdout_information(msg=f'[{get_current_time()}] Please follow the YOLO pose model training in the terminal from where SimBA was launched ...', source=self.__class__.__name__)
        stdout_information(msg=f'[{get_current_time()}] Results will be stored in the {self.save_path} directory ..', source=self.__class__.__name__)
        with redirect_stdout(sys.__stdout__), redirect_stderr(sys.__stderr__):
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
    parser.add_argument('--weights_path', type=str, default=None, help='Path to the trained YOLO model weights (e.g., yolo11n-pose.pt). Omit to download default starter weights.')
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
                          workers=args.workers,
                          patience=args.patience)
    yolo_fitter.run()





# fitter = FitYolo(weights_path=r"D:\maplight_tg2576_yolo\yolo_mdl\original_weight_oct\best.pt",
#                  model_yaml=r"D:\maplight_tg2576_yolo\yolo_mdl\map.yaml",
#                  save_path=r"D:\maplight_tg2576_yolo\yolo_mdl\mdl",
#                  epochs=1500,
#                  batch=22,
#                  format=None,
#                  device=0,
#                  imgsz=640)
# fitter.run()




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


# fitter = FitYolo(weights_path=r"D:\maplight_tg2576_yolo\yolo_mdl\original_weight_oct\best.pt",
#                  model_yaml=r"F:\todd_sleap\yolo_dataset\map.yaml",
#                  save_path=r"F:\todd_sleap\yolo_dataset\mdl",
#                  epochs=1500,
#                  batch=22,
#                  format=None,
#                  device=0,
#                  imgsz=640)
# fitter.run()
