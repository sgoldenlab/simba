__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.model.yolo_inference import YoloInference
from simba.plotting.yolo_visualize import COLOR_BY_OPTIONS, YOLOVisualizer
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_string_value_is_valid_video_timestamp,
                                check_int, check_str,
                                check_that_hhmmss_start_is_before_end,
                                check_valid_boolean, check_valid_lst)
from simba.utils.data import get_cpu_pool, terminate_cpu_pool
from simba.utils.enums import Options
from simba.utils.errors import (InvalidInputError, InvalidVideoFileError,
                                SimBAGPUError, SimBAPAckageVersionError)
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data)
from simba.utils.yolo import get_yolo_imgsz_and_batch_size
from simba.video_processors.video_processing import (
    clip_video_in_range, horizontal_video_concatenator, superimpose_freetext)

ENGINE = '.engine'


class YoloModelComparator():
    """
    Compare two or more YOLO models side-by-side on a set of test videos.

    Per-model CSV detections and visualization videos are written to``out_dir/<label>/`` and the final comparison videos to ``out_dir/compare/``.

    .. seealso::
       :class:`~simba.model.yolo_inference.YoloInference` (per-model inference)
       :class:`~simba.plotting.yolo_visualize.YOLOVisualizer` (per-model rendering)
       :func:`~simba.video_processors.video_processing.horizontal_video_concatenator` (side-by-side concat)

    .. video:: _static/img/YoloModelComparator.webm
       :width: 800
       :loop:
       :autoplay:
       :muted:
       :align: center

    :param List[Union[str, os.PathLike]] weights: List of YOLO model weight paths (length >= 2).
    :param Union[str, os.PathLike, List[Union[str, os.PathLike]]] video_path: One of: a single video file path; a directory of videos (searched non-recursively); or a list/tuple of video file paths.
    :param Union[str, os.PathLike] out_dir: Output directory. Per-model CSV/viz subdirectories and a ``compare/`` subdirectory are created underneath.
    :param Optional[List[str]] labels: Display labels (one per model). If None, the weight file name (without extension) of each model is used. Must be unique and match ``len(weights)``.
    :param float threshold: YOLO detection confidence threshold in ``[0.0, 1.0]``.
    :param Union[Literal['cpu'], int] device: Inference device ('cpu' or CUDA index).
    :param Optional[int] batch_size: Frames per YOLO inference batch. If None, it is read per-model from the model metadata (required for ``.engine`` files); falls back to 400 when unavailable (e.g. ``.pt`` weights).
    :param Optional[int] imgsz: Model inference image size. If None, it is read per-model from the model metadata; falls back to 320 when unavailable. For ``.engine`` files the engine's baked-in image size always wins (a mismatching explicit value is overridden with a warning).
    :param int max_detections: Maximum detections per frame (total, across all classes) returned by each model.
    :param Optional[int] max_per_class: Maximum number of detections to retain per class per frame. E.g., if one 'resident' and one 'intruder' is expected, set this to 1. Defaults to None, meaning all detected instances of each class are retained (up to ``max_detections``). Passed through to each model's ``YoloInference``.
    :param Optional[str] palette: Matplotlib palette passed to ``YOLOVisualizer``.
    :param float opacity: Polygon fill opacity in ``[0.0, 1.0]`` for the visualization.
    :param Optional[int] thickness: Polygon line thickness.
    :param int padding: Polygon offset in pixels for the visualization. Defaults to 0 (draw the exact detection box). Positive values expand polygons outward, ``-1`` shrinks them inward.
    :param Optional[Tuple[int, int, int]] outline_color: BGR outline color for polygons.
    :param Literal['class', 'instance'] color_by: Passed to ``YOLOVisualizer``. ``'class'`` (default) colors every instance of a class the same (avoids color flicker when ``max_per_class > 1``, since instances are confidence-ranked per frame, not identity-tracked); ``'instance'`` colors each instance slot separately. Equivalent for single-instance data.
    :param Optional[int] core_cnt: CPU core count for the visualizer. If None, defaults to a quarter of the available cores to leave headroom; ``-1`` = all cores.
    :param bool gpu: Use the NVENC codec when concatenating side-by-side.
    :param bool overwrite: If True (default), always re-render comparison videos. If False, skip videos whose comparison output already exists.
    :param bool overlay_labels: If True, burns each model's ``label`` into the top-left corner of its panel before side-by-side concatenation, so the panels are self-describing.
    :param Optional[Dict[str, str]] time_window: Analysis window as a dict with ``'start'`` and ``'end'`` keys, both in ``HH:MM:SS`` format (e.g. ``{'start': '00:00:05', 'end': '00:00:30'}``). If provided, each video is clipped to the window (saved under ``out_dir/clips/``) and the comparison runs on the clip. If None, the full video is analysed.
    :param bool verbose: Print progress information.

    :example:

    >>> c = YoloModelComparator(
    ...     weights=[r"/mdl/train8/weights/best.pt", r"/mdl/train13/weights/best.pt"],
    ...     video_path=r"/test_videos",
    ...     out_dir=r"/yolo_comparison",
    ...     labels=["train8", "train13"],
    ...     threshold=0.05,
    ...     max_detections=1,
    ...     device=0,
    ... )
    >>> c.run()
    """

    def __init__(self,
                 weights: List[Union[str, os.PathLike]],
                 video_path: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
                 out_dir: Union[str, os.PathLike],
                 labels: Optional[List[str]] = None,
                 threshold: float = 0.05,
                 device: Union[Literal['cpu'], int] = 0,
                 batch_size: Optional[int] = None,
                 imgsz: Optional[int] = None,
                 max_detections: int = 300,
                 max_per_class: Optional[int] = None,
                 palette: Optional[str] = 'Set1',
                 opacity: float = 0.6,
                 thickness: Optional[int] = 2,
                 padding: int = 0,
                 outline_color: Optional[Tuple[int, int, int]] = None,
                 color_by: Literal['class', 'instance'] = 'class',
                 core_cnt: Optional[int] = None,
                 gpu: bool = False,
                 overwrite: bool = True,
                 overlay_labels: bool = True,
                 time_window: Optional[Dict[str, str]] = None,
                 verbose: bool = True):

        if not _is_cuda_available()[0]:
            raise SimBAGPUError(msg='No GPU detected.', source=self.__class__.__name__)
        if YOLO is None:
            raise SimBAPAckageVersionError(msg='ultralytics.YOLO package not detected.', source=self.__class__.__name__)
        check_valid_lst(data=weights, source=f'{self.__class__.__name__} weights', valid_dtypes=(str, os.PathLike), min_len=2)
        for w in weights:
            check_file_exist_and_readable(file_path=w)
        if labels is None:
            labels = [get_fn_ext(filepath=w)[1] for w in weights]
        check_valid_lst(data=labels, source=f'{self.__class__.__name__} labels', valid_dtypes=(str,), exact_len=len(weights))
        if len(set(labels)) != len(labels):
            raise InvalidInputError(msg=f'labels must be unique, got: {labels}', source=self.__class__.__name__)
        if isinstance(video_path, (list, tuple)):
            check_valid_lst(data=video_path, source=f'{self.__class__.__name__} video_path', valid_dtypes=(str, os.PathLike), min_len=1)
            for v in video_path:
                check_file_exist_and_readable(file_path=v)
                get_video_meta_data(video_path=v)
            self.video_paths = list(video_path)
        elif os.path.isfile(video_path):
            check_file_exist_and_readable(file_path=video_path)
            get_video_meta_data(video_path=video_path)
            self.video_paths = [video_path]
        elif os.path.isdir(video_path):
            self.video_paths = find_files_of_filetypes_in_directory(directory=video_path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True, as_dict=False)
        else:
            raise InvalidVideoFileError(msg=f'{video_path} is not a valid video file, directory, or list of video files.', source=self.__class__.__name__)
        check_if_dir_exists(in_dir=out_dir, source=f'{self.__class__.__name__} out_dir', create_if_not_exist=True)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        if batch_size is not None:
            check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        if imgsz is not None:
            check_int(name=f'{self.__class__.__name__} imgsz', value=imgsz, min_value=32)
        check_int(name=f'{self.__class__.__name__} max_detections', value=max_detections, min_value=1)
        if max_per_class is not None:
            check_int(name=f'{self.__class__.__name__} max_per_class', value=max_per_class, min_value=1)
        if core_cnt is None:
            core_cnt = find_core_cnt()[1]
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        check_float(name=f'{self.__class__.__name__} opacity', value=opacity, min_value=0.0, max_value=1.0)
        check_str(name=f'{self.__class__.__name__} color_by', value=color_by, options=COLOR_BY_OPTIONS)
        check_valid_boolean(value=[gpu, overwrite, overlay_labels, verbose], source=self.__class__.__name__)
        self.start_time, self.end_time = None, None
        if time_window is not None:
            if not isinstance(time_window, dict) or set(time_window.keys()) != {'start', 'end'}:
                raise InvalidInputError(msg=f"time_window must be a dict with exactly 'start' and 'end' keys in HH:MM:SS format, got: {time_window}", source=self.__class__.__name__)
            check_if_string_value_is_valid_video_timestamp(value=time_window['start'], name=f'{self.__class__.__name__} time_window start')
            check_if_string_value_is_valid_video_timestamp(value=time_window['end'], name=f'{self.__class__.__name__} time_window end')
            check_that_hhmmss_start_is_before_end(start_time=time_window['start'], end_time=time_window['end'], name=f'{self.__class__.__name__} time_window')
            self.start_time, self.end_time = time_window['start'], time_window['end']

        self.weights, self.labels = list(weights), list(labels)
        self.video_path, self.out_dir = video_path, out_dir
        self.threshold, self.device, self.batch_size, self.imgsz = threshold, device, batch_size, imgsz
        self.max_detections, self.palette, self.opacity = max_detections, palette, opacity
        self.max_per_class, self.color_by = max_per_class, color_by
        self.thickness, self.padding, self.outline_color = thickness, padding, outline_color
        self.core_cnt, self.gpu, self.overwrite, self.verbose = core_cnt, gpu, overwrite, verbose
        self.overlay_labels = overlay_labels
        self.clip_videos = self.start_time is not None
        self.compare_dir = os.path.join(out_dir, 'compare')
        os.makedirs(self.compare_dir, exist_ok=True)
        if self.clip_videos:
            self.clips_dir = os.path.join(out_dir, 'clips')
            os.makedirs(self.clips_dir, exist_ok=True)
        self.csv_dirs, self.viz_dirs = {}, {}
        for label in self.labels:
            csv_dir = os.path.join(out_dir, label, 'csv')
            viz_dir = os.path.join(out_dir, label, 'viz')
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(viz_dir, exist_ok=True)
            self.csv_dirs[label] = csv_dir
            self.viz_dirs[label] = viz_dir

    def _resolve_imgsz_batch(self, weight: Union[str, os.PathLike], label: str) -> Tuple[int, int]:
        """Resolve the effective imgsz/batch_size for a single model. Engine bindings are authoritative; otherwise the user value, then the model metadata, then a default is used."""
        is_engine = get_fn_ext(filepath=weight)[2].lower() == ENGINE
        mdl_imgsz, mdl_batch = get_yolo_imgsz_and_batch_size(model=weight, raise_error=False)
        if is_engine and mdl_imgsz is not None:
            if self.imgsz is not None and self.imgsz != mdl_imgsz and self.verbose:
                stdout_information(msg=f'{label}: passed imgsz={self.imgsz} overridden by engine imgsz={mdl_imgsz}.', source=self.__class__.__name__)
            imgsz = mdl_imgsz
        else:
            imgsz = self.imgsz if self.imgsz is not None else (mdl_imgsz if mdl_imgsz is not None else 320)
        if is_engine and mdl_batch is not None:
            if self.batch_size is not None and self.batch_size != mdl_batch and self.verbose:
                stdout_information(msg=f'{label}: passed batch_size={self.batch_size} overridden by engine batch_size={mdl_batch}.', source=self.__class__.__name__)
            batch_size = mdl_batch
        else:
            batch_size = self.batch_size if self.batch_size is not None else (mdl_batch if mdl_batch is not None else 400)
        return imgsz, batch_size

    def _clip_videos(self) -> List[Union[str, os.PathLike]]:
        """Clip every source video to ``[start_time, end_time]`` into ``out_dir/clips/`` and return the clipped paths. Existing clips are reused when ``overwrite`` is False."""
        clipped_paths = []
        for video_path in self.video_paths:
            _, video_name, _ = get_fn_ext(filepath=video_path)
            save_path = os.path.join(self.clips_dir, f'{video_name}_clipped.mp4')
            if os.path.isfile(save_path) and not self.overwrite:
                if self.verbose:
                    stdout_information(msg=f'{video_name}: clip exists, reusing {save_path}.', source=self.__class__.__name__)
            else:
                clip_video_in_range(file_path=video_path, start_time=self.start_time, end_time=self.end_time, save_path=save_path, overwrite=True, gpu=self.gpu, verbose=self.verbose)
            clipped_paths.append(save_path)
        return clipped_paths

    def run(self):
        timer = SimbaTimer(start=True)
        video_paths = self._clip_videos() if self.clip_videos else self.video_paths
        for label, weight in zip(self.labels, self.weights):
            imgsz, batch_size = self._resolve_imgsz_batch(weight=weight, label=label)
            if self.verbose:
                stdout_information(msg=f'Running YOLO inference for {label} on {len(video_paths)} video(s) (imgsz={imgsz}, batch_size={batch_size})...', source=self.__class__.__name__)
            YoloInference(weights=weight,
                          video_path=video_paths,
                          save_dir=self.csv_dirs[label],
                          device=self.device,
                          batch_size=batch_size,
                          imgsz=imgsz,
                          threshold=self.threshold,
                          max_detections=self.max_detections,
                          max_per_class=self.max_per_class,
                          verbose=self.verbose).run()

        pool = get_cpu_pool(core_cnt=self.core_cnt, verbose=True, source=self.__class__.__name__)
        for video_idx, video_path in enumerate(video_paths):
            _, video_name, _ = get_fn_ext(filepath=video_path)
            out_path = os.path.join(self.compare_dir, f'{video_name}_compare.mp4')
            if os.path.isfile(out_path) and not self.overwrite:
                if self.verbose:
                    stdout_information(msg=f'[{video_idx + 1}/{len(video_paths)}] {video_name}: comparison exists, skipping.', source=self.__class__.__name__)
                continue
            if self.verbose:
                stdout_information(msg=f'[{video_idx + 1}/{len(video_paths)}] Rendering {video_name}...', source=self.__class__.__name__)
            label_font_size = max(int(get_video_meta_data(video_path=video_path)['height'] / 15), 12) if self.overlay_labels else None
            viz_paths = []
            for label in self.labels:
                csv_path = os.path.join(self.csv_dirs[label], f'{video_name}.csv')
                check_file_exist_and_readable(file_path=csv_path)
                visualizer = YOLOVisualizer(data_path=csv_path,
                                            video_path=video_path,
                                            pool=pool,
                                            save_dir=self.viz_dirs[label],
                                            palette=self.palette,
                                            core_cnt=self.core_cnt,
                                            threshold=self.threshold,
                                            padding=self.padding,
                                            thickness=self.thickness,
                                            opacity=self.opacity,
                                            outline_color=self.outline_color,
                                            color_by=self.color_by,
                                            verbose=self.verbose)
                visualizer.run()
                viz_path = os.path.join(self.viz_dirs[label], f'{video_name}.mp4')
                if self.overlay_labels:
                    superimpose_freetext(video_path=viz_path, text=label, font_size=label_font_size, position='top_left', save_dir=self.viz_dirs[label], gpu=self.gpu)
                    viz_path = os.path.join(self.viz_dirs[label], f'{video_name}_text_superimposed.mp4')
                viz_paths.append(viz_path)
            horizontal_video_concatenator(video_paths=viz_paths,
                                          save_path=out_path,
                                          height_idx=0,
                                          gpu=self.gpu,
                                          verbose=self.verbose)

        terminate_cpu_pool(pool=pool, force=False, source=self.__class__.__name__)
        timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f'Saved {len(video_paths)} comparison video(s) for {len(self.weights)} model(s) in {self.compare_dir}.', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)






#if __name__ == '__main__':
# comparator = YoloModelComparator(weights=[r"G:\netholabs\yolo_comparator_test\mdls\yolo_0518_best.pt",
#                                           r"G:\netholabs\yolo_comparator_test\mdls\yolo_0519_best.pt"],
#                                  video_path=r"D:\troubleshooting\sleap_import_test\project_folder\videos\2016, 2017, 2015, 2014.mp4",
#                                  out_dir=r"D:\troubleshooting\sleap_two_animals\project_folder\videos\output",
#                                  labels=["yolo_0518_best", "train13"],
#                                  threshold=0.001,
#                                  max_detections=10,
#                                  max_per_class=4,
#                                  overwrite=True,
#                                  padding=0,
#                                  time_window={'start': '00:01:00', 'end': '00:01:10'},
#                                  overlay_labels=True,
#                                  device=0)
# comparator.run()
