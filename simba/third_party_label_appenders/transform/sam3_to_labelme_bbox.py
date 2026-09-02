"""
Generate a labelme bounding-box project from videos using SAM3.

Takes a directory of videos and a text prompt, samples N random frames per video,
runs SAM3 semantic segmentation, and writes the detected bounding boxes as a flat
labelme directory holding one ``.json`` file per image alongside the image itself.

This is the labelme counterpart of
:class:`~simba.third_party_label_appenders.transform.sam3_to_yolo_bbox.SAM3ToYoloBBox`,
intended for the case where the SAM3 detections are a starting point to be corrected
by hand before training rather than a finished training set.
"""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2

try:
    from typing import Literal
except:
    from typing_extensions import Literal

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch
    from ultralytics.models.sam import SAM3SemanticPredictor
    from ultralytics.utils.metrics import box_iou
except:
    torch, SAM3SemanticPredictor, box_iou = None, None, None

import numpy as np
import yaml

from simba.third_party_label_appenders.transform.utils import arr_to_b64
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_instance, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_boolean, check_valid_lst,
                                check_valid_tuple)
from simba.utils.errors import (InvalidInputError, NoFilesFoundError,
                                SimBAPAckageVersionError)
from simba.utils.printing import (SimbaTimer, stdout_information,
                                  stdout_success, stdout_warning)
from simba.utils.read_write import (create_directory,
                                    find_all_videos_in_directory, get_fn_ext,
                                    get_pkg_version, get_video_meta_data,
                                    read_frm_of_video, recursive_file_search)

LABELME_VERSION = '5.3.1'
VIDEO_EXTENSIONS = [".avi", ".mp4", ".mov", ".flv", ".m4v", ".webm", ".h264"]


class SAM3ToLabelmeBBox:
    r"""
    Sample N random frames from each video in a directory, run SAM3 with a text prompt, and write the resulting bounding boxes as a labelme project.

    For each sampled frame SAM3 returns boxes matching the text prompt; boxes are kept if their confidence is
    ``>= conf`` (optionally size-filtered, de-duplicated by ``iou_threshold`` / ``containment_threshold``, capped at ``max_detections``),
    optionally grown by ``buffer_pct``, clipped to the frame, and written as labelme ``rectangle`` shapes holding
    the two corner points in absolute pixel coordinates - one shape per box.

    Frames where no detection survives the filters are discarded, or saved as un-annotated images with an empty
    ``shapes`` list if ``save_negatives`` is set.

    Output is a single flat directory holding one ``.json`` file and one ``.png`` image per sampled frame, which
    is the layout labelme expects when opening a directory. Unlike the YOLO writer there is no train/val split
    and no ``map.yaml``: class names are written into each shape directly.

    .. image:: _static/img/SAM3ToLabelmeBBox.png
       :alt: The generated project opened in labelme - a SAM3 bounding box around a mouse, ready to be adjusted by hand, with the class listed in the label list and every sampled frame in the file list
       :width: 700
       :align: center

    .. note::
       To convert the hand-corrected labelme directory into a YOLO training project, see
       :class:`~simba.third_party_label_appenders.transform.labelme_to_yolo.LabelmeBoundingBoxes2YoloBoundingBoxes`.

    .. seealso::

       * :class:`~simba.third_party_label_appenders.transform.sam3_to_yolo_bbox.SAM3ToYoloBBox` - the same SAM3 detection stack writing a YOLO detection project instead.
       * :class:`~simba.third_party_label_appenders.transform.yolo_to_labelme.Yolo2Labelme` - convert an existing YOLO bbox project to labelme.

    :param Union[str, os.PathLike, List[Union[str, os.PathLike]]] video_data: Input videos. Accepts: (1) a directory containing video files (combine with ``recursive=True`` to also search subdirectories), (2) a path to a single video file, or (3) a list whose items may be video file paths, directories, or a mix of both. Each directory in the list is scanned for videos (honouring ``recursive``).
    :param Union[str, os.PathLike] sam_path: Path to SAM3 model weights (e.g. sam3.pt).
    :param Union[str, os.PathLike] save_dir: Output directory for the labelme project. Created if it does not exist.
    :param str txt_prompt: Text prompt for SAM3 (e.g. "mouse", "mouse tail").
    :param int n_frames: Number of random frames to sample from each video.
    :param Tuple[str, ...] names: Class names in SAM3 class-index order, used as the labelme shape labels. Default ``('animal',)``.
    :param float conf: SAM3 confidence threshold. Default 0.25.
    :param int sam_imgsz: Image size for SAM3 inference. Default 644.
    :param bool greyscale: If True, save extracted frames in greyscale. Default False.
    :param Optional[Union[Tuple[int, int, int], bool]] clahe: If True, applies CLAHE with default params. If tuple of (clip_limit, tile_x, tile_y), applies CLAHE with those params. Default False.
    :param float buffer_pct: Fraction to expand each box by (e.g. 0.1 adds 10% of width/height on each side). Applied after the ``conf`` and size filters but before the ``iou_threshold`` / ``containment_threshold`` overlap checks, so expansion that pushes two boxes into overlap is caught by them. Default 0.0.
    :param int consecutive_miss_limit: If this many consecutive frames yield no detection, skip to the next video. Default 100.
    :param Optional[int] max_detections: Maximum number of detections to keep per frame (sorted by confidence descending). If None, all detections above ``conf`` are kept. Default None.
    :param Optional[Tuple[int, int]] min_size: If provided, a ``(height, width)`` tuple in pixels. Only bounding boxes at or above this size (measured on the raw SAM3 detection, before ``buffer_pct`` expansion and image clipping) are retained; smaller boxes are discarded. If None, no lower size bound is applied. Default None.
    :param Optional[Tuple[int, int]] max_size: If provided, a ``(height, width)`` tuple in pixels. Only bounding boxes at or below this size (measured on the raw SAM3 detection, before ``buffer_pct`` expansion and image clipping) are retained; larger boxes are discarded. Useful for rejecting detections that span most of the frame, e.g. when the text prompt latches onto the arena rather than the animal. Can be combined with ``min_size`` to keep boxes within a size band. If None, no upper size bound is applied. Default None.
    :param Optional[float] containment_threshold: If provided, a containment threshold in ``(0, 1]`` that catches duplicates ``iou_threshold`` structurally cannot: two boxes of very different size have a low IoU even when one sits entirely inside the other (a 30x30 box inside a 100x100 box has an IoU of 0.09), so a pure-IoU rule leaves nested boxes in place. Containment is measured as the intersection divided by the area of the **smaller** of the two boxes, i.e. the fraction of the smaller box covered by the larger one, and is therefore insensitive to their size difference; a box whose containment with an already-kept box reaches this value is dropped as part of the same observation, keeping the higher-confidence box. The threshold is inclusive, so ``1.0`` drops only boxes fully inside another, ``0.8`` also drops mostly-nested boxes. Combine with ``iou_threshold`` to catch both near-identical and nested duplicates. Measured on the final boxes, i.e. after ``buffer_pct`` expansion and frame clipping - a large ``buffer_pct`` can nest two boxes that were separate in the raw detection, and it is the expanded box that ends up in the ``.json``. If None, no containment check is performed. Default None.
    :param Optional[float] iou_threshold: If provided, an intersection-over-union threshold in ``[0, 1]`` used to treat overlapping detections as a single observation: with boxes ordered by confidence descending, any box whose IoU with an already-kept box exceeds this value is discarded as a duplicate of that same animal (greedy non-maximum suppression, keeping the higher-confidence box). IoU is computed with :func:`ultralytics.utils.metrics.box_iou` on the final boxes, i.e. after ``buffer_pct`` expansion and frame clipping, so the threshold applies to the geometry actually written to the ``.json`` and drawn by ``preview``. Suppressed boxes do not consume a ``max_detections`` slot. Useful when the text prompt returns several near-identical boxes for one animal. If None, no de-duplication is performed. Default None.
    :param Union[bool, float] save_negatives: Whether frames where no detection survives the filters are saved as background (negative) samples, i.e. an image paired with a ``.json`` holding an empty ``shapes`` list, which labelme opens as an un-annotated image ready to be labelled by hand. ``False`` (default) discards such frames entirely - no image, no json. ``True`` saves every negative frame encountered, which can far outnumber the positives at a low ``conf``. A float in ``(0, 1]`` caps negatives per video at that fraction of ``n_frames``, rounded to the nearest whole frame (e.g. ``0.1`` with ``n_frames=50`` saves at most 5 negatives per video). Negatives do not count toward ``n_frames`` and still count toward ``consecutive_miss_limit``. Default False.
    :param Optional[int] max_negative: Maximum number of background (negative) samples to save per video, as an absolute frame count rather than the fraction of ``n_frames`` taken by ``save_negatives``. Applied together with ``save_negatives``, so the effective per-video allowance is the smaller of the two - ``save_negatives=True`` with ``max_negative=5`` saves at most 5 negatives per video. Useful for keeping a project from being dominated by empty frames when the prompt misses often. If None, only ``save_negatives`` limits the count. Default None.
    :param bool recursive: If True and ``video_data`` is a directory, search it and all subdirectories for videos. Ignored if ``video_data`` is a file path or a list. Default False.
    :param Optional[int] seed: Random seed for reproducible frame sampling.
    :param bool visualize: If True, saves annotated images with bounding-box overlays to a ``visualizations`` subfolder inside ``save_dir``. Boxes are read back from the written labelme shapes, so these images verify what is on disk. Default False.
    :param Optional[int] min_frame_gap: Minimum number of frames between sampled frames. Enforces temporal diversity so samples are spread across the video rather than clustered. If ``None``, frames are sampled purely at random. Default ``None``.
    :param bool shuffle_videos: If True, randomize the order in which videos are processed. Default False.
    :param float io_timeout: Seconds to keep retrying file I/O (read/write) when the operation fails (e.g. temporary drive disconnect). Default 30.0.
    :param bool preview: If True, opens a ``cv2`` window displaying each evaluated frame at its original resolution with any detected bounding boxes drawn. Frames with no detection are labelled ``NO DETECTION``. Press ``q`` to abort. Useful for spotting false negatives. See ``visualize`` to save these same annotated frames to disk. Default False.
    :param Optional[Tuple[str, ...]] skip_substr: If provided, any video whose filename contains one of these substrings (case-insensitive) is skipped. Default None.
    :param Optional[Tuple[int, int]] video_size: If provided, a ``(height, width)`` tuple. Only videos matching this exact resolution are kept; all others are skipped. Default None.
    :param bool img_data: If True, embeds each image in its ``.json`` file as a JPEG-encoded base64 string. Note that this re-encodes the image, so the embedded copy is lossy relative to the ``.png`` written next to it. Default True.
    :param str labelme_version: Version number encoded in the json files. Default ``'5.3.1'``.
    :param Optional[Dict[str, Any]] labelme_config: If provided, a ``.labelmerc`` YAML config is written into ``save_dir`` and these keys are merged into it, overriding the defaults. labelme applies it only when launched with ``--config <save_dir>/.labelmerc``. The defaults written are ``labels`` (taken from ``names``, so the classes are pre-listed in the label dialog), ``with_image_data`` (matching ``img_data``, so hand-edits re-save the way the project was written) and ``shape_color: auto``. Per-shape colours cannot be used: labelme >= 4 ignores ``line_color``/``fill_color`` inside the json, so a colour map must go here as ``{'shape_color': 'manual', 'label_colors': {'animal': [255, 0, 0]}}``. Pass ``{}`` to write the defaults only. If None, no config file is written. Default None.
    :param Optional[Dict[Any, Any]] flags: Flags included in the json files. Default None, which writes an empty dict.
    :param bool verbose: If True, print progress updates. Default True.

    :raises SimBAGPUError: If no NVIDIA GPU is detected (via ``nvidia-smi``).
    :raises InvalidInputError: If ``max_size`` is smaller than ``min_size`` in either dimension.
    :raises SimBAPAckageVersionError: If ``ultralytics`` is not installed, or ``SAM3SemanticPredictor`` cannot be imported.

    :example:

    >>> runner = SAM3ToLabelmeBBox(video_data=r'/path/to/videos', sam_path=r'/path/to/sam3.pt', save_dir=r'/path/to/labelme_project', txt_prompt='mouse', n_frames=50)
    >>> runner.run()
    """

    def __init__(self,
                 video_data: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
                 sam_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 txt_prompt: str = 'mouse',
                 n_frames: int = 50,
                 names: Tuple[str, ...] = ('animal',),
                 conf: float = 0.25,
                 sam_imgsz: int = 644,
                 greyscale: bool = False,
                 clahe: Optional[Union[Tuple[int, int, int], bool]] = False,
                 buffer_pct: float = 0.0,
                 consecutive_miss_limit: int = 100,
                 max_detections: Optional[int] = None,
                 min_size: Optional[Tuple[int, int]] = None,
                 max_size: Optional[Tuple[int, int]] = None,
                 iou_threshold: Optional[float] = None,
                 containment_threshold: Optional[float] = None,
                 save_negatives: Union[bool, float] = False,
                 max_negative: Optional[int] = None,
                 recursive: bool = False,
                 seed: Optional[int] = None,
                 visualize: bool = False,
                 min_frame_gap: Optional[int] = None,
                 shuffle_videos: bool = False,
                 io_timeout: float = 30.0,
                 preview: bool = False,
                 skip_substr: Optional[Tuple[str, ...]] = None,
                 video_size: Optional[Tuple[int, int]] = None,
                 img_data: bool = True,
                 labelme_version: str = LABELME_VERSION,
                 flags: Optional[Dict[Any, Any]] = None,
                 labelme_config: Optional[Dict[str, Any]] = None,
                 verbose: bool = True):

        check_nvidea_gpu_available(raise_error=True)
        _ = get_pkg_version(pkg='ultralytics', raise_error=True)
        if SAM3SemanticPredictor is None:
            raise SimBAPAckageVersionError(msg='Could not import SAM3SemanticPredictor from ultralytics.models.sam. Install a compatible ultralytics build with SAM3 support.', source=self.__class__.__name__)

        check_instance(source=f'{self.__class__.__name__} video_data', instance=video_data, accepted_types=(str, os.PathLike, list))
        if isinstance(video_data, list):
            check_valid_lst(data=video_data, source=f'{self.__class__.__name__} video_data', valid_dtypes=(str, os.PathLike), min_len=1)
            for v in video_data:
                if os.path.isdir(v):
                    check_if_dir_exists(in_dir=v, source=f'{self.__class__.__name__} video_data')
                else:
                    check_file_exist_and_readable(file_path=v)
        elif os.path.isfile(video_data):
            check_file_exist_and_readable(file_path=video_data)
        else:
            check_if_dir_exists(in_dir=video_data, source=f'{self.__class__.__name__} video_data')
        check_file_exist_and_readable(file_path=sam_path)
        check_if_dir_exists(in_dir=os.path.dirname(save_dir), source=f'{self.__class__.__name__} save_dir')
        check_str(name=f'{self.__class__.__name__} txt_prompt', value=txt_prompt)
        check_int(name=f'{self.__class__.__name__} n_frames', value=n_frames, min_value=1)
        check_valid_tuple(x=names, source=f'{self.__class__.__name__} names', minimum_length=1, valid_dtypes=(str,))
        check_float(name=f'{self.__class__.__name__} conf', value=conf, min_value=0.0001, max_value=1.0)
        check_int(name=f'{self.__class__.__name__} imgsz', value=sam_imgsz, min_value=32)
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=img_data, source=f'{self.__class__.__name__} img_data')
        check_str(name=f'{self.__class__.__name__} labelme_version', value=labelme_version)
        check_float(name=f'{self.__class__.__name__} buffer_pct', value=buffer_pct, min_value=0.0, max_value=1.0)
        check_int(name=f'{self.__class__.__name__} consecutive_miss_limit', value=consecutive_miss_limit, min_value=1)
        check_valid_boolean(value=recursive, source=f'{self.__class__.__name__} recursive')
        check_valid_boolean(value=visualize, source=f'{self.__class__.__name__} visualize')
        check_valid_boolean(value=shuffle_videos, source=f'{self.__class__.__name__} shuffle_videos')
        if min_frame_gap is not None: check_int(name=f'{self.__class__.__name__} min_frame_gap', value=min_frame_gap, min_value=1)
        check_float(name=f'{self.__class__.__name__} io_timeout', value=io_timeout, min_value=0.0)
        if max_detections is not None: check_int(name=f'{self.__class__.__name__} max_detections', value=max_detections, min_value=1)
        if min_size is not None:
            check_valid_tuple(x=min_size, source=f'{self.__class__.__name__} min_size', accepted_lengths=(2,), valid_dtypes=(int,))
            check_int(name=f'{self.__class__.__name__} min_size height', value=min_size[0], min_value=1)
            check_int(name=f'{self.__class__.__name__} min_size width', value=min_size[1], min_value=1)
        if max_size is not None:
            check_valid_tuple(x=max_size, source=f'{self.__class__.__name__} max_size', accepted_lengths=(2,), valid_dtypes=(int,))
            check_int(name=f'{self.__class__.__name__} max_size height', value=max_size[0], min_value=1)
            check_int(name=f'{self.__class__.__name__} max_size width', value=max_size[1], min_value=1)
            if min_size is not None and (max_size[0] < min_size[0] or max_size[1] < min_size[1]):
                raise InvalidInputError(msg=f'max_size {max_size} cannot be smaller than min_size {min_size} in either dimension, no box could satisfy both.', source=self.__class__.__name__)
        if iou_threshold is not None: check_float(name=f'{self.__class__.__name__} iou_threshold', value=iou_threshold, min_value=0.0, max_value=1.0)
        if containment_threshold is not None: check_float(name=f'{self.__class__.__name__} containment_threshold', value=containment_threshold, min_value=0.0001, max_value=1.0)
        if (iou_threshold is not None or containment_threshold is not None) and (box_iou is None or torch is None):
            raise SimBAPAckageVersionError(msg='iou_threshold and containment_threshold require torch and ultralytics.utils.metrics.box_iou, which could not be imported. Pass iou_threshold=None and containment_threshold=None, or install a compatible torch / ultralytics build.', source=self.__class__.__name__)
        if seed is not None: check_int(name=f'{self.__class__.__name__} seed', value=seed)
        check_valid_boolean(value=preview, source=f'{self.__class__.__name__} preview')
        if skip_substr is not None:
            check_valid_tuple(x=skip_substr, source=f'{self.__class__.__name__} skip_substr', minimum_length=1, valid_dtypes=(str,))
        if video_size is not None:
            check_valid_tuple(x=video_size, source=f'{self.__class__.__name__} video_size', minimum_length=2, valid_dtypes=(int,))
        if flags is not None:
            check_instance(source=f'{self.__class__.__name__} flags', instance=flags, accepted_types=(dict,))
        if labelme_config is not None:
            check_instance(source=f'{self.__class__.__name__} labelme_config', instance=labelme_config, accepted_types=(dict,))
            if labelme_config.get('shape_color', None) == 'manual' and not labelme_config.get('label_colors', None):
                stdout_warning(msg="labelme_config sets shape_color to 'manual' without label_colors, so every shape falls back to default_shape_color. Pass label_colors={'<name>': [R, G, B]} to colour them per class.")
            unknown_colors = [k for k in (labelme_config.get('label_colors', None) or {}) if k not in names]
            if unknown_colors:
                stdout_warning(msg=f'labelme_config label_colors holds label(s) {unknown_colors} that are not in names {names}, they will never be drawn.')
        self.labelme_config = labelme_config
        check_instance(source=f'{self.__class__.__name__} save_negatives', instance=save_negatives, accepted_types=(bool, float, int))
        if not isinstance(save_negatives, bool):
            check_float(name=f'{self.__class__.__name__} save_negatives', value=save_negatives, min_value=0.0, max_value=1.0)
        if max_negative is not None:
            check_int(name=f'{self.__class__.__name__} max_negative', value=max_negative, min_value=1)
            if save_negatives is False:
                stdout_warning(msg=f'max_negative {max_negative} has no effect while save_negatives is False, no background samples are saved at all.')
        self.save_negatives, self.max_negative = save_negatives, max_negative
        if save_negatives is True:
            self.negative_limit = float('inf')
        elif save_negatives is False:
            self.negative_limit = 0
        else:
            self.negative_limit = int(round(float(save_negatives) * n_frames))
            if self.negative_limit == 0:
                stdout_warning(msg=f'save_negatives {save_negatives} of n_frames {n_frames} rounds to 0 negative frames per video, no background samples will be saved.')
        if max_negative is not None:
            self.negative_limit = min(self.negative_limit, max_negative)
        self.preview = preview
        self.min_size, self.max_size, self.iou_threshold, self.containment_threshold = min_size, max_size, iou_threshold, containment_threshold
        self.skip_substr = skip_substr
        self.video_size = video_size
        self.video_data, self.sam_path, self.save_dir, self.txt_prompt = video_data, sam_path, save_dir, txt_prompt
        self.n_frames, self.names, self.conf, self.imgsz = n_frames, names, conf, sam_imgsz
        self.greyscale, self.clahe, self.buffer_pct, self.consecutive_miss_limit, self.max_detections, self.seed, self.verbose, self.visualize, self.min_frame_gap, self.io_timeout = greyscale, clahe, buffer_pct, consecutive_miss_limit, max_detections, seed, verbose, visualize, min_frame_gap, io_timeout
        self.img_data, self.labelme_version = img_data, labelme_version
        self.flags = {} if flags is None else flags
        if isinstance(video_data, list):
            self.video_paths = {}
            for v in video_data:
                if os.path.isdir(v):
                    if recursive:
                        dir_videos = recursive_file_search(directory=v, extensions=VIDEO_EXTENSIONS, as_dict=True, raise_error=True)
                    else:
                        dir_videos = find_all_videos_in_directory(directory=v, as_dict=True, raise_error=True)
                    self.video_paths.update(dir_videos)
                else:
                    self.video_paths[get_fn_ext(filepath=v)[1]] = str(v)
        elif isinstance(video_data, (str, os.PathLike)) and os.path.isfile(video_data):
            self.video_paths = {get_fn_ext(filepath=video_data)[1]: str(video_data)}
        elif recursive:
            self.video_paths = recursive_file_search(directory=video_data, extensions=VIDEO_EXTENSIONS, as_dict=True, raise_error=True)
        else:
            self.video_paths = find_all_videos_in_directory(directory=video_data, as_dict=True, raise_error=True)
        if shuffle_videos:
            items = list(self.video_paths.items())
            random.shuffle(items)
            self.video_paths = dict(items)
        if self.skip_substr is not None:
            skip_lower = tuple(s.lower() for s in self.skip_substr)
            before_cnt = len(self.video_paths)
            self.video_paths = {k: v for k, v in self.video_paths.items() if not any(s in k.lower() for s in skip_lower)}
            if self.verbose and before_cnt != len(self.video_paths):
                stdout_information(msg=f'skip_substr filtered {before_cnt - len(self.video_paths)} of {before_cnt} videos (remaining: {len(self.video_paths)})')
        if self.video_size is not None:
            target_h, target_w = self.video_size
            before_cnt = len(self.video_paths)
            filtered = {}
            for k, v in self.video_paths.items():
                try:
                    meta = get_video_meta_data(video_path=v)
                    if int(meta['height']) == target_h and int(meta['width']) == target_w:
                        filtered[k] = v
                    elif self.verbose:
                        stdout_information(msg=f'video_size filter: skipping {k} ({int(meta["width"])}x{int(meta["height"])}), expected {target_w}x{target_h}')
                except Exception as e:
                    if self.verbose:
                        stdout_warning(msg=f'video_size filter: could not read {k} ({e}), skipping...')
            self.video_paths = filtered
            if len(self.video_paths) == 0:
                raise NoFilesFoundError(msg=f'No videos found with resolution {target_w}x{target_h}. {before_cnt} videos were checked.', source=self.__class__.__name__)

    def _write_labelme_json(self, path: str, shapes: List[Dict[str, Any]], img: np.ndarray, img_name: str, img_ext: str):
        """Write one labelme annotation file. ``imageData`` embeds a JPEG-encoded copy of the image when ``img_data`` is True."""
        out = {'version': self.labelme_version,
               'flags': self.flags,
               'shapes': shapes,
               'imagePath': f'{img_name}{img_ext}',
               'imageData': arr_to_b64(img) if self.img_data else None,
               'imageHeight': int(img.shape[0]),
               'imageWidth': int(img.shape[1])}
        with open(path, 'w') as f:
            json.dump(out, f)

    def _annotate_frame(self, img: np.ndarray, shapes: List[Dict[str, Any]], video_name: str, frame_idx: int, confs: Optional[List[float]] = None) -> np.ndarray:
        """Draw the labelme rectangles in ``shapes`` onto ``img`` at its original resolution. Shared by ``preview`` and ``visualize`` so both render identical annotations. Greyscale images are promoted to BGR so the overlays stay colored."""
        vis = img.copy() if img.ndim > 2 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_h = vis.shape[0]
        for shape_cnt, shape in enumerate(shapes):
            (x1, y1), (x2, y2) = shape['points']
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = shape['label'] if confs is None or shape_cnt >= len(confs) else f'{shape["label"]} {confs[shape_cnt]:.2f}'
            cv2.putText(vis, label, (int(x1), max(int(y1) - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if len(shapes) == 0:
            cv2.putText(vis, 'NO DETECTION', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(vis, f'{video_name} | frm {frame_idx}', (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return vis

    def _preview_frame(self, frame: np.ndarray, shapes: List[Dict[str, Any]], video_name: str, frame_idx: int, confs: Optional[List[float]] = None) -> bool:
        """Show a preview window with detections drawn at original resolution. Auto-advances. Returns False if user pressed 'q' to quit."""
        cv2.imshow('SAM3 Preview', self._annotate_frame(img=frame, shapes=shapes, video_name=video_name, frame_idx=frame_idx, confs=confs))
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            return False
        return True

    def _io_with_retry(self, func, *args, **kwargs):
        deadline = time.time() + self.io_timeout
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if time.time() >= deadline:
                    raise
                if self.verbose:
                    stdout_warning(msg=f'I/O error ({e}), retrying for {max(0, deadline - time.time()):.0f}s ...')
                time.sleep(1)

    def _write_labelme_config(self) -> str:
        """Write a ``.labelmerc`` into ``save_dir`` holding the class list and any caller overrides. Returns its path."""
        config = {'labels': list(self.names), 'with_image_data': bool(self.img_data), 'shape_color': 'auto'}
        config.update(self.labelme_config)
        config_path = os.path.join(self.save_dir, '.labelmerc')
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        return config_path

    def _write_sample(self, video_path: str, video_name: str, video_progress: str, frame_idx: int, shapes: List[Dict[str, Any]], confs: Optional[List[float]] = None) -> Optional[int]:
        """Write one sample - image, labelme json and optional ``visualize`` overlay - into the flat output directory. An empty ``shapes`` list writes an un-annotated json, i.e. a background (negative) sample. Returns the number of shapes written, or None if the frame could not be read after retries."""
        try:
            img_out = self._io_with_retry(read_frm_of_video, video_path=video_path, frame_index=frame_idx, greyscale=self.greyscale, clahe=self.clahe)
        except Exception as e:
            if self.verbose:
                stdout_warning(msg=f'Video {video_progress} ({video_name}), frame idx {frame_idx}: could not read frame for output after retries ({e}), skipping video...')
            return None
        sample_name = f'{video_name}_frm{frame_idx:08d}'
        self._io_with_retry(cv2.imwrite, os.path.join(self.save_dir, f'{sample_name}.png'), img_out)
        self._io_with_retry(self._write_labelme_json, os.path.join(self.save_dir, f'{sample_name}.json'), shapes, img_out, sample_name, '.png')
        if self.visualize:
            vis_img = self._annotate_frame(img=img_out, shapes=shapes, video_name=video_name, frame_idx=frame_idx, confs=confs)
            self._io_with_retry(cv2.imwrite, os.path.join(self.vis_dir, f'{sample_name}.png'), vis_img)
        return len(shapes)

    def run(self):
        timer = SimbaTimer(start=True)
        if self.seed is not None: random.seed(self.seed)
        create_directory(paths=[self.save_dir], overwrite=False)

        if self.labelme_config is not None:
            config_path = self._write_labelme_config()
            if self.verbose:
                stdout_information(msg=f'labelme config written to {config_path} - open the project with: labelme --config "{config_path}" "{self.save_dir}"', source=self.__class__.__name__)

        if self.verbose and (self.iou_threshold is not None or self.containment_threshold is not None):
            stdout_information(msg=f'Overlap de-duplication active - iou_threshold: {self.iou_threshold}, containment_threshold: {self.containment_threshold}')

        overrides = dict(conf=self.conf, task='segment', mode='predict', imgsz=self.imgsz, model=str(self.sam_path), half=True, save=False, verbose=False)
        predictor = SAM3SemanticPredictor(overrides=overrides)

        self.vis_dir = os.path.join(self.save_dir, 'visualizations')
        if self.visualize:
            create_directory(paths=[self.vis_dir], overwrite=False)

        video_cnt, total_videos = 0, len(self.video_paths)
        sample_cnt, shape_cnt, negative_cnt = 0, 0, 0
        for video_name, video_path in self.video_paths.items():
            video_cnt += 1
            try:
                video_meta = self._io_with_retry(get_video_meta_data, video_path=video_path)
            except Exception as e:
                if self.verbose:
                    stdout_warning(msg=f'Video {video_cnt}/{total_videos} ({video_name}): could not read video ({e}), skipping...')
                continue
            total_frames = int(video_meta['frame_count'])
            img_w, img_h = int(video_meta['width']), int(video_meta['height'])
            candidate_indices = list(range(total_frames))
            random.shuffle(candidate_indices)

            if self.verbose:
                stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}): targeting {self.n_frames} valid frames from {total_frames} total...')
            valid_cnt, consecutive_misses, video_neg_cnt = 0, 0, 0
            used_indices = []
            video_progress = f'{video_cnt}/{total_videos}'
            for frame_idx in candidate_indices:
                if valid_cnt >= self.n_frames:
                    break
                if consecutive_misses >= self.consecutive_miss_limit:
                    if self.verbose:
                        stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}): {self.consecutive_miss_limit} consecutive misses, skipping to next video...')
                    break
                if self.min_frame_gap is not None and any(abs(frame_idx - u) < self.min_frame_gap for u in used_indices):
                    continue
                try:
                    frame = self._io_with_retry(read_frm_of_video, video_path=video_path, frame_index=frame_idx)
                except Exception as e:
                    if self.verbose:
                        stdout_warning(msg=f'Video {video_cnt}/{total_videos} ({video_name}), frame idx {frame_idx}: could not read frame after retries ({e}), skipping video...')
                    break
                if frame is None:
                    consecutive_misses += 1
                    continue

                predictor.set_image(frame)
                results = predictor(text=[self.txt_prompt])
                r = results[0] if isinstance(results, list) and len(results) > 0 else results

                boxes = self._retained_boxes(result=r, img_w=img_w, img_h=img_h)
                shapes = self._boxes_to_labelme_shapes(boxes=boxes)
                confs = [b[-1] for b in boxes]

                if len(shapes) == 0:
                    consecutive_misses += 1
                    if self.preview:
                        if not self._preview_frame(frame=frame, shapes=[], video_name=video_name, frame_idx=frame_idx):
                            break
                    miss_reason = 'no detection found' if (r.boxes is None or len(r.boxes) == 0) else 'no detection passed the conf/size/overlap filters'
                    if video_neg_cnt < self.negative_limit:
                        written = self._write_sample(video_path=video_path, video_name=video_name, video_progress=video_progress, frame_idx=frame_idx, shapes=[])
                        if written is None:
                            break
                        negative_cnt += 1
                        sample_cnt += 1
                        video_neg_cnt += 1
                        used_indices.append(frame_idx)
                        if self.verbose:
                            limit_str = 'unlimited' if self.negative_limit == float('inf') else self.negative_limit
                            stdout_information(msg=f'Video {video_progress} ({video_name}), frame idx {frame_idx}: {miss_reason}, saved as background sample ({video_neg_cnt}/{limit_str} for this video)')
                    elif self.verbose:
                        stdout_information(msg=f'Video {video_progress} ({video_name}), frame idx {frame_idx}: {miss_reason} (consecutive misses: {consecutive_misses}/{self.consecutive_miss_limit})')
                    continue

                if self.preview:
                    if not self._preview_frame(frame=frame, shapes=shapes, video_name=video_name, frame_idx=frame_idx, confs=confs):
                        break
                consecutive_misses = 0
                written = self._write_sample(video_path=video_path, video_name=video_name, video_progress=video_progress, frame_idx=frame_idx, shapes=shapes, confs=confs)
                if written is None:
                    break
                sample_cnt += 1
                shape_cnt += written

                used_indices.append(frame_idx)
                valid_cnt += 1
                if self.verbose:
                    stdout_information(msg=f'Video {video_progress} ({video_name}), frame {valid_cnt}/{self.n_frames} collected (frame idx {frame_idx}, total samples: {sample_cnt})')
            if self.verbose:
                stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}): collected {valid_cnt}/{self.n_frames} valid labeled frames')

        if self.preview:
            cv2.destroyAllWindows()
        if sample_cnt == 0:
            raise NoFilesFoundError(msg='No boxes detected in any sampled frame. No project created.', source=self.__class__.__name__)

        timer.stop_timer()
        negative_msg = f' {negative_cnt} of the images are background samples with no annotations.' if negative_cnt > 0 else ''
        stdout_success(msg=f'Labelme bbox project created at {self.save_dir}. {sample_cnt} image(s), {shape_cnt} annotation(s).{negative_msg}', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)

    def _suppress_duplicates(self, boxes: List[Tuple[int, float, float, float, float, float]]) -> List[Tuple[int, float, float, float, float, float]]:
        """Greedy non-maximum suppression of ``boxes`` (final ``buffer_pct``-expanded, frame-clipped coordinates, sorted by confidence descending): a box is dropped as part of the same observation as an already-kept box if their IoU exceeds ``iou_threshold``, or if their containment - the intersection over the area of the smaller of the two, which stays high for a nested box where IoU is low - reaches ``containment_threshold``. Pairwise IoU comes from :func:`ultralytics.utils.metrics.box_iou`."""
        xyxy = torch.tensor([b[1:5] for b in boxes], dtype=torch.float32)
        iou = box_iou(xyxy, xyxy).numpy() if self.iou_threshold is not None else None
        containment = None
        if self.containment_threshold is not None:
            intersection = (torch.min(xyxy[:, None, 2:], xyxy[None, :, 2:]) - torch.max(xyxy[:, None, :2], xyxy[None, :, :2])).clamp_(0).prod(2)
            area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            containment = (intersection / torch.min(area[:, None], area[None, :]).clamp_(min=1e-7)).numpy()
        keep = []
        for box_idx in range(len(boxes)):
            duplicate = False
            for kept_idx in keep:
                if iou is not None and iou[box_idx, kept_idx] > self.iou_threshold:
                    duplicate = True
                elif containment is not None and containment[box_idx, kept_idx] >= self.containment_threshold:
                    duplicate = True
                if duplicate:
                    break
            if not duplicate:
                keep.append(box_idx)
        if self.verbose and len(keep) < len(boxes):
            stdout_information(msg=f'Overlap filter: {len(boxes) - len(keep)} of {len(boxes)} detections dropped as duplicates of a higher-confidence box (iou_threshold: {self.iou_threshold}, containment_threshold: {self.containment_threshold})')
        return [boxes[i] for i in keep]

    def _retained_boxes(self, result, img_w: int, img_h: int) -> List[Tuple[int, float, float, float, float, float]]:
        """The boxes surviving ``conf``, ``min_size``, ``max_size``, ``iou_threshold``, ``containment_threshold`` and ``max_detections``, as ``(class_id, x1, y1, x2, y2, confidence)`` sorted by confidence descending. ``conf`` and the size bounds are applied to the raw SAM3 detection; boxes are then expanded by ``buffer_pct`` and clipped to the frame, and the overlap thresholds and ``max_detections`` cap are applied to that final geometry - i.e. to the boxes actually written as labelme shapes."""
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return []
        box_indices = list(range(len(result.boxes)))
        box_indices.sort(key=lambda i: float(result.boxes.conf[i].cpu()), reverse=True)
        candidates = []
        for box_idx in box_indices:
            conf = float(result.boxes.conf[box_idx].cpu())
            if conf < self.conf:
                continue
            if self.max_detections is not None and self.iou_threshold is None and self.containment_threshold is None and len(candidates) >= self.max_detections:
                break

            cls_id = 0
            det_cls = int(result.boxes.cls[box_idx].cpu())
            if det_cls < len(self.names):
                cls_id = det_cls

            xyxy = result.boxes.xyxy[box_idx].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            if self.min_size is not None and ((y2 - y1) < self.min_size[0] or (x2 - x1) < self.min_size[1]):
                continue
            if self.max_size is not None and ((y2 - y1) > self.max_size[0] or (x2 - x1) > self.max_size[1]):
                continue
            candidates.append((cls_id, x1, y1, x2, y2, conf))

        boxes = []
        for cls_id, x1, y1, x2, y2, conf in candidates:
            if self.buffer_pct > 0:
                bw, bh = x2 - x1, y2 - y1
                x1 -= bw * self.buffer_pct
                y1 -= bh * self.buffer_pct
                x2 += bw * self.buffer_pct
                y2 += bh * self.buffer_pct
            x1 = max(0.0, min(float(img_w), x1))
            y1 = max(0.0, min(float(img_h), y1))
            x2 = max(0.0, min(float(img_w), x2))
            y2 = max(0.0, min(float(img_h), y2))
            boxes.append((cls_id, x1, y1, x2, y2, conf))

        if (self.iou_threshold is not None or self.containment_threshold is not None) and len(boxes) > 1:
            boxes = self._suppress_duplicates(boxes=boxes)
        if self.max_detections is not None:
            boxes = boxes[:self.max_detections]

        return boxes

    def _boxes_to_labelme_shapes(self, boxes: List[Tuple[int, float, float, float, float, float]]) -> List[Dict[str, Any]]:
        """Convert retained ``(class_id, x1, y1, x2, y2, conf)`` boxes into labelme ``rectangle`` shapes holding absolute pixel corner points."""
        shapes = []
        for cls_id, x1, y1, x2, y2, _ in boxes:
            shapes.append({'label': self.names[cls_id],
                           'points': [[x1, y1], [x2, y2]],
                           'group_id': None,
                           'description': "",
                           'shape_type': 'rectangle',
                           'flags': {}})
        return shapes


# runner = SAM3ToLabelmeBBox(video_data=r"G:\netholabs\6.01.005", sam_path=r'D:\sam3\sam3.pt', save_dir=r"G:\netholabs\labelme_project", txt_prompt='mouse', n_frames=50)
# runner.run()

# if __name__ == "__main__":
#     runner = SAM3ToLabelmeBBox(video_data=r'I:\netholabs\yolo_cage_21_22',
#                             sam_path=r'D:\sam3\sam3.pt',
#                             save_dir=r'I:\netholabs\labelme_cage21_22',
#                             txt_prompt='black mouse body',
#                             n_frames=10,
#                             verbose=True,
#                             conf=0.025,
#                             max_detections=1,
#                             buffer_pct=0.15,
#                             recursive=True,
#                             iou_threshold=0.5,
#                             containment_threshold=0.5,
#                             max_negative=3,
#                             save_negatives=True,
#                             consecutive_miss_limit=25,
#                             skip_substr=('mosaic',),
#                             min_size=(5, 5),
#                             shuffle_videos=True,
#                             visualize=False,
#                             preview=True)
#     runner.run()