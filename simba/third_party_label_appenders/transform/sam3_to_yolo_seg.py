import os
import random
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from typing import Literal
except:
    from typing_extensions import Literal

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch
    from ultralytics.models.sam import SAM3Predictor, SAM3SemanticPredictor
except:
    SAM3Predictor, SAM3SemanticPredictor = None, None

from simba.mixins.plotting_mixin import PlottingMixin
from simba.third_party_label_appenders.converters import create_yolo_yaml
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_instance, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_boolean, check_valid_lst,
                                check_valid_tuple)
from simba.utils.data import create_color_palette, resample_geometry_vertices
from simba.utils.errors import (InvalidInputError, NoFilesFoundError,
                                SimBAPAckageVersionError)
from simba.utils.printing import (SimbaTimer, stdout_information,
                                  stdout_success, stdout_warning)
from simba.utils.read_write import (create_directory,
                                    find_all_videos_in_directory,
                                    get_pkg_version, get_video_meta_data,
                                    read_frm_of_video, recursive_file_search)
from simba.utils.yolo import (create_yolo_sample_visualizations,
                              load_yolo_model)


class SAM3ToYoloSeg:
    """
    Sample N random frames from each video in a directory, run SAM3 with a text prompt, and write the resulting masks as a YOLO segmentation project.

    For each sampled frame, every SAM3 mask above ``conf`` has its outline traced as a polygon, optionally
    de-duplicated by ``iou_threshold`` and resampled to ``vertice_cnt`` vertices, and written as
    ``class x1 y1 x2 y2 ... xn yn`` normalized to ``[0, 1]`` - one line per mask.

    If ``detector_path`` is given, a trained YOLO **detection** model grounds the objects instead of the text prompt: the detector decides
    *what* each object is and roughly *where*, and SAM3 only traces the outline inside each detector box (box prompt, no text). Class labels and
    ``map.yaml`` then come from the detector's ``names`` (``txt_prompt`` and ``names`` are ignored), so classes SAM3 cannot recognise from a word
    are still segmented. Every object left unlabelled in a training image is learned as background, so a doubtful object skips the frame rather
    than being dropped: see ``certain_conf``, ``uncertain_drop_classes`` and ``skip_frame_on_bad_mask``.

    .. image:: _static/img/sam3_to_yolo_seg.webp
       :alt: Text-prompted SAM3 segmentation to YOLO polygon labels
       :width: 700
       :align: center

    .. note::
       To fit a YOLO segmentation model, see :class:`~simba.model.yolo_fit.FitYolo`.
       For YOLO segmentation inference, see :class:`~simba.model.yolo_seg_inference.YOLOSegmentationInference`.

    .. seealso::

       * :class:`~simba.third_party_label_appenders.transform.merge_yolo_projects.MergeYoloProjects` — merge several ``map.yaml`` projects (same classes and task) into one dataset.
       * :class:`~simba.third_party_label_appenders.transform.sam3_to_yolo_bbox.SAM3ToYoloBBox` — SAM3 to YOLO **bounding-box** labels from the same predictor stack.
       * :class:`~simba.third_party_label_appenders.transform.sam3_to_labelme_bbox.SAM3ToLabelmeBBox` — SAM3 bounding boxes written as a labelme project for hand-correction.

    :raises SimBAGPUError: If no NVIDIA GPU is detected (via ``nvidia-smi``).
    :raises SimBAPAckageVersionError: If ``ultralytics`` is not installed, or ``SAM3SemanticPredictor`` cannot be imported.
    :raises InvalidInputError: If ``names`` has neither length 1 nor one entry per ``txt_prompt``.

    :param Union[str, os.PathLike] video_dir: Directory containing input videos (combine with ``recursive=True`` to also search subdirectories).
    :param Union[str, os.PathLike] sam_path: Path to SAM3 model weights (e.g. sam3.pt).
    :param Union[str, os.PathLike] save_dir: Root output directory for the YOLO project.
    :param Union[str, Tuple[str, ...], List[str]] txt_prompt: Text prompt(s) for SAM3 (e.g. ``"mouse"``, or ``("mouse", "rat")``). Multiple prompts are run together in one SAM3 pass and each is treated as its own class, in order: masks found by prompt ``i`` are written with class id ``i``. If ``names`` holds a single class, all prompts map to that class (useful for synonyms); otherwise ``names`` must have one entry per prompt.
    :param int n_frames: Number of random frames to sample from each video.
    :param Tuple[str, ...] names: Class names in index order. Must have length 1 or one entry per ``txt_prompt``. Default ``('animal',)``.
    :param float train_val_split: Fraction allocated to training (0.1-0.9). Default 0.7.
    :param float conf: SAM3 confidence threshold. With ``detector_path``, the minimum SAM3 mask quality score; a box whose mask scores lower fails the mask quality check. Default 0.5.
    :param int sam_imgsz: Image size for SAM3 inference. Default 640.
    :param bool greyscale: If True, save extracted frames in greyscale. Default False.
    :param Optional[Union[Tuple[int, int, int], bool]] clahe: If True, applies CLAHE with default params. If tuple of (clip_limit, tile_x, tile_y), applies CLAHE with those params. Default False.
    :param Optional[int] max_detections: Maximum number of masks to keep per frame (sorted by confidence descending, after ``iou_threshold`` de-duplication, so suppressed masks do not consume a slot). If None, all masks above ``conf`` are kept. Default None.
    :param bool recursive: If True, search ``video_dir`` and all its subdirectories for videos. Default False.
    :param bool shuffle_videos: If True, randomize the order in which videos are processed. Default False.
    :param Optional[int] vertice_cnt: If not None, resample each mask polygon to this many vertices. Default 40.
    :param Optional[float] iou_threshold: If provided, an intersection-over-union threshold in ``[0, 1]`` used to treat overlapping masks as a single observation: with masks ordered by confidence descending, any mask whose IoU with an already-kept mask exceeds this value is discarded as a duplicate of that same animal (greedy non-maximum suppression, keeping the higher-confidence mask). IoU is computed between the **rasterized mask polygons**, not their bounding boxes, so two animals that cross - and therefore have heavily overlapping boxes but little shared area - are both retained. It is measured on the traced outline, before ``vertice_cnt`` resampling, so the resampling cannot alter the overlap. Useful when the text prompt returns several near-identical masks for one animal. If None, no de-duplication is performed and masks are written in detector order. Default None.
    :param Optional[Union[str, os.PathLike]] detector_path: Optional path to trained YOLO detection weights. If given, detector boxes ground SAM3 instead of ``txt_prompt`` (see above). Default None.
    :param float det_conf: Detector confidence threshold. Only used with ``detector_path``. Default 0.25.
    :param Optional[float] certain_conf: Detector boxes with ``det_conf <= conf < certain_conf`` count as uncertain and skip the frame (unless their class is in ``uncertain_drop_classes``). Only used with ``detector_path``. If None, all boxes ``>= det_conf`` are kept. Default None.
    :param Optional[List[str]] uncertain_drop_classes: Detector classes whose uncertain boxes are dropped instead of skipping the frame (e.g. ``['food_pellet', 'nest']`` for small, numerous or fuzzy objects that would otherwise reject most frames). Only used with ``detector_path`` and ``certain_conf``. Default None.
    :param Optional[float] min_box_fill: Masks whose area is less than this fraction of their detector box fail the mask quality check (SAM3 traced only part of the object). Only used with ``detector_path``. Default None.
    :param bool skip_frame_on_bad_mask: If True, a frame is skipped when any detector box fails the mask quality check (``conf`` / ``min_box_fill``); if False, only that mask is dropped. Only used with ``detector_path``. Default True.
    :param Optional[float] cross_class_iou_threshold: Mask IoU above which two masks of **different** classes are treated as one object detected twice (e.g. one mouse detected as both ``mouse`` and ``house``); the lower detector confidence is dropped. Kept high so genuine partial overlaps (a nest under a house, a mouse inside a dome) survive. Only used with ``detector_path``; without it, ``iou_threshold`` applies to every pair. Default 0.85.
    :param Optional[int] max_skipped_frames: If not None, stop sampling a video once this many evaluated frames produced no label (no detection, or skipped by the detector-mode checks) and move to the next video. Guards against scanning every frame of a video whose objects never pass (e.g. a static object whose mask always fails). Default None.
    :param Optional[float] simplify_epsilon: If not None, simplify each polygon with Douglas-Peucker (:func:`cv2.approxPolyDP`), allowing a deviation of this fraction of the polygon's perimeter. Removes the wiggles SAM3 traces along straight edges (e.g. ``0.01`` turns a bottle into a clean outline of ~10 vertices); values above ~0.02 start cutting off real shape. Applied before ``vertice_cnt``; set ``vertice_cnt=None`` to keep only the simplified vertices. Default None.
    :param Optional[List[str]] amodal_classes: Detector classes of **static** objects (e.g. ``['house', 'tunnel', 'nest', 'water_bottle', 'pellet_feeder', 'wood']``) whose masks are completed behind occluders (amodal masks). Before sampling a video, a clean plate of it is built (per-pixel median of ``amodal_n_frames`` frames, with ``plate_mask_classes`` objects masked out) and segmented once into full-extent reference masks; in a sampled frame where the object's mask is a clipped version of its reference and the missing part lies under another object (e.g. a mouse walking in front), the label is the reference plus the visible mask. Only used with ``detector_path``. Default None.
    :param int amodal_n_frames: Number of evenly spaced frames per video whose median forms the clean plate for ``amodal_classes``. Default 60.
    :param Optional[List[str]] plate_mask_classes: Detector classes of moving objects (e.g. ``['mouse']``) left out of the clean plate's median, so that an animal resting in one place for much of the video does not remain on the plate. If None, a plain median is used. Only used with ``amodal_classes``. Default None.
    :param Optional[int] seed: Random seed for reproducible frame sampling.
    :param bool visualize: If True, saves annotated images with segmentation polygon overlays to a ``visualizations`` subfolder inside ``save_dir``. Useful for verifying SAM3 annotation quality. Default False.
    :param float io_timeout: Seconds to keep retrying file I/O (read/write) when the operation fails (e.g. temporary drive disconnect). Default 30.0.
    :param bool preview: If True, opens a ``cv2`` window displaying each evaluated frame at its original resolution with any detected segmentation polygons drawn, each class in its own color (same palette as ``visualize``) with a color-to-class legend and per-class detection counts in the top-left corner. Frames with no detection are labelled ``NO DETECTION``. Press ``q`` to abort. Useful for spotting false negatives. Default False.
    :param bool verbose: If True, print progress updates. Default True.

    :example:

    >>> runner = SAM3ToYoloSeg(video_dir=r'/path/to/videos', sam_path=r'/path/to/sam3.pt', save_dir=r'/path/to/yolo_project', txt_prompt='mouse', n_frames=50)
    >>> runner.run()

    >>> # Detector-grounded: classes from the detector, SAM3 traces each detector box
    >>> runner = SAM3ToYoloSeg(video_dir=r'/path/to/videos', sam_path=r'/path/to/sam3.pt', save_dir=r'/path/to/yolo_project', detector_path=r'/path/to/yolo26s_cage_objects_v1.pt', n_frames=20, certain_conf=0.35, uncertain_drop_classes=['food_pellet', 'nest'], min_box_fill=0.2, iou_threshold=0.5)
    >>> runner.run()
    """

    def __init__(self,
                 video_dir: Union[str, os.PathLike],
                 sam_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 txt_prompt: Union[str, Tuple[str, ...], List[str]] = 'mouse',
                 n_frames: int = 50,
                 names: Tuple[str, ...] = ('animal',),
                 train_val_split: float = 0.7,
                 conf: float = 0.50,
                 sam_imgsz: int = 644,
                 greyscale: bool = False,
                 clahe: Optional[Union[Tuple[int, int, int], bool]] = False,
                 vertice_cnt: Optional[int] = 40,
                 max_detections: Optional[int] = None,
                 recursive: bool = False,
                 shuffle_videos: bool = False,
                 iou_threshold: Optional[float] = None,
                 detector_path: Optional[Union[str, os.PathLike]] = None,
                 det_conf: float = 0.25,
                 certain_conf: Optional[float] = None,
                 uncertain_drop_classes: Optional[List[str]] = None,
                 min_box_fill: Optional[float] = None,
                 skip_frame_on_bad_mask: bool = True,
                 cross_class_iou_threshold: Optional[float] = 0.85,
                 max_skipped_frames: Optional[int] = None,
                 simplify_epsilon: Optional[float] = None,
                 amodal_classes: Optional[List[str]] = None,
                 amodal_n_frames: int = 60,
                 plate_mask_classes: Optional[List[str]] = None,
                 seed: Optional[int] = None,
                 visualize: bool = False,
                 io_timeout: float = 30.0,
                 preview: bool = False,
                 verbose: bool = True):

        check_nvidea_gpu_available(raise_error=True)
        _ = get_pkg_version(pkg="ultralytics", raise_error=True)
        if SAM3SemanticPredictor is None:
            raise SimBAPAckageVersionError(msg="Could not import SAM3SemanticPredictor from ultralytics.models.sam. Install a compatible ultralytics build with SAM3 support.", source=self.__class__.__name__,)
        check_if_dir_exists(in_dir=video_dir, source=f'{self.__class__.__name__} video_dir')
        check_file_exist_and_readable(file_path=sam_path)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        check_instance(source=f'{self.__class__.__name__} txt_prompt', instance=txt_prompt, accepted_types=(str, tuple, list))
        if isinstance(txt_prompt, str):
            check_str(name=f'{self.__class__.__name__} txt_prompt', value=txt_prompt)
            txt_prompt = [txt_prompt]
        else:
            check_valid_lst(data=list(txt_prompt), source=f'{self.__class__.__name__} txt_prompt', valid_dtypes=(str,), min_len=1)
            for prompt in txt_prompt: check_str(name=f'{self.__class__.__name__} txt_prompt', value=prompt)
            txt_prompt = list(txt_prompt)
        check_int(name=f'{self.__class__.__name__} n_frames', value=n_frames, min_value=1)
        check_valid_tuple(x=names, source=f'{self.__class__.__name__} names', minimum_length=1, valid_dtypes=(str,))
        if len(names) != 1 and len(names) != len(txt_prompt):
            raise InvalidInputError(msg=f'names {names} must have length 1 (all prompts map to one class) or one entry per txt_prompt {txt_prompt}, got {len(names)} names for {len(txt_prompt)} prompts.', source=self.__class__.__name__)
        check_float(name=f'{self.__class__.__name__} train_val_split', value=train_val_split, min_value=0.1, max_value=0.9)
        check_float(name=f'{self.__class__.__name__} conf', value=conf, min_value=0.01, max_value=1.0)
        check_int(name=f'{self.__class__.__name__} imgsz', value=sam_imgsz, min_value=32)
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        if vertice_cnt is not None: check_int(name=f'{self.__class__.__name__} vertice_cnt', value=vertice_cnt, min_value=3)
        if max_detections is not None: check_int(name=f'{self.__class__.__name__} max_detections', value=max_detections, min_value=1)
        check_valid_boolean(value=recursive, source=f'{self.__class__.__name__} recursive')
        check_valid_boolean(value=shuffle_videos, source=f'{self.__class__.__name__} shuffle_videos')
        if iou_threshold is not None: check_float(name=f'{self.__class__.__name__} iou_threshold', value=iou_threshold, min_value=0.0, max_value=1.0)
        if max_skipped_frames is not None: check_int(name=f'{self.__class__.__name__} max_skipped_frames', value=max_skipped_frames, min_value=1)
        if simplify_epsilon is not None: check_float(name=f'{self.__class__.__name__} simplify_epsilon', value=simplify_epsilon, min_value=0.0, max_value=0.5)
        if seed is not None: check_int(name=f'{self.__class__.__name__} seed', value=seed)
        check_valid_boolean(value=visualize, source=f'{self.__class__.__name__} visualize')
        check_float(name=f'{self.__class__.__name__} io_timeout', value=io_timeout, min_value=0.0)
        check_valid_boolean(value=preview, source=f'{self.__class__.__name__} preview')
        self.detector = None
        if detector_path is not None:
            if SAM3Predictor is None:
                raise SimBAPAckageVersionError(msg="Could not import SAM3Predictor from ultralytics.models.sam. Install a compatible ultralytics build with SAM3 support.", source=self.__class__.__name__)
            check_file_exist_and_readable(file_path=detector_path)
            check_float(name=f'{self.__class__.__name__} det_conf', value=det_conf, min_value=0.01, max_value=1.0)
            if certain_conf is not None: check_float(name=f'{self.__class__.__name__} certain_conf', value=certain_conf, min_value=det_conf, max_value=1.0)
            if min_box_fill is not None: check_float(name=f'{self.__class__.__name__} min_box_fill', value=min_box_fill, min_value=0.0, max_value=1.0)
            check_valid_boolean(value=skip_frame_on_bad_mask, source=f'{self.__class__.__name__} skip_frame_on_bad_mask')
            if cross_class_iou_threshold is not None: check_float(name=f'{self.__class__.__name__} cross_class_iou_threshold', value=cross_class_iou_threshold, min_value=0.0, max_value=1.0)
            self.detector = load_yolo_model(weights_path=detector_path, verbose=verbose)
            names = tuple(self.detector.names[k] for k in sorted(self.detector.names.keys()))
            if uncertain_drop_classes is not None: check_valid_lst(data=uncertain_drop_classes, source=f'{self.__class__.__name__} uncertain_drop_classes', valid_dtypes=(str,), valid_values=list(names), min_len=1)
            if amodal_classes is not None:
                check_valid_lst(data=amodal_classes, source=f'{self.__class__.__name__} amodal_classes', valid_dtypes=(str,), valid_values=list(names), min_len=1)
                check_int(name=f'{self.__class__.__name__} amodal_n_frames', value=amodal_n_frames, min_value=5)
                if plate_mask_classes is not None: check_valid_lst(data=plate_mask_classes, source=f'{self.__class__.__name__} plate_mask_classes', valid_dtypes=(str,), valid_values=list(names), min_len=1)
        elif amodal_classes is not None:
            raise InvalidInputError(msg='amodal_classes requires detector_path.', source=self.__class__.__name__)
        self.preview = preview
        self.video_dir, self.sam_path, self.save_dir, self.txt_prompt = video_dir, sam_path, save_dir, txt_prompt
        self.n_frames, self.names, self.train_val_split, self.conf, self.imgsz = n_frames, names, train_val_split, conf, sam_imgsz
        self.greyscale, self.clahe, self.vertice_cnt, self.seed, self.verbose, self.visualize, self.io_timeout = greyscale, clahe, vertice_cnt, seed, verbose, visualize, io_timeout
        self.iou_threshold, self.max_detections, self.max_skipped_frames, self.simplify_epsilon = iou_threshold, max_detections, max_skipped_frames, simplify_epsilon
        # Without a detector, iou_threshold applies to every pair of masks, whatever their class.
        self.cross_class_iou_threshold = cross_class_iou_threshold if self.detector is not None else iou_threshold
        self.det_conf, self.certain_conf, self.min_box_fill, self.skip_frame_on_bad_mask = det_conf, certain_conf, min_box_fill, skip_frame_on_bad_mask
        self.uncertain_drop_ids = {names.index(n) for n in uncertain_drop_classes} if (self.detector is not None and uncertain_drop_classes is not None) else set()
        self.amodal_ids = {names.index(n) for n in amodal_classes} if amodal_classes is not None else set()
        self.plate_mask_ids = {names.index(n) for n in plate_mask_classes} if (amodal_classes is not None and plate_mask_classes is not None) else set()
        self.amodal_n_frames, self.amodal_refs, self.amodal_cnt, self.amodal_plate = amodal_n_frames, [], 0, None
        self.skip_reason = None
        self.name_map = {name: idx for idx, name in enumerate(names)}
        self.preview_layouts = {}
        # One distinct color per class: tab10 / tab20 (sampled at exactly their own count, so no repeats), evenly spaced hsv hues beyond 20.
        palette = 'tab10' if len(names) <= 10 else 'tab20' if len(names) <= 20 else 'hsv'
        palette_cnt = {'tab10': 10, 'tab20': 20}.get(palette, len(names))
        self.class_colors = [tuple(int(c) for c in clr) for clr in create_color_palette(pallete_name=palette, increments=palette_cnt - 1)][:max(len(names), 1)]
        if recursive:
            self.video_paths = recursive_file_search(directory=video_dir, extensions=[".avi", ".mp4", ".mov", ".flv", ".m4v", ".webm", ".h264"], as_dict=True, raise_error=True)
        else:
            self.video_paths = find_all_videos_in_directory(directory=video_dir, as_dict=True, raise_error=True)
        if shuffle_videos:
            items = list(self.video_paths.items())
            random.shuffle(items)
            self.video_paths = dict(items)

    @staticmethod
    def _write_label(path: str, content: str):
        with open(path, 'w') as f:
            f.write(content)

    def _preview_frame(self, frame: np.ndarray, label_str: str, video_name: str, frame_idx: int) -> bool:
        """Show a preview window with the YOLO polygons in ``label_str`` drawn at original resolution, one color per class, plus a color-to-class legend listing each class with its detection count in the frame. Auto-advances. Returns False if user pressed 'q' to quit."""
        vis = frame.copy() if frame.ndim > 2 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        img_h, img_w = vis.shape[:2]
        lines = [l for l in label_str.strip().split('\n') if len(l.split()) >= 7] if label_str else []
        cls_cnts = {cls_id: 0 for cls_id in range(len(self.names))}
        line_thickness = max(2, int(min(img_w, img_h) / 300))
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            cls_cnts[cls_id] = cls_cnts.get(cls_id, 0) + 1
            coords = np.array(parts[1:], dtype=np.float64).reshape(-1, 2) * (img_w, img_h)
            cv2.polylines(vis, [np.round(coords).astype(np.int32)], isClosed=True, color=self.class_colors[cls_id % len(self.class_colors)], thickness=line_thickness)
        if (img_w, img_h) not in self.preview_layouts:
            legend_txt = [f'{n} (00)' for n in self.names]
            font_size, _, row_h = PlottingMixin().get_optimal_font_scales(text=legend_txt, accepted_px_width=int(img_w / 4), accepted_px_height=int(img_h / (2 * max(len(self.names), 10))), text_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)
            thickness = max(1, int(round(font_size * 2)))
            txt_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0][0] for t in legend_txt)
            pad = max(4, int(row_h / 3))
            self.preview_layouts[(img_w, img_h)] = (font_size, thickness, row_h, pad, txt_w)
        font_size, thickness, row_h, pad, txt_w = self.preview_layouts[(img_w, img_h)]
        swatch = int(row_h * 0.8)
        legend_w, legend_h = pad * 3 + swatch + txt_w, pad * 2 + row_h * len(self.names)
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (legend_w, legend_h), (0, 0, 0), -1)
        vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)
        for cls_id, name in enumerate(self.names):
            y_top = pad + row_h * cls_id + (row_h - swatch) // 2
            cv2.rectangle(vis, (pad, y_top), (pad + swatch, y_top + swatch), self.class_colors[cls_id % len(self.class_colors)], -1)
            txt_clr = (255, 255, 255) if cls_cnts[cls_id] > 0 else (130, 130, 130)
            cv2.putText(vis, f'{name} ({cls_cnts[cls_id]})', (pad * 2 + swatch, y_top + swatch), cv2.FONT_HERSHEY_SIMPLEX, font_size, txt_clr, thickness)
        if len(lines) == 0:
            cv2.putText(vis, 'NO DETECTION', (pad, legend_h + row_h * 2), cv2.FONT_HERSHEY_SIMPLEX, font_size * 2, (0, 0, 255), thickness * 2)
        cv2.putText(vis, f'{video_name} | frm {frame_idx}', (pad, img_h - pad), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)
        cv2.imshow('SAM3 Preview', vis)
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

    def run(self):
        timer = SimbaTimer(start=True)
        if self.seed is not None: random.seed(self.seed)
        img_train_dir, img_val_dir = os.path.join(self.save_dir, 'images', 'train'), os.path.join(self.save_dir, 'images', 'val')
        lbl_train_dir, lbl_val_dir = os.path.join(self.save_dir, 'labels', 'train'), os.path.join(self.save_dir, 'labels', 'val')
        create_directory(paths=[img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir], overwrite=False)

        map_path = os.path.join(self.save_dir, 'map.yaml')
        create_yolo_yaml(path=self.save_dir, train_path=img_train_dir, val_path=img_val_dir, names=self.name_map, save_path=map_path)
        if self.verbose:
            stdout_information(msg=f'map.yaml written to {map_path}')

        if self.detector is None:
            overrides = dict(conf=self.conf, task='segment', mode='predict', imgsz=self.imgsz, model=str(self.sam_path), half=True, save=False, verbose=False)
            predictor = SAM3SemanticPredictor(overrides=overrides)
        else:
            # conf=0.0: mask scores are checked per box in _detector_label, so a failed box can be attributed (and its frame skipped).
            # compile=None: ultralytics 8.4.9 passes the default compile=False to torch.compile as a mode string and crashes.
            overrides = dict(conf=0.0, task='segment', mode='predict', imgsz=self.imgsz, model=str(self.sam_path), half=True, save=False, verbose=False, compile=None)
            predictor = SAM3Predictor(overrides=overrides)

        all_samples, video_cnt, total_videos = [], 0, len(self.video_paths)
        train_cnt, val_cnt = 0, 0
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
            if self.amodal_ids:
                with torch.inference_mode():
                    self.amodal_refs = self._build_amodal_references(video_path=video_path, frame_count=total_frames, predictor=predictor, img_w=img_w, img_h=img_h)
                if self.verbose:
                    stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}): amodal reference masks for {len(self.amodal_refs)} static objects ({", ".join(self.names[c] for c, _, _ in self.amodal_refs) or "none"})')

            if self.verbose:
                stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}): targeting {self.n_frames} valid frames from {total_frames} total...')
            valid_cnt, skip_counts = 0, {}
            for frame_idx in candidate_indices:
                if valid_cnt >= self.n_frames:
                    break
                try:
                    frame = self._io_with_retry(read_frm_of_video, video_path=video_path, frame_index=frame_idx)
                except Exception as e:
                    if self.verbose:
                        stdout_warning(msg=f'Video {video_cnt}/{total_videos} ({video_name}), frame idx {frame_idx}: could not read frame after retries ({e}), skipping video...')
                    break
                if frame is None:
                    continue

                # inference_mode: ultralytics' set_image runs outside it and keeps autograd buffers (~15 GB peak at imgsz 1008 vs ~1.4 GB).
                with torch.inference_mode():
                    if self.detector is None:
                        predictor.set_image(frame)
                        results = predictor(text=self.txt_prompt)
                        r = results[0] if isinstance(results, list) and len(results) > 0 else results
                        label_str = '' if (r.masks is None or len(r.masks) == 0) else self._masks_to_yolo_label(r, img_w, img_h)
                    else:
                        label_str = self._detector_label(frame=frame, predictor=predictor, img_w=img_w, img_h=img_h)
                if self.preview:
                    if not self._preview_frame(frame=frame, label_str=label_str, video_name=video_name, frame_idx=frame_idx):
                        break
                if not label_str:
                    skip_reason = self.skip_reason if self.detector is not None else 'no detections'
                    skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1
                    if self.max_skipped_frames is not None and sum(skip_counts.values()) >= self.max_skipped_frames:
                        if self.verbose:
                            stdout_warning(msg=f'Video {video_cnt}/{total_videos} ({video_name}): {self.max_skipped_frames} frames skipped, moving to next video...')
                        break
                    continue

                try:
                    img_out = self._io_with_retry(read_frm_of_video, video_path=video_path, frame_index=frame_idx, greyscale=self.greyscale, clahe=self.clahe)
                except Exception as e:
                    if self.verbose:
                        stdout_warning(msg=f'Video {video_cnt}/{total_videos} ({video_name}), frame idx {frame_idx}: could not read frame for output after retries ({e}), skipping video...')
                    break
                sample_name = f'{video_name}_frm{frame_idx:08d}'

                is_train = random.random() < self.train_val_split
                if is_train:
                    img_dir, lbl_dir = img_train_dir, lbl_train_dir
                    train_cnt += 1
                else:
                    img_dir, lbl_dir = img_val_dir, lbl_val_dir
                    val_cnt += 1
                self._io_with_retry(cv2.imwrite, os.path.join(img_dir, f'{sample_name}.png'), img_out)
                self._io_with_retry(self._write_label, os.path.join(lbl_dir, f'{sample_name}.txt'), label_str)

                if self.visualize:
                    all_samples.append((sample_name, img_out, label_str))

                valid_cnt += 1
                if self.verbose:
                    stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}), frame {valid_cnt}/{self.n_frames} collected (frame idx {frame_idx}, total samples: {train_cnt + val_cnt}, split: {"train" if is_train else "val"}{f", amodal completed: {self.amodal_cnt}" if self.amodal_ids else ""})')
            if self.verbose:
                stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}): collected {valid_cnt}/{self.n_frames} valid labeled frames, skipped: {skip_counts if skip_counts else "none"}')

        if self.preview:
            cv2.destroyAllWindows()
        total_samples = train_cnt + val_cnt
        if total_samples == 0:
            raise NoFilesFoundError(msg='No masks detected in any sampled frame. No project created.', source=self.__class__.__name__)
        if self.visualize:
            create_yolo_sample_visualizations(samples=all_samples, save_dir=os.path.join(self.save_dir, 'visualizations'), names=self.names, verbose=self.verbose, source=self.__class__.__name__)
        timer.stop_timer()
        stdout_success(msg=f'YOLO segmentation project created at {self.save_dir}. ' f'{train_cnt} train, {val_cnt} val samples.', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)

    def _mask_iou_suppress(self, candidates: List[Tuple[int, float, np.ndarray]], img_w: int, img_h: int) -> List[Tuple[int, float, np.ndarray]]:
        """
        Greedy non-maximum suppression of ``(class_id, confidence, polygon)`` candidates sorted by confidence descending.

        Overlap is the true mask IoU: each traced outline is rasterized with :func:`cv2.fillPoly` and the
        intersection and union are counted in pixels. A mask whose IoU with an already-kept mask exceeds
        ``iou_threshold`` (same class) or ``cross_class_iou_threshold`` (different classes) is dropped as a duplicate
        detection of the same observation. A None threshold never suppresses. Bounding-box IoU is
        deliberately not used - two animals crossing have heavily overlapping boxes but little shared mask area.
        """
        masks, rects = [], []
        for _, _, pts in candidates:
            m = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
            masks.append(m.astype(bool))
            x, y, w, h = cv2.boundingRect(np.round(pts).astype(np.int32))
            rects.append((x, y, x + w, y + h))
        keep = []
        for cand_idx in range(len(candidates)):
            is_duplicate = False
            for kept_idx in keep:
                threshold = self.iou_threshold if candidates[cand_idx][0] == candidates[kept_idx][0] else self.cross_class_iou_threshold
                (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = rects[cand_idx], rects[kept_idx]
                if threshold is None or max(ax1, bx1) >= min(ax2, bx2) or max(ay1, by1) >= min(ay2, by2):
                    continue
                intersection = int(np.logical_and(masks[cand_idx], masks[kept_idx]).sum())
                if intersection == 0:
                    continue
                union = int(np.logical_or(masks[cand_idx], masks[kept_idx]).sum())
                if union > 0 and (intersection / union) > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(cand_idx)
        return [candidates[i] for i in keep]

    def _masks_to_yolo_label(self, result, img_w: int, img_h: int) -> str:
        candidates = []
        for mask_idx in range(len(result.masks)):
            try:
                polygon = result.masks.xy[mask_idx].astype(np.float64)
            except (IndexError, AttributeError):
                continue
            if polygon.shape[0] < 3:
                continue

            conf = 0.0
            if result.boxes is not None and mask_idx < len(result.boxes.conf):
                conf = float(result.boxes.conf[mask_idx].cpu())
                if conf < self.conf:
                    continue

            cls_id = 0
            if result.boxes is not None and mask_idx < len(result.boxes.cls):
                det_cls = int(result.boxes.cls[mask_idx].cpu())
                if det_cls < len(self.names):
                    cls_id = det_cls

            seen = set()
            unique_idx = []
            for i, pt in enumerate(polygon):
                key = (pt[0], pt[1])
                if key not in seen:
                    seen.add(key)
                    unique_idx.append(i)
            pts = polygon[unique_idx]
            if pts.shape[0] < 3:
                continue
            candidates.append((cls_id, conf, pts))
        return self._candidates_to_label(candidates=candidates, img_w=img_w, img_h=img_h)

    def _detector_label(self, frame: np.ndarray, predictor, img_w: int, img_h: int) -> str:
        """Detector boxes -> SAM3 box-prompted masks -> YOLO label string. Returns an empty string when the frame should be skipped, with the reason in ``self.skip_reason``."""
        self.skip_reason = None
        det = self.detector.predict(frame, conf=self.det_conf, verbose=False)[0]
        if det.boxes is None or len(det.boxes) == 0:
            self.skip_reason = 'no detections'
            return ''
        boxes = det.boxes.xyxy.cpu().numpy()
        det_cls = det.boxes.cls.cpu().numpy().astype(np.int32)
        det_conf = det.boxes.conf.cpu().numpy()
        if self.certain_conf is not None:
            uncertain = det_conf < self.certain_conf
            blocking = uncertain & ~np.isin(det_cls, list(self.uncertain_drop_ids))
            if blocking.any():
                self.skip_reason = f'uncertain {self.names[det_cls[np.argmax(blocking)]]}'
                return ''
            boxes, det_cls, det_conf = boxes[~uncertain], det_cls[~uncertain], det_conf[~uncertain]
            if len(boxes) == 0:
                self.skip_reason = 'no certain detections'
                return ''
        masks, box_idxs, mask_scores = self._segment_boxes(frame=frame, predictor=predictor, boxes=boxes, img_w=img_w, img_h=img_h)
        others_cnt = masks.sum(axis=0, dtype=np.uint16) if self.amodal_refs else None

        candidates, passed = [], set()
        self.amodal_cnt = 0
        for mask, box_idx, mask_score in zip(masks, box_idxs, mask_scores):
            if mask_score < self.conf:
                continue
            # Largest outer contour of the full mask (``masks.xy`` can join disconnected fragments into one polygon).
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                continue
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if contour.shape[0] < 3 or area <= 0:
                continue
            if self.min_box_fill is not None:
                x1, y1, x2, y2 = boxes[box_idx]
                if area / max((x2 - x1) * (y2 - y1), 1.0) < self.min_box_fill:
                    continue
            passed.add(int(box_idx))
            if self.amodal_refs and int(det_cls[box_idx]) in self.amodal_ids:
                completed = self._amodal_complete(mask=mask.astype(bool), cls_id=int(det_cls[box_idx]), box=boxes[box_idx], occluders=(others_cnt - mask) > 0)
                if completed is not None:
                    contours, _ = cv2.findContours(completed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    contour = max(contours, key=cv2.contourArea)
                    self.amodal_cnt += 1
            candidates.append((int(det_cls[box_idx]), float(det_conf[box_idx]), contour.reshape(-1, 2).astype(np.float64)))
        failed = [i for i in range(len(boxes)) if i not in passed]
        if self.skip_frame_on_bad_mask and failed:
            self.skip_reason = f'bad mask on {self.names[det_cls[failed[0]]]}'
            return ''
        label_str = self._candidates_to_label(candidates=candidates, img_w=img_w, img_h=img_h)
        if not label_str:
            self.skip_reason = 'no masks retained'
        return label_str

    @staticmethod
    def _segment_boxes(frame: np.ndarray, predictor, boxes: np.ndarray, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SAM3 box-prompted masks for ``boxes``: returns (masks (N, H, W) uint8, index of the prompting box per mask, SAM3 mask score per mask)."""
        predictor.set_image(frame)
        sam = predictor(bboxes=boxes)[0]
        if sam.masks is None or len(sam.masks) == 0:
            return np.zeros((0, img_h, img_w), np.uint8), np.zeros(0, np.int32), np.zeros(0)
        # SAM writes the index of the prompting box into ``cls``
        return sam.masks.data.cpu().numpy().astype(np.uint8), sam.boxes.cls.cpu().numpy().astype(np.int32), sam.boxes.conf.cpu().numpy()

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
        iw, ih = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iw * ih
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

    def _build_amodal_references(self, video_path: Union[str, os.PathLike], frame_count: int, predictor, img_w: int, img_h: int) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Reference (full-extent) masks for the static ``amodal_classes`` objects of one video, segmented on a clean plate of the video.

        The plate is the per-pixel (greyscale) median of ``amodal_n_frames`` evenly spaced frames, with pixels under detected ``plate_mask_classes``
        objects (padded boxes, e.g. ``mouse``) left out of each frame's vote: moving animals vanish and the static objects they occlude are fully
        visible. Pixels covered in (almost) every frame fall back to the plain median and count as unknown; a reference that is more than 10%
        unknown is dropped. The plate is segmented once (detector boxes -> SAM3). Returns ``[(class_id, box, reference_mask), ...]`` at full resolution.
        """
        frames, valid = [], []
        for frame_idx in np.linspace(0, frame_count - 1, self.amodal_n_frames).astype(int):
            try:
                frame = self._io_with_retry(read_frm_of_video, video_path=video_path, frame_index=int(frame_idx))
            except Exception:
                continue
            if frame is None:
                continue
            frame_valid = np.ones((img_h, img_w), dtype=bool)
            if self.plate_mask_ids:
                det = self.detector.predict(frame, conf=self.det_conf, verbose=False)[0]
                for (x1, y1, x2, y2), cls_id in zip(det.boxes.xyxy.cpu().numpy(), det.boxes.cls.cpu().numpy().astype(np.int32)):
                    if cls_id in self.plate_mask_ids:  # 20% padding covers tails and motion blur
                        pad_x, pad_y = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
                        frame_valid[max(0, int(y1 - pad_y)):int(y2 + pad_y), max(0, int(x1 - pad_x)):int(x2 + pad_x)] = False
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame)
            valid.append(frame_valid)
        if len(frames) < 5:
            return []
        frames, valid = np.stack(frames), np.stack(valid)
        unknown = valid.sum(axis=0) < 5
        masked = frames.astype(np.float32)
        masked[~valid] = np.nan
        with np.errstate(all='ignore'):
            plate = np.nanmedian(masked, axis=0)
        plate = np.where(unknown, np.median(frames, axis=0), plate).astype(np.uint8)
        self.amodal_plate = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)

        det = self.detector.predict(self.amodal_plate, conf=self.det_conf if self.certain_conf is None else self.certain_conf, verbose=False)[0]
        if det.boxes is None or len(det.boxes) == 0:
            return []
        det_cls = det.boxes.cls.cpu().numpy().astype(np.int32)
        keep = np.isin(det_cls, list(self.amodal_ids))
        if not keep.any():
            return []
        boxes, det_cls = det.boxes.xyxy.cpu().numpy()[keep], det_cls[keep]
        masks, box_idxs, mask_scores = self._segment_boxes(frame=self.amodal_plate, predictor=predictor, boxes=boxes, img_w=img_w, img_h=img_h)
        refs = []
        for mask, box_idx, mask_score in zip(masks, box_idxs, mask_scores):
            mask = mask.astype(bool)
            if mask_score < self.conf or not mask.any() or np.logical_and(mask, unknown).sum() > 0.1 * mask.sum():
                continue
            refs.append((int(det_cls[box_idx]), boxes[box_idx], mask))
        return refs

    def _amodal_complete(self, mask: np.ndarray, cls_id: int, box: np.ndarray, occluders: np.ndarray) -> Optional[np.ndarray]:
        """
        Return ``reference | mask`` when ``mask`` is a clipped view of its object's reference mask with the missing part covered by another
        object's mask (occluded), else None. Clipped = visible area below 85% of the reference and at least 90% of it inside the reference
        (so the object has not moved); occluded = at least 30% of the missing pixels lie under (a 13 px dilation of) other masks.
        """
        matches = [(self._box_iou(box, ref_box), ref) for ref_cls, ref_box, ref in self.amodal_refs if ref_cls == cls_id]
        iou, ref = max(matches, key=lambda m: m[0], default=(0.0, None))
        if iou < 0.3:
            return None
        area, ref_area = int(mask.sum()), int(ref.sum())
        if area >= 0.85 * ref_area or np.logical_and(mask, ref).sum() < 0.9 * area:
            return None
        missing = ref & ~mask
        occluders = cv2.dilate(occluders.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
        if np.logical_and(missing, occluders).sum() < 0.3 * missing.sum():
            return None
        return ref | mask

    def _candidates_to_label(self, candidates: List[Tuple[int, float, np.ndarray]], img_w: int, img_h: int) -> str:
        """De-duplicate, cap and resample ``(class_id, confidence, polygon)`` candidates and format them as a YOLO segmentation label string."""
        if (self.iou_threshold is not None or self.cross_class_iou_threshold is not None) and len(candidates) > 1:
            candidates = sorted(candidates, key=lambda c: c[1], reverse=True)
            candidates = self._mask_iou_suppress(candidates=candidates, img_w=img_w, img_h=img_h)
        if self.max_detections is not None:
            candidates = sorted(candidates, key=lambda c: c[1], reverse=True)[:self.max_detections]

        lines = []
        for cls_id, _, pts in candidates:
            if self.simplify_epsilon is not None:
                contour = pts.astype(np.float32).reshape(-1, 1, 2)
                pts = cv2.approxPolyDP(contour, self.simplify_epsilon * cv2.arcLength(contour, True), True).reshape(-1, 2).astype(np.float64)
                if pts.shape[0] < 3:
                    continue
            if self.vertice_cnt is not None:
                pts = resample_geometry_vertices(vertices=[pts], vertice_cnt=self.vertice_cnt)[0].astype(np.float64)
                _, unique_idx = np.unique(pts, axis=0, return_index=True)
                pts = pts[np.sort(unique_idx)]
                if pts.shape[0] < 3:
                    continue

            norm_x = np.clip(pts[:, 0] / img_w, 0, 1)
            norm_y = np.clip(pts[:, 1] / img_h, 0, 1)
            coords = np.column_stack((norm_x, norm_y)).flatten()
            coord_str = ' '.join(f'{v:.6f}' for v in coords)
            lines.append(f'{cls_id} {coord_str}')

        return '\n'.join(lines) + '\n' if lines else ''


# names = ('mouse', 'house', 'tunnel', 'nest', 'nose_port', 'reward_dispenser', 'food_pellet', 'pellet_feeder', 'water_bottle', 'wood')
# txt_prompt = ('mouse', 'house', 'tunnel', 'nest', 'nose port', 'reward dispenser', 'food pellet', 'pellet feeder', 'water bottle', 'wood')
#
# runner = SAM3ToYoloSeg(video_dir=r'F:\netholabs\projects_lp_compressed_13.4.2028\videos\6.01.001_2026_03_11_08_26_00_000\area_1',
#                        sam_path=r'D:\sam3\sam3.pt',
#                        save_dir=r'I:\netholabs\sam3_2_yolo_seg_test',
#                        txt_prompt=txt_prompt,
#                        preview=True,
#                        names=names,
#                        n_frames=50,
#                        verbose=True)
#
# runner.run()


### EXAMPLE
# VIDEO_DIR = r'E:\my_videos'
# MDL_PATH = r'D:\sam3\sam3.pt'
# SAVE_DIR = r'E:\yolo_seg_project'
#
# runner = SAM3ToYoloSeg(
#     video_dir=VIDEO_DIR,
#     sam_path=MDL_PATH,
#     save_dir=SAVE_DIR,
#     txt_prompt='mouse',
#     n_frames=50,
#     names=('mouse',),
#     train_val_split=0.7,
#     conf=0.25,
# )
# runner.run()
