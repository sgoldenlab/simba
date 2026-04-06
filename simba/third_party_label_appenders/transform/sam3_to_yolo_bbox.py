"""
Generate a YOLO bounding-box (detection) project from videos using SAM3.

Takes a directory of videos and a text prompt, samples N random frames per video,
runs SAM3 semantic segmentation, and writes the detected bounding boxes as a
YOLO-format detection project with ``images/``, ``labels/``, and ``map.yaml``.

To merge this project with others that share the same class names and task type, use
:class:`~simba.third_party_label_appenders.transform.merge_yolo_projects.MergeYoloProjects`
(see also :mod:`simba.third_party_label_appenders.transform.sam3_to_yolo_seg`).
"""

import os
import random
from typing import Optional, Tuple, Union

import cv2

try:
    from typing import Literal
except:
    from typing_extensions import Literal

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except:
    SAM3SemanticPredictor = None

import numpy as np

from simba.third_party_label_appenders.converters import create_yolo_yaml
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_boolean, check_valid_tuple)
from simba.utils.errors import NoFilesFoundError, SimBAPAckageVersionError
from simba.utils.printing import (SimbaTimer, stdout_information,
                                  stdout_success, stdout_warning)
from simba.utils.read_write import (create_directory,
                                    find_all_videos_in_directory,
                                    get_pkg_version, get_video_meta_data,
                                    read_frm_of_video, recursive_file_search)
from simba.utils.yolo import create_yolo_sample_visualizations


class SAM3ToYoloBBox:
    """
    Sample N random frames from each video in a directory, run SAM3 with a text prompt, and write the resulting bounding boxes as a YOLO detection project.

    .. note::
       To fit a YOLO detection model, see :class:`~simba.model.yolo_fit.FitYolo`.

    .. seealso::

       * :class:`~simba.third_party_label_appenders.transform.merge_yolo_projects.MergeYoloProjects` — merge multiple YOLO ``map.yaml`` projects into one training set.
       * :class:`~simba.third_party_label_appenders.transform.sam3_to_yolo_seg.SAM3ToYoloSeg` — SAM3 to YOLO **segmentation** labels from the same predictor stack.

    :param Union[str, os.PathLike] video_dir: Directory containing input videos.
    :param Union[str, os.PathLike] sam_path: Path to SAM3 model weights (e.g. sam3.pt).
    :param Union[str, os.PathLike] save_dir: Root output directory for the YOLO project.
    :param str txt_prompt: Text prompt for SAM3 (e.g. "mouse", "mouse tail").
    :param int n_frames: Number of random frames to sample from each video.
    :param Tuple[str, ...] names: Class names in index order. Default ``('animal',)``.
    :param float train_val_split: Fraction allocated to training (0.1-0.9). Default 0.7.
    :param float conf: SAM3 confidence threshold. Default 0.25.
    :param int sam_imgsz: Image size for SAM3 inference. Default 640.
    :param bool greyscale: If True, save extracted frames in greyscale. Default False.
    :param Optional[Union[Tuple[int, int, int], bool]] clahe: If True, applies CLAHE with default params. If tuple of (clip_limit, tile_x, tile_y), applies CLAHE with those params. Default False.
    :param float buffer_pct: Fraction to expand each box by (e.g. 0.1 adds 10% of width/height on each side). Default 0.0.
    :param int consecutive_miss_limit: If this many consecutive frames yield no detection, skip to the next video. Default 100.
    :param bool recursive: If True, search video_dir and all subdirectories for videos. Default False.
    :param Optional[int] seed: Random seed for reproducible frame sampling.
    :param Optional[int] max_detections: Maximum number of detections to keep per frame (sorted by confidence descending). If None, all detections above ``conf`` are kept. Default None.
    :param bool visualize: If True, saves annotated images with bounding-box overlays to a ``visualizations`` subfolder inside ``save_dir``. Useful for verifying SAM3 annotation quality. Default False.
    :param bool verbose: If True, print progress updates. Default True.

    :raises SimBAGPUError: If no NVIDIA GPU is detected (via ``nvidia-smi``).
    :raises SimBAPAckageVersionError: If ``ultralytics`` is not installed, or ``SAM3SemanticPredictor`` cannot be imported.

    :example:
    >>> runner = SAM3ToYoloBBox(video_dir=r'/path/to/videos', sam_path=r'/path/to/sam3.pt', save_dir=r'/path/to/yolo_project', txt_prompt='mouse', n_frames=50)
    >>> runner.run()
    """

    def __init__(self,
                 video_dir: Union[str, os.PathLike],
                 sam_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 txt_prompt: str = 'mouse',
                 n_frames: int = 50,
                 names: Tuple[str, ...] = ('animal',),
                 train_val_split: float = 0.7,
                 conf: float = 0.25,
                 sam_imgsz: int = 644,
                 greyscale: bool = False,
                 clahe: Optional[Union[Tuple[int, int, int], bool]] = False,
                 buffer_pct: float = 0.0,
                 consecutive_miss_limit: int = 100,
                 max_detections: Optional[int] = None,
                 recursive: bool = False,
                 seed: Optional[int] = None,
                 visualize: bool = False,
                 verbose: bool = True):

        check_nvidea_gpu_available(raise_error=True)
        _ = get_pkg_version(pkg='ultralytics', raise_error=True)
        if SAM3SemanticPredictor is None:
            raise SimBAPAckageVersionError(msg='Could not import SAM3SemanticPredictor from ultralytics.models.sam. Install a compatible ultralytics build with SAM3 support.', source=self.__class__.__name__)

        check_if_dir_exists(in_dir=video_dir, source=f'{self.__class__.__name__} video_dir')
        check_file_exist_and_readable(file_path=sam_path)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        check_str(name=f'{self.__class__.__name__} txt_prompt', value=txt_prompt)
        check_int(name=f'{self.__class__.__name__} n_frames', value=n_frames, min_value=1)
        check_valid_tuple(x=names, source=f'{self.__class__.__name__} names', minimum_length=1, valid_dtypes=(str,))
        check_float(name=f'{self.__class__.__name__} train_val_split', value=train_val_split, min_value=0.1, max_value=0.9)
        check_float(name=f'{self.__class__.__name__} conf', value=conf, min_value=0.01, max_value=1.0)
        check_int(name=f'{self.__class__.__name__} imgsz', value=sam_imgsz, min_value=32)
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_float(name=f'{self.__class__.__name__} buffer_pct', value=buffer_pct, min_value=0.0, max_value=1.0)
        check_int(name=f'{self.__class__.__name__} consecutive_miss_limit', value=consecutive_miss_limit, min_value=1)
        check_valid_boolean(value=recursive, source=f'{self.__class__.__name__} recursive')
        check_valid_boolean(value=visualize, source=f'{self.__class__.__name__} visualize')
        if max_detections is not None: check_int(name=f'{self.__class__.__name__} max_detections', value=max_detections, min_value=1)
        if seed is not None: check_int(name=f'{self.__class__.__name__} seed', value=seed)
        self.video_dir, self.sam_path, self.save_dir, self.txt_prompt = video_dir, sam_path, save_dir, txt_prompt
        self.n_frames, self.names, self.train_val_split, self.conf, self.imgsz = n_frames, names, train_val_split, conf, sam_imgsz
        self.greyscale, self.clahe, self.buffer_pct, self.consecutive_miss_limit, self.max_detections, self.seed, self.verbose, self.visualize = greyscale, clahe, buffer_pct, consecutive_miss_limit, max_detections, seed, verbose, visualize
        self.name_map = {name: idx for idx, name in enumerate(names)}
        if recursive:
            self.video_paths = recursive_file_search(directory=video_dir, extensions=[".avi", ".mp4", ".mov", ".flv", ".m4v", ".webm", ".h264"], as_dict=True, raise_error=True)
        else:
            self.video_paths = find_all_videos_in_directory(directory=video_dir, as_dict=True, raise_error=True)

    def run(self):
        timer = SimbaTimer(start=True)
        if self.seed is not None: random.seed(self.seed)
        img_train_dir, img_val_dir = os.path.join(self.save_dir, 'images', 'train'), os.path.join(self.save_dir, 'images', 'val')
        lbl_train_dir, lbl_val_dir = os.path.join(self.save_dir, 'labels', 'train'), os.path.join(self.save_dir, 'labels', 'val')
        create_directory(paths=[img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir], overwrite=False)

        overrides = dict(conf=self.conf, task='segment', mode='predict', imgsz=self.imgsz, model=str(self.sam_path), half=True, save=False, verbose=False)
        predictor = SAM3SemanticPredictor(overrides=overrides)

        all_samples, video_cnt, total_videos = [], 0, len(self.video_paths)
        for video_name, video_path in self.video_paths.items():
            video_cnt += 1
            video_meta = get_video_meta_data(video_path=video_path)
            total_frames = int(video_meta['frame_count'])
            img_w, img_h = int(video_meta['width']), int(video_meta['height'])
            candidate_indices = list(range(total_frames))
            random.shuffle(candidate_indices)

            if self.verbose:
                stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}): targeting {self.n_frames} valid frames from {total_frames} total...')
            valid_cnt, consecutive_misses = 0, 0
            for frame_idx in candidate_indices:
                if valid_cnt >= self.n_frames:
                    break
                if consecutive_misses >= self.consecutive_miss_limit:
                    if self.verbose:
                        stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}): {self.consecutive_miss_limit} consecutive misses, skipping to next video...')
                    break
                try:
                    frame = read_frm_of_video(video_path=video_path, frame_index=frame_idx)
                except Exception as e:
                    if self.verbose:
                        stdout_warning(msg=f'Video {video_cnt}/{total_videos} ({video_name}), frame idx {frame_idx}: could not read frame ({e}), skipping...')
                    consecutive_misses += 1
                    continue
                if frame is None:
                    consecutive_misses += 1
                    continue

                predictor.set_image(frame)
                results = predictor(text=[self.txt_prompt])
                r = results[0] if isinstance(results, list) and len(results) > 0 else results

                if r.boxes is None or len(r.boxes) == 0:
                    consecutive_misses += 1
                    if self.verbose:
                        stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}), frame idx {frame_idx}: no detection found (consecutive misses: {consecutive_misses}/{self.consecutive_miss_limit})')
                    continue

                label_str = self._boxes_to_yolo_label(r, img_w, img_h)
                if not label_str:
                    consecutive_misses += 1
                    if self.verbose:
                        stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}), frame idx {frame_idx}: detections below conf threshold (consecutive misses: {consecutive_misses}/{self.consecutive_miss_limit})')
                    continue

                consecutive_misses = 0
                img_out = read_frm_of_video(video_path=video_path, frame_index=frame_idx, greyscale=self.greyscale, clahe=self.clahe)
                sample_name = f'{video_name}_frm{frame_idx:08d}'
                all_samples.append((sample_name, img_out, label_str))
                valid_cnt += 1
                if self.verbose:
                    stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}), frame {valid_cnt}/{self.n_frames} collected (frame idx {frame_idx}, total samples: {len(all_samples)})')
            if self.verbose:
                stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}): collected {valid_cnt}/{self.n_frames} valid labeled frames')

        if len(all_samples) == 0:

            raise NoFilesFoundError(msg='No boxes detected in any sampled frame. No project created.', source=self.__class__.__name__)

        if self.verbose:
            stdout_information(msg=f'All videos processed. {len(all_samples)} total samples collected. Shuffling and splitting...')
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * self.train_val_split)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        if self.verbose:
            stdout_information(msg=f'Split: {len(train_samples)} train, {len(val_samples)} val. Writing to disk...')

        for split_name, samples, img_dir, lbl_dir in [('train', train_samples, img_train_dir, lbl_train_dir), ('val', val_samples, img_val_dir, lbl_val_dir)]:
            for cnt, (sample_name, img, label_str) in enumerate(samples):
                cv2.imwrite(os.path.join(img_dir, f'{sample_name}.png'), img)
                with open(os.path.join(lbl_dir, f'{sample_name}.txt'), 'w') as f:
                    f.write(label_str)
                if self.verbose and (cnt + 1) % 10 == 0:
                    stdout_information(msg=f'Writing {split_name}: {cnt + 1}/{len(samples)}...')
            if self.verbose:
                stdout_information(msg=f'Writing {split_name}: {len(samples)}/{len(samples)} complete.')

        map_path = os.path.join(self.save_dir, 'map.yaml')
        create_yolo_yaml(path=self.save_dir, train_path=img_train_dir, val_path=img_val_dir, names=self.name_map, save_path=map_path)

        if self.visualize:
            create_yolo_sample_visualizations(samples=all_samples, save_dir=os.path.join(self.save_dir, 'visualizations'), names=self.names, verbose=self.verbose, source=self.__class__.__name__)

        timer.stop_timer()
        stdout_success(msg=f'YOLO bbox detection project created at {self.save_dir}. ' f'{len(train_samples)} train, {len(val_samples)} val samples.', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)

    def _boxes_to_yolo_label(self, result, img_w: int, img_h: int) -> str:
        box_indices = list(range(len(result.boxes)))
        box_indices.sort(key=lambda i: float(result.boxes.conf[i].cpu()), reverse=True)
        lines = []
        for box_idx in box_indices:
            if float(result.boxes.conf[box_idx].cpu()) < self.conf:
                continue
            if self.max_detections is not None and len(lines) >= self.max_detections:
                break

            cls_id = 0
            det_cls = int(result.boxes.cls[box_idx].cpu())
            if det_cls < len(self.names):
                cls_id = det_cls

            xyxy = result.boxes.xyxy[box_idx].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            if self.buffer_pct > 0:
                bw, bh = x2 - x1, y2 - y1
                x1 -= bw * self.buffer_pct
                y1 -= bh * self.buffer_pct
                x2 += bw * self.buffer_pct
                y2 += bh * self.buffer_pct
            x_center = ((x1 + x2) / 2.0) / img_w
            y_center = ((y1 + y2) / 2.0) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            lines.append(f'{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}')

        return '\n'.join(lines) + '\n' if lines else ''




# runner = SAM3ToYoloBBox(video_dir=r'E:\open_video\open_field_2',
#                         sam_path=r'D:\sam3\sam3.pt',
#                         save_dir=r'E:\open_video\open_field_2\yolo_bbox_project',
#                         txt_prompt='mouse',
#                         n_frames=25,
#                         verbose=True,
#                         conf=0.25,
#                         max_detections=1,
#                         buffer_pct=0.15,
#                         recursive=False)
# runner.run()


### EXAMPLE
# VIDEO_DIR = r'E:\my_videos'
# MDL_PATH = r'D:\sam3\sam3.pt'
# SAVE_DIR = r'E:\yolo_bbox_project'
#
# runner = SAM3ToYoloBBox(
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
