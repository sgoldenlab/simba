import os
import random
from typing import Optional, Tuple, Union

import cv2
import numpy as np

try:
    from typing import Literal
except:
    from typing_extensions import Literal

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except:
    SAM3SemanticPredictor = None

from simba.third_party_label_appenders.converters import create_yolo_yaml
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_boolean, check_valid_tuple)
from simba.utils.data import resample_geometry_vertices
from simba.utils.errors import NoFilesFoundError, SimBAPAckageVersionError
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (create_directory,
                                    find_all_videos_in_directory, get_fn_ext,
                                    get_pkg_version, get_video_meta_data,
                                    read_frm_of_video)


class SAM3ToYoloSeg:
    """
    Sample N random frames from each video in a directory, run SAM3 with a text prompt, and write the resulting masks as a YOLO segmentation project.

    .. note::
       To fit a YOLO segmentation model, see :class:`~simba.model.yolo_fit.FitYolo`.
       For YOLO segmentation inference, see :class:`~simba.model.yolo_seg_inference.YOLOSegmentationInference`.

    .. seealso::

       * :class:`~simba.third_party_label_appenders.transform.merge_yolo_projects.MergeYoloProjects` — merge several ``map.yaml`` projects (same classes and task) into one dataset.

    :raises SimBAGPUError: If no NVIDIA GPU is detected (via ``nvidia-smi``).
    :raises SimBAPAckageVersionError: If ``ultralytics`` is not installed, or ``SAM3SemanticPredictor`` cannot be imported.

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
    :param Optional[int] vertice_cnt: If not None, resample each mask polygon to this many vertices. Default 40.
    :param Optional[int] seed: Random seed for reproducible frame sampling.
    :param bool verbose: If True, print progress updates. Default True.

    :example:
    >>> runner = SAM3ToYoloSeg(video_dir=r'/path/to/videos', sam_path=r'/path/to/sam3.pt', save_dir=r'/path/to/yolo_project', txt_prompt='mouse', n_frames=50)
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
                 vertice_cnt: Optional[int] = 40,
                 seed: Optional[int] = None,
                 verbose: bool = True):

        check_nvidea_gpu_available(raise_error=True)
        _ = get_pkg_version(pkg="ultralytics", raise_error=True)
        if SAM3SemanticPredictor is None:
            raise SimBAPAckageVersionError(msg="Could not import SAM3SemanticPredictor from ultralytics.models.sam. Install a compatible ultralytics build with SAM3 support.", source=self.__class__.__name__,)
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
        if vertice_cnt is not None: check_int(name=f'{self.__class__.__name__} vertice_cnt', value=vertice_cnt, min_value=3)
        if seed is not None: check_int(name=f'{self.__class__.__name__} seed', value=seed)
        self.video_dir, self.sam_path, self.save_dir, self.txt_prompt = video_dir, sam_path, save_dir, txt_prompt
        self.n_frames, self.names, self.train_val_split, self.conf, self.imgsz = n_frames, names, train_val_split, conf, sam_imgsz
        self.greyscale, self.clahe, self.vertice_cnt, self.seed, self.verbose = greyscale, clahe, vertice_cnt, seed, verbose
        self.name_map = {name: idx for idx, name in enumerate(names)}
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
            valid_cnt = 0
            for frame_idx in candidate_indices:
                if valid_cnt >= self.n_frames:
                    break
                frame = read_frm_of_video(video_path=video_path, frame_index=frame_idx)
                if frame is None:
                    continue

                predictor.set_image(frame)
                results = predictor(text=[self.txt_prompt])
                r = results[0] if isinstance(results, list) and len(results) > 0 else results

                if r.masks is None or len(r.masks) == 0:
                    continue

                label_str = self._masks_to_yolo_label(r, img_w, img_h)
                if not label_str:
                    continue

                img_out = read_frm_of_video(video_path=video_path, frame_index=frame_idx, greyscale=self.greyscale, clahe=self.clahe)
                sample_name = f'{video_name}_frm{frame_idx:08d}'
                all_samples.append((sample_name, img_out, label_str))
                valid_cnt += 1
                if self.verbose:
                    stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}), frame {valid_cnt}/{self.n_frames} collected (frame idx {frame_idx}, total samples: {len(all_samples)})')
            if self.verbose:
                stdout_information(msg=f'Video {video_cnt}/{total_videos} ({video_name}): collected {valid_cnt}/{self.n_frames} valid labeled frames')

        if len(all_samples) == 0:
            raise NoFilesFoundError(msg='No masks detected in any sampled frame. No project created.', source=self.__class__.__name__)

        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * self.train_val_split)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]

        for samples, img_dir, lbl_dir in [(train_samples, img_train_dir, lbl_train_dir), (val_samples, img_val_dir, lbl_val_dir)]:
            for sample_name, img, label_str in samples:
                cv2.imwrite(os.path.join(img_dir, f'{sample_name}.png'), img)
                with open(os.path.join(lbl_dir, f'{sample_name}.txt'), 'w') as f:
                    f.write(label_str)

        map_path = os.path.join(self.save_dir, 'map.yaml')
        create_yolo_yaml(path=self.save_dir, train_path=img_train_dir, val_path=img_val_dir, names=self.name_map, save_path=map_path)

        timer.stop_timer()
        stdout_success(msg=f'YOLO segmentation project created at {self.save_dir}. ' f'{len(train_samples)} train, {len(val_samples)} val samples.', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)

    def _masks_to_yolo_label(self, result, img_w: int, img_h: int) -> str:
        lines = []
        for mask_idx in range(len(result.masks)):
            try:
                polygon = result.masks.xy[mask_idx].astype(np.float64)
            except (IndexError, AttributeError):
                continue
            if polygon.shape[0] < 3:
                continue

            if result.boxes is not None and mask_idx < len(result.boxes.conf):
                if float(result.boxes.conf[mask_idx].cpu()) < self.conf:
                    continue

            cls_id = 0
            if result.boxes is not None and mask_idx < len(result.boxes.cls):
                det_cls = int(result.boxes.cls[mask_idx].cpu())
                if det_cls < len(self.names):
                    cls_id = det_cls

            pts = np.unique(polygon, axis=0)
            if pts.shape[0] < 3:
                continue
            center = np.mean(pts, axis=0)
            angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
            pts = pts[np.argsort(angles)]

            if self.vertice_cnt is not None:
                pts = resample_geometry_vertices(vertices=[pts.astype(np.int32)], vertice_cnt=self.vertice_cnt)[0].astype(np.float64)

            norm_x = np.clip(pts[:, 0] / img_w, 0, 1)
            norm_y = np.clip(pts[:, 1] / img_h, 0, 1)
            coords = np.column_stack((norm_x, norm_y)).flatten()
            coord_str = ' '.join(f'{v:.6f}' for v in coords)
            lines.append(f'{cls_id} {coord_str}')

        return '\n'.join(lines) + '\n' if lines else ''




# runner = SAM3ToYoloSeg(video_dir=r'E:\litpose_yolo\pi\videos',
#                        sam_path=r'D:\sam3\sam3.pt',
#                        save_dir=r'E:\litpose_yolo\yolo_from_sam3',
#                        txt_prompt='mouse',
#                        n_frames=75,
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
