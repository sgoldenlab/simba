import platform
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import functools
import multiprocessing
import os
from collections import ChainMap

import cv2
import pandas as pd
from numba import int64, jit, njit, prange, uint8
from shapely.geometry import Polygon

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_if_valid_img,
                                check_instance, check_int, check_str,
                                check_valid_array, check_valid_lst)
from simba.utils.enums import Defaults, Formats, GeometryEnum, Options
from simba.utils.errors import ArrayError, FrameRangeError, InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data,
                                    read_frm_of_video)


class ImageMixin(object):
    """
    Methods to slice and compute attributes of images and comparing those attributes across sequential images.

    This can be helpful when the behaviors studied are very subtle and the signal is very low in relation to the
    noise within the pose-estimated data. In these use-cases, we cannot use pose-estimated data directly, and we instead
    study histograms, contours and other image metrics within images derived from the intersection of geometries
    (like a circle around the nose) across sequential images. Often these methods are called using image masks created from
    pose-estimated points within ``simba.mixins.geometry_mixin.GeometryMixin`` methods.

    .. important::
        If there is non-pose related noise in the environment (e.g., there are non-experiment related light sources that goes on and off, or other image noise that doesn't necesserily affect pose-estimation reliability),
        this will negatively affect the reliability of most image attribute comparisons.

    """

    def __init__(self):
        pass

    @staticmethod
    def brightness_intensity(imgs: List[np.ndarray], ignore_black: Optional[bool] = True) -> List[float]:
        """
        Compute the average brightness intensity within each image within a list.

        For example, (i) create a list of images containing a light cue ROI, (ii) compute brightness in each image, (iii) perform kmeans on brightness, and get the frames when the light cue is on vs off.

        :param List[np.ndarray] imgs: List of images as arrays to calculate average brightness intensity within.
        :param Optional[bool] ignore_black: If True, ignores black pixels. If the images are sliced non-rectangular geometric shapes created by ``slice_shapes_in_img``, then pixels that don't belong to the shape has been masked in black.
        :returns List[float]: List of floats of size len(imgs) with brightness intensities.

        :example:
        >>> img = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
        >>> ImageMixin.brightness_intensity(imgs=[img], ignore_black=False)
        >>> [159.0]
        """
        results = []
        check_instance(
            source=f"{ImageMixin().brightness_intensity.__name__} imgs",
            instance=imgs,
            accepted_types=list,
        )
        for cnt, img in enumerate(imgs):
            check_instance(
                source=f"{ImageMixin().brightness_intensity.__name__} img {cnt}",
                instance=img,
                accepted_types=np.ndarray,
            )
            if len(img) == 0:
                results.append(0)
            else:
                if ignore_black:
                    results.append(np.ceil(np.average(img[img != 0])))
                else:
                    results.append(np.ceil(np.average(img)))
        return results

    @staticmethod
    def gaussian_blur(img: np.ndarray, kernel_size: Optional[Tuple] = (9, 9)):
        check_if_valid_img(data=img, source=ImageMixin.gaussian_blur.__name__)
        check_instance(source=ImageMixin.gaussian_blur.__name__, instance=kernel_size, accepted_types=(tuple,))
        check_valid_lst(data=list(kernel_size), source=ImageMixin.gaussian_blur.__name__, valid_dtypes=(int,), exact_len=2,)
        return cv2.GaussianBlur(img, kernel_size, 0)

    @staticmethod
    def erode(
        img: np.ndarray,
        kernel_size: Optional[Tuple] = (3, 3),
        iterations: Optional[int] = 3,
    ):
        check_if_valid_img(data=img, source=ImageMixin.gaussian_blur.__name__)
        check_instance(
            source=ImageMixin.gaussian_blur.__name__,
            instance=kernel_size,
            accepted_types=(tuple,),
        )
        check_valid_lst(
            data=list(kernel_size),
            source=ImageMixin.gaussian_blur.__name__,
            valid_dtypes=(int,),
            exact_len=2,
        )
        check_int(name=ImageMixin.erode.__name__, value=iterations, min_value=1)
        return cv2.erode(img, np.ones((3, 3), np.uint8), iterations=3)

    @staticmethod
    def get_histocomparison(
        img_1: np.ndarray,
        img_2: np.ndarray,
        method: Optional[
            Literal[
                "chi_square",
                "correlation",
                "intersection",
                "bhattacharyya",
                "hellinger",
                "chi_square_alternative",
                "kl_divergence",
            ]
        ] = "correlation",
        absolute: Optional[bool] = True,
    ):
        """
        :example:
        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/3.png').astype(np.uint8)
        >>> ImageMixin.get_histocomparison(img_1=img_1, img_2=img_2, method='chi_square_alternative')
        """
        check_if_valid_img(
            source=f"{ImageMixin.get_histocomparison.__name__} img_1", data=img_1
        )
        check_if_valid_img(
            source=f"{ImageMixin.get_histocomparison.__name__} img_2", data=img_2
        )
        check_str(
            name=f"{ImageMixin().get_histocomparison.__name__} method",
            value=method,
            options=list(GeometryEnum.HISTOGRAM_COMPARISON_MAP.value.keys()),
        )
        method = GeometryEnum.HISTOGRAM_COMPARISON_MAP.value[method]
        if absolute:
            return abs(
                cv2.compareHist(
                    img_1.astype(np.float32), img_2.astype(np.float32), method
                )
            )
        else:
            return cv2.compareHist(
                img_1.astype(np.float32), img_2.astype(np.float32), method
            )

    @staticmethod
    def get_contourmatch(img_1: np.ndarray,
                         img_2: np.ndarray,
                         mode: Optional[Literal["all", "exterior"]] = "all",
                         method: Optional[Literal["simple", "none", "l2", "kcos"]] = "simple",
                         canny: Optional[bool] = True) -> float:

        """
        Calculate contour similarity between two images.

        :param np.ndarray img_1: First input image (numpy array).
        :param np.ndarray img_2: Second input image (numpy array).
        :param Optional[Literal['all', 'exterior']] method: Method for contour extraction. Options: 'all' (all contours) or 'exterior' (only exterior contours). Defaults to 'all'.
        :return float: Contour similarity score between the two images. Lower values indicate greater similarity, and higher values indicate greater dissimilarity.

        :example:
        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/3.png').astype(np.uint8)
        >>> ImageMixin.get_contourmatch(img_1=img_1, img_2=img_2, method='exterior')
        """

        check_if_valid_img(
            source=f"{ImageMixin.get_contourmatch.__name__} img_1", data=img_1
        )
        check_if_valid_img(
            source=f"{ImageMixin.get_contourmatch.__name__} img_2", data=img_2
        )
        check_str(
            name=f"{ImageMixin().get_contourmatch.__name__} mode",
            value=mode,
            options=list(GeometryEnum.CONTOURS_MODE_MAP.value.keys()),
        )
        check_str(
            name=f"{ImageMixin.find_contours.__name__} method",
            value=method,
            options=list(GeometryEnum.CONTOURS_RETRIEVAL_MAP.value.keys()),
        )
        if canny:
            img_1 = ImageMixin().canny_edge_detection(img=img_1)
            img_2 = ImageMixin().canny_edge_detection(img=img_2)
        img_1_contours = ImageMixin().find_contours(img=img_1, mode=mode, method=method)
        img_2_contours = ImageMixin().find_contours(img=img_2, mode=mode, method=method)
        return cv2.matchShapes(
            img_1_contours[0], img_2_contours[0], cv2.CONTOURS_MATCH_I1, 0.0
        )

    @staticmethod
    def slice_shapes_in_img(
        img: Union[np.ndarray, Tuple[cv2.VideoCapture, int]],
        geometries: List[Union[Polygon, np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Slice regions of interest (ROIs) from an image based on provided shapes.

        .. note::
           Use for slicing one or several static geometries from a single image. If you have several images,
           and shifting geometries across images, consider ``simba.mixins.image_mixin.ImageMixin.slice_shapes_in_imgs``
           which uses CPU multiprocessing.

        :param Union[np.ndarray, Tuple[cv2.VideoCapture, int]] img: Either an image in numpy array format OR a tuple with cv2.VideoCapture object and the frame index.
        :param List[Union[Polygon, np.ndarray]] img: A list of shapes either as vertices in a numpy array, or as shapely Polygons.
        :returns List[np.ndarray]: List of sliced ROIs from the input image.

        >>> img = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/img_comparisons_4/1.png')
        >>> img_video = cv2.VideoCapture('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1.mp4')
        >>> data_path = '/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/csv/outlier_corrected_movement_location/Example_1.csv'
        >>> data = pd.read_csv(data_path, nrows=4, usecols=['Nose_x', 'Nose_y']).fillna(-1).values.astype(np.int64)
        >>> shapes = []
        >>> for frm_data in data: shapes.append(GeometryMixin().bodyparts_to_circle(frm_data, 100))
        >>> ImageMixin().slice_shapes_in_img(img=(img_video, 1), shapes=shapes)
        """

        result = []
        check_instance(
            source=f"{ImageMixin().slice_shapes_in_img.__name__} img",
            instance=img,
            accepted_types=(tuple, np.ndarray),
        )
        check_instance(
            source=f"{ImageMixin().slice_shapes_in_img.__name__} shapes",
            instance=geometries,
            accepted_types=list,
        )
        for shape_cnt, shape in enumerate(geometries):
            check_instance(
                source=f"{ImageMixin().slice_shapes_in_img.__name__} shapes {shape_cnt}",
                instance=shape,
                accepted_types=(Polygon, np.ndarray),
            )
        if isinstance(img, tuple):
            check_instance(
                source=f"{ImageMixin().slice_shapes_in_img.__name__} img tuple first entry",
                instance=img[0],
                accepted_types=cv2.VideoCapture,
            )
            frm_cnt = int(img[0].get(cv2.CAP_PROP_FRAME_COUNT))
            check_int(
                name=f"{ImageMixin().slice_shapes_in_img.__name__} video frame count",
                value=img[1],
                max_value=frm_cnt,
                min_value=0,
            )
            img[0].set(1, img[1])
            _, img = img[0].read()
        corrected_shapes = []
        for shape in geometries:
            if isinstance(shape, np.ndarray):
                shape[shape < 0] = 0
                corrected_shapes.append(shape)
            else:
                shape = np.array(shape.exterior.coords).astype(np.int64)
                shape[shape < 0] = 0
                corrected_shapes.append(shape)
        shapes = corrected_shapes
        del corrected_shapes
        for shape_cnt, shape in enumerate(shapes):
            x, y, w, h = cv2.boundingRect(shape)
            roi_img = img[y : y + h, x : x + w].copy()
            mask = np.zeros_like(roi_img, np.uint8)
            cv2.drawContours(
                mask, [shape - (x, y)], -1, (255, 255, 255), -1, cv2.LINE_AA
            )
            result.append(cv2.bitwise_and(roi_img, mask))
        return result

    @staticmethod
    def canny_edge_detection(
        img: np.ndarray,
        threshold_1: int = 30,
        threshold_2: int = 200,
        aperture_size: int = 3,
        l2_gradient: bool = False,
    ) -> np.ndarray:
        """
        Apply Canny edge detection to the input image.
        """
        check_if_valid_img(source=f"{ImageMixin.img_moments.__name__}", data=img)
        check_int(
            name=f"{ImageMixin.img_moments.__name__} threshold_1",
            value=threshold_1,
            min_value=1,
        )
        check_int(
            name=f"{ImageMixin.img_moments.__name__} threshold_2",
            value=threshold_2,
            min_value=1,
        )
        check_int(
            name=f"{ImageMixin.img_moments.__name__} aperture_size",
            value=aperture_size,
            min_value=1,
        )
        if len(img.shape) >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(
            img,
            threshold1=threshold_1,
            threshold2=threshold_2,
            apertureSize=aperture_size,
            L2gradient=l2_gradient,
        )

    @staticmethod
    def img_moments(img: np.ndarray, hu_moments: Optional[bool] = False) -> np.ndarray:
        """
        Compute image moments.

        :param np.ndarray img: The input image.
        :param  Optional[bool] img: If True, returns the 7 Hu moments. Else, returns the moments.
        :returns np.ndarray: A 24x1 2d-array if hu_moments is False, 7x1 2d-array if hu_moments is True.

        :example:
        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
        >>> ImageMixin.img_moments(img=img_1, hu_moments=True)
        >>> [[ 1.01270313e-03], [ 8.85983106e-10], [ 4.67680675e-13], [ 1.00442018e-12], [-4.64181508e-25], [-2.49036749e-17], [ 5.08375216e-25]]
        """
        check_if_valid_img(source=f"{ImageMixin.img_moments.__name__}", data=img)
        if len(img.shape) >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not hu_moments:
            return np.array(list(cv2.moments(img).values())).reshape(-1, 1)
        else:
            return cv2.HuMoments(cv2.moments(img))

    @staticmethod
    def find_contours(
        img: np.ndarray,
        mode: Optional[Literal["all", "exterior"]] = "all",
        method: Optional[Literal["simple", "none", "l1", "kcos"]] = "simple",
    ) -> np.ndarray:
        """
        Find contours in the input image.

        :param np.ndarray img: Input image as a NumPy array.
        :param Optional[Literal['all', 'exterior']] img: Contour retrieval mode. E.g., which contours should be kept. Default is 'all'.
        :param Optional[Literal['simple', 'none', 'l1', 'kcos']]: Contour approximation method. Default is 'simple'.
        """

        check_if_valid_img(source=f"{ImageMixin.find_contours.__name__} img", data=img)
        check_str(
            name=f"{ImageMixin.find_contours.__name__} mode",
            value=mode,
            options=list(GeometryEnum.CONTOURS_MODE_MAP.value.keys()),
        )
        check_str(
            name=f"{ImageMixin.find_contours.__name__} method",
            value=method,
            options=list(GeometryEnum.CONTOURS_RETRIEVAL_MAP.value.keys()),
        )
        mode = GeometryEnum.CONTOURS_MODE_MAP.value[mode]
        method = GeometryEnum.CONTOURS_RETRIEVAL_MAP.value[method]
        if len(img.shape) >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if mode in [0, 1]:
            return cv2.findContours(img, mode, method)[1]
        else:
            cnts, hierarchy = cv2.findContours(img, mode, method)[-2:]  # TODO
            interior_contours = []
            for i in range(len(cnts)):
                if (
                    hierarchy[0][i][3] == -1
                ):  # Contour with no parent (interior contour)
                    interior_contours.append(cnts[i])

    @staticmethod
    def orb_matching_similarity_(img_1: np.ndarray,
                                 img_2: np.ndarray,
                                 method: Literal["knn", "match", "radius"] = "knn",
                                 mask: Optional[np.ndarray] = None,
                                 threshold: Optional[int] = 0.75) -> int:


        """Perform ORB feature matching between two sets of images.

        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/10.png').astype(np.uint8)
        >>> ImageMixin().orb_matching_similarity_(img_1=img_1, img_2=img_2, method='radius')
        >>> 4
        """

        kp1, des1 = cv2.ORB_create().detectAndCompute(img_1, mask)
        kp2, des2 = cv2.ORB_create().detectAndCompute(img_2, mask)
        sliced_matches = None
        if method == "knn":
            matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
            sliced_matches = [m for m, n in matches if m.distance < threshold * n.distance]
        if method == "match":
            matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
            sliced_matches = [match for match in matches if match.distance <= threshold]
        if method == "radius":
            matches = cv2.BFMatcher().radiusMatch(des1, des2, maxDistance=threshold)
            sliced_matches = [item for sublist in matches for item in sublist]
        return len(sliced_matches)

    @staticmethod
    def _template_matching_cpu_helper(
        data: np.ndarray, video_path: Union[str, os.PathLike], target_frm: np.ndarray
    ):
        """Helper called from ``simba.mixins.image_mixin.ImageMixins.template_matching_cpu()``"""
        cap = cv2.VideoCapture(video_path)
        start, end, current = data[0], data[-1], data[0]
        cap.set(1, start)
        results = {}
        while current < end:
            print(f"Processing frame {current}...")
            _, img = cap.read()
            result = cv2.matchTemplate(img, target_frm, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            results[current] = {"p": np.max(result), "loc": max_loc}
            current += 1
        return results

    @staticmethod
    def template_matching_cpu(
        video_path: Union[str, os.PathLike],
        img: np.ndarray,
        core_cnt: Optional[int] = -1,
        return_img: Optional[bool] = False,
    ) -> Tuple[int, dict, Union[None, np.ndarray]]:
        """
        Perform template matching on CPU using multiprocessing for parallelization.

        E.g., having a cropped image, find the image and frame number in a video it most likely has been cropped from.

        :param Union[str, os.PathLike] video_path: Path to the video file on disk.
        :param np.ndarray img: Template image for matching. E.g., a cropped image from ``video_path``.
        :param Optional[int] core_cnt: Number of CPU cores to use for parallel processing. Default is -1 (max available cores).
        :param Optional[bool] return_img: Whether to return the annotated best match image with rectangle around matched template area. Default is False.
        :returns Tuple[ int, dict, Union[None, np.ndarray]]: A tuple containing: (i) int: frame index of the frame with the best match.
                 (ii) dict: Dictionary containing results (probability and match location) for each frame.
                 (iii) Union[None, np.ndarray]: Annotated image with rectangles around matches (if return_img is True), otherwise None.

        :example:
        >>> img = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Screenshot 2024-01-17 at 12.45.55 PM.png')
        >>> results = ImageMixin().template_matching_cpu(video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Together_1.avi', img=img, return_img=True)
        """

        results, found_img = [], None
        check_if_valid_img(data=img)
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        frame_cnt = get_video_meta_data(video_path=video_path)["frame_count"]
        frm_idx = np.arange(0, frame_cnt + 1)
        chunk_size = len(frm_idx) // core_cnt
        remainder = len(frm_idx) % core_cnt
        split_frm_idx = [
            frm_idx[
                i * chunk_size
                + min(i, remainder) : (i + 1) * chunk_size
                + min(i + 1, remainder)
            ]
            for i in range(core_cnt)
        ]
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                ImageMixin()._template_matching_cpu_helper,
                video_path=video_path,
                target_frm=img,
            )
            for cnt, result in enumerate(
                pool.imap(constants, split_frm_idx, chunksize=1)
            ):
                results.append(result)
        pool.terminate()
        pool.join()
        results = dict(ChainMap(*results))

        max_value, max_frm = -np.inf, None
        for k, v in results.items():
            if v["p"] > max_value:
                max_value = v["p"]
                max_frm = k

        if return_img:
            h, w, _ = img.shape
            found_img = read_frm_of_video(video_path=video_path, frame_index=max_frm)
            loc = results[max_frm]["loc"]
            found_img = cv2.rectangle(
                found_img,
                (int(loc[0]), int(loc[1])),
                (int(loc[0]) + w, int(loc[1] + h)),
                (0, 255, 0),
                2,
            )
        return max_frm, results, found_img

    def template_matching_gpu(self):
        # TODO
        pass

    @staticmethod
    def img_to_bw(
        img: np.ndarray,
        lower_thresh: Optional[int] = 20,
        upper_thresh: Optional[int] = 250,
        invert: Optional[bool] = True,
    ) -> np.ndarray:
        """
        Convert an image to black and white (binary).

        .. image:: _static/img/img_to_bw.png
           :width: 600
           :align: center

        .. note::
           If converting multiple images from colour to black and white, consider ``simba.mixins.image_mixin.ImageMixin.img_stack_to_bw()``


        :param np.ndarray img: Input image as a NumPy array.
        :param Optional[int] lower_thresh: Lower threshold value for binary conversion. Pixels below this value become black. Default is 20.
        :param Optional[int] upper_thresh: Upper threshold value for binary conversion. Pixels above this value become white. Default is 250.
        :param Optional[bool] invert: Flag indicating whether to invert the binary image (black becomes white and vice versa). Default is True.
        :return np.ndarray: Binary black and white image.
        """
        check_if_valid_img(data=img, source=ImageMixin().img_to_bw.__name__)
        check_int(
            name=f"{ImageMixin().img_to_bw.__name__} lower_thresh",
            value=lower_thresh,
            max_value=255,
            min_value=1,
        )
        check_int(
            name=f"{ImageMixin().img_to_bw.__name__} upper_thresh",
            value=upper_thresh,
            max_value=255,
            min_value=1,
        )
        if len(img) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not invert:
            return cv2.threshold(img, lower_thresh, upper_thresh, cv2.THRESH_BINARY)[1]
        else:
            return ~cv2.threshold(img, lower_thresh, upper_thresh, cv2.THRESH_BINARY)[1]

    @staticmethod
    @jit(nopython=True)
    def img_stack_to_bw(
        imgs: np.ndarray, lower_thresh: int, upper_thresh: int, invert: bool
    ):
        """
        Convert a stack of color images into black and white format.

        .. note::
           If converting a single image, consider ``simba.mixins.image_mixin.ImageMixin.img_to_bw()``

        :param np.ndarray img: 4-dimensional array of color images.
        :param Optional[int] lower_thresh: Lower threshold value for binary conversion. Pixels below this value become black. Default is 20.
        :param Optional[int] upper_thresh: Upper threshold value for binary conversion. Pixels above this value become white. Default is 250.
        :param Optional[bool] invert: Flag indicating whether to invert the binary image (black becomes white and vice versa). Default is True.
        :return np.ndarray: 4-dimensional array with black and white image.

        :example:
        >>> imgs = ImageMixin.read_img_batch_from_video(video_path='/Users/simon/Downloads/3A_Mouse_5-choice_MouseTouchBasic_a1.mp4', start_frm=0, end_frm=100)
        >>> imgs = np.stack(imgs.values(), axis=0)
        >>> bw_imgs = ImageMixin.img_stack_to_bw(imgs=imgs, upper_thresh=255, lower_thresh=20, invert=False)
        """

        results = np.full((imgs.shape[:3]), np.nan)
        for cnt in range(imgs.shape[0]):
            arr = imgs[cnt]
            m, n, _ = arr.shape
            img_mean = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    total = 0.0
                    for k in range(arr.shape[2]):
                        total += arr[i, j, k]
                    img_mean[i, j] = total / arr.shape[2]
            img = np.where(img_mean < lower_thresh, 0, img_mean)
            img = np.where(img > upper_thresh, 1, img)
            if invert:
                img = 1 - img
            results[cnt] = img
        return results

    @staticmethod
    def segment_img_horizontal(
        img: np.ndarray,
        pct: int,
        lower: Optional[bool] = True,
        both: Optional[bool] = False,
    ) -> np.ndarray:
        """
        Segment a horizontal part of the input image.

        This function segments either the lower, upper, or both lower and upper part of the input image based on the specified percentage.

        .. image:: _static/img/segment_img_horizontal.png
           :width: 800
           :align: center

        :param np.ndarray img: Input image as a NumPy array.
        :param int pct: Percentage of the image to be segmented. If `lower` is True, it represents the lower part; if False, it represents the upper part.
        :param Optional[bool] lower: Flag indicating whether to segment the lower part (True) or upper part (False) of the image. Default is True.
        :param Optional[bool] both: If True, **removes** both the upper pct and lower pct and keeps middle part.
        :return np.array: Segmented part of the image.

        :example:
        >>> img = cv2.imread('/Users/simon/Desktop/test.png')
        >>> img = ImageMixin.segment_img_horizontal(img=img, pct=10, both=True)
        """

        check_if_valid_img(
            data=img, source=ImageMixin().segment_img_horizontal.__name__
        )
        check_int(
            name=f"{ImageMixin().segment_img_horizontal.__name__} pct",
            value=pct,
            min_value=1,
            max_value=99,
        )
        sliced_height = int(img.shape[0] * pct / 100)
        if both:
            return img[sliced_height : img.shape[0] - sliced_height, :]
        elif lower:
            return img[img.shape[0] - sliced_height :, :]
        else:
            return img[:sliced_height, :]

    @staticmethod
    @jit(nopython=True, parallel=True)
    def segment_img_stack_horizontal(
        imgs: np.ndarray, pct: int, lower: bool, both: bool
    ) -> np.ndarray:
        """
        Segment a horizontal part of all images in stack.

        :example:
        >>> imgs = ImageMixin.read_img_batch_from_video(video_path='/Users/simon/Downloads/3A_Mouse_5-choice_MouseTouchBasic_a1.mp4', start_frm=0, end_frm=400)
        >>> imgs = np.stack(imgs.values(), axis=0)
        >>> sliced_imgs = ImageMixin.segment_img_stack_horizontal(imgs=imgs, pct=50, lower=True, both=False)
        """
        results = []
        for cnt in range(imgs.shape[0]):
            img = imgs[cnt]
            sliced_height = int(img.shape[0] * pct / 100)
            if both:
                sliced_img = img[sliced_height : img.shape[0] - sliced_height, :]
            elif lower:
                sliced_img = img[img.shape[0] - sliced_height :, :]
            else:
                sliced_img = img[:sliced_height, :]
            results.append(sliced_img)
        stacked_results = np.full(
            (len(results), results[0].shape[0], results[0].shape[1], 3), np.nan
        )
        for i in prange(len(results)):
            stacked_results[i] = results[i]
        return results

    @staticmethod
    def segment_img_vertical(
        img: np.ndarray,
        pct: int,
        left: Optional[bool] = True,
        both: Optional[bool] = False,
    ) -> np.ndarray:
        """
        Segment a vertical part of the input image.

        This function segments either the left, right or both the left and right part of  input image based on the specified percentage.

        .. image:: _static/img/segment_img_vertical.png
           :width: 800
           :align: center

        :param np.ndarray img: Input image as a NumPy array.
        :param int pct: Percentage of the image to be segmented. If `lower` is True, it represents the lower part; if False, it represents the upper part.
        :param Optional[bool] lower: Flag indicating whether to segment the lower part (True) or upper part (False) of the image. Default is True.
        :param Optional[bool] both: If True, **removes** both the left pct and right pct and keeps middle part.
        :return np.array: Segmented part of the image.
        """

        check_if_valid_img(data=img, source=ImageMixin().segment_img_vertical.__name__)
        check_int(
            name=f"{ImageMixin().segment_img_vertical.__name__} pct",
            value=pct,
            min_value=1,
            max_value=99,
        )
        sliced_width = int(img.shape[1] * pct / 100)
        if both:
            return img[:, sliced_width : img.shape[1] - sliced_width]
        elif left:
            return img[:, :sliced_width]
        else:
            return img[:, img.shape[1] - sliced_width :]

    @staticmethod
    def add_img_border_and_flood_fill(
        img: np.array, invert: Optional[bool] = False, size: Optional[int] = 1
    ) -> np.ndarray:
        """
        Add a border to the input image and perform flood fill.

        E.g., Used to remove any black pixel areas connected to the border of the image. Used to remove noise
        if noise is defined as being connected to the edges of the image.

        .. image:: _static/img/add_img_border_and_flood_fill.png
           :width: 400
           :align: center

        :param np.ndarray img: Input image as a numpy array.
        :param Optional[bool] invert: If false, make black border and floodfill black pixels with white. If True, make white border and floodfill white pixels with black. Default False.
        :param Optional[bool] size: Size of border. Default 1 pixel.
        """

        check_if_valid_img(
            data=img, source=ImageMixin().add_img_border_and_flood_fill.__name__
        )
        check_int(
            name=f"{ImageMixin().add_img_border_and_flood_fill.__name__} size",
            value=size,
            min_value=1,
        )
        if len(img.shape) > 2:
            raise InvalidInputError(
                msg="Floodfill requires 2d image",
                source=ImageMixin().add_img_border_and_flood_fill.__name__,
            )
        if not invert:
            img = cv2.copyMakeBorder(
                img, size, size, size, size, cv2.BORDER_CONSTANT, value=0
            )
            mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
            img = cv2.floodFill(
                img, mask=mask, seedPoint=(0, 0), newVal=(255, 255, 255)
            )[1]

        else:
            img = cv2.copyMakeBorder(
                img, size, size, size, size, cv2.BORDER_CONSTANT, value=255
            )
            mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
            img = cv2.floodFill(img, mask=mask, seedPoint=(0, 0), newVal=(0, 0, 0))[1]

        return img[size:-size, size:-size]

    @staticmethod
    def _image_reader_helper(img_paths: List[str]):
        """Multiprocessing helper for ``ImageMixin().read_all_img_in_dir``"""
        results = {}
        for img_path in img_paths:
            results[get_fn_ext(filepath=img_path)[1]] = cv2.imread(img_path)
        return results

    @staticmethod
    def read_all_img_in_dir(
        dir: Union[str, os.PathLike], core_cnt: Optional[int] = -1
    ) -> Dict[str, np.ndarray]:
        """
        Helper to read in all images within a directory using multiprocessing.
        Returns a dictionary with the image name as key and the images in array format as values.

        :example:
        >>> imgs = ImageMixin().read_all_img_in_dir(dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/Together_4_cropped_frames')
        """
        check_if_dir_exists(in_dir=dir)
        file_paths = find_files_of_filetypes_in_directory(
            directory=dir,
            extensions=list(Options.ALL_IMAGE_FORMAT_OPTIONS.value),
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        chunk_size = len(file_paths) // core_cnt
        file_paths = [
            file_paths[
                i * chunk_size
                + min(i, len(file_paths) % core_cnt) : (i + 1) * chunk_size
                + min(i + 1, len(file_paths) % core_cnt)
            ]
            for i in range(core_cnt)
        ]
        imgs = {}
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, result in enumerate(
                pool.imap(ImageMixin()._image_reader_helper, file_paths, chunksize=1)
            ):
                imgs.update(result)
        pool.join()
        pool.terminate()
        return imgs

    @staticmethod
    @njit([(uint8[:, :, :, :], uint8[:, :, :, :]), (uint8[:, :, :], uint8[:, :, :])])
    def img_stack_mse(imgs_1: np.ndarray, imgs_2: np.ndarray) -> np.ndarray:
        """
        Pairwise comparison of images in two stacks of equal length using mean squared errors.

        .. note::
           Useful for noting subtle changes, each imgs_2 equals imgs_1 with images shifted by 1. Images has to be in uint8 format.
           Also see ``img_sliding_mse``.


        :param np.ndarray imgs_1: First three (non-color) or four (color) dimensional stack of images in array format.
        :param np.ndarray imgs_1: Second three (non-color) or four (color) dimensional stack of images in array format.
        :return np.ndarray: Array of size len(imgs_1) comparing ``imgs_1`` and ``imgs_2`` at each index using mean squared errors at each pixel location.

        :example:
        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/10.png').astype(np.uint8)
        >>> imgs_1 = np.stack((img_1, img_2)); imgs_2 = np.stack((img_2, img_2))
        >>> ImageMixin.img_stack_mse(imgs_1=imgs_1, imgs_2=imgs_2)
        >>> [637,   0]
        >>> imgs = ImageMixin().read_all_img_in_dir(dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/Together_4_cropped_frames')
        >>> imgs_1 = np.stack(imgs.values())
        >>> imgs_2 = np.roll(imgs_1,-1, axis=0)
        >>> mse = ImageMixin().img_stack_mse(imgs_1=imgs_1, imgs_2=imgs_1)
        """

        results = np.full((imgs_1.shape[0]), np.nan)
        for i in range(imgs_1.shape[0]):
            results[i] = np.sum((imgs_1[i] - imgs_2[i]) ** 2) / float(
                imgs_1[i].shape[0] * imgs_2[i].shape[1]
            )
        return results.astype(np.int64)

    @staticmethod
    @njit([(uint8[:, :, :, :], int64), (uint8[:, :, :], int64)])
    def img_sliding_mse(imgs: np.ndarray, slide_size: int = 1) -> np.ndarray:
        """Pairwise comparison of images in sliding windows using mean squared errors

        :example:
        >>> imgs = ImageMixin().read_all_img_in_dir(dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/Together_4_cropped_frames')
        >>> imgs = np.stack(imgs.values())
        >>> mse = ImageMixin().img_sliding_mse(imgs=imgs, slide_size=2)
        """

        results = np.full((imgs.shape[0]), 0)
        for i in prange(slide_size, imgs.shape[0]):
            results[i] = np.sum((imgs[i - slide_size] - imgs[i]) ** 2) / float(
                imgs[i - slide_size].shape[0] * imgs[i].shape[1]
            )
        return results.astype(int64)

    @staticmethod
    def _read_img_batch_from_video_helper(
        frm_idx: np.ndarray, video_path: Union[str, os.PathLike]
    ):
        """Multiprocess helper used by read_img_batch_from_video to read in images from video file."""
        start_idx, end_frm, current_frm = frm_idx[0], frm_idx[-1] + 1, frm_idx[0]
        results = {}
        cap = cv2.VideoCapture(video_path)
        cap.set(1, current_frm)
        while current_frm < end_frm:
            results[current_frm] = cap.read()[1]
            current_frm += 1
        return results

    @staticmethod
    def read_img_batch_from_video(
        video_path: Union[str, os.PathLike],
        start_frm: int,
        end_frm: int,
        core_cnt: Optional[int] = -1,
    ) -> Dict[int, np.ndarray]:
        """
        Read a batch of frames from a video file. This method reads frames from a specified range of frames within a video file using multiprocessing.

        :param Union[str, os.PathLike] video_path: Path to the video file.
        :param int start_frm: Starting frame index.
        :param int end_frm: Ending frame index.
        :param Optionalint] core_cnt: Number of CPU cores to use for parallel processing. Default is -1, indicating using all available cores.
        :param Optional[bool] greyscale: If True, reads the images as greyscale. If False, then as original color scale. Default: True.
        :returns Dict[int, np.ndarray]: A dictionary containing frame indices as keys and corresponding frame arrays as values.

        :example:
        >>> ImageMixin().read_img_batch_from_video(video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Together_1.avi', start_frm=0, end_frm=50)
        """
        check_file_exist_and_readable(file_path=video_path)
        video_meta_data = get_video_meta_data(video_path=video_path)
        check_int(
            name=ImageMixin().__class__.__name__,
            value=start_frm,
            min_value=0,
            max_value=video_meta_data["frame_count"],
        )
        check_int(
            name=ImageMixin().__class__.__name__,
            value=end_frm,
            min_value=0,
            max_value=video_meta_data["frame_count"],
        )
        check_int(name=ImageMixin().__class__.__name__, value=core_cnt, min_value=-1)
        if core_cnt < 0:
            core_cnt = multiprocessing.cpu_count()
        if end_frm <= start_frm:
            FrameRangeError(
                msg=f"Start frame ({start_frm}) has to be before end frame ({end_frm})",
                source=ImageMixin().__class__.__name__,
            )
        frm_lst = np.array_split(np.arange(start_frm, end_frm + 1), core_cnt)
        results = {}
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                ImageMixin()._read_img_batch_from_video_helper, video_path=video_path
            )
            for cnt, result in enumerate(pool.imap(constants, frm_lst, chunksize=1)):
                results.update(result)
        pool.join()
        pool.terminate()
        return results

    @staticmethod
    def img_emd(
        imgs: List[np.ndarray] = None,
        img_1: Optional[np.ndarray] = None,
        img_2: Optional[np.ndarray] = None,
        lower_bound: Optional[float] = 0.5,
    ):
        """
        Compute Wasserstein distance between two images represented as numpy arrays.

        :example:
        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1_frames/24.png', 0).astype(np.float32)
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1_frames/1984.png', 0).astype(np.float32)
        >>> img_emd(img_1=img_1, img_2=img_3, lower_bound=0.5)
        >>> 10.658767700195312
        """
        check_float(
            name=f"{ImageMixin.img_emd.__name__} lower bound",
            min_value=0.0,
            value=lower_bound,
        )
        if (img_1 is None and img_1 is None and imgs is None) or (
            img_1 is not None and img_2 is None
        ):
            raise InvalidInputError(
                msg="Pass img_1 and img_2 OR imgs", source=ImageMixin.__class__.__name__
            )
        if (img_1 is not None) and (img_2 is not None):
            check_if_valid_img(data=img_1, source=ImageMixin.img_emd.__name__)
            check_if_valid_img(data=img_2, source=ImageMixin.img_emd.__name__)
            if img_1.ndim > 2:
                img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            if img_2.ndim > 2:
                img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
        else:
            check_valid_lst(
                data=imgs,
                source=ImageMixin.img_emd.__name__,
                valid_dtypes=(np.ndarray,),
                exact_len=2,
            )
            check_if_valid_img(data=imgs[0], source=ImageMixin.img_emd.__name__)
            check_if_valid_img(data=imgs[1], source=ImageMixin.img_emd.__name__)
            if imgs[0].ndim > 2:
                img_1 = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
            if imgs[1].ndim > 2:
                img_2 = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY)
        return cv2.EMD(
            img_1.astype(np.float32),
            img_2.astype(np.float32),
            cv2.DIST_L2,
            lowerBound=lower_bound,
        )[0]

    @staticmethod
    @jit(nopython=True)
    def img_matrix_mse(imgs: np.ndarray) -> np.ndarray:
        """
        Compute the mean squared error (MSE) matrix table for a stack of images.

        This function calculates the MSE between each pair of images in the input array
        and returns a symmetric matrix where each element (i, j) represents the MSE
        between the i-th and j-th images. Useful for image similarities and anomalities.

        :param np.ndarray imgs: A stack of images represented as a numpy array.
        :return np.ndarray: The MSE matrix table.

        :example:
        >>> imgs = ImageMixin().read_img_batch_from_video(video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Together_1.avi', start_frm=0, end_frm=50)
        >>> imgs = np.stack(list(imgs.values()))
        >>> ImageMixin().img_matrix_mse(imgs=imgs)
        """
        results = np.full((imgs.shape[0], imgs.shape[0]), 0.0)
        for i in prange(imgs.shape[0]):
            for j in range(i + 1, imgs.shape[0]):
                val = np.sum((imgs[i] - imgs[j]) ** 2) / float(
                    imgs[i].shape[0] * imgs[j].shape[1]
                )
                results[i, j] = val
                results[j, i] = val
        return results.astype(np.int32)

    @staticmethod
    @njit("(uint8[:, :, :, :],)", fastmath=True)
    def img_stack_to_greyscale(imgs: np.ndarray):
        """
        Jitted conversion of a 4D stack of color images (RGB format) to grayscale.

        .. image:: _static/img/img_stack_to_greyscale.png
           :width: 600
           :align: center

        :parameter np.ndarray imgs: A 4D array representing color images. It should have the shape (num_images, height, width, 3) where the last dimension represents the color channels (R, G, B).
        :returns np.ndarray: A 3D array containing the grayscale versions of the input images. The shape of the output array is (num_images, height, width).

        :example:
        >>> imgs = ImageMixin().read_img_batch_from_video( video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Together_1.avi', start_frm=0, end_frm=100)
        >>> imgs = np.stack(list(imgs.values()))
        >>> imgs_gray = ImageMixin.img_stack_to_greyscale(imgs=imgs)
        """
        results = np.full((imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.nan).astype(
            np.uint8
        )
        for i in prange(imgs.shape[0]):
            vals = (
                0.07 * imgs[i][:, :, 2]
                + 0.72 * imgs[i][:, :, 1]
                + 0.21 * imgs[i][:, :, 0]
            )
            results[i] = vals.astype(np.uint8)
        return results

    @staticmethod
    def _slice_shapes_in_imgs_array_helper(
        data: Tuple[np.ndarray, np.ndarray]
    ) -> List[np.ndarray]:
        """
        Private multiprocess helper called from ``simba.mixins.image_mixin.ImageMixins.slice_shapes_in_imgs()`` to slice shapes from
        an array of images.
        """
        img, in_shapes = data[0], data[1]
        shapes, results = [], []
        for shape in in_shapes:
            shape = np.array(shape.exterior.coords).astype(np.int64)
            shape[shape < 0] = 0
            shapes.append(shape)
        for shape_cnt, shape in enumerate(shapes):
            x, y, w, h = cv2.boundingRect(shape)
            roi_img = img[y : y + h, x : x + w].copy()
            mask = np.zeros_like(roi_img, np.uint8)
            cv2.drawContours(
                mask, [shape - (x, y)], -1, (255, 255, 255), -1, cv2.LINE_AA
            )
            results.append(cv2.bitwise_and(roi_img, mask))
        return results

    @staticmethod
    def pad_img_stack(
        image_dict: Dict[int, np.ndarray], pad_value: Optional[int] = 0
    ) -> Dict[int, np.ndarray]:
        """
        Pad images in a dictionary stack to have the same dimensions (the same dimension is represented by the largest image in the stack)

        :param Dict[int, np.ndarray] image_dict: A dictionary mapping integer keys to numpy arrays representing images.
        :param Optional[int] pad_value: The value used for padding. Defaults to 0 (black)
        :return Dict[int, np.ndarray]: A dictionary mapping integer keys to numpy arrays representing padded images.
        """
        check_instance(
            source=ImageMixin.pad_img_stack.__name__,
            instance=image_dict,
            accepted_types=(dict,),
        )
        check_int(
            name=f"{ImageMixin.pad_img_stack.__name__} pad_value",
            value=pad_value,
            max_value=255,
            min_value=0,
        )
        max_height = max(image.shape[0] for image in image_dict.values())
        max_width = max(image.shape[1] for image in image_dict.values())
        padded_images = {}
        for key, image in image_dict.items():
            pad_height = max_height - image.shape[0]
            pad_width = max_width - image.shape[1]
            padded_image = np.pad(
                image,
                ((0, pad_height), (0, pad_width), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )
            padded_images[key] = padded_image
        return padded_images

    @staticmethod
    def img_stack_to_video(
        imgs: Dict[int, np.ndarray],
        save_path: Union[str, os.PathLike],
        fps: int,
        verbose: Optional[bool] = True,
    ):
        """
        Convert a dictionary of images into a video file.

        .. note::
           The input dict can be greated with ImageMixin().slice_shapes_in_imgs()

        :param Dict[int, np.ndarray] imgs: A dictionary containing frames of the video, where the keys represent frame indices and the values are numpy arrays representing the images.
        :param Union[str, os.PathLike] save_path: The path to save the output video file.
        :param int fps: Frames per second (FPS) of the output video.
        :param Optional[bool] verbose: If True, prints progress messages. Defaults to True.
        """

        check_instance(
            source=ImageMixin.img_stack_to_video.__name__,
            instance=imgs,
            accepted_types=(dict,),
        )
        img_sizes = set()
        for k, v in imgs.items():
            img_sizes.add(v.shape)
        if len(list(img_sizes)) > 1:
            imgs = ImageMixin.pad_img_stack(imgs)
        imgs = np.stack(imgs.values())
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        writer = cv2.VideoWriter(
            save_path, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0])
        )
        for i in range(imgs.shape[0]):
            if verbose:
                print(f"Writing img {i + 1}...")
            writer.write(imgs[i])
        writer.release()
        stdout_success(msg=f"Video {save_path} complete")

    @staticmethod
    def _slice_shapes_in_video_file_helper(
        data: np.ndarray, video_path: Union[str, os.PathLike], verbose: bool
    ):
        cap = cv2.VideoCapture(video_path)
        start_frm, current_frm, end_frm = data[0][0], data[0][0], data[-1][0]
        cap.set(1, start_frm)
        results = {}
        idx_cnt = 0
        while current_frm <= end_frm:
            if verbose:
                print(f"Processing frame {current_frm}...")
            img = cap.read(current_frm)[1].astype(np.uint8)
            shape = np.array(data[idx_cnt][1].exterior.coords).astype(np.int64)
            shape[shape < 0] = 0
            x, y, w, h = cv2.boundingRect(shape)
            roi_img = img[y : y + h, x : x + w].copy()
            mask = np.zeros_like(roi_img, np.uint8)
            cv2.drawContours(
                mask, [shape - (x, y)], -1, (255, 255, 255), -1, cv2.LINE_AA
            )
            results[current_frm] = cv2.bitwise_and(roi_img, mask).astype(np.uint8)
            current_frm += 1
            idx_cnt += 1
        return results

    def slice_shapes_in_imgs(
        self,
        imgs: Union[np.ndarray, os.PathLike],
        shapes: Union[np.ndarray, List[Polygon]],
        core_cnt: Optional[int] = -1,
        verbose: Optional[bool] = False,
    ) -> Dict[int, np.ndarray]:
        """
        Slice regions from a stack of images or a video file, where the regions are based on defined shapes. Uses multiprocessing.

        For example, given a stack of N images, and N*X geometries representing the region around the animal body-part(s),
        slice out the X geometries from each of the N images and return the sliced areas.

        .. image:: _static/img/slice_shapes_in_imgs_4.gif
           :width: 400
           :align: center

        .. image:: _static/img/bodyparts_to_circle_1.gif
           :width: 400
           :align: center

        .. image:: _static/img/bodyparts_to_circle_2.gif
           :width: 400
           :align: center

        :example I:
        >>> imgs = ImageMixin().read_img_batch_from_video( video_path='/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1.mp4', start_frm=0, end_frm=10)
        >>> imgs = np.stack(list(imgs.values()))
        >>> imgs_gray = ImageMixin().img_stack_to_greyscale(imgs=imgs)
        >>> data = pd.read_csv('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/csv/outlier_corrected_movement_location/Example_1.csv', nrows=11).fillna(-1)
        >>> nose_array, tail_array = data.loc[0:10, ['Nose_x', 'Nose_y']].values.astype(np.float32), data.loc[0:10, ['Tail_base_x', 'Tail_base_y']].values.astype(np.float32)
        >>> nose_shapes, tail_shapes = [], []
        >>> for frm_data in nose_array: nose_shapes.append(GeometryMixin().bodyparts_to_circle(frm_data, 80))
        >>> for frm_data in tail_array: tail_shapes.append(GeometryMixin().bodyparts_to_circle(frm_data, 80))
        >>> shapes = np.array(np.vstack([nose_shapes, tail_shapes]).T)
        >>> sliced_images = ImageMixin().slice_shapes_in_imgs(imgs=imgs_gray, shapes=shapes)

        :example II:
        >>> video_path = '/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1_clipped.mp4'
        >>> data_path = r'/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/csv/outlier_corrected_movement_location/Example_1_clipped.csv'
        >>> df = pd.read_csv(data_path, usecols=['Nose_x', 'Nose_y', 'Tail_base_x', 'Tail_base_y']).fillna(0).values.astype(int)
        >>> data = df.reshape(len(df), -1, int(df.shape[1]/2))
        >>> geometries = GeometryMixin().multiframe_bodyparts_to_line(data=data, buffer=30, px_per_mm=4.1)
        >>> imgs = ImageMixin().slice_shapes_in_imgs(imgs=video_path, shapes=geometries)


        """
        timer = SimbaTimer(start=True)
        check_int(
            name=f"{ImageMixin().slice_shapes_in_imgs.__name__} core count",
            value=core_cnt,
            min_value=-1,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        check_instance(
            source=ImageMixin().slice_shapes_in_imgs.__name__,
            instance=imgs,
            accepted_types=(np.ndarray, str),
        )
        check_instance(
            source=ImageMixin().slice_shapes_in_imgs.__name__,
            instance=shapes,
            accepted_types=(np.ndarray, list),
        )
        if isinstance(shapes, np.ndarray):
            check_valid_array(
                data=shapes,
                source=ImageMixin().slice_shapes_in_imgs.__name__,
                accepted_ndims=(2,),
                accepted_dtypes=[Polygon],
            )
        else:
            check_valid_lst(
                data=shapes,
                source=ImageMixin().slice_shapes_in_imgs.__name__,
                valid_dtypes=[Polygon],
            )
            shapes = np.array(shapes)
        if isinstance(imgs, np.ndarray):
            check_valid_array(
                data=imgs,
                source=ImageMixin().slice_shapes_in_imgs.__name__,
                accepted_ndims=(4, 3),
            )
            if shapes.shape[0] != imgs.shape[0]:
                raise ArrayError(
                    msg=f"The image array ({imgs.shape[0]}) and shapes array ({shapes.shape[0]}) have unequal length.",
                    source=ImageMixin().slice_shapes_in_imgs.__name__,
                )
        else:
            check_file_exist_and_readable(file_path=imgs)
            video_meta_data = get_video_meta_data(video_path=imgs)
            if shapes.shape[0] != video_meta_data["frame_count"]:
                raise ArrayError(
                    msg=f'The image array ({video_meta_data["frame_count"]}) and shapes array ({shapes.shape[0]}) have unequal length.',
                    source=ImageMixin().slice_shapes_in_imgs.__name__,
                )
        results = []
        if isinstance(imgs, np.ndarray):
            with multiprocessing.Pool(
                core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
            ) as pool:
                for cnt, result in enumerate(
                    pool.imap(
                        self._slice_shapes_in_imgs_array_helper,
                        zip(imgs, shapes),
                        chunksize=1,
                    )
                ):
                    results.append(result)
        else:
            shapes = np.array_split(
                np.column_stack((np.arange(len(shapes)), shapes)), core_cnt
            )
            with multiprocessing.Pool(
                core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
            ) as pool:
                constants = functools.partial(
                    self._slice_shapes_in_video_file_helper,
                    video_path=imgs,
                    verbose=verbose,
                )
                for cnt, result in enumerate(pool.imap(constants, shapes, chunksize=1)):
                    results.append(result)
                results = dict(ChainMap(*results))
        pool.join()
        pool.terminate()
        timer.stop_timer()
        stdout_success(
            msg="Geometry image slicing complete.",
            elapsed_time=timer.elapsed_time_str,
            source=self.__class__.__name__,
        )
        return results

    #
    # def segment_img_horizontal(img: np.ndarray, pct: int, lower: Optional[bool] = True,
    #                            both: Optional[bool] = False


# img = cv2.imread('/Users/simon/Desktop/test.png')
# img = ImageMixin.segment_img_vertical(img=img, pct=20, both=True)
# cv2.imshow('sdsdf', img)
# cv2.waitKey(5000)


# bw_img = ImageMixin.img_to_bw(img=img, invert=False)
# cv2.imshow('sdsdf', bw_img)
# cv2.waitKey(5000)

# imgs = ImageMixin().read_img_batch_from_video( video_path='/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1.mp4', start_frm=0, end_frm=10)
# imgs = np.stack(list(imgs.values()))
# imgs_gray = ImageMixin().img_stack_to_greyscale(imgs=imgs)
# data = pd.read_csv('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/csv/outlier_corrected_movement_location/Example_1.csv', nrows=11).fillna(-1)
# nose_array, tail_array = data.loc[0:10, ['Nose_x', 'Nose_y']].values.astype(np.float32), data.loc[0:10, ['Tail_base_x', 'Tail_base_y']].values.astype(np.float32)
# nose_shapes, tail_shapes = [], []
# from simba.mixins.geometry_mixin import GeometryMixin
# for frm_data in nose_array: nose_shapes.append(GeometryMixin().bodyparts_to_circle(frm_data, 80))
# for frm_data in tail_array: tail_shapes.append(GeometryMixin().bodyparts_to_circle(frm_data, 80))
# shapes = np.array(np.vstack([nose_shapes, tail_shapes]).T)
# sliced_images = ImageMixin().slice_shapes_in_imgs(imgs=imgs_gray, shapes=shapes)

# imgs_.shape
# imgs = ImageMixin().read_all_img_in_dir(dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/Together_4_cropped_frames')
# imgs_1 = np.stack(imgs.values())
# mse = ImageMixin().img_sliding_mse(imgs=imgs_1, slide_size=2)
# #imgs_2 = np.roll(imgs_1,-1, axis=0)
# #mse = ImageMixin().img_mse(imgs_1=imgs_1, imgs_2=imgs_1)
# mse = ImageMixin().img_sliding_mse(imgs=imgs_1, slide_size=2)

# img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
# img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/10.png').astype(np.uint8)
#
# img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
# img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
#
# imgs_1 = np.stack((img_1, img_2))
# imgs_2 = np.stack((img_2, img_2))
# ImageMixin.img_mse(imgs_1=imgs_1, imgs_2=imgs_2)

# ImageMixin.img_mse_test(imgs_1=imgs_1)


# res = ImageMixin.img_moments(img=img_1, hu_moments=True)
# img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
# img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/10.png').astype(np.uint8)
# ImageMixin.get_contourmatch(img_1=img_1, img_2=img_2, mode='exterior')


# img = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/img_comparisons_4/1.png')
# img_video = cv2.VideoCapture('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1.mp4')
# data_path = '/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/csv/outlier_corrected_movement_location/Example_1.csv'
# data = pd.read_csv(data_path, nrows=4, usecols=['Nose_x', 'Nose_y']).fillna(-1).values.astype(np.int64)
# shapes = []
# for frm_data in data: shapes.append(GeometryMixin().bodyparts_to_circle(frm_data, 100))
#
# ImageMixin().slice_shapes_in_img(img=(img_video, 1), shapes=shapes)


# img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/3.png').astype(np.uint8)
# ImageMixin.get_contourmatch(img_1=img_1, img_2=img_2, method='exterior')

# img = cv2.VideoCapture('/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/zebrafish/project_folder/videos/20200730_AB_7dpf_850nm_0003.mp4')
# ImageMixin.slice_shapes_in_img(img=(img, 1))


# img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
# img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/1.png').astype(np.uint8)
# ImageMixin.get_contourmatch(img_1=img_1, img_2=img_2, method='all', canny=True)
#
