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
from numba import float64, int64, jit, njit, prange, uint8
from shapely.geometry import MultiPolygon, Polygon
from skimage.metrics import structural_similarity

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_if_valid_img,
                                check_if_valid_rgb_tuple, check_instance,
                                check_int, check_str, check_valid_array,
                                check_valid_boolean, check_valid_lst,
                                check_valid_tuple, is_img_bw, is_img_greyscale)
from simba.utils.enums import Defaults, Formats, GeometryEnum, Options
from simba.utils.errors import (ArrayError, FFMPEGCodecGPUError,
                                FrameRangeError, InvalidInputError,
                                NotDirectoryError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data,
                                    read_frm_of_video,
                                    read_img_batch_from_video_gpu, write_df)


class ImageMixin(object):
    """
    Methods to slice and compute attributes of images from tracking data and comparing those image attributes across sequential images.

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

        .. seealso::
           For GPU acceleration, see :func:`simba.data_processors.cuda.image.img_stack_brightness`.
           For geometry based brightness, see :func:`simba.mixins.geometry_mixin.GeometryMixin.get_geometry_brightness_intensity`

        :param List[np.ndarray] imgs: List of images as arrays to calculate average brightness intensity within.
        :param Optional[bool] ignore_black: If True, ignores black pixels. If the images are sliced non-rectangular geometric shapes created by ``slice_shapes_in_img``, then pixels that don't belong to the shape has been masked in black.
        :returns: List of floats of size len(imgs) with brightness intensities.
        :rtype: List[float]

        :example:
        >>> img = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
        >>> ImageMixin.brightness_intensity(imgs=[img], ignore_black=False)
        >>> [159.0]
        """
        results = []
        check_instance(source=f"{ImageMixin().brightness_intensity.__name__} imgs", instance=imgs, accepted_types=list)
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
    def gaussian_blur(img: np.ndarray, kernel_size: Optional[Tuple] = (9, 9)) -> np.ndarray:
        """
        Applies a Gaussian blur to an input image using the specified kernel size.

        Gaussian blurring is used to reduce image noise and detail by smoothing the image. It applies a weighted average
        where more importance is given to the central pixels, creating a soft blur effect.

        :param np.ndarray img: Input image as a NumPy array. The image should be a valid 2D (grayscale) or 3D (color) array.
        :param Optional[Tuple] kernel_size: A tuple (height, width) representing the size of the Gaussian kernel. The values must be positive odd integers. Default is (9, 9).
        :return: A NumPy array representing the blurred image with the same dimensions as the input image.
        :rtype: np.ndarray

        """
        check_if_valid_img(data=img, source=ImageMixin.gaussian_blur.__name__)
        check_instance(source=ImageMixin.gaussian_blur.__name__, instance=kernel_size, accepted_types=(tuple,))
        check_valid_lst(data=list(kernel_size), source=ImageMixin.gaussian_blur.__name__, valid_dtypes=(int,), exact_len=2,)
        return cv2.GaussianBlur(img, kernel_size, 0)

    @staticmethod
    def erode(img: np.ndarray,
              kernel_size: Optional[Tuple[int, int]] = (3, 3),
              iterations: Optional[int] = 3) -> np.ndarray:
        """
        Applies morphological erosion to the input image using the specified kernel size and number of iterations.

        :param np.ndarray img: A 2D or 3D NumPy array representing the input image on which erosion will be applied. It should be in the form of a binary or greyscale image.
        :param Optional[Tuple[int, int]] kernel_size: A tuple (width, height) specifying the size of the kernel to be used for erosion. The default kernel size is (3, 3).
        :param Optional[int] iterations: The number of times the erosion operation is applied. The default value is 3.
        :return: A NumPy array of the same shape as the input `img` representing the eroded image.
        :rtype: np.ndarray

        """

        check_if_valid_img(data=img, source=ImageMixin.gaussian_blur.__name__)
        check_instance(source=ImageMixin.gaussian_blur.__name__, instance=kernel_size, accepted_types=(tuple,))
        check_valid_lst(data=list(kernel_size), source=ImageMixin.gaussian_blur.__name__, valid_dtypes=(int,), exact_len=2)
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
            options=tuple(GeometryEnum.HISTOGRAM_COMPARISON_MAP.value.keys()),
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
        :return: Contour similarity score between the two images. Lower values indicate greater similarity, and higher values indicate greater dissimilarity.
        :rtype: float

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
        return cv2.matchShapes(img_1_contours[0], img_2_contours[0], cv2.CONTOURS_MATCH_I1, 0.0)

    @staticmethod
    def slice_shapes_in_img(img: Union[np.ndarray, Tuple[cv2.VideoCapture, int]], geometries: List[Union[Polygon, np.ndarray]]) -> List[np.ndarray]:
        """
        Slice regions of interest (ROIs) from an image based on provided shapes.

        .. note::
           Use for slicing one or several static geometries from a single image. If you have several images,
           and shifting geometries across images, consider ``simba.mixins.image_mixin.ImageMixin.slice_shapes_in_imgs``
           which uses CPU multiprocessing.

        .. seealso::
           For GPU acceleration, see :func:`simba.data_processors.cuda.image.slice_imgs`
           For single core method, see :func:`simba.mixins.image_mixin.ImageMixin.slice_shapes_in_imgs`

        .. video:: _static/img/slice_imgs_gpu.webm
           :width: 800
           :autoplay:
           :loop:

        :param Union[np.ndarray, Tuple[cv2.VideoCapture, int]] img: Either an image in numpy array format OR a tuple with cv2.VideoCapture object and the frame index.
        :param List[Union[Polygon, np.ndarray]] img: A list of shapes either as vertices in a numpy array, or as shapely Polygons.
        :returns: List of sliced ROIs from the input image.
        :rtype: List[np.ndarray]

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
            cv2.drawContours(mask, [shape - (x, y)], -1, (255, 255, 255), -1, cv2.LINE_AA)
            result.append(cv2.bitwise_and(roi_img, mask))
        return result

    @staticmethod
    def canny_edge_detection(img: np.ndarray,
                             threshold_1: Optional[int] = 30,
                             threshold_2: Optional[int] = 200,
                             aperture_size: Optional[int] = 3,
                             l2_gradient: Optional[bool] = False) -> np.ndarray:
        """
        Applies Canny edge detection to the input image using specified thresholds, aperture size, and L2 gradient option.

        Canny edge detection is an edge detection algorithm that uses gradient values to identify sharp changes in intensity
        in an image, which correspond to edges.

        .. note::
           High sensitivity: threshold_1 = 10 threshold_2 = 60
           Ultra-high sensitivity: threshold_1 = 5 threshold_2 = 40

        :param np.ndarray img: A 2D or 3D NumPy array representing the input image. If the image has 3 channels (RGB or BGR), it will be converted to grayscale.
        :param Optional[int] threshold_1: The lower threshold value for the Canny edge detection. Default is 30.
        :param Optional[int] threshold_2: The upper threshold value for the Canny edge detection. Default is 200.
        :param Optional[int] aperture_size: The aperture size for the Sobel operator used during edge detection. It must be an odd number between 3 and 7. Default is 3. Larger values reduce sensitivity to fine details.
        :param Optional[bool] l2_gradient:  If set to True, the L2 norm is used to calculate the gradient magnitude. If False, the L1 norm is used. Default is False.
        :return: A NumPy array representing the detected edges in the image.
        :rtype: np.ndarray
        """

        check_if_valid_img(source=f"{ImageMixin.canny_edge_detection.__name__}", data=img)
        check_int(
            name=f"{ImageMixin.canny_edge_detection.__name__} threshold_1",
            value=threshold_1,
            min_value=1,
        )
        check_int(
            name=f"{ImageMixin.canny_edge_detection.__name__} threshold_2",
            value=threshold_2,
            min_value=1,
        )
        check_int(
            name=f"{ImageMixin.canny_edge_detection.__name__} aperture_size",
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
        Compute image moments or Hu moments from the given image.

        Image moments are statistical properties of an image that provide information about its shape and structure.
        Hu moments are a set of seven invariant moments used for image pattern recognition, which are invariant to
        image transformations such as scaling, translation, and rotation.

        .. note::
           See wiki ` https://en.wikipedia.org/wiki/Image_moment#Hu_invariant_moments <See https://en.wikipedia.org/wiki/Image_moment#Hu_invariant_moments>`__.

        :param np.ndarray img: The input image as a 2D or 3D NumPy array. If the image has multiple channels (e.g., RGB or BGR), it will be converted to grayscale.
        :param  Optional[bool] hu_moments: If set to True, the function computes and returns the 7 Hu moments.  If False, it returns the standard moments of the image. Default is False.
        :returns: A 24x1 2D-array if `hu_moments` is False (representing standard moments), or a 7x1 2D-array if `hu_moments` is True (representing Hu moments).
        :rtype: np.ndarray

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
    def find_contours(img: np.ndarray,
                      mode: Optional[Literal["all", "exterior"]] = "all",
                      method: Optional[Literal["simple", "none", "l1", "kcos"]] = "simple") -> np.ndarray:

        """
        Find contours in an image.

        .. seealso::
           For contour comparisons, see :func:`simba.mixins.image_mixin.ImageMixin.get_contourmatch`

        :param np.ndarray img: Input image as a NumPy array.
        :param Optional[Literal['all', 'exterior']] img: Contour retrieval mode. E.g., which contours should be kept. Default is 'all'.
        :param Optional[Literal['simple', 'none', 'l1', 'kcos']]: Contour approximation method. Default is 'simple'.
        :rtype: np.ndarray
        """

        check_if_valid_img(source=f"{ImageMixin.find_contours.__name__} img", data=img)
        check_str(name=f"{ImageMixin.find_contours.__name__} mode", value=mode, options=list(GeometryEnum.CONTOURS_MODE_MAP.value.keys()))
        check_str(name=f"{ImageMixin.find_contours.__name__} method", value=method, options=list(GeometryEnum.CONTOURS_RETRIEVAL_MAP.value.keys()))
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
                if (hierarchy[0][i][3] == -1):  # Contour with no parent (interior contour)
                    interior_contours.append(cnts[i])
            return interior_contours

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
        :returns: A tuple containing: (i) int: frame index of the frame with the best match.
                 (ii) dict: Dictionary containing results (probability and match location) for each frame.
                 (iii) Union[None, np.ndarray]: Annotated image with rectangles around matches (if return_img is True), otherwise None.
        :rtype: Tuple[ int, dict, Union[None, np.ndarray]]

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

        .. seealso::
           If converting multiple images from colour to black and white, consider :func:`simba.mixins.image_mixin.ImageMixin.img_stack_to_bw` for multi-core method or
           :func:`simba.data_processors.cuda.image.img_stack_to_bw` for GPU acceleration.

        :param np.ndarray img: Input image as a NumPy array.
        :param Optional[int] lower_thresh: Lower threshold value for binary conversion. Pixels below this value become black. Default is 20.
        :param Optional[int] upper_thresh: Upper threshold value for binary conversion. Pixels above this value become white. Default is 250.
        :param Optional[bool] invert: Flag indicating whether to invert the binary image (black becomes white and vice versa). Default is True.
        :return: Binary black and white image.
        :rtype: np.ndarray
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
            return np.ascontiguousarray(cv2.threshold(img, lower_thresh, upper_thresh, cv2.THRESH_BINARY)[1])
        else:
            return np.ascontiguousarray(~cv2.threshold(img, lower_thresh, upper_thresh, cv2.THRESH_BINARY)[1])

    @staticmethod
    @jit(nopython=True)
    def img_stack_to_bw(imgs: np.ndarray, lower_thresh: int, upper_thresh: int, invert: bool):
        """
        Convert a stack of color images into black and white format.

        .. note::
           If converting a single image, consider ``simba.mixins.image_mixin.ImageMixin.img_to_bw()``. For GPU acceleration,
           see :func:`simba.data_processors.cuda.image.img_stack_to_bw`.

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

        .. seealso::
           :func:`simba.mixins.image_mixin.ImageMixin.segment_img_stack_horizontal`, :func:`simba.mixins.image_mixin.ImageMixin.segment_img_vertical`,
           :func:`simba.data_processors.cuda.image.segment_img_stack_horizontal`, :func:`simba.data_processors.cuda.image.segment_img_stack_vertical`

        :param np.ndarray img: Input image as a NumPy array.
        :param int pct: Percentage of the image to be segmented. If `lower` is True, it represents the lower part; if False, it represents the upper part.
        :param Optional[bool] lower: Flag indicating whether to segment the lower part (True) or upper part (False) of the image. Default is True.
        :param Optional[bool] both: If True, **removes** both the upper pct and lower pct and keeps middle part.
        :return: Segmented part of the image.
        :rtype: np.ndarray

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
    def segment_img_stack_horizontal(imgs: np.ndarray, pct: int, lower: bool, both: bool) -> np.ndarray:
        """
        Segment a horizontal part of all images in stack.

        .. seealso::
           :func:`simba.mixins.image_mixin.ImageMixin.segment_img_horizontal`
           :func:`simba.data_processors.cuda.image.segment_img_stack_horizontal`,

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

        .. seealso::
           :func:`simba.data_processors.cuda.image.segment_img_stack_vertical`
           :func:`simba.data_processors.cuda.image.segment_img_stack_horizontal`

        :param np.ndarray img: Input image as a NumPy array.
        :param int pct: Percentage of the image to be segmented. If `lower` is True, it represents the lower part; if False, it represents the upper part.
        :param Optional[bool] lower: Flag indicating whether to segment the lower part (True) or upper part (False) of the image. Default is True.
        :param Optional[bool] both: If True, **removes** both the left pct and right pct and keeps middle part.
        :return: Segmented part of the image.
        :rtype: np.ndarray
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
    def read_all_img_in_dir(dir: Union[str, os.PathLike], core_cnt: Optional[int] = -1) -> Dict[str, np.ndarray]:
        """
        Helper to read in all images within a directory using multiprocessing.

        :param Union[str, os.PathLike] dir: Diretory holding the input images.
        :param Optional[int] core_cnt: Number of CPU cores to use to read in images. Default to ``-1`` denoting all available cores.
        :returns: Returns a dictionary with the image name as key and the images in array format as values.
        :rtype: Dict[str, np.ndarray]

        :example:
        >>> imgs = ImageMixin().read_all_img_in_dir(dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/Together_4_cropped_frames')
        """
        check_if_dir_exists(in_dir=dir)
        file_paths = find_files_of_filetypes_in_directory(directory=dir, extensions=list(Options.ALL_IMAGE_FORMAT_OPTIONS.value), raise_error=True)
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
        Jitted pairwise comparison of images in two stacks of equal length using mean squared errors.

        .. note::
           Useful for noting subtle changes, each imgs_2 equals imgs_1 with images shifted by 1. Images has to be in uint8 format.
           Also see ``img_sliding_mse``.

        .. seealso::
           For time-series comparison and GPU acceleration, see :func:`simba.data_processors.cuda.image.stack_sliding_mse`.
           For time-series comparison and multicore acceleration, see:func:`simba.mixins.image_mixin.ImageMixin.img_sliding_mse`

        :param np.ndarray imgs_1: First three (non-color) or four (color) dimensional stack of images in array format.
        :param np.ndarray imgs_1: Second three (non-color) or four (color) dimensional stack of images in array format.
        :return: Array of size len(imgs_1) comparing ``imgs_1`` and ``imgs_2`` at each index using mean squared errors at each pixel location.
        :rtype: np.ndarray

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
    @njit([(uint8[:, :, :, :], float64, float64), (uint8[:, :, :], float64, float64)])
    def img_sliding_mse(imgs: np.ndarray,
                        slide_length: Optional[float] = 1.0,
                        sample_rate: Optional[float] = 1.0) -> np.ndarray:
        """
        Jitted compute the mean squared error (MSE) between pairs of images in a sliding window manner.

        This function performs pairwise comparisons of images using mean squared errors (MSE).
        It slides a window of the specified size over the sequence of images and computes the MSE
        between each image and the image that is `slide_size` positions before it.

        .. seealso::
           For GPU acceleration, see :func:`simba.data_processors.cuda.image.stack_sliding_mse`.
           To compare images in two stacks, see :func:`simba.mixins.image_mixin.ImageMixin.img_stack_mse`

        .. image:: _static/img/img_sliding_mse.webp
           :width: 600
           :align: center

        :param imgs: 3d or 4d A numpy array of images.
        :param slide_size: The size of the sliding window (default is 1).
        :return: A numpy array of MSE values for each pair of images in the sliding window.
        :rtype: np.ndarray


        :example:
        >>> imgs = ImageMixin().read_all_img_in_dir(dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/Together_4_cropped_frames')
        >>> imgs = np.stack(imgs.values())
        >>> mse = ImageMixin().img_sliding_mse(imgs=imgs, slide_length=2)
        """

        slide_length = int(max(0.0, slide_length * sample_rate))
        results = np.full((imgs.shape[0]), 0)
        for i in prange(slide_length, imgs.shape[0]):
            results[i] = np.sum((imgs[i - slide_length] - imgs[i]) ** 2) / float(imgs[i - slide_length].shape[0] * imgs[i].shape[1])
        return results.astype(int64)

    @staticmethod
    def _read_img_batch_from_video_helper(frm_idx: np.ndarray, video_path: Union[str, os.PathLike], greyscale: bool, verbose: bool, black_and_white: bool):

        """Multiprocess helper used by read_img_batch_from_video to read in images from video file."""
        start_idx, end_frm, current_frm = frm_idx[0], frm_idx[-1] + 1, frm_idx[0]
        results = {}
        cap = cv2.VideoCapture(video_path)
        cap.set(1, current_frm)
        while current_frm < end_frm:
            if verbose:
                print(f'Reading frame idx {current_frm}...')
            img = cap.read()[1]
            if greyscale or black_and_white:
                img = ImageMixin.img_to_greyscale(img=img)
            if black_and_white:
                img = np.where(img > 127, 255, 0).astype(np.uint8)
            results[current_frm] = img
            current_frm += 1
        return results

    @staticmethod
    def read_img_batch_from_video(video_path: Union[str, os.PathLike],
                                  start_frm: int,
                                  end_frm: int,
                                  greyscale: bool = False,
                                  black_and_white: bool = False,
                                  core_cnt: int = -1,
                                  verbose: bool = False) -> Dict[int, np.ndarray]:
        """
        Read a batch of frames from a video file. This method reads frames from a specified range of frames within a video file using multiprocessing.

        .. seealso::
           For GPU acceleration, see :func:`simba.utils.read_write.read_img_batch_from_video_gpu`

        .. note::
          When black-and-white videos are saved as MP4, there can be some small errors in pixel values during compression. A video with only (0, 255) pixel values therefore gets other pixel values, around 0 and 255, when read in again.
          If you expect that the video you are reading in is black and white, set ``black_and_white`` to True to round any of these wonly value sto 0 and 255.

        :param Union[str, os.PathLike] video_path: Path to the video file.
        :param int start_frm: Starting frame index.
        :param int end_frm: Ending frame index.
        :param Optionalint] core_cnt: Number of CPU cores to use for parallel processing. Default is -1, indicating using all available cores.
        :param Optional[bool] greyscale: If True, reads the images as greyscale. If False, then as original color scale. Default: False.
        :param bool black_and_white: If True, returns the images in black and white. Default False.
        :returns: A dictionary containing frame indices as keys and corresponding frame arrays as values.
        :rtype: Dict[int, np.ndarray]

        :example:
        >>> ImageMixin().read_img_batch_from_video(video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Together_1.avi', start_frm=0, end_frm=50)
        """
        if platform.system() == "Darwin":
            if not multiprocessing.get_start_method(allow_none=True):
                multiprocessing.set_start_method("fork", force=True)

        check_file_exist_and_readable(file_path=video_path)
        video_meta_data = get_video_meta_data(video_path=video_path)
        check_int(name=ImageMixin().__class__.__name__,value=start_frm, min_value=0,max_value=video_meta_data["frame_count"])
        check_int(name=ImageMixin().__class__.__name__, value=end_frm, min_value=start_frm+1, max_value=video_meta_data["frame_count"])
        check_int(name=ImageMixin().__class__.__name__, value=core_cnt, min_value=-1)
        check_valid_boolean(value=[greyscale, black_and_white], source=f'{ImageMixin().__class__.__name__} greyscale black_and_white')
        if core_cnt < 0:
            core_cnt = multiprocessing.cpu_count()
        if end_frm <= start_frm:
            FrameRangeError(msg=f"Start frame ({start_frm}) has to be before end frame ({end_frm})", source=ImageMixin().__class__.__name__)
        frm_lst = np.array_split(np.arange(start_frm, end_frm + 1), core_cnt)
        results = {}
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(ImageMixin()._read_img_batch_from_video_helper, video_path=video_path, greyscale=greyscale, black_and_white=black_and_white, verbose=verbose)
            for cnt, result in enumerate(pool.imap(constants, frm_lst, chunksize=1)):
                results.update(result)

        pool.join()
        pool.terminate()
        return results

    @staticmethod
    def img_emd(imgs: List[np.ndarray] = None,
                img_1: Optional[np.ndarray] = None,
                img_2: Optional[np.ndarray] = None,
                lower_bound: Optional[float] = 0.5,
                verbose: Optional[bool] = False):
        """
        Compute Wasserstein distance between two images represented as numpy arrays.

        .. math::
           EMD(P, Q) = \inf_{\gamma \in \Gamma(P, Q)} \int |x - y| d\gamma(x, y)

        where :math:`P` and :math:`Q` are the distributions (image histograms), and :math:`\Gamma(P, Q)` represents the set of all possible joint distributions between :math:`P` and :math:`Q`. The goal is to minimize the cost of transforming :math:`P` into :math:`Q`.


        .. note::
           Long runtime for larger images. Consider down-sampling videos / images before caluclating wasserstein / earth mover distances.

        .. image:: _static/img/img_emd.webp
           :width: 600
           :align: center

        :param List[np.ndarray] imgs: A list containing two images as NumPy arrays. Alternatively, you can pass `img_1` and `img_2` directly.
        :param Optional[np.ndarray] img_1: The first image (optional if `imgs` is provided).
        :param Optional[np.ndarray] img_2:  The second image (optional if `imgs` is provided).
        :param Optional[float] lower_bound: Lower bound on the EMD computation. Default is 0.5.
        :param Optional[bool] verbose: If True, prints additional information such as elapsed time.
        :return: The Earth Mover's Distance (Wasserstein distance) between the two images.
        :rtype: float

        :example:
        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1_frames/24.png', 0).astype(np.float32)
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1_frames/1984.png', 0).astype(np.float32)
        >>> ImageMixin.img_emd(img_1=img_1, img_2=img_2, lower_bound=0.5)
        >>> 10.658767700195312
        """
        timer = SimbaTimer(start=True)
        check_float(name=f"{ImageMixin.img_emd.__name__} lower bound", min_value=0.0, value=lower_bound)
        if (img_1 is None and img_1 is None and imgs is None) or (img_1 is not None and img_2 is None):
            raise InvalidInputError(msg="Pass img_1 and img_2 OR imgs", source=ImageMixin.__class__.__name__)
        if (img_1 is not None) and (img_2 is not None):
            check_if_valid_img(data=img_1, source=ImageMixin.img_emd.__name__)
            check_if_valid_img(data=img_2, source=ImageMixin.img_emd.__name__)
            if img_1.ndim > 2:
                img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            if img_2.ndim > 2:
                img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
        else:
            check_valid_lst(data=imgs, source=ImageMixin.img_emd.__name__, valid_dtypes=(np.ndarray,), exact_len=2,)
            check_if_valid_img(data=imgs[0], source=ImageMixin.img_emd.__name__)
            check_if_valid_img(data=imgs[1], source=ImageMixin.img_emd.__name__)
            if imgs[0].ndim > 2:
                img_1 = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
            if imgs[1].ndim > 2:
                img_2 = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY)
        timer.stop_timer()
        emd = cv2.EMD(img_1.astype(np.float32), img_2.astype(np.float32), cv2.DIST_L2, lowerBound=lower_bound)[0]
        if verbose:
            stdout_success(msg='EMD complete', elapsed_time=timer.elapsed_time_str)
        return emd

    @staticmethod
    def create_uniform_img(size: Tuple[int, int],
                           color: Tuple[int, int, int] = (0, 0, 0),
                           save_path: Optional[Union[str, os.PathLike]] = None) -> Union[None, np.ndarray]:

        """
        Creates an image of specified size and color, and optionally saves it to a file.

        .. image:: _static/img/create_uniform_img.webp
           :width: 600
           :align: center

        :param Tuple[int, int] size: A tuple of two integers representing the width and height of the image.
        :param Tuple[int, int, int] color: A tuple of three integers representing the RGB color (e.g., (255, 0, 0) for red). Defaults to black (0, 0, 0).
        :param Optional[Union[str, os.PathLike]] save_path: a string representing the file path to save the image.  If not provided, the function returns the image as a numpy array.
        :return: If save_path is provided, the function saves the image to the specified path and returns None. f save_path is not provided, the function returns the image as a numpy ndarray.
        :rtype: Union[None, np.ndarray]

        :example:
        >>> from simba.utils.data import create_color_palette
        >>> clrs = create_color_palette(pallete_name='inferno', increments=4, as_int=True)
        >>> imgs_stack = np.full((5, 10, 10, 3), -1)
        >>> for cnt, i in enumerate(clrs): imgs_stack[cnt] = ImageMixin.create_uniform_img(size=(10, 10), color=tuple(i))

        """

        check_valid_tuple(x=size, accepted_lengths=(2,), valid_dtypes=(int,))
        check_if_valid_rgb_tuple(data=color)
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = color[::-1]
        if save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(save_path))
            cv2.imwrite(save_path, img)
        else:
            return img

    @staticmethod
    @jit(nopython=True)
    def img_matrix_mse(imgs: np.ndarray) -> np.ndarray:
        """
        Compute the mean squared error (MSE) matrix table for a stack of images.

        This function calculates the MSE between each pair of images in the input array
        and returns a symmetric matrix where each element (i, j) represents the MSE
        between the i-th and j-th images. Useful for image similarities and anomalities.

        .. image:: _static/img/img_matrix_mse.webp
           :width: 600
           :align: center

        .. seealso::
           For time-series and GPU acceleration, see :func:`simba.data_processors.cuda.image.stack_sliding_mse`.
           For time-series and multicore CPU solution, see :func:`simba.mixins.image_mixin.ImageMixin.img_sliding_mse`
           To compare two images, see :func:`simba.mixins.image_mixin.ImageMixin.img_stack_mse`


        :param np.ndarray imgs: A stack of images represented as a numpy array.
        :return: The MSE matrix table.
        :rtype: np.ndarray

        :example:
        >>> imgs = ImageMixin().read_img_batch_from_video(video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Together_1.avi', start_frm=0, end_frm=50)
        >>> imgs = np.stack(list(imgs.values()))
        >>> ImageMixin().img_matrix_mse(imgs=imgs)
        """
        results = np.full((imgs.shape[0], imgs.shape[0]), 0.0)
        for i in prange(imgs.shape[0]):
            for j in range(i + 1, imgs.shape[0]):
                val = np.sum((imgs[i] - imgs[j]) ** 2) / float(imgs[i].shape[0] * imgs[j].shape[1])
                results[i, j] = val
                results[j, i] = val
        return results.astype(np.int32)

    @staticmethod
    def img_to_greyscale(img: np.ndarray) -> np.ndarray:
        """
        Convert a single color image to greyscale.

        The function takes an RGB image and converts it to a greyscale image using a weighted sum approach.
        If the input image is already in greyscale (2D array), it is returned as is.

        .. seealso::
           For CuPy based GPU acceleration, see :func:`simba.data_processors.cuda.image.img_stack_to_grayscale_cupy`
           For numba CUDA based GPU acceleration, see :func:`simba.data_processors.cuda.image.img_stack_to_grayscale_cuda`
           For numba based multicore solution, see :func:`simba.mixins.image_mixin.ImageMixin.img_stack_to_greyscale`

        :param np.ndarray img: Input image represented as a NumPy array. For a color image, the array should have three channels (RGB).
        :return: The greyscale image as a 2D NumPy array.
        :rtype: np.ndarray
        """
        check_if_valid_img(data=img, source=ImageMixin.img_to_greyscale.__name__)
        if len(img.shape) != 2:
            return (0.07 * img[:, :, 2] + 0.72 * img[:, :, 1] + 0.21 * img[:, :, 0]).astype(np.uint8)
        else:
            return img.astype(np.uint8)

    @staticmethod
    @njit("(uint8[:, :, :, :],)", fastmath=True)
    def img_stack_to_greyscale(imgs: np.ndarray):
        """
        Jitted conversion of a 4D stack of color images (RGB format) to grayscale.

        .. image:: _static/img/img_stack_to_greyscale.png
           :width: 600
           :align: center

        .. seealso::
           For CuPy based GPU acceleration, see :func:`simba.data_processors.cuda.image.img_stack_to_grayscale_cupy`
           For numba CUDA based GPU acceleration, see :func:`simba.data_processors.cuda.image.img_stack_to_grayscale_cuda`
           For single image conversion, see :func:`simba.mixins.image_mixin.ImageMixin.img_to_greyscale`

        :param np.ndarray imgs: A 4D array representing color images. It should have the shape (num_images, height, width, 3) where the last dimension represents the color channels (R, G, B).
        :returns: A 3D array containing the grayscale versions of the input images. The shape of the output array is (num_images, height, width).
        :rtype: np.ndarray

        :example:
        >>> imgs = ImageMixin().read_img_batch_from_video( video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Together_1.avi', start_frm=0, end_frm=100)
        >>> imgs = np.stack(list(imgs.values()))
        >>> imgs_gray = ImageMixin.img_stack_to_greyscale(imgs=imgs)
        """
        results = np.full((imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.nan).astype(np.uint8)
        for i in prange(imgs.shape[0]):
            vals = (0.07 * imgs[i][:, :, 2] + 0.72 * imgs[i][:, :, 1] + 0.21 * imgs[i][:, :, 0])
            results[i] = vals.astype(np.uint8)
        return results

    @staticmethod
    def _slice_shapes_in_imgs_array_helper(data: Tuple[int, int, np.ndarray, Polygon],
                                           verbose: bool,
                                           bg_color: Tuple[int, int, int]) -> Dict[int, np.ndarray]:
        """
        Private multiprocess helper called from ``simba.mixins.image_mixin.ImageMixins.slice_shapes_in_imgs()`` to slice shapes from
        an array of images.
        """
        results = {}
        for obs in data:
            batch_id, img_idx, img, shape = obs
            if verbose:
                print(f"Processing frame {img_idx}... (batch: {batch_id})")
            if shape.is_empty or not shape.is_valid:
                results[img_idx] = None
            else:
                roi = np.array(shape.exterior.coords).astype(np.int64)
                roi[roi < 0] = 0
                x, y, w, h = cv2.boundingRect(roi)
                roi_img = img[y : y + h, x : x + w].copy()
                mask = np.zeros_like(roi_img, np.uint8)
                if any(dim == 0 for dim in mask.shape):
                    results[img_idx] = None
                else:
                    cv2.drawContours(mask, [roi - (x, y)], -1, bg_color, -1, cv2.LINE_AA)
                    result = cv2.bitwise_and(roi_img, mask)
                    results[img_idx] = result
        return results

    @staticmethod
    def pad_img_stack(image_dict: Dict[int, np.ndarray], pad_value: Optional[int] = 0) -> Dict[int, np.ndarray]:
        """
        Pad images in a dictionary stack to have the same dimensions (the same dimension is represented by the largest image in the stack)

        .. image:: _static/img/pad_img_stack.webp
           :width: 400
           :align: center

        .. seealso::
           To read in a dictionary of images from a video using GPU acceleration, see :func:`simba.utils.read_write.read_img_batch_from_video_gpu`
           To read in a dictionary of images from a video using multicore acceleration, see :func:`simba.mixins.image_mixin.ImageMixin.read_img_batch_from_video`
           To read in a dictionary of images from a directory using multicore acceleration, use :func:`simba.mixins.image_mixin.ImageMixin.read_all_img_in_dir`

        :param Dict[int, np.ndarray] image_dict: A dictionary mapping integer keys to numpy arrays representing images.
        :param Optional[int] pad_value: The value (between 0-255) used for padding. Defaults to 0 (black)
        :return: A dictionary mapping integer keys to numpy arrays representing padded images.
        :rtype: Dict[int, np.ndarray]
        """

        check_instance(source=ImageMixin.pad_img_stack.__name__, instance=image_dict,accepted_types=(dict,))
        check_int(name=f"{ImageMixin.pad_img_stack.__name__} pad_value", value=pad_value, max_value=255, min_value=0)
        max_height = max(image.shape[0] for image in image_dict.values() if image is not None)
        max_width = max(image.shape[1] for image in image_dict.values() if image is not None)
        valid_images = [img for img in image_dict.values() if img is not None]
        if not valid_images:
            raise InvalidInputError(msg="No valid images found in `image_dict`.")
        first_valid = valid_images[0]
        has_channels = first_valid.ndim == 3
        for img in valid_images:
            if has_channels != (img.ndim == 3):
                raise ValueError("Inconsistent image dimensions: mix of grayscale and color images.")
        channels = first_valid.shape[2] if has_channels else None
        padded_images = {}
        for key, image in image_dict.items():
            if image is None:
                if has_channels:
                    empty_image = np.full((max_height, max_width, channels), pad_value, dtype=np.uint8)
                else:
                    empty_image = np.full((max_height, max_width), pad_value, dtype=np.uint8)
                padded_images[key] = empty_image
                continue
            check_if_valid_img(data=image, source=ImageMixin.pad_img_stack.__name__, raise_error=True)
            pad_height = max_height - image.shape[0]
            pad_width = max_width - image.shape[1]
            if image.ndim == 3:
                padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant", constant_values=pad_value)
            else:
                padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode="constant", constant_values=pad_value)
            padded_images[key] = padded_image
        return padded_images

    @staticmethod
    def img_stack_to_video(imgs: Dict[int, np.ndarray],
                           fps: Union[int, float],
                           save_path: Union[str, os.PathLike],
                           verbose: Optional[bool] = True) -> None:
        """
        Convert a dictionary of images into a video file.

        .. note::
           The input dictionary ``imgs`` can be created with ``simba.mixins.ImageMixin.slice_shapes_in_imgs``.

        .. seealso::
           For GPU acceleration, see :func:`simba.utils.read_write.img_stack_to_video`.

        :param Dict[int, np.ndarray] imgs: A dictionary containing frames of the video, where the keys represent frame indices and the values are numpy arrays representing the images.
        :param Union[str, os.PathLike] save_path: The path to save the output video file.
        :param int fps: Frames per second (FPS) of the output video.
        :param Optional[bool] verbose: If True, prints progress messages. Defaults to True.
        """

        timer = SimbaTimer(start=True)
        check_instance(source=ImageMixin.img_stack_to_video.__name__, instance=imgs, accepted_types=(dict,))
        check_float(name=f'{ImageMixin().img_stack_to_video.__name__} fps', value=fps, min_value=10e-6)
        img_sizes = set()
        for k, v in imgs.items():
            img_sizes.add(v.shape)
        if len(list(img_sizes)) > 1:
            imgs = ImageMixin.pad_img_stack(imgs)
        imgs = np.stack(list(imgs.values()))
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        writer = cv2.VideoWriter(save_path, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0]))
        for i in range(imgs.shape[0]):
            if verbose:
                print(f"Writing img {i + 1} / {imgs.shape[0]}...")
            writer.write(imgs[i])
        writer.release()
        timer.stop_timer()
        if verbose:
            stdout_success(msg=f"Video {save_path} complete", elapsed_time=timer.elapsed_time_str)

    @staticmethod
    def _slice_shapes_in_video_file_helper(data: List[Tuple[int, Polygon]],
                                           video_path: Union[str, os.PathLike],
                                           bg_color: Tuple[int, int, int],
                                           verbose: bool):

        cap = cv2.VideoCapture(video_path)
        batch_id, start_frm, current_frm, end_frm = data[0][0], data[0][1], data[0][1], data[-1][1]
        cap.set(1, start_frm)
        results = {}
        idx_cnt = 0
        while current_frm <= end_frm:
            if verbose:
                print(f"Processing frame {current_frm}... (batch: {batch_id})")
            img = read_frm_of_video(video_path=cap, frame_index=current_frm)
            shape = data[idx_cnt][2]
            if shape.is_empty or not shape.is_valid:
                results[current_frm] = None
            else:
                shape = np.array(shape.exterior.coords).astype(np.int64)
                shape[shape < 0] = 0
                x, y, w, h = cv2.boundingRect(shape)
                roi_img = img[y : y + h, x : x + w].copy()
                mask = np.zeros_like(roi_img, np.uint8)
                if any(dim == 0 for dim in mask.shape):
                    results[current_frm] = None
                else:
                    cv2.drawContours(mask, [shape - (x, y)], -1, bg_color, -1, cv2.LINE_AA)
                    results[current_frm] = cv2.bitwise_and(roi_img, mask).astype(np.uint8)
            current_frm += 1
            idx_cnt += 1
        return results

    def slice_shapes_in_imgs(self,
                             imgs: Union[np.ndarray, os.PathLike],
                             shapes: Union[np.ndarray, List[Polygon]],
                             core_cnt: Optional[int] = -1,
                             verbose: Optional[bool] = False,
                             bg_color: Optional[Tuple[int, int, int]] = (255, 255, 255)) -> Dict[int, np.ndarray]:
        """
        Slice regions from a stack of images or a video file, where the regions are based on defined shapes. Uses multiprocessing.

        For example, given a stack of N images, and N*X geometries representing the region around the animal body-part(s),
        slice out the X geometries from each of the N images and return the sliced areas.

        .. seealso::
           For GPU acceleration, see :func:`simba.data_processors.cuda.image.slice_imgs`
           For single core process, use :func:`simba.mixins.image_mixin.ImageMixin.slice_shapes_in_img`

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
        >>> imgs = ImageMixin().read_img_batch_from_video(video_path='/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1.mp4', start_frm=0, end_frm=10)
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
        check_int(name=f"{ImageMixin().slice_shapes_in_imgs.__name__} core count",value=core_cnt,min_value=-1)
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        check_instance(source=ImageMixin().slice_shapes_in_imgs.__name__, instance=imgs, accepted_types=(np.ndarray, str))
        check_instance( source=ImageMixin().slice_shapes_in_imgs.__name__, instance=shapes, accepted_types=(np.ndarray, list))
        check_if_valid_rgb_tuple(data=bg_color)
        if isinstance(shapes, np.ndarray):
            check_valid_array(data=shapes, source=ImageMixin().slice_shapes_in_imgs.__name__, accepted_ndims=(2,), accepted_dtypes=[Polygon])
        else:
            check_valid_lst(data=shapes, source=ImageMixin().slice_shapes_in_imgs.__name__, valid_dtypes=[Polygon])
            shapes = np.array(shapes)
        if isinstance(imgs, np.ndarray):
            check_valid_array(data=imgs, source=ImageMixin().slice_shapes_in_imgs.__name__, accepted_ndims=(4, 3))
            if shapes.shape[0] != imgs.shape[0]:
                raise ArrayError(msg=f"The image array ({imgs.shape[0]}) and shapes array ({shapes.shape[0]}) have unequal length.", source=ImageMixin().slice_shapes_in_imgs.__name__)
        else:
            check_file_exist_and_readable(file_path=imgs)
            video_meta_data = get_video_meta_data(video_path=imgs)
            if shapes.shape[0] != video_meta_data["frame_count"]:
                raise ArrayError( msg=f'The image array ({video_meta_data["frame_count"]}) and shapes array ({shapes.shape[0]}) have unequal length.', source=ImageMixin().slice_shapes_in_imgs.__name__)
        if isinstance(imgs, np.ndarray):
            results = []
            shapes = [(img_idx, img, shape) for img_idx, (img, shape) in enumerate(zip(imgs, shapes))]
            k, m = divmod(len(shapes), core_cnt)
            shapes = [[(i, *tup) for tup in shapes[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]] for i in range(core_cnt)]
            with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
                constants = functools.partial(self._slice_shapes_in_imgs_array_helper, bg_color=bg_color, verbose=verbose)
                for cnt, result in enumerate(pool.imap(constants, shapes, chunksize=1)):
                    results.append(result)
            results = dict(ChainMap(*results))

        else:
            results = []
            shapes = [(img_idx, shape) for img_idx, shape in enumerate(shapes)]
            k, m = divmod(len(shapes), core_cnt)
            shapes = [[(i, *tup) for tup in shapes[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]] for i in range(core_cnt)]
            with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
                constants = functools.partial( self._slice_shapes_in_video_file_helper, video_path=imgs, bg_color=bg_color, verbose=verbose)
                for cnt, result in enumerate(pool.imap(constants, shapes, chunksize=1)):
                    results.append(result)
                results = dict(ChainMap(*results))
        pool.join()
        pool.terminate()
        results = dict(sorted(results.items(), key=lambda item: int(item[0])))
        timer.stop_timer()
        stdout_success(msg="Geometry image slicing complete.", elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)
        return results

    @staticmethod
    def structural_similarity_index(img_1: np.ndarray, img_2: np.ndarray) -> float:
        """
        Compute the Structural Similarity Index (SSI) between two images.

        The function evaluates the SSI between two input images `img_1` and `img_2`. If the images have different numbers
        of channels, they are converted to greyscale before computing the SSI. If the images are multi-channel (e.g., RGB),
        the SSI is computed for each channel.

        .. seealso::
           For matrix SSI of image stack, see :func:`simba.mixins.image_mixin.ImageMixin.structural_similarity_matrix`
           for time-series based comparison, see :func:`simba.mixins.image_mixin.ImageMixin.sliding_structural_similarity_index`

        :param np.ndarray img_1: The first input image represented as a NumPy array.
        :param np.ndarray img_2: The second input image represented as a NumPy array.
        :returns: The SSI value representing the similarity between the two images.
        :rtype: float
        """

        check_if_valid_img(data=img_1, source=f'{ImageMixin.structural_similarity_index.__name__} img_1')
        check_if_valid_img(data=img_2, source=f'{ImageMixin.structural_similarity_index.__name__} img_2')
        multichannel = False
        if img_1.ndim != img_2.ndim:
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
        if img_1.ndim > 2: multichannel = True
        return abs(structural_similarity(im1=img_1.astype(np.uint8), im2=img_2.astype(np.uint8), multichannel=multichannel))

    @staticmethod
    def sliding_structural_similarity_index(imgs: List[np.ndarray],
                                            stride: Optional[int] = 1,
                                            verbose: Optional[bool] = False) -> np.ndarray:

        """
        Computes the Structural Similarity Index (SSI) between consecutive images in an array with a specified stride.

        The function evaluates the SSI between pairs of images in the input array `imgs` using a sliding window approach
        with the specified `stride`. The SSI is computed for each pair of images and the results are stored in an output
        array. If the images are multi-channel (e.g., RGB), the SSI is computed for each channel.

        High SSI values (close to 1) indicate high similarity between images, while low SSI values (close to 0 or negative)
        indicate low similarity.

        .. seealso::
           For matrix SSI of image stack, see :func:`simba.mixins.image_mixin.ImageMixin.structural_similarity_matrix`
           For simple comparison of two images, see :func:`simba.mixins.image_mixin.ImageMixin.structural_similarity_index`

        :param np.ndarray imgs: A list of images. Each element in the list is expected to be a numpy array representing an image.
        :param Optional[int] stride: The number of images to skip between comparisons. Default is 1.
        :param Optional[bool] verbose: If True, prints progress messages. Default is False.
        :return: A numpy array containing the SSI values for each pair of images.
        :rtype: np.ndarray

        :example:
        >>> imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/test')
        >>> imgs = {k: imgs[k] for k in sorted(imgs, key=lambda x: int(x.split('.')[0]))}
        >>> imgs = list(imgs.values())
        >>> results = ImageMixin.sliding_structural_similarity_index(imgs=imgs, stride=1, verbose=True)
        """

        check_valid_lst(data=imgs, valid_dtypes=(np.ndarray,), min_len=2)
        check_int(name=f'{ImageMixin.sliding_structural_similarity_index.__name__} stride', min_value=1, max_value=len(imgs), value=stride)
        ndims, multichannel = set(), False
        for i in imgs:
            check_if_valid_img(data=i, source=ImageMixin.sliding_structural_similarity_index.__name__)
            ndims.add(i.ndim)
        if len(list(ndims)) > 1:
            imgs = ImageMixin.img_stack_to_greyscale(imgs=imgs)
        if imgs[0].ndim > 2: multichannel = True
        results = np.zeros((len(imgs)), np.float32)
        for cnt, i in enumerate(range(stride, len(imgs))):
            img_1, img_2 = imgs[i - stride], imgs[i]
            results[i] = structural_similarity(im1=img_1, im2=img_2, multichannel=multichannel)
            if verbose:
                print(f'SSI computed ({cnt + 1}/{len(imgs) - stride})')
        return results

    @staticmethod
    def structural_similarity_matrix(imgs: List[np.array], verbose: Optional[bool] = False) -> np.ndarray:
        """
        Computes a matrix of Structural Similarity Index (SSI) values for a list of images.

        This function takes a list of images and computes the SSI between each pair of images and produce a symmetric matrix.

        .. seealso::
           For time-series based comparison, see :func:`simba.mixins.image_mixin.ImageMixin.sliding_structural_similarity_index`
           For simple comparison of two images, see :func:`simba.mixins.image_mixin.ImageMixin.structural_similarity_index`

        :param List[np.array] imgs: A list of images represented as numpy arrays. If not all images are greyscale or color, they are converted and processed as greyscale.
        :param Optional[bool] verbose: If True, prints progress messages showing which SSI values have been computed.  Default is False.
        :return: A square numpy array where the element at [i, j] represents the SSI between imgs[i] and imgs[j].
        :rtype: np.ndarray

        :example:
        >>> imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/test')
        >>> imgs = {k: imgs[k] for k in sorted(imgs, key=lambda x: int(x.split('.')[0]))}
        >>> imgs = list(imgs.values())[0:10]
        >>> results = ImageMixin.structural_similarity_matrix(imgs=imgs)
        """

        check_valid_lst(data=imgs, valid_dtypes=(np.ndarray,), min_len=2)
        ndims, multichannel = set(), False
        for i in imgs:
            check_if_valid_img(data=i, source=ImageMixin.structural_similarity_matrix.__name__)
            ndims.add(i.ndim)
        if len(list(ndims)) > 1:
            imgs = ImageMixin.img_stack_to_greyscale(imgs=imgs)
        if imgs[0].ndim > 2: multichannel = True
        results = np.ones((len(imgs), len(imgs)), np.float32)
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                if verbose:
                    print(f'SSI matrix position ({i}, {j}) complete...')
                val = structural_similarity(im1=imgs[i], im2=imgs[j], multichannel=multichannel)
                results[i, j] = val
                results[j, i] = val
        return results

    @staticmethod
    @njit(["(uint8[:, :], uint8[:, :])",
           "(uint8[:, :, :], uint8[:, :, :])"])
    def cross_correlation_similarity(img_1: np.ndarray, img_2: np.ndarray) -> float:
        """
        Computes the Normalized Cross-Correlation (NCC) similarity between two images.

        The NCC measures the similarity between two images by calculating the correlation coefficient of their pixel values. The output value ranges from -1 to 1, where 1 indicates perfect positive correlation, 0 indicates no correlation, and -1 indicates perfect negative correlation.

        .. seealso::
           For time-series based NCC comparisons, see :func:`simba.mixins.image_mixin.ImageMixin.sliding_cross_correlation_similarity`
           For matrix based NCC comparisons, see :func:`simba.mixins.image_mixin.ImageMixin.cross_correlation_matrix`

        :param np.ndarray img_1: The first input image. It can be a 2D grayscale image or a 3D color image.
        :param np.ndarray img_2:  The second input image. It must have the same dimensions as img_1.
        :return: The NCC value representing the similarity between the two images. Returns 0.0 if the denominator is zero, indicating no similarity.
        :rtype: float

        :example:
        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/a.png').astype(np.uint8)
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/f.png').astype(np.uint8)
        >>> ImageMixin.cross_correlation_similarity(img_1=img_1, img_2=img_2)
        """

        img1_flat = img_1.flatten()
        img2_flat = img_2.flatten()
        mean_1, mean_2 = np.mean(img1_flat), np.mean(img2_flat)
        N = np.sum((img1_flat - mean_1) * (img2_flat - mean_2))
        D = np.sqrt(np.sum((img1_flat - mean_1) ** 2) * np.sum((img2_flat - mean_2) ** 2))
        if D == 0:
            return 0.0
        else:
            return N / D

    @staticmethod
    @njit(["(uint8[:, :, :], int64)",
           "(uint8[:, :, :, :], int64)"])
    def sliding_cross_correlation_similarity(imgs: np.ndarray,
                                             stride: int) -> np.ndarray:
        """
        Computes the Normalized Cross-Correlation (NCC) similarity for a sequence of images using a sliding window approach.

        This function calculates the NCC between each image and the image that is `stride` positions before it in the sequence. The result is an array of NCC values representing
        the similarity between successive images.

        .. seealso::
           For simple two image NCC comparison, see :func:`simba.mixins.image_mixin.ImageMixin.cross_correlation_similarity`
           For matrix based NCC comparisons, see :func:`simba.mixins.image_mixin.ImageMixin.cross_correlation_matrix`

        :param np.ndarray imgs: A 3D array (for grayscale images) or a 4D array (for color images) containing the sequence of images.  Each image should have the same size.
        :param int stride: The stride length for comparing images. Determines how many steps back in the sequence each image is compared to.
        :return: A 1D array of NCC values representing the similarity between each image and the image `stride` positions before it. The length of the array is the same as the number of images.
        :rtype: np.ndarray

        :example:
        >>> imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/08102021_DOT_Rat11_12_frames')
        >>> imgs = {k: imgs[k] for k in sorted(imgs, key=lambda x: int(x.split('.')[0]))}
        >>> imgs = np.stack(list(imgs.values()))
        >>> results = ImageMixin.sliding_cross_correlation_similarity(imgs=imgs, stride=1)
        """
        results = np.ones((imgs.shape[0]), dtype=np.float32)
        for i in prange(stride, imgs.shape[0]):
            img1_flat, img2_flat = imgs[i - stride].flatten(), imgs[i].flatten()
            mean_1, mean_2 = np.mean(img1_flat), np.mean(img2_flat)
            N = np.sum((img1_flat - mean_1) * (img2_flat - mean_2))
            D = np.sqrt(np.sum((img1_flat - mean_1) ** 2) * np.sum((img2_flat - mean_2) ** 2))
            if D == 0:
                results[i] = 0.0
            else:
                results[i] = N / D
        return results

    @staticmethod
    @njit(["(uint8[:, :, :],)",
           "(uint8[:, :, :, :],)"])
    def cross_correlation_matrix(imgs: np.array) -> np.ndarray:
        """
        Computes the cross-correlation matrix for a given array of images.

        This function calculates the cross-correlation coefficient between each pair of images in the input array.
        The cross-correlation coefficient is a measure of similarity between two images, with values ranging from
        -1 (completely dissimilar) to 1 (identical).

        The function uses the `numba` library for Just-In-Time (JIT) compilation to optimize performance, and
        `prange` for parallel execution over the image pairs.

        .. seealso::
           For simple two image NCC comparison, see :func:`simba.mixins.image_mixin.ImageMixin.cross_correlation_similarity`
           For time-series based NCC comparisons, see :func:`simba.mixins.image_mixin.ImageMixin.sliding_cross_correlation_similarity`

        .. note::
           Use greyscale images for faster runtime. Ideally should be moved to GPU.


        :param np.array imgs: A 3D (or 4D) numpy array of images where the first dimension indexes the images,
                              and the remaining dimensions are the image dimensions (height, width, [channels]).
                              - For grayscale images: shape should be (n_images, height, width)
                              - For color images: shape should be (n_images, height, width, channels)

        :return: A 2D numpy array representing the cross-correlation matrix, where the element at [i, j]
                          contains the cross-correlation coefficient between the i-th and j-th images.
        :rtype: np.array

        :example:
        >>> imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/test')
        >>> imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/08102021_DOT_Rat11_12_frames')
        >>> imgs = {k: imgs[k] for k in sorted(imgs, key=lambda x: int(x.split('.')[0]))}
        >>> imgs = np.stack(list(imgs.values()))
        >>> imgs = ImageMixin.img_stack_to_greyscale(imgs=imgs)
        >>> results = ImageMixin.cross_correlation_matrix(imgs=imgs)
        """

        results = np.ones((imgs.shape[0], imgs.shape[0]), dtype=np.float32)
        for i in prange(imgs.shape[0]):
            img1_flat = imgs[i].flatten()
            mean_1 = np.mean(img1_flat)
            for j in range(i + 1, imgs.shape[0]):
                img2_flat = imgs[j].flatten()
                mean_2 = np.mean(img2_flat)
                N = np.sum((img1_flat - mean_1) * (img2_flat - mean_2))
                D = np.sqrt(np.sum((img1_flat - mean_1) ** 2) * np.sum((img2_flat - mean_2) ** 2))
                if D == 0:
                    val = 0.0
                else:
                    val = N / D
                results[i, j] = val
                results[j, i] = val
        return results

    @staticmethod
    def close(x: Union[List[np.ndarray], np.ndarray],
              kernel: Tuple[int, int],
              iterations: int = 3) -> Union[List[np.ndarray], np.ndarray]:

        """
        Performs morphological closing on the provided image(s). Closing is a dilation operation followed by erosion,
        commonly used to fill small holes or gaps in an image.

        :param Union[List[np.ndarray], np.ndarray] x: Input image or list of images to process. Each image must be greyscale or black-and-white.
        :param Tuple[int, int] kernel: Tuple specifying the size of the structuring element used for the closing operation.
        :param int iterations: Number of times the closing operation is applied. Defaults to 3.
        :return: Processed image or list of processed images after morphological closing.
        :rtype: Union[List[np.ndarray], np.ndarray]
        """

        imgs, results = [], []
        if isinstance(x, np.ndarray):
            check_if_valid_img(data=x, source=ImageMixin.close.__name__, raise_error=True)
            if not is_img_bw(img=x) and not is_img_greyscale(img=x):
                raise InvalidInputError(msg='The image is invalid. Greyscale or black-and-white image is requires', source=ImageMixin.close.__name__.__name__)
            imgs.append(x)
        elif isinstance(x, list):
            for cnt, i in enumerate(x):
                check_if_valid_img(data=i, source=f'{ImageMixin.close.__name__.__name__} {cnt}', raise_error=True)
                if not is_img_bw(img=i) and not is_img_greyscale(img=i):
                    raise InvalidInputError(msg='The image is invalid. Greyscale or black-and-white image is requires', source=ImageMixin.close.__name__.__name__)
                imgs.append(i)
        else:
            raise InvalidInputError(msg=f'x is not a valid input. Require list of arrays or array, got {type(x)}',  source=ImageMixin.close.__name__.__name__)
        check_valid_tuple(x=kernel, source=f'{ImageMixin.close.__name__} kernel', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
        check_int(name=f'{ImageMixin.close.__name__} iterations', value=iterations, min_value=1, raise_error=True)
        for img in imgs:
            results.append(cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.array(kernel), iterations=iterations))
        if len(results) == 1:
            return results[0]
        else:
            return results

    @staticmethod
    def find_first_non_uniform_clr_frm(video_path: Union[str, os.PathLike, cv2.VideoCapture],
                                       start_idx: Optional[int] = 0,
                                       end_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Find the first frame of non-uniform color in a video.

        .. note::
           Helpful in the :func:`simba.ui.px_to_mm_ui.GetPixelsPerMillimeterInterface` to ensure that a viable frame is pulled up.

        :param Union[str, os.PathLike, cv2.VideoCapture] video_path: The path to a video file on disk, or a cv2.VideoCapture object.
        :param Optional[int] start_idx: The first frame (where to start searching for the non-uniform color image). Default: 0 which equals the first frame. None also equals start searching at the first frame.
        :param Optional[int] end_idx: The last frame (where to end searching for the non-uniform color image). Default: None, which equals 1s into the video.
        :returns: The first non-uniform color image in the video as np.ndarray and the index of the first frame.
        :rtype: np.ndarray
        """

        check_instance(source='find_first_non_uniform_clr_frm', instance=video_path, accepted_types=(str, cv2.VideoCapture))
        video_meta_data = get_video_meta_data(video_path=video_path)
        if isinstance(start_idx, int) and isinstance(end_idx, int):
            if start_idx >= end_idx:
                raise FrameRangeError(msg=f'Start frame ({start_idx}) has to be before the end frame ({end_idx})', source='find_first_non_uniform_clr_frm')
        if isinstance(start_idx, int):
            check_int(name='start_idx', value=start_idx, min_value=0)
            if start_idx > (video_meta_data['frame_count'] - 1):
                start_idx = int(video_meta_data['frame_count'] - 1)
        elif start_idx is None:
            start_idx = 0
        else:
            raise FrameRangeError(msg=f'Start frame idx {(start_idx)} has to be None or an integer', source='find_first_non_uniform_clr_frm')
        if isinstance(end_idx, int):
            check_int(name='end_idx', value=start_idx, min_value=0)
            if end_idx > (video_meta_data['frame_count']):
                end_idx = int(video_meta_data['frame_count'])
        elif end_idx is None:
            end_idx = int(video_meta_data['fps'])
        else:
            raise FrameRangeError(msg=f'End frame idx {(end_idx)} has to be None or an integer', source='find_first_non_uniform_clr_frm')

        first_frm = read_frm_of_video(video_path=video_path, frame_index=0)
        for frame_index in range(int(start_idx), int(end_idx)):
            frame = read_frm_of_video(video_path=video_path, frame_index=frame_index)
            if frame.ndim == 2:
                clr_cnt = np.unique(frame).shape[0]
            else:
                clr_cnt = np.unique(frame.reshape(-1, frame.shape[-1]), axis=0).shape[0]
            if clr_cnt > 1:
                return frame, frame_index
        return first_frm, 0

    @staticmethod
    def is_video_color(video: Union[str, os.PathLike, cv2.VideoCapture]):
        """
        Determines whether a video is in color or greyscale.

        :param Union[str, os.PathLike, cv2.VideoCapture] video: The video source, either a cv2.VideoCapture object or a path to a file on disk.
        :return: Returns `True` if the video is in color (has more than one channel), and `False` if the video is greyscale (single channel).
        :rtype: bool
        """

        frm, frm_idx = ImageMixin.find_first_non_uniform_clr_frm(video_path=video)
        
        # If frame has only 2 dimensions, it's definitely greyscale
        if frm.ndim == 2:
            return False
        
        # If frame has 3 dimensions, check if it's actually greyscale
        # (some greyscale videos are stored as 3-channel with identical values)
        if frm.ndim == 3:
            # Check if all channels are identical (indicating greyscale)
            if frm.shape[2] == 3:  # BGR format
                # Compare B, G, and R channels
                if np.array_equal(frm[:, :, 0], frm[:, :, 1]) and np.array_equal(frm[:, :, 1], frm[:, :, 2]):
                    return False  # All channels identical = greyscale
                else:
                    return True   # Channels different = color
            elif frm.shape[2] == 1:
                return False  # Single channel = greyscale
            else:
                return True   # Other multi-channel formats = color
        
        # Default case: assume greyscale
        return False

    @staticmethod
    @jit(nopython=True)
    def resize_img_stack(imgs: np.ndarray,
                         scale_factor: float = 0.5) -> np.ndarray:
        """
        Resizes a stack of images by applying a scaling factor to each image in the stack. Uses bilinear interpolation.

        .. note::
           Pass gresyscale images.

        :param np.ndarray imgs: 3D numpy array of shape (N, H, W). All images are expected to have the same shape.
        :param float scale_factor: A float that determines the scaling factor for resizing each image. A value of 0.5 will reduce the size by  half.
        :return: A 3D numpy array of the resized images, with shape (N, Nh, Nw), where Nh and Nw are the new height and width calculated by applying the `scale_factor` to the original height and width.
        :rtype: np.ndarray

        :example:
        >>> VIDEO_PATH = r"D:\EPM_2\EPM_1.mp4"
        >>> img = read_img_batch_from_video(video_path=VIDEO_PATH, greyscale=True, start_frm=0, end_frm=15, core_cnt=1)
        >>> imgs = np.stack(list(img.values()))
        >>> resized_img = resize_img_stack(imgs=imgs)
        """

        Nh, Nw = round(imgs.shape[1] * scale_factor), round(imgs.shape[2] * scale_factor)
        results = np.empty((imgs.shape[0], Nh, Nw), dtype=imgs.dtype)
        for t in prange(imgs.shape[0]):
            for i in range(Nh):
                for j in range(Nw):
                    orig_x = min(int(j / scale_factor), imgs.shape[2] - 1)
                    orig_y = min(int(i / scale_factor), imgs.shape[1] - 1)
                    if orig_x + 1 < imgs.shape[2] and orig_y + 1 < imgs.shape[1]:
                        dx = (j / scale_factor) - orig_x
                        dy = (i / scale_factor) - orig_y
                        top_left = imgs[t, orig_y, orig_x]
                        top_right = imgs[t, orig_y, orig_x + 1]
                        bottom_left = imgs[t, orig_y + 1, orig_x]
                        bottom_right = imgs[t, orig_y + 1, orig_x + 1]
                        top = top_left + dx * (top_right - top_left)
                        bottom = bottom_left + dx * (bottom_right - bottom_left)
                        results[t, i, j] = top + dy * (bottom - top)
                    else:
                        results[t, i, j] = imgs[t, orig_y, orig_x]
        return results


    @staticmethod
    def resize_img_dict(imgs: Dict[str, np.ndarray],
                        size: Union[Literal['min', 'max'], Tuple[int, int]],
                        interpolation: Optional[int] = cv2.INTER_LINEAR) -> Dict[str, np.ndarray]:

        """
        Resize a dictionary of images to a specified size.

        .. seealso::

           To read in a dictionary of images from a video using GPU acceleration, see :func:`simba.utils.read_write.read_img_batch_from_video_gpu`
           To read in a dictionary of images from a video using multicore acceleration, see :func:`simba.mixins.image_mixin.ImageMixin.read_img_batch_from_video`
           To read in a dictionary of images from a directory using multicore acceleration, use :func:`simba.mixins.image_mixin.ImageMixin.read_all_img_in_dir`

        :param Dict[str, np.ndarray] imgs: A dictionary where keys are image names (strings) and values are NumPy arrays representing the images.
        :param Union[Literal['min', 'max'], Tuple[int, int]] size: The target size for the resizing operation. It can be: - `'min'`: Resize all images to the smallest height and width found among the input images. - `'max'`: Resize all images to the largest height and width found among the input images. - Tuple of two integers `(height, width)`: Explicitly specify the target size for all images.
        :param interpolation: Interpolation method to use for resizing. This can be one of OpenCV's interpolation methods.
        :return: A dictionary of resized images, where the keys match the original dictionary, and the values are the resized images as NumPy arrays.
        :rtype: Dict[str, np.ndarray]
        """

        check_instance(source=ImageMixin.resize_img_dict.__name__, instance=imgs, accepted_types=(dict,))
        check_instance(source=ImageMixin.resize_img_dict.__name__, instance=size, accepted_types=(tuple, str,))
        results = {}
        if size == 'min':
            target_h, target_w = np.inf, np.inf
            for k, v in imgs.items():
                target_h, target_w = min(v.shape[0], target_h), min(v.shape[1], target_w)
        elif size == 'max':
            target_h, target_w = -np.inf, -np.inf
            for k, v in imgs.items():
                target_h, target_w = max(v.shape[0], target_h), max(v.shape[1], target_w)
        elif isinstance(size, tuple):
            check_valid_tuple(x=size, accepted_lengths=(2,), valid_dtypes=(int,))
            check_int(name=ImageMixin.resize_img_dict.__name__, value=size[0], min_value=1)
            check_int(name=ImageMixin.resize_img_dict.__name__, value=size[1], min_value=1)
            target_h, target_w = size[0], size[1]
        else:
            raise InvalidInputError(msg=f'{size} is not a valid size argument.', source=ImageMixin.resize_img_dict.__name__)

        for k, v in imgs.items():
            check_if_valid_img(data=v, source=ImageMixin.resize_img_dict.__name__)
            results[k] = cv2.resize(v, dsize=(target_w, target_h), fx=0, fy=0, interpolation=interpolation)

        return results

    @staticmethod
    def non_local_mean_denoising_image(img: np.ndarray,
                                       sigma: int = 30,
                                       template_window: float = 0.02,
                                       search_window: float = 0.10) -> np.ndarray:

        """
        Applies Non-Local Means (NLM) denoising to a grayscale or color image using OpenCV.

        .. note::
           Pretty slow.

        .. image:: _static/img/non_local_mean_denoising.webp
           :width: 400
           :align: center

        :param np.ndarray img: Input image (grayscale or color) as a NumPy array.
        :param int sigma: Strength of the filter. Higher values remove more noise but may blur details. Default 30.
        :param float template_window: Size of the local patch for denoising, relative to the larger image dimension.  Must be between `1e-5` and `1.0`. Default is `0.02`.
        :param float search_window: Size of the area where similar patches are searched, relative to the larger image dimension. Must be between `1e-5` and `1.0`. Default is `0.10`.
        :return: Denoised image as a NumPy array with the same shape as the input.
        :rtype: np.ndarray
        """

        check_if_valid_img(data=img, source=f'{ImageMixin.non_local_mean_denoising_image.__name__} img')
        check_int(name=f'{ImageMixin.non_local_mean_denoising_image.__name__} sigma', value=sigma, min_value=1)
        check_float(f'{ImageMixin.non_local_mean_denoising_image.__name__} template_window', min_value=10e-6, max_value=1.0, value=template_window)
        check_float(f'{ImageMixin.non_local_mean_denoising_image.__name__} search_window', min_value=10e-6, max_value=1.0, value=search_window)
        img_h, img_w = img.shape[0], img.shape[1]
        template_window = int(max(img_h, img_w) * template_window)
        search_window_size = int(max(img_h, img_w) * search_window)
        if img.ndim == 2:
            img = cv2.fastNlMeansDenoising(img, dst=None, h=sigma, templateWindowSize=template_window, searchWindowSize=search_window_size)
        else:
            img = cv2.fastNlMeansDenoisingColored(img, None, h=sigma, templateWindowSize=template_window, searchWindowSize=search_window_size)
        return img

    @staticmethod
    def non_local_mean_denoising_sequence(imgs: np.ndarray,
                                          sigma: int = 30,
                                          img_to_denoise_idx: Optional[int] = None) -> np.ndarray:

        """
        Applies Non-Local Means (NLM) denoising to a stack of images or video frames to reduce noise, using a temporal window for multi-frame denoising.

        .. note::
           Pretty slow.

        .. image:: _static/img/non_local_mean_denoising.webp
           :width: 400
           :align: center

        .. seealso::
           For single images, see :func:`~simba.mixins.image_mixin.ImageMixin.non_local_mean_denoising_image`

        :param np.ndarray imgs: A 3D or 4D NumPy array of images or video frames.  If the input is a 3D array, it represents a single image stack (height, width, num_frames). If the input is a 4D array, it represents a batch of video frames (num_frames, height, width, num_channels).
        :param int sigma: The filtering strength parameter. A higher value corresponds to stronger denoising  and more smoothing of the image. The default is 30.
        :return: Denoised video or image stack.
        :rtype: np.ndarray If input is a 3D array (grayscale), the output is a 3D array. If input is a 4D array (colored), the output is a 4D arra
        """

        check_valid_array(data=imgs, source=f'{ImageMixin.non_local_mean_denoising_sequence.__name__} imgs', accepted_ndims=(3, 4,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_axis_0=2)
        check_int(name=f'{ImageMixin.non_local_mean_denoising_sequence.__name__} sigma', value=sigma, min_value=1)
        if img_to_denoise_idx is None:
            img_to_denoise_idx = int(np.floor(imgs.shape[0] / 2))
        else:
            check_int(name=f'{ImageMixin.non_local_mean_denoising_sequence.__name__} img_to_denoise_idx', value=img_to_denoise_idx, max_value=imgs.shape[0] - 1, min_value=0)
        temporal_window_size = int((imgs.shape[0] / 2))
        temporal_window_size = temporal_window_size if temporal_window_size % 2 == 1 else temporal_window_size - 1
        if imgs.shape[0] <= 2:
            temporal_window_size = 1
        if imgs.ndim == 3:
            imgs = [imgs[:, :, i] for i in range(imgs.shape[2])]
            denoised_img = cv2.fastNlMeansDenoisingMulti(imgs, imgToDenoiseIndex=img_to_denoise_idx, temporalWindowSize=temporal_window_size, h=sigma)
        else:
            imgs = [imgs[i] for i in range(imgs.shape[0])]
            denoised_img = cv2.fastNlMeansDenoisingColoredMulti(imgs, imgToDenoiseIndex=img_to_denoise_idx, temporalWindowSize=temporal_window_size, h=sigma)

        return denoised_img














#x = ImageMixin.get_blob_locations(video_path=r"C:\troubleshooting\RAT_NOR\project_folder\videos\2022-06-20_NOB_DOT_4_downsampled_bg_subtracted.mp4", gpu=True)
# imgs = ImageMixin().read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples')
# imgs = np.stack(imgs.values())
# mse = ImageMixin().img_sliding_mse(imgs=imgs, slide_size=2)


# img_1 = cv2.imread('/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/.png', 0).astype(np.float32)
# img_2 = cv2.imread('/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/4.png', 0).astype(np.float32)
# ImageMixin.img_emd(img_1=img_1, img_2=img_2, lower_bound=0.5, verbose=True)

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

#
# from simba.utils.data import create_color_palette
# clrs = create_color_palette(pallete_name='inferno', increments=4, as_int=True)
# imgs_stack = np.full((5, 10, 10, 3), -1)
# for cnt, i in enumerate(clrs): imgs_stack[cnt] = ImageMixin.create_uniform_img(size=(10, 10), color=tuple(i))
#
#
#
#
#
# ImageMixin.img_matrix_mse(imgs=imgs_stack)


#if

#ImageMixin.get_blob_locations(video_path=r"C:\troubleshooting\mitra\test\temp\501_MA142_Gi_Saline_0515.mp4")