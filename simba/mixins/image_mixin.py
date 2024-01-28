import time

import numpy as np
from typing import List, Optional, Union, Tuple, Dict
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from shapely.geometry import Polygon
import pandas as pd
import cv2
import os
from collections import ChainMap
import multiprocessing
import functools
from numba import njit, uint8, prange, int64, jit, float64

from simba.utils.errors import CountError, InvalidInputError, FrameRangeError
from simba.utils.checks import check_instance, check_str, check_int, check_if_valid_img, check_if_dir_exists, check_file_exist_and_readable
from simba.utils.enums import GeometryEnum, Defaults, Options
from simba.utils.read_write import find_core_cnt, read_frm_of_video, get_video_meta_data, find_files_of_filetypes_in_directory, get_fn_ext

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
    def brightness_intensity(imgs: List[np.ndarray],
                             ignore_black: Optional[bool] = True) -> List[float]:

        """
        Compute the average brightness intensity within each image within a list.

        :param List[np.ndarray] imgs: List of images as arrays to calculate average brightness intensity within.
        :param Optional[bool] ignore_black: If True, ignores black pixels. If the images are sliced non-rectangular geometric shapes created by ``slice_shapes_in_img``, then pixels that don't belong to the shape has been masked in black.
        :returns List[float]: List of floats of size len(imgs) with brightness intensities.


        :example:
        >>> img = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
        >>> ImageMixin.brightness_intensity(imgs=[img], ignore_black=False)
        >>> [159.0]
        """
        results = []
        check_instance(source=f'{ImageMixin().brightness_intensity.__name__} imgs', instance=imgs, accepted_types=list)
        for cnt, img in enumerate(imgs):
            check_instance(source=f'{ImageMixin().brightness_intensity.__name__} img {cnt}', instance=img, accepted_types=np.ndarray)
            if len(img) == 0:
                results.append(0)
            else:
                if ignore_black:
                    results.append(np.ceil(np.average(img[img != 0])))
                else:
                    results.append(np.ceil(np.average(img)))
        return results

    @staticmethod
    def get_histocomparison(img_1: np.ndarray,
                            img_2: np.ndarray,
                            method: Optional[Literal['chi_square', 'correlation', 'intersection', 'bhattacharyya', 'hellinger', 'chi_square_alternative', 'kl_divergence']] = 'correlation',
                            absolute: Optional[bool] = True):
        """
        :example:
        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/3.png').astype(np.uint8)
        >>> ImageMixin.get_histocomparison(img_1=img_1, img_2=img_2, method='chi_square_alternative')
        """
        check_if_valid_img(source=f'{ImageMixin.get_histocomparison.__name__} img_1', data=img_1)
        check_if_valid_img(source=f'{ImageMixin.get_histocomparison.__name__} img_2', data=img_2)
        check_str(name=f'{ImageMixin().get_histocomparison.__name__} method', value=method, options=list(GeometryEnum.HISTOGRAM_COMPARISON_MAP.value.keys()))
        method = GeometryEnum.HISTOGRAM_COMPARISON_MAP.value[method]
        if absolute:
            return abs(cv2.compareHist(img_1.astype(np.float32), img_2.astype(np.float32), method))
        else:
            return cv2.compareHist(img_1.astype(np.float32), img_2.astype(np.float32), method)


    @staticmethod
    def get_contourmatch(img_1: np.ndarray,
                         img_2: np.ndarray,
                         mode: Optional[Literal['all', 'exterior']] = 'all',
                         method: Optional[Literal['simple', 'none', 'l2', 'kcos']] = 'simple',
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

        check_if_valid_img(source=f'{ImageMixin.get_contourmatch.__name__} img_1', data=img_1)
        check_if_valid_img(source=f'{ImageMixin.get_contourmatch.__name__} img_2', data=img_2)
        check_str(name=f'{ImageMixin().get_contourmatch.__name__} mode', value=mode, options=list(GeometryEnum.CONTOURS_MODE_MAP.value.keys()))
        check_str(name=f'{ImageMixin.find_contours.__name__} method', value=method, options=list(GeometryEnum.CONTOURS_RETRIEVAL_MAP.value.keys()))
        if canny:
            img_1 = ImageMixin().canny_edge_detection(img=img_1)
            img_2 = ImageMixin().canny_edge_detection(img=img_2)
        img_1_contours = ImageMixin().find_contours(img=img_1, mode=mode, method=method)
        img_2_contours = ImageMixin().find_contours(img=img_2, mode=mode, method=method)
        return cv2.matchShapes(img_1_contours[0], img_2_contours[0], cv2.CONTOURS_MATCH_I1, 0.0)

    @staticmethod
    def slice_shapes_in_img(img: Union[np.ndarray, Tuple[cv2.VideoCapture, int]],
                            geometries: List[Union[Polygon, np.ndarray]]) -> List[np.ndarray]:

        """
        Slice regions of interest (ROIs) from an image based on provided shapes.

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
        check_instance(source=f'{ImageMixin().slice_shapes_in_img.__name__} img', instance=img, accepted_types=(tuple, np.ndarray))
        check_instance(source=f'{ImageMixin().slice_shapes_in_img.__name__} shapes', instance=geometries, accepted_types=list)
        for shape_cnt, shape in enumerate(geometries): check_instance(source=f'{ImageMixin().slice_shapes_in_img.__name__} shapes {shape_cnt}', instance=shape,  accepted_types=(Polygon, np.ndarray))
        if isinstance(img, tuple):
            check_instance(source=f'{ImageMixin().slice_shapes_in_img.__name__} img tuple first entry', instance=img[0], accepted_types=cv2.VideoCapture)
            frm_cnt = int(img[0].get(cv2.CAP_PROP_FRAME_COUNT))
            check_int(name=f'{ImageMixin().slice_shapes_in_img.__name__} video frame count', value=img[1], max_value=frm_cnt, min_value=0)
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
        shapes = corrected_shapes; del corrected_shapes
        for shape_cnt, shape in enumerate(shapes):
            x, y, w, h = cv2.boundingRect(shape)
            roi_img = img[y:y + h, x:x + w].copy()
            mask = np.zeros(roi_img.shape[:2], np.uint8)
            cv2.drawContours(mask, [shape - shape.min(axis=0)], -1, (255, 255, 255), -1, cv2.LINE_AA)
            bg = np.ones_like(roi_img, np.uint8)
            cv2.bitwise_not(bg, bg, mask=mask)
            roi_img = bg + cv2.bitwise_and(roi_img, roi_img, mask=mask)
            result.append(roi_img)
        return result

    @staticmethod
    def canny_edge_detection(img: np.ndarray,
                             threshold_1: int = 30,
                             threshold_2: int = 200,
                             aperture_size: int = 3,
                             l2_gradient: bool = False) -> np.ndarray:
        """
        Apply Canny edge detection to the input image.
        """
        check_if_valid_img(source=f'{ImageMixin.img_moments.__name__}', data=img)
        check_int(name=f'{ImageMixin.img_moments.__name__} threshold_1', value=threshold_1, min_value=1)
        check_int(name=f'{ImageMixin.img_moments.__name__} threshold_2', value=threshold_2, min_value=1)
        check_int(name=f'{ImageMixin.img_moments.__name__} aperture_size', value=aperture_size, min_value=1)
        if len(img.shape) >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(img, threshold1=threshold_1, threshold2=threshold_2, apertureSize=aperture_size, L2gradient=l2_gradient)

    @staticmethod
    def img_moments(img: np.ndarray,
                    hu_moments: Optional[bool] = False) -> np.ndarray:
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
        check_if_valid_img(source=f'{ImageMixin.img_moments.__name__}', data=img)
        if len(img.shape) >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not hu_moments:
            return np.array(list(cv2.moments(img).values())).reshape(-1, 1)
        else:
            return cv2.HuMoments(cv2.moments(img))

    @staticmethod
    def find_contours(img: np.ndarray,
                      mode: Optional[Literal['all', 'exterior']] = 'all',
                      method:  Optional[Literal['simple', 'none', 'l1', 'kcos']] = 'simple') -> np.ndarray:
        """
        Find contours in the input image.

        :param np.ndarray img: Input image as a NumPy array.
        :param Optional[Literal['all', 'exterior']] img: Contour retrieval mode. E.g., which contours should be kept. Default is 'all'.
        :param Optional[Literal['simple', 'none', 'l1', 'kcos']]: Contour approximation method. Default is 'simple'.
        """


        check_if_valid_img(source=f'{ImageMixin.find_contours.__name__} img', data=img)
        check_str(name=f'{ImageMixin.find_contours.__name__} mode', value=mode, options=list(GeometryEnum.CONTOURS_MODE_MAP.value.keys()))
        check_str(name=f'{ImageMixin.find_contours.__name__} method', value=method, options=list(GeometryEnum.CONTOURS_RETRIEVAL_MAP.value.keys()))
        mode = GeometryEnum.CONTOURS_MODE_MAP.value[mode]
        method = GeometryEnum.CONTOURS_RETRIEVAL_MAP.value[method]
        if len(img.shape) >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if mode in [0, 1]:
            return cv2.findContours(img, mode, method)[1]
        else:
            cnts, hierarchy = cv2.findContours(img, mode, method)[-2:] #TODO
            interior_contours = []
            for i in range(len(cnts)):
                if hierarchy[0][i][3] == -1:  # Contour with no parent (interior contour)
                    interior_contours.append(cnts[i])

    @staticmethod
    def orb_matching_similarity_(img_1: np.ndarray,
                                 img_2: np.ndarray,
                                 method: Literal['knn', 'match', 'radius'] = 'knn',
                                 mask: Optional[np.ndarray] = None,
                                 threshold: Optional[int] = 0.75) -> int:
        """ Perform ORB feature matching between two sets of images.

        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/10.png').astype(np.uint8)
        >>> ImageMixin().orb_matching_similarity_(img_1=img_1, img_2=img_2, method='radius')
        >>> 4
        """

        kp1, des1 = cv2.ORB_create().detectAndCompute(img_1, mask)
        kp2, des2 = cv2.ORB_create().detectAndCompute(img_2, mask)
        sliced_matches = None
        if method == 'knn':
            matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
            sliced_matches = [m for m, n in matches if m.distance < threshold * n.distance]
        if method == 'match':
            matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
            sliced_matches = [match for match in matches if match.distance <= threshold]
        if method == 'radius':
            matches = cv2.BFMatcher().radiusMatch(des1, des2, maxDistance=threshold)
            sliced_matches = [item for sublist in matches for item in sublist]
        return len(sliced_matches)

    @staticmethod
    def _template_matching_cpu_helper(data: np.ndarray,
                                      video_path: Union[str, os.PathLike],
                                      target_frm: np.ndarray):
        """ Helper called from ``simba.mixins.image_mixin.ImageMixins.template_matching_cpu()`` """
        cap = cv2.VideoCapture(video_path)
        start, end, current = data[0], data[-1], data[0]
        cap.set(1, start); results = {}
        while current < end:
            print(f'Processing frame {current}...')
            _, img = cap.read()
            result = cv2.matchTemplate(img, target_frm, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            results[current] = {'p': np.max(result), 'loc': max_loc}
            current += 1
        return results

    @staticmethod
    def template_matching_cpu(video_path: Union[str, os.PathLike],
                              img: np.ndarray,
                              core_cnt: Optional[int] = -1,
                              return_img: Optional[bool] = False) -> Tuple[int, dict, Union[None, np.ndarray]]:
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
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        frame_cnt = get_video_meta_data(video_path=video_path)['frame_count']
        frm_idx = np.arange(0, frame_cnt + 1)
        chunk_size = len(frm_idx) // core_cnt
        remainder = len(frm_idx) % core_cnt
        split_frm_idx = [frm_idx[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(core_cnt)]
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(ImageMixin()._template_matching_cpu_helper, video_path=video_path, target_frm=img)
            for cnt, result in enumerate(pool.imap(constants, split_frm_idx, chunksize=1)):
                results.append(result)
        pool.terminate(); pool.join()
        results = dict(ChainMap(*results))

        max_value, max_frm = -np.inf, None
        for k, v in results.items():
            if v['p'] > max_value:
                max_value = v['p']; max_frm = k

        if return_img:
            h, w, _ = img.shape
            found_img = read_frm_of_video(video_path=video_path, frame_index=max_frm)
            loc = results[max_frm]['loc']
            found_img = cv2.rectangle(found_img, (int(loc[0]), int(loc[1])), (int(loc[0]) + w, int(loc[1] + h)), (0, 255, 0), 2)
        return max_frm, results, found_img

    def template_matching_gpu(self):
        #TODO
        pass

    @staticmethod
    def img_to_bw(img: np.ndarray,
                  lower_thresh: Optional[int] = 20,
                  upper_thresh: Optional[int] = 250,
                  invert: Optional[bool] = True) -> np.ndarray:

        """
        Convert an image to black and white (binary).

        :param np.ndarray img: Input image as a NumPy array.
        :param Optional[int] lower_thresh: Lower threshold value for binary conversion. Pixels below this value become black. Default is 20.
        :param Optional[int] upper_thresh: Upper threshold value for binary conversion. Pixels above this value become white. Default is 250.
        :param Optional[bool] invert: Flag indicating whether to invert the binary image (black becomes white and vice versa). Default is True.
        :return np.ndarray: Binary black and white image.
        """
        check_if_valid_img(data=img, source=ImageMixin().segment_img_horizontal.__name__)
        check_int(name=f'{ImageMixin().segment_img_horizontal.__name__} lower_thresh', value=lower_thresh, max_value=255, min_value=1)
        check_int(name=f'{ImageMixin().segment_img_horizontal.__name__} upper_thresh', value=upper_thresh, max_value=255, min_value=1)
        if len(img) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not invert:
            return cv2.threshold(img, lower_thresh, upper_thresh, cv2.THRESH_BINARY)[1]
        else:
            return ~cv2.threshold(img, lower_thresh, upper_thresh, cv2.THRESH_BINARY)[1]

    @staticmethod
    def segment_img_horizontal(img: np.ndarray,
                               pct: int,
                               lower: Optional[bool] = True,
                               both: Optional[bool] = False) -> np.ndarray:
        """
        Segment a horizontal part of the input image.

        This function segments either the lower, upper, or both lower and upper part of the input image based on the specified percentage.

        :param np.ndarray img: Input image as a NumPy array.
        :param int pct: Percentage of the image to be segmented. If `lower` is True, it represents the lower part; if False, it represents the upper part.
        :param Optional[bool] lower: Flag indicating whether to segment the lower part (True) or upper part (False) of the image. Default is True.
        :return np.array: Segmented part of the image.
        """

        check_if_valid_img(data=img, source=ImageMixin().segment_img_horizontal.__name__)
        check_int(name=f'{ImageMixin().segment_img_horizontal.__name__} pct', value=pct, min_value=1, max_value=99)
        sliced_height = int(img.shape[0] * pct / 100)
        if both:
            return img[sliced_height:img.shape[0] - sliced_height, :]
        elif lower:
            return img[img.shape[0] - sliced_height:, :]
        else:
            return img[:sliced_height, :]

    @staticmethod
    def segment_img_vertical(img: np.ndarray,
                             pct: int,
                             left: Optional[bool] = True,
                             both: Optional[bool] = False) -> np.ndarray:

        """
        Segment a vertical part of the input image.

        This function segments either the left, right or both the left and right part of  input image based on the specified percentage.

        :param np.ndarray img: Input image as a NumPy array.
        :param int pct: Percentage of the image to be segmented. If `lower` is True, it represents the lower part; if False, it represents the upper part.
        :param Optional[bool] lower: Flag indicating whether to segment the lower part (True) or upper part (False) of the image. Default is True.
        :return np.array: Segmented part of the image.
        """

        check_if_valid_img(data=img, source=ImageMixin().segment_img_vertical.__name__)
        check_int(name=f'{ImageMixin().segment_img_vertical.__name__} pct', value=pct, min_value=1, max_value=99)
        sliced_width = int(img.shape[1] * pct / 100)
        if both:
            return img[:, sliced_width:img.shape[1] - sliced_width]
        elif left:
            return img[:, :sliced_width]
        else:
            return img[:, img.shape[1] - sliced_width:]

    @staticmethod
    def add_img_border_and_flood_fill(img: np.array,
                                      invert: Optional[bool] = False,
                                      size: Optional[int] = 1) -> np.ndarray:
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

        check_if_valid_img(data=img, source=ImageMixin().add_img_border_and_flood_fill.__name__)
        check_int(name=f'{ImageMixin().add_img_border_and_flood_fill.__name__} size', value=size, min_value=1)
        if len(img.shape) > 2: raise InvalidInputError(msg='Floodfill requires 2d image', source=ImageMixin().add_img_border_and_flood_fill.__name__)
        if not invert:
            img = cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_CONSTANT, value=0)
            mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
            img = cv2.floodFill(img, mask=mask, seedPoint=(0, 0), newVal=(255, 255, 255))[1]

        else:
            img = cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_CONSTANT, value=255)
            mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
            img = cv2.floodFill(img, mask=mask, seedPoint=(0, 0), newVal=(0, 0, 0))[1]

        return img[size:-size, size:-size]

    @staticmethod
    def _image_reader_helper(img_paths: List[str]):
        """ Multiprocessing helper for ``ImageMixin().read_all_img_in_dir``"""
        results = {}
        for img_path in img_paths: results[get_fn_ext(filepath=img_path)[1]] = cv2.imread(img_path)
        return results

    @staticmethod
    def read_all_img_in_dir(dir: Union[str, os.PathLike],
                            core_cnt: Optional[int] = -1) -> Dict[str, np.ndarray]:
        """
        Helper to read in all images within a directory using multiprocessing.
        Returns a dictionary with the image name as key and the images in array format as values.

        :example:
        >>> imgs = ImageMixin().read_all_img_in_dir(dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/Together_4_cropped_frames')
        """
        check_if_dir_exists(in_dir=dir)
        file_paths = find_files_of_filetypes_in_directory(directory=dir, extensions=list(Options.ALL_IMAGE_FORMAT_OPTIONS.value), raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        chunk_size = len(file_paths) // core_cnt
        file_paths = [file_paths[i * chunk_size + min(i, len(file_paths) % core_cnt):(i + 1) * chunk_size + min(i + 1, len(file_paths) % core_cnt)] for i in range(core_cnt)]
        imgs = {}
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(ImageMixin()._image_reader_helper, file_paths, chunksize=1)):
                imgs.update(result)
        return imgs

    @staticmethod
    @njit([(uint8[:, :, :, :], uint8[:, :, :, :]),
           (uint8[:, :, :], uint8[:, :, :])])
    def img_stack_mse(imgs_1: np.ndarray, imgs_2: np.ndarray) -> np.ndarray:
        """
        Pairwise comparison of images in two stacks of equal length using mean squared errors.

        .. note::
           Images has to be in uint8 format.

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
            results[i] = np.sum((imgs_1[i] - imgs_2[i]) ** 2) / float(imgs_1[i].shape[0] * imgs_2[i].shape[1])
        return results.astype(np.int64)

    @staticmethod
    @njit([(uint8[:, :, :, :], int64),
           (uint8[:, :, :], int64)])
    def img_sliding_mse(imgs: np.ndarray, slide_size: int = 1) -> np.ndarray:
        """Pairwise comparison of images in sliding windows using mean squared errors

        :example:
        >>> imgs = ImageMixin().read_all_img_in_dir(dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/Together_4_cropped_frames')
        >>> imgs = np.stack(imgs.values())
        >>> mse = ImageMixin().img_sliding_mse(imgs=imgs, slide_size=2)
        """

        results = np.full((imgs.shape[0]), 0)
        for i in prange(slide_size, imgs.shape[0]):
            results[i] = np.sum((imgs[i-slide_size] - imgs[i]) ** 2) / float(imgs[i-slide_size].shape[0] * imgs[i].shape[1])
        return results.astype(int64)

    @staticmethod
    def _read_img_batch_from_video_helper(frm_idx: np.ndarray,
                                          video_path: Union[str, os.PathLike]):
        start_idx, end_frm, current_frm = frm_idx[0], frm_idx[-1]+1, frm_idx[0]
        results = {}
        cap = cv2.VideoCapture(video_path)
        cap.set(1, current_frm)
        while current_frm < end_frm:
            results[current_frm] = cap.read()[1]
            current_frm += 1
        return results

    @staticmethod
    def read_img_batch_from_video(video_path: Union[str, os.PathLike], start_frm: int, end_frm: int, core_cnt: Optional[int] = -1) -> Dict[int, np.ndarray]:
        """
        Read a batch of frames from a video file. This method reads frames from a specified range of frames within a video file using multiprocessing.

        :param Union[str, os.PathLike] video_path: Path to the video file.
        :param int start_frm: Starting frame index.
        :param int end_frm: Ending frame index.
        :param int core_cnt: Number of CPU cores to use for parallel processing. Default is -1, indicating using all available cores.
        :returns Dict[int, np.ndarray]: A dictionary containing frame indices as keys and corresponding frame arrays as values.

        :example:
        >>> ImageMixin().read_img_batch_from_video(video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Together_1.avi', start_frm=0, end_frm=50)
        """
        check_file_exist_and_readable(file_path=video_path)
        video_meta_data = get_video_meta_data(video_path=video_path)
        check_int(name=ImageMixin().__class__.__name__, value=start_frm, min_value=0, max_value=video_meta_data['frame_count'])
        check_int(name=ImageMixin().__class__.__name__, value=end_frm, min_value=0, max_value=video_meta_data['frame_count'])
        check_int(name=ImageMixin().__class__.__name__, value=core_cnt, min_value=-1)
        if core_cnt < 0: core_cnt = multiprocessing.cpu_count()
        if end_frm <= start_frm: FrameRangeError(msg=f'Start frame ({start_frm}) has to be before end frame ({end_frm})', source=ImageMixin().__class__.__name__)
        frm_lst = np.array_split(np.arange(start_frm, end_frm+1), core_cnt)
        results = {}
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(ImageMixin()._read_img_batch_from_video_helper,
                                          video_path=video_path)
            for cnt, result in enumerate(pool.imap(constants, frm_lst, chunksize=1)):
                results.update(result)
        return results

    @staticmethod
    @jit(nopython=True)
    def img_matrix_mse(imgs: np.ndarray) -> np.ndarray:
        """
        :example:
        >>> imgs = ImageMixin().read_img_batch_from_video(video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Together_1.avi', start_frm=0, end_frm=50)
        >>> imgs = np.stack(list(imgs.values()))
        >>> ImageMixin().img_matrix_mse(imgs=imgs)
        """
        results = np.full((imgs.shape[0], imgs.shape[0]), 0.0)
        for i in prange(imgs.shape[0]):
            for j in range(i+1, imgs.shape[0]):
                val = np.sum((imgs[i] - imgs[j]) ** 2) / float(imgs[i].shape[0] * imgs[j].shape[1])
                results[i, j] = val
                results[j, i] = val
        return results.astype(np.int32)

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

#ImageMixin.img_mse_test(imgs_1=imgs_1)


#res = ImageMixin.img_moments(img=img_1, hu_moments=True)
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

