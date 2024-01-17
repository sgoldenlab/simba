from typing import List, Optional, Tuple, Union

import numpy as np

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from copy import deepcopy

import cv2
import pandas as pd
from numba import jit, njit, uint8
from shapely.geometry import Polygon

from simba.utils.checks import (check_if_valid_img, check_instance, check_int,
                                check_str)
from simba.utils.enums import GeometryEnum


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
    def brightness_intensity(
        imgs: List[np.ndarray], ignore_black: Optional[bool] = True
    ) -> List[float]:
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
    def get_contourmatch(
        img_1: np.ndarray,
        img_2: np.ndarray,
        mode: Optional[Literal["all", "exterior"]] = "all",
        method: Optional[Literal["simple", "none", "l2", "kcos"]] = "simple",
        canny: Optional[bool] = True,
    ) -> float:
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
            mask = np.zeros(roi_img.shape[:2], np.uint8)
            cv2.drawContours(
                mask, [shape - shape.min(axis=0)], -1, (255, 255, 255), -1, cv2.LINE_AA
            )
            bg = np.ones_like(roi_img, np.uint8)
            cv2.bitwise_not(bg, bg, mask=mask)
            roi_img = bg + cv2.bitwise_and(roi_img, roi_img, mask=mask)
            result.append(roi_img)
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

    def find_contours(
        self,
        img: np.ndarray,
        mode: Optional[Literal["all", "exterior"]] = "all",
        method: Optional[Literal["simple", "none", "l1", "kcos"]] = "simple",
    ) -> np.ndarray:
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
        return cv2.findContours(img, mode, method)[1]

    @staticmethod
    @njit([(uint8[:, :, :, :], uint8[:, :, :, :]), (uint8[:, :, :], uint8[:, :, :])])
    def img_mse(imgs_1: np.ndarray, imgs_2: np.ndarray) -> np.ndarray:
        """
        Pairwise comparison of images in two stacks using mean squared errors.

        .. note::
           Images has to be in uint8 format.

        :param np.ndarray imgs_1: First three (non-color) or four (color) dimensional stack of images in array format.
        :param np.ndarray imgs_1: Second three (non-color) or four (color) dimensional stack of images in array format.
        :return np.ndarray: Array of size len(imgs_1) comparing ``imgs_1`` and ``imgs_2`` at each index using mean squared errors at each pixel location.

        :example:
        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/0.png').astype(np.uint8)
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/khan/project_folder/videos/stitched_frames/10.png').astype(np.uint8)
        >>> imgs_1 = np.stack((img_1, img_2)); imgs_2 = np.stack((img_2, img_2))
        >>> ImageMixin.img_mse(imgs_1=imgs_1, imgs_2=imgs_2)
        >>> [637,   0]
        """

        results = np.full((imgs_1.shape[0]), np.nan)
        for i in range(imgs_1.shape[0]):
            results[i] = np.sum((imgs_1[i] - imgs_2[i]) ** 2) / float(
                imgs_1[i].shape[0] * imgs_2[i].shape[1]
            )
        return results.astype(np.int64)

    @staticmethod
    def orb_matching_similarity_(
        img_1: np.ndarray,
        img_2: np.ndarray,
        method: Literal["knn", "match", "radius"] = "knn",
        mask: Optional[np.ndarray] = None,
        threshold: Optional[int] = 0.75,
    ) -> int:
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
            sliced_matches = [
                m for m, n in matches if m.distance < threshold * n.distance
            ]
        if method == "match":
            matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
            sliced_matches = [match for match in matches if match.distance <= threshold]
        if method == "radius":
            matches = cv2.BFMatcher().radiusMatch(des1, des2, maxDistance=threshold)
            sliced_matches = [item for sublist in matches for item in sublist]
        return len(sliced_matches)


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
