import math

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.affinity import scale
from shapely.geometry import Polygon

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.errors import NoROIDataError, ParametersFileError
from simba.utils.printing import stdout_success


class ROISizeStandardizer(ConfigReader, FeatureExtractionMixin):
    """
    Standardize ROI sizes according to a reference video.

    .. note::
       Example: You select a "baseline" video, say that this baseline video has a pixel per millimeter of `10`.
       Say there are a further two videos in the project with ROIs, and these videos has pixels per millimeter of `9` and `11`.
       At runtime, the area of the rectangles, circles and polygons in the two additional videos get their ROI areas increased/decreased
       with 10% while the baseline video ROIs are unchanged.

    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter: str join_bouts_within_delta: Name of baseline video without extension, e.g., `Video_1`.

    :example:
    >>> test = ROISizeStandardizer(config_path='/Users/simon/Desktop/envs/troubleshooting/DLC_2_Black_animals/project_folder/project_config.ini', reference_video='Together_1')
    >>> test.run()
    >>> test.save()
    """

    def __init__(self, config_path: str, reference_video: str):
        ConfigReader.__init__(self, config_path=config_path)
        FeatureExtractionMixin.__init__(self)
        self.reference_video_name = reference_video
        self.read_roi_data()
        self.checks()

    def checks(self):
        if self.reference_video_name not in self.video_names_w_rois:
            raise NoROIDataError(
                msg=f"The reference video {self.reference_video_name} does not have any defined ROIs.",
                source=self.__class__.__name__,
            )
        for video_name in self.video_names_w_rois:
            if video_name not in list(self.video_info_df["Video"]):
                raise ParametersFileError(
                    msg=f"Found defined ROIs for video {video_name}, but this video does not have pixels per millimeter defined in the project_folder/logs/video_info.csv file.",
                    source=self.__class__.__name__,
                )

    def compute_correction_factors(self):
        self.correction_factors = {}
        for video_name in self.video_names_w_rois:
            px_per_mm = self.read_video_info(video_name=video_name)[1]
            self.correction_factors[video_name] = round(
                (((px_per_mm / self.reference_px_per_mm) - 1) * 100), 4
            )

    def find_scale_factor(self, video_name: str, correction_factor: float):
        if self.correction_factors[video_name] < 0:
            return math.sqrt(1 - abs(correction_factor) / 100)
        else:
            return math.sqrt(1 + correction_factor / 100)

    def run(self):
        _, self.reference_px_per_mm, _ = self.read_video_info(
            video_name=self.reference_video_name
        )
        self.compute_correction_factors()
        self.video_names_to_change = [
            x for x in self.video_names_w_rois if x != self.reference_video_name
        ]

        updated_rectangles, updated_polygons, updated_circles = (
            pd.DataFrame(columns=self.rectangles_df.columns),
            pd.DataFrame(columns=self.polygon_df.columns),
            pd.DataFrame(columns=self.circles_df.columns),
        )
        for video_name in self.video_names_to_change:
            rectangles = self.rectangles_df[self.rectangles_df["Video"] == video_name]
            circles = self.circles_df[self.circles_df["Video"] == video_name]
            polygons = self.polygon_df[self.polygon_df["Video"] == video_name]

            # UPDATE RECTANGLES
            for idx, rectangle in rectangles.iterrows():
                if self.correction_factors[video_name] != 0:
                    vertices = np.array(
                        [
                            rectangle["Tags"]["Top left tag"],
                            rectangle["Tags"]["Top right tag"],
                            rectangle["Tags"]["Bottom left tag"],
                            rectangle["Tags"]["Bottom right tag"],
                        ]
                    )
                    scale_factor = self.find_scale_factor(
                        video_name=video_name,
                        correction_factor=self.correction_factors[video_name],
                    )
                    new_vertices = np.unique(
                        np.array(
                            scale(
                                Polygon(vertices),
                                xfact=scale_factor,
                                yfact=scale_factor,
                            ).exterior.coords
                        ).astype(np.int16),
                        axis=0,
                    )

                    top_left = np.min(new_vertices, axis=0)
                    bottom_right = np.max(new_vertices, axis=0)

                    rectangle["topLeftX"], rectangle["topLeftY"] = (
                        top_left[0],
                        top_left[1],
                    )
                    rectangle["Bottom_right_X"], rectangle["Bottom_right_Y"] = (
                        bottom_right[0],
                        bottom_right[1],
                    )
                    rectangle["width"], rectangle["height"] = int(
                        rectangle["Bottom_right_X"] - rectangle["topLeftX"]
                    ), int(rectangle["Bottom_right_Y"] - rectangle["topLeftY"])
                    (
                        rectangle["Tags"]["Top left tag"],
                        rectangle["Tags"]["Top right tag"],
                    ) = (int(rectangle["topLeftX"]), int(rectangle["topLeftY"])), (
                        int(rectangle["Bottom_right_X"]),
                        int(rectangle["topLeftY"]),
                    )
                    (
                        rectangle["Tags"]["Bottom left tag"],
                        rectangle["Tags"]["Bottom right tag"],
                    ) = (rectangle["topLeftX"], rectangle["Bottom_right_Y"]), (
                        rectangle["Bottom_right_X"],
                        rectangle["Bottom_right_Y"],
                    )
                    rectangle["Tags"]["Left tag"] = (
                        int(rectangle["topLeftX"]),
                        int(rectangle["topLeftY"] + rectangle["height"] / 2),
                    )
                    rectangle["Tags"]["Right tag"] = (
                        int(rectangle["topLeftX"] + rectangle["width"]),
                        int(rectangle["topLeftY"] + rectangle["height"] / 2),
                    )
                    rectangle["Tags"]["Top tag"] = (
                        int(rectangle["topLeftX"] + rectangle["width"] / 2),
                        int(rectangle["topLeftY"]),
                    )
                    rectangle["Tags"]["Bottom tag"] = (
                        int(rectangle["topLeftX"] + rectangle["width"] / 2),
                        int(rectangle["topLeftY"] + rectangle["height"]),
                    )
                    updated_rectangles.loc[idx] = rectangle

            # UPDATE POLYGONS
            for idx, polygon in polygons.iterrows():
                if self.correction_factors[video_name] != 0:
                    scale_factor = self.find_scale_factor(
                        video_name=video_name,
                        correction_factor=self.correction_factors[video_name],
                    )
                    new_vertices = np.array(
                        scale(
                            Polygon(polygon["vertices"]),
                            xfact=scale_factor,
                            yfact=scale_factor,
                        ).exterior.coords
                    )
                    new_vertices = ConvexHull(new_vertices).points.astype(np.int32)
                    polygon["vertices"] = new_vertices
                    for cnt, i in enumerate(polygon["vertices"]):
                        polygon["Tags"][f"Tag_{cnt}"] = i
                    updated_polygons.loc[idx] = polygon

            # UPDATE CIRCLES
            for idx, circle in circles.iterrows():
                if self.correction_factors[video_name] != 0:
                    circle["radius"] = int(
                        circle["radius"] * (self.correction_factors[video_name] / 100)
                    )
                    circle["Tags"]["Border tag"] = (
                        int(circle["centerX"] - circle["radius"]),
                        circle["centerY"],
                    )
                    updated_circles.loc[idx] = circle

            self.rectangles_df.update(updated_rectangles)
            self.polygon_df.update(updated_polygons)
            self.circles_df.update(updated_circles)

    def save(self):
        store = pd.HDFStore(self.roi_coordinates_path, mode="w")
        store["rectangles"] = self.rectangles_df
        store["circleDf"] = self.circles_df
        store["polygons"] = self.polygon_df
        store.close()
        stdout_success(
            msg=f"ROI size definitions standardized according to pixels per millimeter in video {self.reference_video_name}",
            source=self.__class__.__name__,
        )


# test = ROISizeStandardizer(config_path='/Users/simon/Desktop/envs/troubleshooting/DLC_2_Black_animals/project_folder/project_config.ini', reference_video='Together_1')
# test.run()
# test.save()


#
# test = ROISizeStandardizer(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', reference_video='Together_1')
# test.run()
# test.save()
