__author__ = "Simon Nilsson"

import itertools
import os
from typing import Optional

import numpy as np
import shapely.wkt
from joblib import Parallel, delayed
from shapely.geometry import LineString, Point, Polygon

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.printing import stdout_success
from simba.utils.read_write import find_core_cnt, get_fn_ext, read_df, write_df


class AnimalBoundaryFinder(ConfigReader, FeatureExtractionMixin):
    """
    Compute boundaries (animal-anchored) ROIs for animals in each frame. Result is saved
    as a pickle in the ``project_folder/logs`` directory of the SimBA project.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter str roi_type: shape type of ROI. OPTIONS: "ENTIRE ANIMAL", "SINGLE BODY-PART SQUARE", "SINGLE BODY-PART CIRCLE". For
                             more information/examples, see `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/anchored_rois.md/>`_.
    :parameter bool force_rectangle: If True, forces roi shape into minimum bounding rectangle. If False, then polygon.
    :parameter Optional[dict] or None body_parts: If roi_type is 'SINGLE BODY-PART CIRCLE' or 'SINGLE BODY-PART SQUARE', then body-parts to anchor the ROI to
                                        with keys as animal names and values as body-parts. E.g., body_parts={'Animal_1': 'Head_1', 'Animal_2': 'Head_2'}.
    :parameter Optional[int] parallel_offset: Offset of ROI from the animal outer bounds in millimeter. If None, then no offset.

    .. notes:
       `Bounding boxes tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/anchored_rois.md/>`_.

    Examples
    ----------
    >>> animal_boundary_finder = AnimalBoundaryFinder(config_path='project_folder/project_config.ini', roi_type='SINGLE BODY-PART CIRCLE',body_parts={'Animal_1': 'Head_1', 'Animal_2': 'Head_2'}, force_rectangle=False, parallel_offset=15)
    >>> animal_boundary_finder.run()
    """

    def __init__(
        self,
        config_path: str,
        roi_type: str or None,
        force_rectangle: bool,
        body_parts: Optional[dict] = None,
        parallel_offset: Optional[int] = None,
    ):
        ConfigReader.__init__(self, config_path=config_path)
        FeatureExtractionMixin.__init__(self)
        self.parallel_offset_mm, self.roi_type, self.force_rectangle = (
            parallel_offset,
            roi_type,
            force_rectangle,
        )
        if self.parallel_offset_mm == 0:
            self.parallel_offset_mm += 1
        self.body_parts = body_parts
        check_if_filepath_list_is_empty(
            filepaths=self.outlier_corrected_paths,
            error_msg="ZERO files found in project_folder/outlier_corrected_movement_location directory",
        )
        self.save_path = os.path.join(self.project_path, "logs", "anchored_rois.pickle")
        self.cpus, self.cpus_to_use = find_core_cnt()
        if (self.roi_type == "SINGLE BODY-PART CIRCLE") or (
            self.roi_type == "SINGLE BODY-PART SQUARE"
        ):
            self.center_bp_names = {}
            for animal, body_part in self.body_parts.items():
                self.center_bp_names[animal] = [body_part + "_x", body_part + "_y"]

    def _save_results(self):
        write_df(df=self.polygons, file_type="pickle", save_path=self.save_path)
        stdout_success(
            msg=f"Animal shapes for {len(self.outlier_corrected_paths)} videos saved at {self.save_path}"
        )

    def _find_polygons(self, point_array: np.array):
        if self.roi_type == "ENTIRE ANIMAL":
            animal_shape = LineString(point_array.tolist()).buffer(self.offset_px)
        elif self.roi_type == "SINGLE BODY-PART CIRCLE":
            animal_shape = Point(point_array).buffer(self.offset_px)
        elif self.roi_type == "SINGLE BODY-PART SQUARE":
            top_left = Point(
                int(point_array[0] - self.offset_px),
                int(point_array[1] - self.offset_px),
            )
            top_right = Point(
                int(point_array[0] + self.offset_px),
                int(point_array[1] - self.offset_px),
            )
            bottom_left = Point(
                int(point_array[0] - self.offset_px),
                int(point_array[1] + self.offset_px),
            )
            bottom_right = Point(
                int(point_array[0] + self.offset_px),
                int(point_array[1] + self.offset_px),
            )
            animal_shape = Polygon([top_left, top_right, bottom_left, bottom_right])
        if self.force_rectangle:
            animal_shape = Polygon(
                self.minimum_bounding_rectangle(
                    points=np.array(animal_shape.exterior.coords)
                )
            )
        animal_shape = shapely.wkt.loads(
            shapely.wkt.dumps(animal_shape, rounding_precision=1)
        ).simplify(0)
        return animal_shape

    def run(self):
        self.polygons = {}
        for file_cnt, file_path in enumerate(self.outlier_corrected_paths):
            _, self.video_name, _ = get_fn_ext(file_path)
            _, px_per_mm, _ = self.read_video_info(video_name=self.video_name)
            self.offset_px = px_per_mm * self.parallel_offset_mm
            self.polygons[self.video_name] = {}
            self.data_df = read_df(
                file_path=file_path, file_type=self.file_type
            ).astype(int)
            for animal_cnt, animal in enumerate(self.animal_bp_dict.keys()):
                print(
                    f"Analyzing shapes in video {self.video_name} ({file_cnt+1}/{len(self.outlier_corrected_paths)}), animal {animal} ({animal_cnt+1}/{len(list(self.animal_bp_dict.keys()))})..."
                )
                if self.roi_type == "ENTIRE ANIMAL":
                    animal_x_cols, animal_y_cols = (
                        self.animal_bp_dict[animal]["X_bps"],
                        self.animal_bp_dict[animal]["Y_bps"],
                    )
                    animal_df = self.data_df[
                        [
                            x
                            for x in itertools.chain.from_iterable(
                                itertools.zip_longest(animal_x_cols, animal_y_cols)
                            )
                            if x
                        ]
                    ]
                    animal_arr = np.reshape(
                        animal_df.values, (-1, len(animal_x_cols), 2)
                    )
                if (self.roi_type == "SINGLE BODY-PART SQUARE") or (
                    self.roi_type == "SINGLE BODY-PART CIRCLE"
                ):
                    animal_arr = self.data_df[self.center_bp_names[animal]].values

                self.polygons[self.video_name][animal] = Parallel(
                    n_jobs=self.cpus_to_use, verbose=1, backend="threading"
                )(delayed(self._find_polygons)(x) for x in animal_arr)
        self._save_results()
