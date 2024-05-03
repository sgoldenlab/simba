__author__ = "Simon Nilsson"

import itertools
import os
from typing import Any, Dict, List, Union

import cv2
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.roi_tools.ROI_feature_analyzer import ROIFeatureCreator
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_keys_exist_in_dict, check_int,
                                check_valid_array, check_valid_dataframe,
                                check_valid_lst,
                                check_video_and_data_frm_count_align)
from simba.utils.data import slice_roi_dict_for_video
from simba.utils.enums import Formats, Keys, TextOptions
from simba.utils.errors import (BodypartColumnNotFoundError, NoFilesFoundError,
                                ROICoordinatesNotFoundError)
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, get_video_meta_data, read_df
from simba.utils.warnings import DuplicateNamesWarning

ROI_CENTERS = "roi_centers"
ROI_EAR_TAGS = "roi_ear_tags"
DIRECTIONALITY = "directionality"
DIRECTIONALITY_STYLE = "directionality_style"
BORDER_COLOR = "border_color"
POSE = "pose_estimation"
ANIMAL_NAMES = "animal_names"

STYLE_KEYS = [
    ROI_CENTERS,
    ROI_EAR_TAGS,
    DIRECTIONALITY,
    BORDER_COLOR,
    POSE,
    DIRECTIONALITY_STYLE,
    ANIMAL_NAMES,
]


class ROIfeatureVisualizer(ConfigReader):
    """
    Visualizing features that depend on the relationships between the location of the animals and user-defined
    ROIs. E.g., distances to centroids of ROIs, if animals are directing towards ROIs, and if animals are within ROIs.

    .. note::
       For improved run-time, see :meth:`simba.ROI_feature_visualizer_mp.ROIfeatureVisualizerMultiprocess` for multiprocess class.
       `Tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-5-visualizing-roi-features>`__.

    .. image:: _static/img/roi_visualize.png
       :width: 400
       :align: center

    .. image:: _static/img/ROIfeatureVisualizer_1.png
       :width: 700
       :align: center

    .. image:: _static/img/ROIfeatureVisualizer_2.png
       :width: 700
       :align: center

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike] video_path: Path to video file to overlay ROI features on.
    :param List[str] body_parts: List of body-parts to use as proxy for animal location(s).
    :param Dict[str, Any] style_attr: User-defined styles (sizes, colors etc.)

    :example:
    >>> style_attr = {'roi_centers': True, 'roi_ear_tags': True, 'directionality': True, 'directionality_style': 'funnel', 'border_color': (0, 0, 0), 'pose_estimation': True, 'animal_names': True}
    >>> test = ROIfeatureVisualizer(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini', video_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_DOT_4.mp4', style_attr=style_attr, body_parts=['Nose'])
    >>> test.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        video_path: Union[str, os.PathLike],
        body_parts: List[str],
        style_attr: Dict[str, Any],
    ):

        check_file_exist_and_readable(file_path=config_path)
        check_file_exist_and_readable(file_path=video_path)
        check_if_keys_exist_in_dict(
            data=style_attr,
            key=STYLE_KEYS,
            name=f"{self.__class__.__name__} style_attr",
        )
        _, self.video_name, _ = get_fn_ext(video_path)
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(
                expected_file_path=self.roi_coordinates_path
            )
        self.read_roi_data()
        self.roi_dict, shape_names = slice_roi_dict_for_video(
            data=self.roi_dict, video_name=self.video_name
        )
        self.data_path = os.path.join(
            self.outlier_corrected_dir, f"{self.video_name}.{self.file_type}"
        )
        if not os.path.isfile(self.data_path):
            raise NoFilesFoundError(
                msg=f"SIMBA ERROR: Could not find the file at path {self.data_path}. Make sure the data file exist to create ROI visualizations",
                source=self.__class__.__name__,
            )
        if not os.path.exists(self.roi_features_save_dir):
            os.makedirs(self.roi_features_save_dir)
        self.save_path = os.path.join(
            self.roi_features_save_dir, f"{self.video_name}.mp4"
        )
        check_valid_lst(
            data=body_parts,
            source=f"{self.__class__.__name__} body-parts",
            valid_dtypes=(str,),
            min_len=1,
        )
        for bp in body_parts:
            if bp not in self.body_parts_lst:
                raise BodypartColumnNotFoundError(
                    msg=f"The body-part {bp} is not a valid body-part in the SimBA project. Options: {self.body_parts_lst}",
                    source=self.__class__.__name__,
                )
        self.roi_feature_creator = ROIFeatureCreator(
            config_path=config_path,
            body_parts=body_parts,
            append_data=False,
            data_path=self.data_path,
        )
        self.roi_feature_creator.run()
        self.bp_lk = self.roi_feature_creator.bp_lk
        self.animal_bp_names = [f"{v[0]} {v[1]}" for v in self.bp_lk.values()]
        self.animal_names = [v[0] for v in self.bp_lk.values()]
        self.video_meta_data = get_video_meta_data(video_path)
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.cap = cv2.VideoCapture(video_path)
        self.max_dim = max(
            self.video_meta_data["width"], self.video_meta_data["height"]
        )
        self.circle_size = int(
            TextOptions.RADIUS_SCALER.value
            / (TextOptions.RESOLUTION_SCALER.value / self.max_dim)
        )
        self.font_size = float(
            TextOptions.FONT_SCALER.value
            / (TextOptions.RESOLUTION_SCALER.value / self.max_dim)
        )
        self.spacing_scale = int(
            TextOptions.SPACE_SCALER.value
            / (TextOptions.RESOLUTION_SCALER.value / self.max_dim)
        )
        check_video_and_data_frm_count_align(
            video=video_path, data=self.data_path, name=video_path, raise_error=False
        )
        self.style_attr = style_attr
        self.direct_viable = self.roi_feature_creator.roi_directing_viable
        self.data_df = read_df(file_path=self.data_path, file_type=self.file_type).reset_index(drop=True)
        self.shape_dicts = self.__create_shape_dicts()
        self.directing_df = self.roi_feature_creator.dr

    def __calc_text_locs(self):
        add_spacer = TextOptions.FIRST_LINE_SPACING.value
        self.loc_dict = {}
        for animal_cnt, animal_data in self.bp_lk.items():
            animal, animal_bp, _ = animal_data
            animal_name = f"{animal} {animal_bp}"
            self.loc_dict[animal_name] = {}
            self.loc_dict[animal] = {}
            for shape in self.shape_names:
                self.loc_dict[animal_name][shape] = {}
                self.loc_dict[animal_name][shape][
                    "in_zone_text"
                ] = f"{shape} {animal_name} in zone"
                self.loc_dict[animal_name][shape][
                    "distance_text"
                ] = f"{shape} {animal_name} distance"
                self.loc_dict[animal_name][shape]["in_zone_text_loc"] = (
                    (self.video_meta_data["width"] + 5),
                    (
                        self.video_meta_data["height"]
                        - (self.video_meta_data["height"] + 10)
                        + self.spacing_scale * add_spacer
                    ),
                )
                self.loc_dict[animal_name][shape]["in_zone_data_loc"] = (
                    int(self.img_w_border_w - (self.img_w_border_w / 8)),
                    (
                        self.video_meta_data["height"]
                        - (self.video_meta_data["height"] + 10)
                        + self.spacing_scale * add_spacer
                    ),
                )
                add_spacer += 1
                self.loc_dict[animal_name][shape]["distance_text_loc"] = (
                    (self.video_meta_data["width"] + 5),
                    (
                        self.video_meta_data["height"]
                        - (self.video_meta_data["height"] + 10)
                        + self.spacing_scale * add_spacer
                    ),
                )
                self.loc_dict[animal_name][shape]["distance_data_loc"] = (
                    int(self.img_w_border_w - (self.img_w_border_w / 8)),
                    (
                        self.video_meta_data["height"]
                        - (self.video_meta_data["height"] + 10)
                        + self.spacing_scale * add_spacer
                    ),
                )
                add_spacer += 1
                if self.direct_viable and self.style_attr[DIRECTIONALITY]:
                    self.loc_dict[animal][shape] = {}
                    self.loc_dict[animal][shape][
                        "directing_text"
                    ] = f"{shape} {animal} facing"
                    self.loc_dict[animal][shape]["directing_text_loc"] = (
                        (self.video_meta_data["width"] + 5),
                        (
                            self.video_meta_data["height"]
                            - (self.video_meta_data["height"] + 10)
                            + self.spacing_scale * add_spacer
                        ),
                    )
                    self.loc_dict[animal][shape]["directing_data_loc"] = (
                        int(self.img_w_border_w - (self.img_w_border_w / 8)),
                        (
                            self.video_meta_data["height"]
                            - (self.video_meta_data["height"] + 10)
                            + self.spacing_scale * add_spacer
                        ),
                    )
                    add_spacer += 1

    def __create_shape_dicts(self):
        shape_dicts = {}
        for shape, df in self.roi_dict.items():
            if not df["Name"].is_unique:
                df = df.drop_duplicates(subset=["Name"], keep="first")
                DuplicateNamesWarning(
                    msg=f'Some of your ROIs with the same shape ({shape}) has the same names for video {self.video_name}. E.g., you have two rectangles named "My rectangle". SimBA prefers ROI shapes with unique names. SimBA will keep one of the unique shape names and drop the rest.',
                    source=self.__class__.__name__,
                )
            d = df.set_index("Name").to_dict(orient="index")
            shape_dicts = {**shape_dicts, **d}
        return shape_dicts

    def __insert_texts(self, shape_df):
        for cnt, animal_data in self.bp_lk.items():
            animal, animal_bp, _ = animal_data
            animal_name = f"{animal} {animal_bp}"
            for _, shape in shape_df.iterrows():
                shape_name, shape_color = shape["Name"], shape["Color BGR"]
                cv2.putText(
                    self.img_w_border,
                    self.loc_dict[animal_name][shape_name]["in_zone_text"],
                    self.loc_dict[animal_name][shape_name]["in_zone_text_loc"],
                    self.font,
                    self.font_size,
                    shape_color,
                    1,
                )
                cv2.putText(
                    self.img_w_border,
                    self.loc_dict[animal_name][shape_name]["distance_text"],
                    self.loc_dict[animal_name][shape_name]["distance_text_loc"],
                    self.font,
                    self.font_size,
                    shape_color,
                    1,
                )
                if self.direct_viable:
                    cv2.putText(
                        self.img_w_border,
                        self.loc_dict[animal][shape_name]["directing_text"],
                        self.loc_dict[animal][shape_name]["directing_text_loc"],
                        self.font,
                        self.font_size,
                        shape_color,
                        1,
                    )

    def run(self):
        self.frame_cnt = 0
        while self.cap.isOpened():
            ret, self.img = self.cap.read()
            if ret:
                self.img_w_border = cv2.copyMakeBorder(self.img, 0, 0, 0, self.video_meta_data["width"], borderType=cv2.BORDER_CONSTANT, value=self.style_attr[BORDER_COLOR])
                if self.frame_cnt == 0:
                    self.img_w_border_h, self.img_w_border_w = (self.img_w_border.shape[0], self.img_w_border.shape[1])
                    self.__calc_text_locs()
                    self.writer = cv2.VideoWriter(self.save_path, self.fourcc, self.video_meta_data["fps"], (self.img_w_border_w, self.img_w_border_h))
                self.__insert_texts(self.roi_dict[Keys.ROI_RECTANGLES.value])
                self.__insert_texts(self.roi_dict[Keys.ROI_CIRCLES.value])
                self.__insert_texts(self.roi_dict[Keys.ROI_POLYGONS.value])
                if self.style_attr[POSE]:
                    for animal_name, bp_data in self.animal_bp_dict.items():
                        for bp_cnt, bp in enumerate(zip(bp_data["X_bps"], bp_data["Y_bps"])):
                            bp_cords = self.data_df.loc[self.frame_cnt, list(bp)].values.astype(np.int64)
                            cv2.circle(
                                self.img_w_border,
                                (bp_cords[0], bp_cords[1]),
                                0,
                                self.animal_bp_dict[animal_name]["colors"][bp_cnt],
                                self.circle_size,
                            )
                if self.style_attr[ANIMAL_NAMES]:
                    for animal_name, bp_data in self.animal_bp_dict.items():
                        headers = [bp_data["X_bps"][-1], bp_data["Y_bps"][-1]]
                        bp_cords = self.data_df.loc[
                            self.frame_cnt, headers
                        ].values.astype(np.int64)
                        cv2.putText(
                            self.img_w_border,
                            animal_name,
                            (bp_cords[0], bp_cords[1]),
                            self.font,
                            self.font_size,
                            self.animal_bp_dict[animal_name]["colors"][0],
                            1,
                        )

                self.img_w_border = PlottingMixin.roi_dict_onto_img(
                    img=self.img_w_border,
                    roi_dict=self.roi_dict,
                    circle_size=self.circle_size,
                    show_tags=self.style_attr[ROI_EAR_TAGS],
                    show_center=self.style_attr[ROI_CENTERS],
                )

                for animal_name, shape_name in itertools.product(
                    self.animal_bp_names, self.shape_names
                ):
                    in_zone_col_name = f"{shape_name} {animal_name} in zone"
                    distance_col_name = f"{shape_name} {animal_name} distance"
                    in_zone_value = str(
                        bool(
                            self.roi_feature_creator.out_df.loc[
                                self.frame_cnt, in_zone_col_name
                            ]
                        )
                    )
                    distance_value = round(
                        self.roi_feature_creator.out_df.loc[
                            self.frame_cnt, distance_col_name
                        ],
                        2,
                    )
                    cv2.putText(
                        self.img_w_border,
                        in_zone_value,
                        self.loc_dict[animal_name][shape_name]["in_zone_data_loc"],
                        self.font,
                        self.font_size,
                        self.shape_dicts[shape_name]["Color BGR"],
                        1,
                    )
                    cv2.putText(
                        self.img_w_border,
                        str(distance_value),
                        self.loc_dict[animal_name][shape_name]["distance_data_loc"],
                        self.font,
                        self.font_size,
                        self.shape_dicts[shape_name]["Color BGR"],
                        1,
                    )
                if self.direct_viable and self.style_attr[DIRECTIONALITY]:
                    for animal_name, shape_name in itertools.product(
                        self.animal_names, self.shape_names
                    ):
                        facing_col_name = f"{shape_name} {animal_name} facing"
                        facing_value = self.roi_feature_creator.out_df.loc[
                            self.frame_cnt, facing_col_name
                        ]
                        cv2.putText(
                            self.img_w_border,
                            str(bool(facing_value)),
                            self.loc_dict[animal_name][shape_name][
                                "directing_data_loc"
                            ],
                            self.font,
                            self.font_size,
                            self.shape_dicts[shape_name]["Color BGR"],
                            1,
                        )
                        if facing_value:
                            self.img_w_border = PlottingMixin.insert_directing_line(
                                directing_df=self.directing_df,
                                img=self.img_w_border,
                                shape_name=shape_name,
                                animal_name=animal_name,
                                frame_id=self.frame_cnt,
                                color=self.shape_dicts[shape_name]["Color BGR"],
                                thickness=self.shape_dicts[shape_name]["Thickness"],
                                style=self.style_attr[DIRECTIONALITY_STYLE],
                            )
                self.frame_cnt += 1
                self.writer.write(np.uint8(self.img_w_border))
                print(
                    f"Frame: {self.frame_cnt} / {self.video_meta_data['frame_count']}. Video: {self.video_name} ..."
                )
            else:
                break

        self.timer.stop_timer()
        self.cap.release()
        self.writer.release()
        stdout_success(
            msg=f"Feature video {self.video_name} saved in {self.save_path} directory ...",
            elapsed_time=self.timer.elapsed_time_str,
        )


# style_attr = {'roi_centers': True, 'roi_ear_tags': True, 'directionality': True, 'directionality_style': 'lines', 'border_color': (0, 0, 0), 'pose_estimation': True, 'animal_names': True}
# test = ROIfeatureVisualizer(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini',
#                             video_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_DOT_4.mp4',
#                             style_attr=style_attr,
#                             body_parts=['Nose'])
# test.run()


# style_attr = {'roi_centers': True, 'roi_ear_tags': True, 'directionality': True, 'directionality_style': 'funnel', 'border_color': (0, 128, 0), 'pose_estimation': True, 'animal_names': True}
# test = ROIfeatureVisualizer(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                             video_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi',
#                             style_attr=style_attr,
#                             body_parts=['Nose_1', 'Nose_2'])
# test.run()


# style_attr = {'ROI_centers': True, 'ROI_ear_tags': True, 'Directionality': True, 'Directionality_style': 'Line', 'Border_color': (0, 128, 0), 'Pose_estimation': True}
# test = ROIfeatureVisualizer(config_path='/Users/simon/Desktop/envs/simba_dev/tests/test_data/mouse_open_field/project_folder/project_config.ini', video_name='Video1.mp4', style_attr=style_attr)
# test.run()


# test = ROIfeatureVisualizer(config_path='/Users/simon/Desktop/train_model_project/project_folder/project_config.ini', video_name='Together_1.avi')
# test.run()
# test.save_new_features_files()
