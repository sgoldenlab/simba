__author__ = "Simon Nilsson"

import functools
import itertools
import multiprocessing
import os
import platform
from typing import Any, Dict, Optional, Union

import cv2

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.roi_tools.ROI_feature_analyzer import ROIFeatureCreator
from simba.utils.checks import (check_file_exist_and_readable, check_instance,
                                check_int)
from simba.utils.enums import TextOptions
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext,
                                    get_video_meta_data, read_df,
                                    remove_a_folder)


class ROIfeatureVisualizerMultiprocess(ConfigReader, PlottingMixin):
    """
    Visualize features that depend on the relationships between the location of the animals and user-defined
    ROIs. E.g., distances to centroids of ROIs, cumulative time spent in ROIs, if animals are directing towards ROIs
    etc.

    :param str config_path: Path to SimBA project config file in Configparser format
    :param str video_name: Name of video to create feature visualizations for.
    :param dict style_attr: User-defined styles (sizes, colors etc.)
    :param int cores: Number of cores to use.

    .. note:
       `Tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-5-visualizing-roi-features>`__.

    Examples
    ----------
    >>> style_attr = {'ROI_centers': True, 'ROI_ear_tags': True, 'Directionality': True, 'Directionality_style': 'Funnel', 'Border_color': (0, 128, 0), 'Pose_estimation': True}
    >>> _ = ROIfeatureVisualizerMultiprocess(config_path='test_/project_folder/project_config.ini', video_name='Together_1.avi', style_attr=style_attr, core_cnt=3).run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        video_name: str,
        style_attr: Dict[str, Any],
        core_cnt: Optional[int] = -1,
    ):

        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        check_int(
            name=f"{self.__class__.__name__} core_cnt",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
        )
        check_instance(
            source=f"{self.__class__.__name__} core_cnt",
            instance=style_attr,
            accepted_types=(dict,),
        )
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        _, self.video_name, _ = get_fn_ext(video_name)
        self.core_cnt, self.style_attr = core_cnt, style_attr
        self.save_path = os.path.join(
            self.roi_features_save_dir, f"{self.video_name}.mp4"
        )
        if not os.path.exists(self.roi_features_save_dir):
            os.makedirs(self.roi_features_save_dir)
        self.save_temp_dir = os.path.join(self.roi_features_save_dir, "temp")
        if os.path.exists(self.save_temp_dir):
            remove_a_folder(folder_dir=self.save_temp_dir)
        os.makedirs(self.save_temp_dir)
        self.roi_feature_creator = ROIFeatureCreator(config_path=config_path)
        self.file_in_path = os.path.join(
            self.outlier_corrected_dir, f"{self.video_name}.{self.file_type}"
        )
        self.video_path = os.path.join(self.project_path, "videos", video_name)
        check_file_exist_and_readable(file_path=self.file_in_path)
        self.roi_feature_creator.features_files = [self.file_in_path]
        self.roi_feature_creator.files_found = [self.file_in_path]
        self.roi_feature_creator.run()
        self.video_meta_data = get_video_meta_data(self.video_path)
        self.scalers = {}
        self.max_dim = max(
            self.video_meta_data["width"], self.video_meta_data["height"]
        )
        self.scalers["circle_size"] = int(
            TextOptions.RADIUS_SCALER.value
            / (TextOptions.RESOLUTION_SCALER.value / self.max_dim)
        )
        self.scalers["font_size"] = float(
            TextOptions.FONT_SCALER.value
            / (TextOptions.RESOLUTION_SCALER.value / self.max_dim)
        )
        self.scalers["spacing_size"] = int(
            TextOptions.SPACE_SCALER.value
            / (TextOptions.RESOLUTION_SCALER.value / self.max_dim)
        )
        self.data_df = read_df(self.file_in_path, self.file_type)
        self.bp_names = self.roi_feature_creator.roi_analyzer.bp_dict
        self.video_recs = self.roi_feature_creator.roi_analyzer.video_recs
        self.video_circs = self.roi_feature_creator.roi_analyzer.video_circs
        self.video_polys = self.roi_feature_creator.roi_analyzer.video_polys
        self.shape_dicts = {}
        for df in [self.video_recs, self.video_circs, self.video_polys]:
            d = df.set_index("Name").to_dict(orient="index")
            self.shape_dicts = {**self.shape_dicts, **d}
        self.video_shapes = list(
            itertools.chain(
                self.video_recs["Name"].unique(),
                self.video_circs["Name"].unique(),
                self.video_polys["Name"].unique(),
            )
        )
        self.roi_directing_viable = self.roi_feature_creator.roi_directing_viable
        if self.roi_directing_viable:
            self.directing_data = self.roi_feature_creator.directing_analyzer.results_df
            self.directing_data = self.directing_data[
                self.directing_data["Video"] == self.video_name
            ]
            if ("Directionality_roi_subset" in style_attr.keys()) and (
                type(style_attr["Directionality_roi_subset"]) == list
            ):
                self.directing_data = self.directing_data[
                    self.directing_data["ROI"].isin(
                        style_attr["Directionality_roi_subset"]
                    )
                ].reset_index(drop=True)
        else:
            self.directing_data = None
        self.roi_feature_creator.out_df.fillna(0, inplace=True)

    def __calc_text_locs(self):
        add_spacer = 2
        self.loc_dict = {}
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap.set(0, 1)
        ret, img = self.cap.read()
        self.img_w_border = cv2.copyMakeBorder(
            img,
            0,
            0,
            0,
            int(self.video_meta_data["width"]),
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        self.img_w_border_h, self.img_w_border_w = (
            self.img_w_border.shape[0],
            self.img_w_border.shape[1],
        )

        for animal_cnt, animal_name in enumerate(self.multi_animal_id_list):
            self.loc_dict[animal_name] = {}
            for shape in self.video_shapes:
                self.loc_dict[animal_name][shape] = {}
                self.loc_dict[animal_name][shape]["in_zone_text"] = "{} {} {}".format(
                    shape, animal_name, "in zone"
                )
                self.loc_dict[animal_name][shape]["distance_text"] = "{} {} {}".format(
                    shape, animal_name, "distance"
                )
                self.loc_dict[animal_name][shape]["in_zone_text_loc"] = (
                    (self.video_meta_data["width"] + 5),
                    (
                        self.video_meta_data["height"]
                        - (self.video_meta_data["height"] + 10)
                        + self.scalers["spacing_size"] * add_spacer
                    ),
                )
                self.loc_dict[animal_name][shape]["in_zone_data_loc"] = (
                    int(self.img_w_border_w - (self.img_w_border_w / 8)),
                    (
                        self.video_meta_data["height"]
                        - (self.video_meta_data["height"] + 10)
                        + self.scalers["spacing_size"] * add_spacer
                    ),
                )
                add_spacer += 1
                self.loc_dict[animal_name][shape]["distance_text_loc"] = (
                    (self.video_meta_data["width"] + 5),
                    (
                        self.video_meta_data["height"]
                        - (self.video_meta_data["height"] + 10)
                        + self.scalers["spacing_size"] * add_spacer
                    ),
                )
                self.loc_dict[animal_name][shape]["distance_data_loc"] = (
                    int(self.img_w_border_w - (self.img_w_border_w / 8)),
                    (
                        self.video_meta_data["height"]
                        - (self.video_meta_data["height"] + 10)
                        + self.scalers["spacing_size"] * add_spacer
                    ),
                )
                add_spacer += 1
                if self.roi_directing_viable and self.style_attr["Directionality"]:
                    self.loc_dict[animal_name][shape]["directing_text"] = (
                        "{} {} {}".format(shape, animal_name, "facing")
                    )
                    self.loc_dict[animal_name][shape]["directing_text_loc"] = (
                        (self.video_meta_data["width"] + 5),
                        (
                            self.video_meta_data["height"]
                            - (self.video_meta_data["height"] + 10)
                            + self.scalers["spacing_size"] * add_spacer
                        ),
                    )
                    self.loc_dict[animal_name][shape]["directing_data_loc"] = (
                        int(self.img_w_border_w - (self.img_w_border_w / 8)),
                        (
                            self.video_meta_data["height"]
                            - (self.video_meta_data["height"] + 10)
                            + self.scalers["spacing_size"] * add_spacer
                        ),
                    )
                    add_spacer += 1

    def run(self):

        self.timer = SimbaTimer(start=True)
        self.__calc_text_locs()
        data_arr, frm_per_core = self.split_and_group_df(
            df=self.roi_feature_creator.out_df,
            splits=self.core_cnt,
            include_split_order=True,
        )
        print(
            f"Creating ROI feature images, multiprocessing (determined chunksize: {self.multiprocess_chunksize}, cores: {self.core_cnt})..."
        )
        with multiprocessing.Pool(
            self.core_cnt, maxtasksperchild=self.maxtasksperchild
        ) as pool:
            constants = functools.partial(
                self.roi_feature_visualizer_mp,
                text_locations=self.loc_dict,
                scalers=self.scalers,
                video_meta_data=self.video_meta_data,
                shape_info=self.shape_dicts,
                style_attr=self.style_attr,
                save_temp_dir=self.save_temp_dir,
                directing_data=self.directing_data,
                video_path=self.video_path,
                directing_viable=self.roi_directing_viable,
                animal_names=self.multi_animal_id_list,
                tracked_bps=self.bp_names,
                animal_bps=self.animal_bp_dict,
            )
            for cnt, result in enumerate(
                pool.imap(constants, data_arr, chunksize=self.multiprocess_chunksize)
            ):
                print(
                    f"Image {int(frm_per_core * (result + 1))}/{len(self.data_df)}, Video {self.video_name}..."
                )
            print(f"Joining {self.video_name} multi-processed video...")
            concatenate_videos_in_folder(
                in_folder=self.save_temp_dir,
                save_path=self.save_path,
                video_format="mp4",
                remove_splits=True,
            )
            self.timer.stop_timer()
            pool.terminate()
            pool.join()
            stdout_success(
                msg=f"Video {self.video_name} complete. Video saved in project_folder/frames/output/ROI_features.",
                elapsed_time=self.timer.elapsed_time_str,
            )


# style_attr = {'ROI_centers': True,
#               'ROI_ear_tags': True,
#               'Directionality': True,
#               'Directionality_style': 'Funnel',
#               'Border_color': (0, 128, 0),
#               'Pose_estimation': True,
#               'Directionality_roi_subset': ['My_polygon']}

# roi_feature_visualizer = ROIfeatureVisualizerMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                                           video_name='Together_1.avi',
#                                                           style_attr=style_attr,
#                                                           core_cnt=3)
# roi_feature_visualizer.run()


# style_attr = {'ROI_centers': True,
#               'ROI_ear_tags': True,
#               'Directionality': True,
#               'Directionality_style': 'Funnel',
#               'Border_color': (0, 128, 0),
#               'Pose_estimation': True}
# roi_feature_visualizer = ROIfeatureVisualizerMultiprocess(config_path='/Users/simon/Desktop/envs/simba_dev/tests/test_data/mouse_open_field/project_folder/project_config.ini',
#                                                           video_name='Video1.mp4',
#                                                           style_attr=style_attr,
#                                                           core_cnt=3)
# roi_feature_visualizer.create_visualization()
#
# style_attr = {'ROI_centers': True, 'ROI_ear_tags': True, 'Directionality': True, 'Directionality_style': 'Funnel', 'Border_color': (0, 128, 0), 'Pose_estimation': True}
# test = ROIfeatureVisualizerMultiprocess(config_path='/Users/simon/Desktop/envs/simba_dev/tests/test_data/mouse_open_field/project_folder/project_config.ini', video_name='Video1.mp4', style_attr=style_attr, core_cnt=5)
# test.create_visualization()
