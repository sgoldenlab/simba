__author__ = "Simon Nilsson"

import os
from typing import List, Union
import cv2
import numpy as np
from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.enums import Formats, Options
from simba.utils.errors import NoSpecifiedOutputError, InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df
from simba.utils.checks import check_all_file_names_are_represented_in_video_log, check_str, check_valid_dict, check_filepaths_in_iterable_exist, check_valid_dataframe, check_int

class HeatMapperClfSingleCore(ConfigReader, PlottingMixin):
    """
    Create heatmaps representing the locations of the classified behavior.

    .. note::
       `GitHub visualizations tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.
       For improved run-time, see :meth:`simba.heat_mapper_clf_mp.HeatMapperClfMultiprocess` for multiprocess class.

    .. image:: _static/img/heatmap.png
       :width: 500
       :align: center

    :param str config_path: path to SimBA project config file in Configparser format
    :param bool final_img_setting: If True, then create a single image representing the last frame of the input video
    :param bool video_setting: If True, then create a video of heatmaps.
    :param bool frame_setting: If True, then create individual heatmap frames.
    :param str clf_name: The name of the classified behavior.
    :param str bodypart: The name of the body-part used to infer the location of the classified behavior
    :param Dict style_attr: Dict containing settings for colormap, bin-size, max scale, and smooothing operations. For example: {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 50, 'max_scale': 'auto'}.

    :example:
    >>> test = HeatMapperClfSingleCore(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini",
    >>>                  style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 50, 'max_scale': 'auto'},
    >>>                  final_img_setting=True,
    >>>                  video_setting=True,
    >>>                  frame_setting=False,
    >>>                  bodypart='Ear_left',
    >>>                  clf_name='straub_tail',
    >>>                  data_paths=[r"C:\troubleshooting\RAT_NOR\project_folder\csv\test\2022-06-20_NOB_DOT_4.csv"])
    >>> test.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 bodypart: str,
                 clf_name: str,
                 data_paths: List[str],
                 style_attr: dict,
                 final_img_setting: bool = True,
                 video_setting: bool = False,
                 frame_setting: bool = False):



        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        if (not frame_setting) and (not video_setting) and (not final_img_setting):
            raise NoSpecifiedOutputError(msg="Please select either heatmap videos, frames, and/or final image.")
        check_filepaths_in_iterable_exist(file_paths=data_paths, name=f'{self.__class__.__name__} data_paths')
        check_str(name=f'{self.__class__.__name__} clf_name', value=clf_name)
        check_str(name=f'{self.__class__.__name__} bodypart', value=bodypart)
        check_valid_dict(x=style_attr, required_keys=('max_scale', 'bin_size', 'shading', 'palette'))
        self.frame_setting, self.video_setting, self.final_img_setting = frame_setting, video_setting, final_img_setting
        self.bin_size, self.max_scale, self.palette, self.shading = (style_attr["bin_size"], style_attr["max_scale"], style_attr["palette"], style_attr["shading"])
        check_str(name=f'{self.__class__.__name__} shading', value=style_attr["shading"], options=Options.HEATMAP_SHADING_OPTIONS.value)
        check_int(name=f'{self.__class__.__name__} bin_size', value=style_attr["bin_size"], min_value=1)
        self.clf_name, self.data_paths, self.bp = clf_name, data_paths, bodypart
        if not os.path.exists(self.heatmap_clf_location_dir): os.makedirs(self.heatmap_clf_location_dir)
        self.bp_lst = [f"{self.bp}_x", f"{self.bp}_y"]
        self.timer = SimbaTimer(start=True)

    def __calculate_max_scale(self, clf_array: np.array):
        return np.round(np.max(np.max(clf_array[-1], axis=0)), 3)

    def run(self):
        print(f"Processing heatmaps for {len(self.data_paths)} video(s)...")
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            print(f'Plotting heatmap classification map for video {self.video_name}...')
            self.video_info, self.px_per_mm, self.fps = self.read_video_info(video_name=self.video_name)
            self.width, self.height = int(self.video_info["Resolution_width"].values[0]), int(self.video_info["Resolution_height"].values[0])
            if self.video_setting:
                self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
                self.video_save_path = os.path.join(self.heatmap_clf_location_dir, f"{self.video_name}.mp4")
                self.writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.fps, (self.width, self.height))
            if self.frame_setting:
                self.save_video_folder = os.path.join(self.heatmap_clf_location_dir, self.video_name)
                if not os.path.exists(self.save_video_folder):
                    os.makedirs(self.save_video_folder)
            self.data_df = read_df(file_path=file_path, file_type=self.file_type)
            check_valid_dataframe(df=self.data_df, required_fields=[self.clf_name] + self.bp_lst, valid_dtypes=Formats.NUMERIC_DTYPES.value)
            bp_data = self.data_df[self.bp_lst].values.astype(np.int32)
            clf_data = self.data_df[self.clf_name].values.astype(np.int32)
            if len(np.unique(clf_data)) == 1:
                raise InvalidInputError(msg=f'Cannot plot heatmap for behavior {self.clf_name} in video {self.video_name}. The behavior is classified as {np.unique(clf_data)} in every single frame.')
            grid, aspect_ratio = GeometryMixin.bucket_img_into_grid_square(img_size=(self.width, self.height), bucket_grid_size_mm=self.bin_size, px_per_mm=self.px_per_mm, add_correction=False)
            clf_data = GeometryMixin().cumsum_bool_geometries(data=bp_data, geometries=grid, bool_data=clf_data, fps=self.fps, verbose=False)
            if self.max_scale == "auto":
                self.max_scale = max(1, self.__calculate_max_scale(clf_array=clf_data))
            if self.final_img_setting:
                file_name = os.path.join(self.heatmap_clf_location_dir, f"{self.video_name}_final_frm.png")
                self.make_location_heatmap_plot(frm_data=clf_data[-1:, :, :][0],
                                                max_scale=self.max_scale,
                                                palette=self.palette,
                                                aspect_ratio=aspect_ratio,
                                                file_name=file_name,
                                                shading=self.shading,
                                                img_size=(self.width, self.height))
                print(f"Final heatmap image saved at {file_name}.")

            if self.video_setting or self.frame_setting:
                for frm_cnt, cumulative_frm_idx in enumerate(range(clf_data.shape[0])):
                    frm_data = clf_data[cumulative_frm_idx, :, :]
                    img = self.make_location_heatmap_plot(frm_data=frm_data,
                                                          max_scale=self.max_scale,
                                                          palette=self.palette,
                                                          aspect_ratio=aspect_ratio,
                                                          shading=self.shading,
                                                          img_size=(self.width, self.height))[:,:,:3]

                    if self.video_setting:
                        self.writer.write(img)
                    if self.frame_setting:
                        frame_save_path = os.path.join(self.save_video_folder, f"{frm_cnt}.png")
                        cv2.imwrite(frame_save_path, img)
                    print(f"Created heatmap frame: {frm_cnt+1} / {len(self.data_df)}. Video: {self.video_name} ({file_cnt + 1}/{len(self.data_paths)})")
            if self.video_setting:
                self.writer.release()
            video_timer.stop_timer()
            print(f"Heatmap plot for video {self.video_name} saved at {self.heatmap_clf_location_dir} (elapsed time: {video_timer.elapsed_time_str}s)...")

        self.timer.stop_timer()
        stdout_success(msg=f"All heatmap visualizations created in {self.heatmap_clf_location_dir} directory", elapsed_time=self.timer.elapsed_time_str)


# test = HeatMapperClfSingleCore(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini",
#                      style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 50, 'max_scale': 'auto'},
#                      final_img_setting=True,
#                      video_setting=True,
#                      frame_setting=False,
#                      bodypart='Ear_left',
#                      clf_name='straub_tail',
#                      data_paths=[r"C:\troubleshooting\RAT_NOR\project_folder\csv\test\2022-06-20_NOB_DOT_4.csv"])
# test.run()





# test = HeatMapperClfSingleCore(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini',
#                      style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 75, 'max_scale': 'auto'},
#                      final_img_setting=False,
#                      video_setting=True,
#                      frame_setting=False,
#                      bodypart='Nose_1',
#                      clf_name='Attack',
#                      files_found=['/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/csv/machine_results/Together_3.csv'])
# test.run()
