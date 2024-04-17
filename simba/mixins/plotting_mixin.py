__author__ = "Simon Nilsson"
import io
import itertools
import os
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import imutils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import plotly.graph_objs as go
import plotly.io as pio
import seaborn as sns
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numba import njit
from PIL import Image

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import simba
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_instance, check_str,
                                check_that_column_exist, check_valid_array,
                                check_valid_lst, check_if_valid_rgb_tuple, check_int, check_float, check_if_keys_exist_in_dict)
from simba.utils.enums import Formats, Options, TextOptions
from simba.utils.errors import InvalidInputError
from simba.utils.lookups import (get_categorical_palettes, get_color_dict,
                                 get_named_colors)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_frm_of_video


class PlottingMixin(object):
    """
    Methods for visualizations
    """

    def __init__(self):
        pass

    def create_gantt_img(
        self,
        bouts_df: pd.DataFrame,
        clf_name: str,
        image_index: int,
        fps: int,
        gantt_img_title: str,
    ):
        """
        Helper to create a single gantt plot based on the data preceeding the input image

        :param pd.DataFrame bouts_df: ataframe holding information on individual bouts created by :meth:`simba.misc_tools.get_bouts_for_gantt`.
        :param str clf_name: Name of the classifier.
        :param int image_index: The count of the image. E.g., ``1000`` will create a gantt image representing frame 1-1000.
        :param int fps: The fps of the input video.
        :param str gantt_img_title: Title of the image.
        :return np.ndarray
        """

        fig, ax = plt.subplots()
        fig.suptitle(gantt_img_title, fontsize=24)
        relRows = bouts_df.loc[bouts_df["End_frame"] <= image_index]
        for i, event in enumerate(relRows.groupby("Event")):
            data_event = event[1][["Start_time", "Bout_time"]]
            ax.broken_barh(data_event.values, (4, 4), facecolors="red")
        xLength = (round(image_index / fps)) + 1
        if xLength < 10:
            xLength = 10
        ax.set_xlim(0, xLength)
        ax.set_ylim([0, 12])
        plt.ylabel(clf_name, fontsize=12)
        plt.yticks([])
        plt.xlabel("time(s)", fontsize=12)
        ax.yaxis.set_ticklabels([])
        ax.grid(True)
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)
        ar = np.asarray(image)
        open_cv_image = ar[:, :, ::-1]
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        open_cv_image = cv2.resize(open_cv_image, (640, 480))
        open_cv_image = np.uint8(open_cv_image)
        buffer_.close()
        plt.close(fig)

        return open_cv_image

    def create_single_color_lst(
        self,
        pallete_name: Literal[Options.PALETTE_OPTIONS],
        increments: int,
        as_rgb_ratio: bool = False,
        as_hex: bool = False,
    ) -> List[Union[str, int, float]]:
        """
        Helper to create a color palette of bgr colors in a list.

        :param str pallete_name: Palette name (e.g., 'jet')
        :param int increments: Numbers of colors in the color palette to create.
        :param bool as_rgb_ratio: If True returns the colors as RGB ratios (0-1).
        :param bool as_hex: If True, returns the colors as HEX.
        :return list

        .. note::
           If as_rgb_ratio **AND** as_hex, then returns HEX.


        """
        if as_hex:
            as_rgb_ratio = True
        cmap = cm.get_cmap(pallete_name, increments + 1)
        color_lst = []
        for i in range(cmap.N):
            rgb = list((cmap(i)[:3]))
            if not as_rgb_ratio:
                rgb = [i * 255 for i in rgb]
            rgb.reverse()
            if as_hex:
                rgb = matplotlib.colors.to_hex(rgb)
            color_lst.append(rgb)
        return color_lst

    def remove_a_folder(self, folder_dir: str) -> None:
        """Helper to remove a directory, use for cleaning up smaller multiprocessed videos following concat"""
        shutil.rmtree(folder_dir, ignore_errors=True)

    def split_and_group_df(
        self,
        df: pd.DataFrame,
        splits: int,
        include_row_index: bool = False,
        include_split_order: bool = True,
    ) -> (List[pd.DataFrame], int):
        """
        Helper to split a dataframe for multiprocessing. If include_split_order, then include the group number
        in split data as a column. If include_row_index, includes a column representing the row index in the array,
        which can be helpful for knowing the frame indexes while multiprocessing videos. Returns split data and approximations of number of observations per split.
        """

        if include_row_index:
            row_indices = np.arange(len(df)).reshape(-1, 1)
            df = np.concatenate((df, row_indices), axis=1)
        data_arr = np.array_split(df, splits)
        if include_split_order:
            for df_cnt in range(len(data_arr)):
                data_arr[df_cnt]["group"] = df_cnt
        obs_per_split = len(data_arr[0])

        return data_arr, obs_per_split

    def make_distance_plot(
        self,
        data: np.array,
        line_attr: Dict[int, str],
        style_attr: Dict[str, Any],
        fps: int,
        save_img: bool = False,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Helper to make a single line plot .png image with N lines.

        :param np.array data: Two-dimensional array where rows represent frames and columns represent intertwined x and y coordinates.
        :param dict line_attr: Line color attributes.
        :param dict style_attr: Plot attributes (size, font size, line width etc).
        :param int fps: Video frame rate.
        :param Optionan[str] save_path: Location to store output .png image. If None, then return image.

        .. note::
           `GitHub tutorial/documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-distance-plots>`__.

        :example:
        >>> fps = 10
        >>> data = np.random.random((100,2))
        >>> line_attr = {0: ['Blue'], 1: ['Red']}
        >>> save_path = '/_tests/final_frm.png'
        >>> style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'y_max': 'auto'}
        >>> self.make_distance_plot(fps=fps, data=data, line_attr=line_attr, style_attr=style_attr, save_path=save_path)
        """
        colors = get_color_dict()
        for j in range(data.shape[1]):
            color = colors[line_attr[j][-1]][::-1]
            color = tuple(x / 255 for x in color)
            plt.plot(
                data[:, j],
                color=color,
                linewidth=style_attr["line width"],
                alpha=style_attr["opacity"],
            )

        timer = SimbaTimer(start=True)
        max_x = len(data)
        if style_attr["y_max"] == "auto":
            max_y = np.amax(data)
        else:
            max_y = float(style_attr["y_max"])
        y_ticks_locs = y_lbls = np.round(np.linspace(0, max_y, 10), 2)
        x_ticks_locs = x_lbls = np.linspace(0, max_x, 5)
        x_lbls = np.round((x_lbls / fps), 1)

        plt.xlabel("time (s)")
        plt.ylabel("distance (cm)")
        plt.xticks(
            x_ticks_locs,
            x_lbls,
            rotation="horizontal",
            fontsize=style_attr["font size"],
        )
        plt.yticks(y_ticks_locs, y_lbls, fontsize=style_attr["font size"])
        plt.ylim(0, max_y)
        plt.suptitle(
            "Animal distances", x=0.5, y=0.92, fontsize=style_attr["font size"] + 4
        )
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        img = PIL.Image.open(buffer_)
        img = np.uint8(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        buffer_.close()
        plt.close()
        img = cv2.resize(img, (style_attr["width"], style_attr["height"]))
        timer.stop_timer()
        if save_img:
            cv2.imwrite(save_path, img)
            stdout_success(
                f"Final distance plot saved at {save_path}",
                elapsed_time=timer.elapsed_time_str,
                source=self.__class__.__name__,
            )
        else:
            return img

    def make_probability_plot(
        self, data: pd.Series, style_attr: dict, clf_name: str, fps: int, save_path: str
    ) -> np.ndarray:
        """
        Make a single classifier probability plot png image.

        :param pd.Series data: row representing frames and field representing classification probabilities.
        :param dict line_attr: Line color attributes.
        :param dict style_attr: Image attributes (size, font size, line width etc).
        :param int fps: Video frame rate.
        :param str ot
        :param str save_path: Location to store output .png image.



        .. notes:
          `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-classification-probabilities>`__.

        :example:
        >>> data = pd.Series(np.random.random((100, 1)).flatten())
        >>> style_attr = {'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'blue', 'circle size': 20}
        >>> clf_name='Attack'
        >>> fps=10
        >>> save_path = '/_test/frames/output/probability_plots/Together_1_final_frame.png'

        >>> _ = self.make_probability_plot(data=data, style_attr=style_attr, clf_name=clf_name, fps=fps, save_path=save_path)
        """

        timer = SimbaTimer()
        timer.start_timer()
        if style_attr["y_max"] == "auto":
            max_y = float(data.max().round(2))
        else:
            max_y = float(style_attr["y_max"])
        max_x = len(data)
        plt.plot(
            list(data), color=style_attr["color"], linewidth=style_attr["line width"]
        )
        plt.plot(
            len(data),
            list(data)[-1],
            "o",
            markersize=style_attr["circle size"],
            color=style_attr["color"],
        )
        plt.ylim([0, max_y])
        plt.ylabel(clf_name, fontsize=style_attr["font size"])

        y_ticks_locs = y_lbls = np.round(np.linspace(0, max_y, 10), 2)
        x_ticks_locs = x_lbls = np.linspace(0, max_x, 5)
        x_lbls = np.round((x_lbls / fps), 1)
        plt.xlabel("Time (s)", fontsize=style_attr["font size"] + 4)
        plt.grid()
        plt.xticks(
            x_ticks_locs,
            x_lbls,
            rotation="horizontal",
            fontsize=style_attr["font size"],
        )
        plt.yticks(y_ticks_locs, y_lbls, fontsize=style_attr["font size"])
        plt.suptitle(
            "{} {}".format(clf_name, "probability"),
            x=0.5,
            y=0.92,
            fontsize=style_attr["font size"] + 4,
        )
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)
        ar = np.asarray(image)
        img = cv2.cvtColor(ar, cv2.COLOR_RGB2BGR)
        img = np.uint8(cv2.resize(img, (style_attr["width"], style_attr["height"])))
        buffer_.close()
        plt.close()
        timer.stop_timer()
        cv2.imwrite(save_path, img)
        stdout_success(
            msg=f"Final distance plot saved at {save_path}",
            elapsed_time=timer.elapsed_time_str,
            source=self.__class__.__name__,
        )

    def make_gantt_plot(
        self,
        data_df: pd.DataFrame,
        bouts_df: pd.DataFrame,
        clf_names: List[str],
        fps: int,
        style_attr: dict,
        video_name: str,
        save_path: str,
    ) -> None:

        video_timer = SimbaTimer(start=True)
        colours = get_named_colors()
        colour_tuple_x = list(np.arange(3.5, 203.5, 5))
        fig, ax = plt.subplots()
        for i, event in enumerate(bouts_df.groupby("Event")):
            for x in clf_names:
                if event[0] == x:
                    ix = clf_names.index(x)
                    data_event = event[1][["Start_time", "Bout_time"]]
                    ax.broken_barh(
                        data_event.values,
                        (colour_tuple_x[ix], 3),
                        facecolors=colours[ix],
                    )

        x_ticks_locs = x_lbls = np.round(np.linspace(0, len(data_df) / fps, 6))
        ax.set_xticks(x_ticks_locs)
        ax.set_xticklabels(x_lbls)
        ax.set_ylim(0, colour_tuple_x[len(clf_names)])
        ax.set_yticks(np.arange(5, 5 * len(clf_names) + 1, 5))
        ax.set_yticklabels(clf_names, rotation=style_attr["font rotation"])
        ax.tick_params(axis="both", labelsize=style_attr["font size"])
        plt.xlabel("Session (s)", fontsize=style_attr["font size"] + 3)
        ax.yaxis.grid(True)
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)
        ar = np.asarray(image)
        open_cv_image = cv2.cvtColor(ar, cv2.COLOR_RGB2BGR)
        open_cv_image = cv2.resize(
            open_cv_image, (style_attr["width"], style_attr["height"])
        )
        frame = np.uint8(open_cv_image)
        buffer_.close()
        plt.close(fig)
        cv2.imwrite(save_path, frame)
        video_timer.stop_timer()
        stdout_success(
            msg=f"Final gantt frame for video {video_name} saved at {save_path}",
            elapsed_time=video_timer.elapsed_time_str,
            source=self.__class__.__name__,
        )

    @staticmethod
    def make_clf_heatmap_plot(
        frm_data: np.array,
        max_scale: float,
        palette: Literal[Options.PALETTE_OPTIONS],
        aspect_ratio: float,
        shading: Literal["gouraud", "flat"],
        clf_name: str,
        img_size: Tuple[int, int],
        file_name: Optional[str] = None,
        final_img: bool = False,
    ):

        cum_df = pd.DataFrame(frm_data).reset_index()
        cum_df = cum_df.melt(
            id_vars="index",
            value_vars=None,
            var_name=None,
            value_name="seconds",
            col_level=None,
        ).rename(columns={"index": "vertical_idx", "variable": "horizontal_idx"})
        cum_df["color"] = (
            (cum_df["seconds"].astype(float) / float(max_scale))
            .round(2)
            .clip(upper=100)
        )
        color_array = np.zeros(
            (
                len(cum_df["vertical_idx"].unique()),
                len(cum_df["horizontal_idx"].unique()),
            )
        )
        for i in range(color_array.shape[0]):
            for j in range(color_array.shape[1]):
                value = cum_df["color"][
                    (cum_df["horizontal_idx"] == j) & (cum_df["vertical_idx"] == i)
                ].values[0]
                color_array[i, j] = value
        color_array = color_array * max_scale
        matplotlib.font_manager._get_font.cache_clear()
        plt.close("all")
        fig = plt.figure()
        im_ratio = color_array.shape[0] / color_array.shape[1]
        plt.pcolormesh(
            color_array,
            shading=shading,
            cmap=palette,
            rasterized=True,
            alpha=1,
            vmin=0.0,
            vmax=float(max_scale),
        )
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        plt.tick_params(axis="both", which="both", length=0)
        cb = plt.colorbar(pad=0.0, fraction=0.023 * im_ratio)
        cb.ax.tick_params(size=0)
        cb.outline.set_visible(False)
        cb.set_label("{} (seconds)".format(clf_name), rotation=270, labelpad=10)
        plt.tight_layout()
        # plt.gca().set_aspect(aspect_ratio)
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)
        image = np.uint8(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        image = cv2.resize(image, img_size)
        buffer_.close()
        plt.close()
        if final_img:
            cv2.imwrite(file_name, image)
            stdout_success(
                msg=f"Final classifier heatmap image saved at at {file_name}",
                source="make_clf_heatmap_plot",
            )
        else:
            return image

    @staticmethod
    def make_location_heatmap_plot(
        frm_data: np.array,
        max_scale: float,
        palette: Literal[Options.PALETTE_OPTIONS],
        aspect_ratio: float,
        shading: str,
        img_size: Tuple[int, int],
        file_name: str or None = None,
        final_img: bool = False,
    ):

        cum_df = pd.DataFrame(frm_data).reset_index()
        cum_df = cum_df.melt(
            id_vars="index",
            value_vars=None,
            var_name=None,
            value_name="seconds",
            col_level=None,
        ).rename(columns={"index": "vertical_idx", "variable": "horizontal_idx"})
        cum_df["color"] = (
            (cum_df["seconds"].astype(float) / float(max_scale))
            .round(2)
            .clip(upper=100)
        )
        color_array = np.zeros(
            (
                len(cum_df["vertical_idx"].unique()),
                len(cum_df["horizontal_idx"].unique()),
            )
        )
        for i in range(color_array.shape[0]):
            for j in range(color_array.shape[1]):
                value = cum_df["color"][
                    (cum_df["horizontal_idx"] == j) & (cum_df["vertical_idx"] == i)
                ].values[0]
                color_array[i, j] = value
        color_array = color_array * max_scale
        matplotlib.font_manager._get_font.cache_clear()
        plt.close("all")
        fig = plt.figure()
        im_ratio = color_array.shape[0] / color_array.shape[1]
        plt.pcolormesh(
            color_array,
            shading=shading,
            cmap=palette,
            rasterized=True,
            alpha=1,
            vmin=0.0,
            vmax=max_scale,
        )
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        plt.tick_params(axis="both", which="both", length=0)
        cb = plt.colorbar(pad=0.0, fraction=0.023 * im_ratio)
        cb.ax.tick_params(size=0)
        cb.outline.set_visible(False)
        cb.set_label("location (seconds)", rotation=270, labelpad=10)
        plt.tight_layout()
        # plt.gca().set_aspect(aspect_ratio)
        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        image = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, img_size)
        image = np.uint8(image)
        plt.close("all")
        if final_img:
            cv2.imwrite(file_name, image)
            stdout_success(
                msg=f"Final location heatmap image saved at at {file_name}",
                source=PlottingMixin.make_location_heatmap_plot.__class__.__name__,
            )
        else:
            return image
        return None

    def get_bouts_for_gantt(
        self, data_df: pd.DataFrame, clf_name: str, fps: int
    ) -> np.ndarray:
        """
        Helper to detect all behavior bouts for a specific classifier.

        :param pd.DataFrame data_df: Pandas Dataframe with classifier prediction data.
        :param str clf_name: Name of the classifier
        :param int fps: The fps of the input video.
        :return  pd.DataFrame: Holding the start time, end time, end frame, bout time etc of each classified bout.
        """

        boutsList, nameList, startTimeList, endTimeList, endFrameList = (
            [],
            [],
            [],
            [],
            [],
        )
        groupDf = pd.DataFrame()
        v = (data_df[clf_name] != data_df[clf_name].shift()).cumsum()
        u = data_df.groupby(v)[clf_name].agg(["all", "count"])
        m = u["all"] & u["count"].ge(1)
        groupDf["groups"] = data_df.groupby(v).apply(
            lambda x: (x.index[0], x.index[-1])
        )[m]
        for indexes, rows in groupDf.iterrows():
            currBout = list(rows["groups"])
            boutTime = ((currBout[-1] - currBout[0]) + 1) / fps
            startTime = (currBout[0] + 1) / fps
            endTime = (currBout[1]) / fps
            endFrame = currBout[1]
            endTimeList.append(endTime)
            startTimeList.append(startTime)
            boutsList.append(boutTime)
            nameList.append(clf_name)
            endFrameList.append(endFrame)

        return pd.DataFrame(
            list(zip(nameList, startTimeList, endTimeList, endFrameList, boutsList)),
            columns=["Event", "Start_time", "End Time", "End_frame", "Bout_time"],
        )

    def resize_gantt(self, gantt_img: np.array, img_height: int) -> np.ndarray:
        """
        Helper to resize image while retaining aspect ratio.
        """

        return imutils.resize(gantt_img, height=img_height)

    @staticmethod
    def roi_feature_visualizer_mp(
        data: pd.DataFrame,
        text_locations: dict,
        scalers: dict,
        save_temp_dir: str,
        video_meta_data: dict,
        shape_info: dict,
        style_attr: dict,
        directing_viable: bool,
        video_path: str,
        animal_names: list,
        tracked_bps: dict,
        animal_bps: dict,
        directing_data: pd.DataFrame,
    ):

        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        font = cv2.FONT_HERSHEY_COMPLEX

        def __insert_texts(shape_info: dict, img: np.array):
            for shape_name, shape_info in shape_info.items():
                for animal_name in animal_names:
                    shape_color = shape_info["Color BGR"]
                    cv2.putText(
                        img,
                        text_locations[animal_name][shape_name]["in_zone_text"],
                        text_locations[animal_name][shape_name]["in_zone_text_loc"],
                        font,
                        scalers["font_size"],
                        shape_color,
                        1,
                    )
                    cv2.putText(
                        img,
                        text_locations[animal_name][shape_name]["distance_text"],
                        text_locations[animal_name][shape_name]["distance_text_loc"],
                        font,
                        scalers["font_size"],
                        shape_color,
                        1,
                    )
                    if directing_viable and style_attr["Directionality"]:
                        cv2.putText(
                            img,
                            text_locations[animal_name][shape_name]["directing_text"],
                            text_locations[animal_name][shape_name][
                                "directing_text_loc"
                            ],
                            font,
                            scalers["font_size"],
                            shape_color,
                            1,
                        )
            return img

        def __insert_shapes(img: np.array, shape_info: dict):
            for shape_name, shape_info in shape_info.items():
                if shape_info["Shape_type"] == "Rectangle":
                    cv2.rectangle(
                        img,
                        (int(shape_info["topLeftX"]), int(shape_info["topLeftY"])),
                        (
                            int(shape_info["Bottom_right_X"]),
                            int(shape_info["Bottom_right_Y"]),
                        ),
                        shape_info["Color BGR"],
                        int(shape_info["Thickness"]),
                    )
                    if style_attr["ROI_centers"]:
                        center_cord = (
                            (int(shape_info["topLeftX"] + (shape_info["width"] / 2))),
                            (int(shape_info["topLeftY"] + (shape_info["height"] / 2))),
                        )
                        cv2.circle(
                            img,
                            center_cord,
                            scalers["circle_size"],
                            shape_info["Color BGR"],
                            -1,
                        )
                    if style_attr["ROI_ear_tags"]:
                        for tag_data in shape_info["Tags"].values():
                            cv2.circle(
                                img,
                                tag_data,
                                scalers["circle_size"],
                                shape_info["Color BGR"],
                                -1,
                            )

                if shape_info["Shape_type"] == "Circle":
                    cv2.circle(
                        img,
                        (int(shape_info["centerX"]), int(shape_info["centerY"])),
                        shape_info["radius"],
                        shape_info["Color BGR"],
                        int(shape_info["Thickness"]),
                    )
                    if style_attr["ROI_centers"]:
                        cv2.circle(
                            img,
                            (int(shape_info["centerX"]), int(shape_info["centerY"])),
                            scalers["circle_size"],
                            shape_info["Color BGR"],
                            -1,
                        )
                    if style_attr["ROI_ear_tags"]:
                        for tag_data in shape_info["Tags"].values():
                            cv2.circle(
                                img,
                                tag_data,
                                scalers["circle_size"],
                                shape_info["Color BGR"],
                                -1,
                            )

                if shape_info["Shape_type"] == "Polygon":
                    cv2.polylines(
                        img,
                        [shape_info["vertices"]],
                        True,
                        shape_info["Color BGR"],
                        thickness=int(shape_info["Thickness"]),
                    )
                    if style_attr["ROI_centers"]:
                        cv2.circle(
                            img,
                            shape_info["Tags"]["Center_tag"],
                            scalers["circle_size"],
                            shape_info["Color BGR"],
                            -1,
                        )
                    if style_attr["ROI_ear_tags"]:
                        for tag_data in shape_info["Tags"].values():
                            cv2.circle(
                                img,
                                tuple(tag_data),
                                scalers["circle_size"],
                                shape_info["Color BGR"],
                                -1,
                            )

            return img

        def __insert_directing_line(
            directing_data: pd.DataFrame,
            shape_name: str,
            frame_cnt: int,
            shape_info: dict,
            img: np.array,
            video_name: str,
            style_attr: dict,
        ):

            r = directing_data.loc[
                (directing_data["Video"] == video_name)
                & (directing_data["ROI"] == shape_name)
                & (directing_data["Animal"] == animal_name)
                & (directing_data["Frame"] == frame_cnt)
            ]
            if len(r) > 0:
                clr = shape_info[shape_name]["Color BGR"]
                thickness = shape_info[shape_name]["Thickness"]
                if style_attr["Directionality_style"] == "Funnel":
                    convex_hull_arr = (
                        np.array(
                            [
                                [r["ROI_edge_1_x"], r["ROI_edge_1_y"]],
                                [r["ROI_edge_2_x"], r["ROI_edge_2_y"]],
                                [r["Eye_x"], r["Eye_y"]],
                            ]
                        )
                        .reshape(-1, 2)
                        .astype(int)
                    )
                    cv2.fillPoly(img, [convex_hull_arr], clr)

                if style_attr["Directionality_style"] == "Lines":
                    cv2.line(
                        img,
                        (int(r["Eye_x"]), int(r["Eye_y"])),
                        (int(r["ROI_x"]), int(r["ROI_y"])),
                        clr,
                        int(thickness),
                    )

            return img

        group_cnt = int(data["group"].values[0])
        start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
        save_path = os.path.join(save_temp_dir, "{}.mp4".format(str(group_cnt)))
        _, video_name, _ = get_fn_ext(filepath=video_path)
        writer = cv2.VideoWriter(
            save_path,
            fourcc,
            video_meta_data["fps"],
            (video_meta_data["width"] * 2, video_meta_data["height"]),
        )

        cap = cv2.VideoCapture(video_path)
        cap.set(1, start_frm)

        while current_frm < end_frm:
            ret, img = cap.read()
            img = cv2.copyMakeBorder(
                img,
                0,
                0,
                0,
                int(video_meta_data["width"]),
                borderType=cv2.BORDER_CONSTANT,
                value=style_attr["Border_color"],
            )
            img = __insert_texts(shape_info=shape_info, img=img)
            if style_attr["Pose_estimation"]:
                for animal, animal_bp_name in tracked_bps.items():
                    bp_cords = data.loc[current_frm, animal_bp_name].values
                    cv2.circle(
                        img,
                        (int(bp_cords[0]), int(bp_cords[1])),
                        0,
                        animal_bps[animal]["colors"][0],
                        scalers["circle_size"],
                    )
                    cv2.putText(
                        img,
                        animal,
                        (int(bp_cords[0]), int(bp_cords[1])),
                        font,
                        scalers["font_size"],
                        animal_bps[animal]["colors"][0],
                        1,
                    )

            img = __insert_shapes(img=img, shape_info=shape_info)
            #
            for animal_name, shape_name in itertools.product(animal_names, shape_info):
                in_zone_col_name = f'{shape_name} {animal_name} {"in zone"}'
                distance_col_name = f'{shape_name} {animal_name} {"distance"}'
                in_zone_value = str(bool(data.loc[current_frm, in_zone_col_name]))
                distance_value = str(round(data.loc[current_frm, distance_col_name], 2))
                cv2.putText(
                    img,
                    in_zone_value,
                    text_locations[animal_name][shape_name]["in_zone_data_loc"],
                    font,
                    scalers["font_size"],
                    shape_info[shape_name]["Color BGR"],
                    1,
                )
                cv2.putText(
                    img,
                    distance_value,
                    text_locations[animal_name][shape_name]["distance_data_loc"],
                    font,
                    scalers["font_size"],
                    shape_info[shape_name]["Color BGR"],
                    1,
                )
                if directing_viable and style_attr["Directionality"]:
                    facing_col_name = "{} {} {}".format(
                        shape_name, animal_name, "facing"
                    )
                    facing_value = bool(data.loc[current_frm, facing_col_name])
                    cv2.putText(
                        img,
                        str(facing_value),
                        text_locations[animal_name][shape_name]["directing_data_loc"],
                        font,
                        scalers["font_size"],
                        shape_info[shape_name]["Color BGR"],
                        1,
                    )
                    if facing_value:
                        img = __insert_directing_line(
                            directing_data=directing_data,
                            shape_name=shape_name,
                            frame_cnt=current_frm,
                            shape_info=shape_info,
                            img=img,
                            video_name=video_name,
                            style_attr=style_attr,
                        )
            writer.write(img)
            current_frm += 1
            print(
                "Multi-processing video frame {} on core {}...".format(
                    str(current_frm), str(group_cnt)
                )
            )
        cap.release()
        writer.release()

        return group_cnt

    @staticmethod
    def directing_animals_mp(
        data: pd.DataFrame,
        directionality_data: pd.DataFrame,
        bp_names: dict,
        style_attr: dict,
        save_temp_dir: str,
        video_path: str,
        video_meta_data: dict,
        colors: list,
    ):

        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        group_cnt = int(data.iloc[0]["group"])
        start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
        save_path = os.path.join(save_temp_dir, "{}.mp4".format(str(group_cnt)))
        _, video_name, _ = get_fn_ext(filepath=video_path)
        writer = cv2.VideoWriter(
            save_path,
            fourcc,
            video_meta_data["fps"],
            (video_meta_data["width"], video_meta_data["height"]),
        )
        cap = cv2.VideoCapture(video_path)
        cap.set(1, start_frm)
        color = colors[0]

        def __draw_individual_lines(animal_img_data: pd.DataFrame, img: np.array):
            color = colors[0]
            for cnt, (i, r) in enumerate(animal_img_data.iterrows()):
                if style_attr["Direction_color"] == "Random":
                    color = random.sample(colors[0], 1)[0]
                cv2.line(
                    img,
                    (int(r["Eye_x"]), int(r["Eye_y"])),
                    (int(r["Animal_2_bodypart_x"]), int(r["Animal_2_bodypart_y"])),
                    color,
                    style_attr["Direction_thickness"],
                )
                if style_attr["Highlight_endpoints"]:
                    cv2.circle(
                        img,
                        (int(r["Eye_x"]), int(r["Eye_y"])),
                        style_attr["Pose_circle_size"] + 2,
                        color,
                        style_attr["Pose_circle_size"],
                    )
                    cv2.circle(
                        img,
                        (int(r["Animal_2_bodypart_x"]), int(r["Animal_2_bodypart_y"])),
                        style_attr["Pose_circle_size"] + 1,
                        color,
                        style_attr["Pose_circle_size"],
                    )

            return img

        while current_frm < end_frm:
            ret, img = cap.read()
            try:
                if ret:
                    if style_attr["Show_pose"]:
                        bp_data = data.loc[current_frm]
                        for cnt, (animal_name, animal_bps) in enumerate(
                            bp_names.items()
                        ):
                            for bp in zip(animal_bps["X_bps"], animal_bps["Y_bps"]):
                                x_bp, y_bp = bp_data[bp[0]], bp_data[bp[1]]
                                cv2.circle(
                                    img,
                                    (int(x_bp), int(y_bp)),
                                    style_attr["Pose_circle_size"],
                                    bp_names[animal_name]["colors"][cnt],
                                    style_attr["Direction_thickness"],
                                )

                    if current_frm in list(directionality_data["Frame_#"].unique()):
                        img_data = directionality_data[
                            directionality_data["Frame_#"] == current_frm
                        ]
                        unique_animals = img_data["Animal_1"].unique()
                        for animal in unique_animals:
                            animal_img_data = img_data[
                                img_data["Animal_1"] == animal
                            ].reset_index(drop=True)
                            if style_attr["Polyfill"]:
                                convex_hull_arr = animal_img_data.loc[
                                    0, ["Eye_x", "Eye_y"]
                                ].values.reshape(-1, 2)
                                for animal_2 in animal_img_data["Animal_2"].unique():
                                    convex_hull_arr = np.vstack(
                                        (
                                            convex_hull_arr,
                                            animal_img_data[
                                                [
                                                    "Animal_2_bodypart_x",
                                                    "Animal_2_bodypart_y",
                                                ]
                                            ][
                                                animal_img_data["Animal_2"] == animal_2
                                            ].values,
                                        )
                                    ).astype("int")
                                    convex_hull_arr = np.unique(convex_hull_arr, axis=0)
                                    if convex_hull_arr.shape[0] >= 3:
                                        if style_attr["Direction_color"] == "Random":
                                            color = random.sample(colors[0], 1)[0]
                                        cv2.fillPoly(img, [convex_hull_arr], color)
                                    else:
                                        img = __draw_individual_lines(
                                            animal_img_data=animal_img_data, img=img
                                        )

                            else:
                                img = __draw_individual_lines(
                                    animal_img_data=animal_img_data, img=img
                                )

                    img = np.uint8(img)

                    current_frm += 1
                    writer.write(img)
                    print(
                        "Multi-processing video frame {} on core {}...".format(
                            str(current_frm), str(group_cnt)
                        )
                    )

                else:
                    cap.release()
                    writer.release()
                    break

            except IndexError:
                cap.release()
                writer.release()
                break

        return group_cnt

    @staticmethod
    def distance_plotter_mp(
        data: np.array,
        video_setting: bool,
        frame_setting: bool,
        video_name: str,
        video_save_dir: str,
        frame_folder_dir: str,
        style_attr: dict,
        line_attr: dict,
        fps: int,
    ):

        group = int(data[0][0])
        line_data = data[:, 2:]
        color_dict = get_color_dict()
        video_writer = None
        if video_setting:
            fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
            video_save_path = os.path.join(video_save_dir, "{}.mp4".format(str(group)))
            video_writer = cv2.VideoWriter(
                video_save_path,
                fourcc,
                fps,
                (style_attr["width"], style_attr["height"]),
            )

        for i in range(line_data.shape[0]):
            frame_id = int(data[i][1])
            for j in range(line_data.shape[1]):
                color = color_dict[line_attr[j][-1]][::-1]
                color = tuple(x / 255 for x in color)
                plt.plot(
                    line_data[0:i, j],
                    color=color,
                    linewidth=style_attr["line width"],
                    alpha=style_attr["opacity"],
                )

            x_ticks_locs = x_lbls = np.round(np.linspace(0, i, 5))
            x_lbls = np.round((x_lbls / fps), 1)
            plt.ylim(0, style_attr["max_y"])
            plt.xlabel("time (s)")
            plt.ylabel("distance (cm)")
            plt.xticks(
                x_ticks_locs,
                x_lbls,
                rotation="horizontal",
                fontsize=style_attr["font size"],
            )
            plt.yticks(
                style_attr["y_ticks_locs"],
                style_attr["y_ticks_lbls"],
                fontsize=style_attr["font size"],
            )
            plt.suptitle(
                "Animal distances", x=0.5, y=0.92, fontsize=style_attr["font size"] + 4
            )

            buffer_ = io.BytesIO()
            plt.savefig(buffer_, format="png")
            buffer_.seek(0)
            img = PIL.Image.open(buffer_)
            img = np.uint8(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
            buffer_.close()
            plt.close()

            img = cv2.resize(img, (style_attr["width"], style_attr["height"]))
            if video_setting:
                video_writer.write(np.uint8(img))
            if frame_setting:
                frm_name = os.path.join(frame_folder_dir, str(frame_id) + ".png")
                cv2.imwrite(frm_name, np.uint8(img))

            print(
                "Distance frame created: {}, Video: {}, Processing core: {}".format(
                    str(frame_id + 1), video_name, str(group + 1)
                )
            )

        return group

    @staticmethod
    def gantt_creator_mp(
        data: np.array,
        frame_setting: bool,
        video_setting: bool,
        video_save_dir: str,
        frame_folder_dir: str,
        bouts_df: pd.DataFrame,
        clf_names: list,
        colors: list,
        color_tuple: tuple,
        fps: int,
        rotation: int,
        font_size: int,
        width: int,
        height: int,
        video_name: str,
    ):

        group, frame_rng = data[0], data[1:]
        start_frm, end_frm, current_frm = frame_rng[0], frame_rng[-1], frame_rng[0]
        video_writer = None
        if video_setting:
            fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
            video_save_path = os.path.join(video_save_dir, f"{group}.mp4")
            video_writer = cv2.VideoWriter(
                video_save_path, fourcc, fps, (width, height)
            )

        while current_frm < end_frm:
            fig, ax = plt.subplots()
            bout_rows = bouts_df.loc[bouts_df["End_frame"] <= current_frm]
            for i, event in enumerate(bout_rows.groupby("Event")):
                for x in clf_names:
                    if event[0] == x:
                        ix = clf_names.index(x)
                        data_event = event[1][["Start_time", "Bout_time"]]
                        ax.broken_barh(
                            data_event.values,
                            (color_tuple[ix], 3),
                            facecolors=colors[ix],
                        )

            x_ticks_locs = x_lbls = np.round(
                np.linspace(0, round((current_frm / fps), 3), 6)
            )
            ax.set_xticks(x_ticks_locs)
            ax.set_xticklabels(x_lbls)
            ax.set_ylim(0, color_tuple[len(clf_names)])
            ax.set_yticks(np.arange(5, 5 * len(clf_names) + 1, 5))
            ax.tick_params(axis="both", labelsize=font_size)
            ax.set_yticklabels(clf_names, rotation=rotation)
            ax.set_xlabel("Session (s)", fontsize=font_size)
            ax.yaxis.grid(True)
            canvas = FigureCanvas(fig)
            canvas.draw()
            mat = np.array(canvas.renderer._renderer)
            img = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            img = np.uint8(cv2.resize(img, (width, height)))
            if video_setting:
                video_writer.write(img)
            if frame_setting:
                frame_save_name = os.path.join(
                    frame_folder_dir, "{}.png".format(str(current_frm))
                )
                cv2.imwrite(frame_save_name, img)
            plt.close(fig)
            current_frm += 1

            print(
                "Gantt frame created: {}, Video: {}, Processing core: {}".format(
                    str(current_frm + 1), video_name, str(group + 1)
                )
            )

        if video_setting:
            video_writer.release()

        return group

    @staticmethod
    def roi_plotter_mp(
        data: pd.DataFrame,
        loc_dict: dict,
        scalers: dict,
        video_meta_data: dict,
        save_temp_directory: str,
        shape_meta_data: dict,
        video_shape_names: list,
        input_video_path: str,
        body_part_dict: dict,
        roi_analyzer_data: object,
        colors: list,
        style_attr: dict,
        animal_ids: list,
        threshold: float,
    ):

        def __insert_texts(shape_df):
            for animal_name in animal_ids:
                for _, shape in shape_df.iterrows():
                    shape_name, shape_color = shape["Name"], shape["Color BGR"]
                    cv2.putText(
                        border_img,
                        loc_dict[animal_name][shape_name]["timer_text"],
                        loc_dict[animal_name][shape_name]["timer_text_loc"],
                        font,
                        scalers["font_size"],
                        shape_color,
                        TextOptions.TEXT_THICKNESS.value,
                    )
                    cv2.putText(
                        border_img,
                        loc_dict[animal_name][shape_name]["entries_text"],
                        loc_dict[animal_name][shape_name]["entries_text_loc"],
                        font,
                        scalers["font_size"],
                        shape_color,
                        TextOptions.TEXT_THICKNESS.value,
                    )

            return border_img

        font = cv2.FONT_HERSHEY_TRIPLEX
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        group_cnt = int(data["group"].values[0])
        start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
        save_path = os.path.join(save_temp_directory, "{}.mp4".format(str(group_cnt)))
        writer = cv2.VideoWriter(
            save_path,
            fourcc,
            video_meta_data["fps"],
            (video_meta_data["width"] * 2, video_meta_data["height"]),
        )
        cap = cv2.VideoCapture(input_video_path)
        cap.set(1, start_frm)

        while current_frm <= end_frm:
            ret, img = cap.read()
            border_img = cv2.copyMakeBorder(
                img,
                0,
                0,
                0,
                int(video_meta_data["width"]),
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            border_img = __insert_texts(roi_analyzer_data.video_recs)
            border_img = __insert_texts(roi_analyzer_data.video_circs)
            border_img = __insert_texts(roi_analyzer_data.video_polys)

            for _, row in roi_analyzer_data.video_recs.iterrows():
                top_left_x, top_left_y, shape_name = (
                    row["topLeftX"],
                    row["topLeftY"],
                    row["Name"],
                )
                bottom_right_x, bottom_right_y = (
                    row["Bottom_right_X"],
                    row["Bottom_right_Y"],
                )
                thickness, color = row["Thickness"], row["Color BGR"]
                cv2.rectangle(
                    border_img,
                    (int(top_left_x), int(top_left_y)),
                    (int(bottom_right_x), int(bottom_right_y)),
                    color,
                    int(thickness),
                )

            for _, row in roi_analyzer_data.video_circs.iterrows():
                center_x, center_y, radius, shape_name = (
                    row["centerX"],
                    row["centerY"],
                    row["radius"],
                    row["Name"],
                )
                thickness, color = row["Thickness"], row["Color BGR"]
                cv2.circle(
                    border_img, (center_x, center_y), radius, color, int(thickness)
                )

            for _, row in roi_analyzer_data.video_polys.iterrows():
                vertices, shape_name = row["vertices"], row["Name"]
                thickness, color = row["Thickness"], row["Color BGR"]
                cv2.polylines(
                    border_img, [vertices], True, color, thickness=int(thickness)
                )

            for animal_cnt, animal_name in enumerate(animal_ids):
                if style_attr["Show_body_part"] or style_attr["Show_animal_name"]:
                    bp_data = data.loc[current_frm, body_part_dict[animal_name]].values
                    if threshold < bp_data[2]:
                        if style_attr["Show_body_part"]:
                            cv2.circle(
                                border_img,
                                (int(bp_data[0]), int(bp_data[1])),
                                scalers["circle_size"],
                                colors[animal_cnt],
                                -1,
                            )
                        if style_attr["Show_animal_name"]:
                            cv2.putText(
                                border_img,
                                animal_name,
                                (int(bp_data[0]), int(bp_data[1])),
                                font,
                                scalers["font_size"],
                                colors[animal_cnt],
                                TextOptions.TEXT_THICKNESS.value,
                            )

                for shape_name in video_shape_names:
                    timer = round(
                        data.loc[
                            current_frm,
                            "{}_{}_cum_sum_time".format(animal_name, shape_name),
                        ],
                        2,
                    )
                    entries = data.loc[
                        current_frm,
                        "{}_{}_cum_sum_entries".format(animal_name, shape_name),
                    ]
                    cv2.putText(
                        border_img,
                        str(timer),
                        loc_dict[animal_name][shape_name]["timer_data_loc"],
                        font,
                        scalers["font_size"],
                        shape_meta_data[shape_name]["Color BGR"],
                        TextOptions.TEXT_THICKNESS.value,
                    )
                    cv2.putText(
                        border_img,
                        str(entries),
                        loc_dict[animal_name][shape_name]["entries_data_loc"],
                        font,
                        scalers["font_size"],
                        shape_meta_data[shape_name]["Color BGR"],
                        TextOptions.TEXT_THICKNESS.value,
                    )

            writer.write(border_img)
            current_frm += 1
            print(
                "Multi-processing video frame {} on core {}...".format(
                    str(current_frm), str(group_cnt)
                )
            )

        cap.release()
        writer.release()

        return group_cnt

    @staticmethod
    def bbox_mp(
        frm_range: list,
        polygon_data: dict,
        animal_bp_dict: dict,
        data_df: Optional[pd.DataFrame],
        intersection_data_df: Optional[pd.DataFrame],
        roi_attributes: dict,
        video_path: str,
        key_points: bool,
        greyscale: bool,
    ):

        cap, current_frame = cv2.VideoCapture(video_path), frm_range[0]
        cap.set(1, frm_range[0])
        img_lst = []
        while current_frame < frm_range[-1]:
            ret, frame = cap.read()
            if ret:
                if key_points:
                    frm_data = data_df.iloc[current_frame]
                if greyscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                for animal_cnt, (animal, animal_data) in enumerate(
                    animal_bp_dict.items()
                ):
                    if key_points:
                        for bp_cnt, (x_col, y_col) in enumerate(
                            zip(animal_data["X_bps"], animal_data["Y_bps"])
                        ):
                            cv2.circle(
                                frame,
                                (frm_data[x_col], frm_data[y_col]),
                                0,
                                roi_attributes[animal]["bbox_clr"],
                                roi_attributes[animal]["keypoint_size"],
                            )
                    animal_polygon = np.array(
                        list(
                            polygon_data[animal][
                                current_frame
                            ].convex_hull.exterior.coords
                        )
                    ).astype(int)
                    if intersection_data_df is not None:
                        intersect = intersection_data_df.loc[
                            current_frame,
                            intersection_data_df.columns.str.startswith(animal),
                        ].sum()
                        if intersect > 0:
                            cv2.polylines(
                                frame,
                                [animal_polygon],
                                1,
                                roi_attributes[animal]["highlight_clr"],
                                roi_attributes[animal]["highlight_clr_thickness"],
                            )
                    cv2.polylines(
                        frame,
                        [animal_polygon],
                        1,
                        roi_attributes[animal]["bbox_clr"],
                        roi_attributes[animal]["bbox_thickness"],
                    )
                img_lst.append(frame)
                current_frame += 1
            else:
                print(
                    "SIMBA WARNING: SimBA tried to grab frame number {} from video {}, but could not find it. The video has {} frames.".format(
                        str(current_frame),
                        video_path,
                        str(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    )
                )
        return img_lst

    @staticmethod
    def path_plot_mp(
        data: np.array,
        video_setting: bool,
        frame_setting: bool,
        video_save_dir: str,
        video_name: str,
        frame_folder_dir: str,
        style_attr: dict,
        print_animal_names: bool,
        animal_attr: dict,
        fps: int,
        video_info: pd.DataFrame,
        clf_attr: dict,
        input_style_attr: dict,
        video_path: Optional[Union[str, os.PathLike]] = None,
    ):

        group = int(data[0][0])
        color_dict = get_color_dict()
        if video_setting:
            fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
            video_save_path = os.path.join(video_save_dir, "{}.mp4".format(str(group)))
            video_writer = cv2.VideoWriter(
                video_save_path,
                fourcc,
                fps,
                (style_attr["width"], style_attr["height"]),
            )

        if input_style_attr is not None:
            if (type(input_style_attr["bg color"]) == dict) and (
                input_style_attr["bg color"]["type"]
            ) == "moving":
                check_file_exist_and_readable(file_path=video_path)
                video_cap = cv2.VideoCapture(video_path)

        for i in range(data.shape[0]):
            if input_style_attr is not None:
                if (type(input_style_attr["bg color"]) == dict) and (
                    input_style_attr["bg color"]["type"]
                ) == "moving":
                    style_attr["bg color"] = input_style_attr["bg color"]
            frame_id = int(data[i][1])
            frame_data = data[i][2:].astype(int)
            frame_data = np.split(frame_data, len(list(animal_attr.keys())), axis=0)
            img = np.zeros(
                (
                    int(video_info["Resolution_height"].values[0]),
                    int(video_info["Resolution_width"].values[0]),
                    3,
                )
            )
            if (type(style_attr["bg color"]) == dict) and (
                style_attr["bg color"]["type"]
            ) == "moving":
                style_attr["bg color"] = read_frm_of_video(
                    video_path=video_cap,
                    opacity=style_attr["bg color"]["opacity"],
                    frame_index=frame_id,
                )

            img[:] = style_attr["bg color"]
            for animal_cnt, animal_data in enumerate(frame_data):
                animal_clr = style_attr["animal clrs"][animal_cnt]
                print(animal_data)
                cv2.line(
                    img, tuple(animal_data), animal_clr, int(style_attr["line width"])
                )
                cv2.circle(
                    img,
                    tuple(animal_data[-1]),
                    0,
                    animal_clr,
                    style_attr["circle size"],
                )
                if print_animal_names:
                    cv2.putText(
                        img,
                        style_attr["animal names"][animal_cnt],
                        tuple(animal_data[-1]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        style_attr["font size"],
                        animal_clr,
                        style_attr["font thickness"],
                    )

            if clf_attr:
                for clf_cnt, clf_name in enumerate(clf_attr["data"].columns):
                    clf_size = int(clf_attr["attr"][clf_cnt][-1].split(": ")[-1])
                    clf_clr = color_dict[clf_attr["attr"][clf_cnt][1]]
                    clf_sliced = clf_attr["data"][clf_name].loc[0:frame_id]
                    clf_sliced_idx = list(clf_sliced[clf_sliced == 1].index)
                    locations = clf_attr["positions"][clf_sliced_idx, :]
                    for i in range(locations.shape[0]):
                        cv2.circle(
                            img,
                            (locations[i][0], locations[i][1]),
                            0,
                            clf_clr,
                            clf_size,
                        )

            img = cv2.resize(img, (style_attr["width"], style_attr["height"]))
            if video_setting:
                video_writer.write(np.uint8(img))
            if frame_setting:
                frm_name = os.path.join(frame_folder_dir, str(frame_id) + ".png")
                cv2.imwrite(frm_name, np.uint8(img))

            print(
                "Path frame created: {}, Video: {}, Processing core: {}".format(
                    str(frame_id + 1), video_name, str(group + 1)
                )
            )
        if video_setting:
            video_writer.release()
        return group

    @staticmethod
    def probability_plot_mp(
        data: list,
        probability_lst: list,
        clf_name: str,
        video_setting: bool,
        frame_setting: bool,
        video_dir: str,
        frame_dir: str,
        highest_p: float,
        fps: int,
        style_attr: dict,
        video_name: str,
    ):

        group, data = data[0], data[1:]
        start_frm, end_frm, current_frm = data[0], data[-1], data[0]

        if video_setting:
            fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
            video_save_path = os.path.join(video_dir, "{}.mp4".format(str(group)))
            video_writer = cv2.VideoWriter(
                video_save_path,
                fourcc,
                fps,
                (style_attr["width"], style_attr["height"]),
            )

        while current_frm < end_frm:
            fig, ax = plt.subplots()
            current_lst = probability_lst[0 : current_frm + 1]
            ax.plot(
                current_lst,
                color=style_attr["color"],
                linewidth=style_attr["line width"],
            )
            ax.plot(
                current_frm,
                current_lst[-1],
                "o",
                markersize=style_attr["circle size"],
                color=style_attr["color"],
            )
            ax.set_ylim([0, highest_p])
            x_ticks_locs = x_lbls = np.linspace(0, current_frm, 5)
            x_lbls = np.round((x_lbls / fps), 1)
            ax.xaxis.set_ticks(x_ticks_locs)
            ax.set_xticklabels(x_lbls, fontsize=style_attr["font size"])
            ax.set_xlabel("Time (s)", fontsize=style_attr["font size"])
            ax.set_ylabel(
                "{} {}".format(clf_name, "probability"),
                fontsize=style_attr["font size"],
            )
            plt.suptitle(clf_name, x=0.5, y=0.92, fontsize=style_attr["font size"] + 4)
            canvas = FigureCanvas(fig)
            canvas.draw()
            mat = np.array(canvas.renderer._renderer)
            image = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            image = np.uint8(
                cv2.resize(image, (style_attr["width"], style_attr["height"]))
            )
            if video_setting:
                video_writer.write(image)
            if frame_setting:
                frame_save_name = os.path.join(
                    frame_dir, "{}.png".format(str(current_frm))
                )
                cv2.imwrite(frame_save_name, image)
            plt.close()
            current_frm += 1

            print(
                "Probability frame created: {}, Video: {}, Processing core: {}".format(
                    str(current_frm + 1), video_name, str(group + 1)
                )
            )

        return group

    def violin_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        save_path: Union[str, os.PathLike],
        font_rotation: Optional[int] = 45,
        font_size: Optional[int] = 10,
        img_size: Optional[tuple] = (13.7, 8.27),
        cut: Optional[int] = 0,
        scale: Optional[Literal["area", "count", "width"]] = "area",
    ):
        named_colors = get_named_colors()
        palette = {}
        for cnt, violin in enumerate(sorted(list(data[x].unique()))):
            palette[violin] = named_colors[cnt]
        plt.figure()
        order = data.groupby(by=[x])[y].median().sort_values().iloc[::-1].index
        figure_FSCTT = sns.violinplot(
            x=x, y=y, data=data, cut=cut, scale=scale, order=order, palette=palette
        )
        figure_FSCTT.set_xticklabels(
            figure_FSCTT.get_xticklabels(), rotation=font_rotation, size=font_size
        )
        figure_FSCTT.figure.set_size_inches(img_size)
        figure_FSCTT.figure.savefig(save_path, bbox_inches="tight")
        stdout_success(
            msg=f"Violin plot saved at {save_path}", source=self.__class__.__name__
        )

    @staticmethod
    @njit("(uint8[:,:,:],)")
    def rotate_img(img: np.ndarray):
        """
        Jitted helper to flip image 90 degrees.

        :example:
        >>> img = cv2.imread('/Users/simon/Desktop/sliding_line_length.png')
        >>> rotated_img = PlottingMixin().rotate_img(img)
        """
        rotated_image = np.transpose(img, (1, 0, 2))
        return np.ascontiguousarray(np.fliplr(rotated_image).astype(np.uint8))

    @staticmethod
    def continuous_scatter(
        data: Union[np.ndarray, pd.DataFrame],
        columns: Optional[List[str]] = ("X", "Y", "Cluster"),
        palette: Optional[str] = "magma",
        show_box: Optional[bool] = False,
        size: Optional[int] = 10,
        title: Optional[str] = None,
        bg_clr: Optional[str] = None,
        save_path: Optional[Union[str, os.PathLike]] = None,
    ):
        """Create a 2D scatterplot with a continuous legend"""

        check_instance(
            source=f"{PlottingMixin.continuous_scatter.__name__} data",
            instance=data,
            accepted_types=(np.ndarray, pd.DataFrame),
        )
        if isinstance(data, pd.DataFrame):
            check_that_column_exist(
                df=data,
                column_name=columns,
                file_name=PlottingMixin.continuous_scatter.__name__,
            )
            data = data[list(columns)]
        else:
            check_valid_array(
                data=data,
                source=PlottingMixin.continuous_scatter.__name__,
                accepted_ndims=(2,),
                max_axis_1=len(columns),
                min_axis_1=len(columns),
            )
            data = pd.DataFrame(data, columns=list(columns))

        fig, ax = plt.subplots()
        if bg_clr is not None:
            if bg_clr not in get_named_colors():
                raise InvalidInputError(
                    msg=f"bg_clr {bg_clr} is not a valid named color. Options: {get_named_colors()}",
                    source=PlottingMixin.continuous_scatter.__name__,
                )
            fig.set_facecolor(bg_clr)
        if not show_box:
            plt.axis("off")
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plot = ax.scatter(
            data[columns[0]], data[columns[1]], c=data[columns[2]], s=size, cmap=palette
        )
        cbar = fig.colorbar(plot)
        cbar.set_label(columns[2], loc="center")
        if title is not None:
            plt.title(
                title,
                ha="center",
                fontsize=15,
                bbox={"facecolor": "orange", "alpha": 0.5, "pad": 0},
            )
        if save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(save_path))
            fig.savefig(save_path)
            plt.close("all")
        else:
            return plot

    @staticmethod
    def categorical_scatter(
        data: Union[np.ndarray, pd.DataFrame],
        columns: Optional[List[str]] = ("X", "Y", "Cluster"),
        palette: Optional[str] = "Set1",
        show_box: Optional[bool] = False,
        size: Optional[int] = 10,
        title: Optional[str] = None,
        save_path: Optional[Union[str, os.PathLike]] = None,
    ):
        """Create a 2D scatterplot with a categorical legend"""
        cmaps = get_categorical_palettes()
        if palette not in cmaps:
            raise InvalidInputError(
                msg=f"{palette} is not a valid palette. Accepted options: {cmaps}.",
                source=PlottingMixin.categorical_scatter.__name__,
            )
        check_instance(
            source=f"{PlottingMixin.categorical_scatter.__name__} data",
            instance=data,
            accepted_types=(np.ndarray, pd.DataFrame),
        )
        if isinstance(data, pd.DataFrame):
            check_that_column_exist(
                df=data,
                column_name=columns,
                file_name=PlottingMixin.categorical_scatter.__name__,
            )
            data = data[list(columns)]
        else:
            check_valid_array(
                data=data,
                source=PlottingMixin.categorical_scatter.__name__,
                accepted_ndims=(2,),
                max_axis_1=len(columns),
                min_axis_1=len(columns),
            )
            data = pd.DataFrame(data, columns=list(columns))

        if not show_box:
            plt.axis("off")
        pct_x = np.percentile(data[columns[0]].values, 75)
        pct_y = np.percentile(data[columns[1]].values, 75)
        plt.xlim(data[columns[0]].min() - pct_x, data[columns[0]].max() + pct_x)
        plt.ylim(data[columns[1]].min() - pct_y, data[columns[1]].max() + pct_y)

        plot = sns.scatterplot(
            data=data,
            x=columns[0],
            y=columns[1],
            hue=columns[2],
            palette=sns.color_palette(palette, len(data[columns[2]].unique())),
            s=size,
        )
        if title is not None:
            plt.title(
                title,
                ha="center",
                fontsize=15,
                bbox={"facecolor": "orange", "alpha": 0.5, "pad": 0},
            )
        if save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close("all")
        else:
            return plot

    @staticmethod
    def joint_plot(
        data: Union[np.ndarray, pd.DataFrame],
        columns: Optional[List[str]] = ("X", "Y", "Cluster"),
        palette: Optional[str] = "Set1",
        kind: Optional[str] = "scatter",
        size: Optional[int] = 10,
        title: Optional[str] = None,
        save_path: Optional[Union[str, os.PathLike]] = None,
    ):
        """
        :example:
        >>> x = np.hstack([np.random.normal(loc=10, scale=4, size=(100, 2)), np.random.randint(0, 1, size=(100, 1))])
        >>> y = np.hstack([np.random.normal(loc=25, scale=4, size=(100, 2)), np.random.randint(1, 2, size=(100, 1))])
        >>> plot = PlottingMixin.joint_plot(data=np.vstack([x, y]), columns=['X', 'Y', 'Cluster'], title='The plot')
        """

        cmaps = get_categorical_palettes()
        if palette not in cmaps:
            raise InvalidInputError(
                msg=f"{palette} is not a valid palette. Accepted options: {cmaps}",
                source=PlottingMixin.joint_plot.__name__,
            )
        check_instance(
            source=f"{PlottingMixin.joint_plot.__name__} data",
            instance=data,
            accepted_types=(np.ndarray, pd.DataFrame),
        )
        check_str(
            name=f"{PlottingMixin.joint_plot.__name__} kind",
            value=kind,
            options=("kde", "reg", "hist", "scatter"),
        )
        if isinstance(data, pd.DataFrame):
            check_that_column_exist(
                df=data,
                column_name=columns,
                file_name=PlottingMixin.joint_plot.__name__,
            )
            data = data[list(columns)]
        else:
            check_valid_array(
                data=data,
                source=PlottingMixin.joint_plot.__name__,
                accepted_ndims=(2,),
                max_axis_1=len(columns),
                min_axis_1=len(columns),
            )
            data = pd.DataFrame(data, columns=list(columns))

        pct_x = np.percentile(data[columns[0]].values, 75)
        pct_y = np.percentile(data[columns[1]].values, 75)
        plot = sns.jointplot(
            data=data,
            x=columns[0],
            y=columns[1],
            hue=columns[2],
            xlim=(data[columns[0]].min() - pct_x, data[columns[0]].max() + pct_x),
            ylim=(data[columns[1]].min() - pct_y, data[columns[1]].max() + pct_y),
            palette=sns.color_palette(palette, len(data[columns[2]].unique())),
            kind=kind,
            marginal_ticks=False,
            s=size,
        )

        if title is not None:
            plot.fig.suptitle(
                title,
                va="baseline",
                ha="center",
                fontsize=15,
                bbox={"facecolor": "orange", "alpha": 0.5, "pad": 0},
            )
        if save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(save_path))
            plot.savefig(save_path)
            plt.close("all")
        else:
            return plot

    @staticmethod
    def line_plot(
        df: pd.DataFrame,
        x: str,
        y: str,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, os.PathLike]] = None,
    ):

        check_instance(
            source=f"{PlottingMixin.line_plot.__name__} df",
            instance=df,
            accepted_types=(pd.DataFrame),
        )
        check_str(
            name=f"{PlottingMixin.line_plot.__name__} x",
            value=x,
            options=tuple(df.columns),
        )
        check_str(
            name=f"{PlottingMixin.line_plot.__name__} y",
            value=y,
            options=tuple(df.columns),
        )

        check_valid_lst(
            data=list(df[y]),
            source=f"{PlottingMixin.line_plot.__name__} y",
            valid_dtypes=(np.float32, np.float64, np.int32, np.int64, int, float),
        )
        sns.set_style("whitegrid", {"grid.linestyle": "--"})
        plot = sns.lineplot(data=df, x=x, y=y)

        if x_label is not None:
            check_str(name=f"{PlottingMixin.line_plot.__name__} x_label", value=x_label)
            plt.xlabel(x_label)
        if y_label is not None:
            check_str(name=f"{PlottingMixin.line_plot.__name__} y_label", value=y_label)
            plt.ylabel(y_label)
        if title is not None:
            check_str(name=f"{PlottingMixin.line_plot.__name__} title", value=title)
            plt.title(title, ha="center", fontsize=15)
        if save_path is not None:
            check_str(
                name=f"{PlottingMixin.line_plot.__name__} save_path", value=save_path
            )
            check_if_dir_exists(in_dir=os.path.dirname(save_path))
            plot.figure.savefig(save_path)
            plt.close("all")
        else:
            return plot

    @staticmethod
    def make_line_plot(
        data: List[np.ndarray],
        colors: List[str],
        show_box: Optional[bool] = True,
        width: Optional[int] = 640,
        height: Optional[int] = 480,
        line_width: Optional[int] = 6,
        font_size: Optional[int] = 8,
        bg_clr: Optional[str] = None,
        x_lbl_divisor: Optional[float] = None,
        title: Optional[str] = None,
        y_lbl: Optional[str] = None,
        x_lbl: Optional[str] = None,
        y_max: Optional[int] = -1,
        line_opacity: Optional[int] = 0.0,
        save_path: Optional[Union[str, os.PathLike]] = None,
    ):

        check_valid_lst(
            data=data,
            source=PlottingMixin.make_line_plot.__name__,
            valid_dtypes=(
                np.ndarray,
                list,
            ),
        )
        check_valid_lst(
            data=colors,
            source=PlottingMixin.make_line_plot.__name__,
            valid_dtypes=(str,),
            exact_len=len(data),
        )
        clr_dict = get_color_dict()
        matplotlib.font_manager._get_font.cache_clear()
        plt.close("all")
        fig, ax = plt.subplots()
        if bg_clr is not None:
            fig.set_facecolor(bg_clr)
        if not show_box:
            plt.axis("off")
        for i in range(len(data)):
            line_clr = clr_dict[colors[i]][::-1]
            line_clr = tuple(x / 255 for x in line_clr)
            flat_data = data[i].flatten()
            plt.plot(
                flat_data, color=line_clr, linewidth=line_width, alpha=line_opacity
            )
        max_x = max([len(x) for x in data])
        if y_max == -1:
            y_max = max([np.max(x) for x in data])
        y_ticks_locs = y_lbls = np.round(np.linspace(0, y_max, 10), 2)
        x_ticks_locs = x_lbls = np.linspace(0, max_x, 5)
        if x_lbl_divisor is not None:
            x_lbls = np.round((x_lbls / x_lbl_divisor), 1)
        if y_lbl is not None:
            plt.ylabel(y_lbl)
        if x_lbl is not None:
            plt.xlabel(x_lbl)
        plt.xticks(x_ticks_locs, x_lbls, rotation="horizontal", fontsize=font_size)
        plt.yticks(y_ticks_locs, y_lbls, fontsize=font_size)
        plt.ylim(0, y_max)
        if title is not None:
            plt.suptitle(title, x=0.5, y=0.92, fontsize=font_size + 4)
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        img = PIL.Image.open(buffer_)
        img = np.uint8(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        buffer_.close()
        plt.close()
        img = cv2.resize(img, (width, height))
        if save_path is not None:
            cv2.imwrite(save_path, img)
            stdout_success(msg=f"Line plot saved at {save_path}")
        else:
            return img

    @staticmethod
    def make_line_plot_plotly(
        data: List[np.ndarray],
        colors: List[str],
        show_box: Optional[bool] = True,
        show_grid: Optional[bool] = False,
        width: Optional[int] = 640,
        height: Optional[int] = 480,
        line_width: Optional[int] = 6,
        font_size: Optional[int] = 8,
        bg_clr: Optional[str] = "white",
        x_lbl_divisor: Optional[float] = None,
        title: Optional[str] = None,
        y_lbl: Optional[str] = None,
        x_lbl: Optional[str] = None,
        y_max: Optional[int] = -1,
        line_opacity: Optional[int] = 0.5,
        save_path: Optional[Union[str, os.PathLike]] = None,
    ):
        """
        Create a line plot using Plotly.

        .. note::
           Plotly can be more reliable than matplotlib on some systems when accessed through multprocessing calls.

           If **not** called though multiprocessing, consider using ``simba.mixins.plotting_mixin.PlottingMixin.make_line_plot()``

           Uses ``kaleido`` for transform image to numpy array or save to disk.

        :param List[np.ndarray] data: List of 1D numpy arrays representing lines.
        :param List[str] colors: List of named colors of size len(data).
        :param bool show_box: Whether to show the plot box (axes, title, etc.).
        :param bool show_grid: Whether to show gridlines on the plot.
        :param int width: Width of the plot in pixels.
        :param int height: Height of the plot in pixels.
        :param int line_width: Width of the lines in the plot.
        :param int font_size: Font size for axis labels and tick labels.
        :param str bg_clr: Background color of the plot.
        :param float x_lbl_divisor: Divisor for adjusting the tick spacing on the x-axis.
        :param str title: Title of the plot.
        :param str y_lbl: Label for the y-axis.
        :param str x_lbl: Label for the x-axis.
        :param int y_max: Maximum value for the y-axis.
        :param float line_opacity: Opacity of the lines in the plot.
        :param Union[str, os.PathLike] save_path: Path to save the plot image. If None, returns a numpy array of the plot.
        :return: If save_path is None, returns a numpy array representing the plot image.

        :example:
        >>> p = np.random.randint(0, 50, (100,))
        >>> y = np.random.randint(0, 50, (200,))
        >>> img = PlottingMixin.make_line_plot_plotly(data=[p, y], show_box=False, font_size=20, bg_clr='white', show_grid=False, x_lbl_divisor=30, colors=['Red', 'Green'], save_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/line_plot/Trial     3_final_img.png')
        """

        def tick_formatter(x):
            if x_lbl_divisor is not None:
                return str(round(x / x_lbl_divisor, 2))
            else:
                return str(x)

        fig = go.Figure()
        clr_dict = get_color_dict()
        if y_max == -1:
            y_max = max([np.max(i) for i in data])
        for i in range(len(data)):
            line_clr = clr_dict[colors[i]]
            line_clr = (
                f"rgba({line_clr[0]}, {line_clr[1]}, {line_clr[2]}, {line_opacity})"
            )
            fig.add_trace(
                go.Scatter(
                    y=data[i].flatten(),
                    mode="lines",
                    line=dict(color=line_clr, width=line_width),
                )
            )

        if not show_box:
            fig.update_layout(
                width=width,
                height=height,
                title=title,
                xaxis_visible=False,
                yaxis_visible=False,
                showlegend=False,
            )
        else:
            if fig["layout"]["xaxis"]["tickvals"] is None:
                tickvals = [i for i in range(data[0].shape[0])]
            else:
                tickvals = fig["layout"]["xaxis"]["tickvals"]
            if x_lbl_divisor is not None:
                ticktext = [tick_formatter(x) for x in tickvals]
            else:
                ticktext = tickvals
            fig.update_layout(
                width=width,
                height=height,
                title=title,
                xaxis=dict(
                    title=x_lbl,
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickmode="auto",
                    tick0=0,
                    dtick=10,
                    tickfont=dict(size=font_size),
                    showgrid=show_grid,
                ),
                yaxis=dict(
                    title=y_lbl,
                    tickfont=dict(size=font_size),
                    range=[0, y_max],
                    showgrid=show_grid,
                ),
                showlegend=False,
            )

        if bg_clr is not None:
            fig.update_layout(plot_bgcolor=bg_clr)
        if save_path is not None:
            pio.write_image(fig, save_path)
            stdout_success(msg=f"Line plot saved at {save_path}")
        else:
            img_bytes = fig.to_image(format="png")
            img = PIL.Image.open(io.BytesIO(img_bytes))
            fig = None
            return np.array(img).astype(np.uint8)

    @staticmethod
    def make_path_plot(data: List[np.ndarray],
                       colors: List[Tuple[int, int, int]],
                       width: Optional[int] = 640,
                       height: Optional[int] = 480,
                       max_lines: Optional[int] = None,
                       bg_clr: Optional[Union[Tuple[int, int, int], np.ndarray]] = (255, 255, 255),
                       circle_size: Optional[Union[int, None]] = 3,
                       font_size: Optional[float] = 2.0,
                       font_thickness: Optional[int] = 2,
                       line_width: Optional[int] = 2,
                       animal_names: Optional[List[str]] = None,
                       clf_attr: Optional[Dict[str, Any]] = None,
                       save_path: Optional[Union[str, os.PathLike]] = None) -> Union[None, np.ndarray]:

        """
        Creates a path plot visualization from the given data.

        .. image:: _static/img/make_path_plot.png
           :width: 500
           :align: center

        :param List[np.ndarray] data: List of numpy arrays containing path data.
        :param List[Tuple[int, int, int]] colors: List of RGB tuples representing colors for each path.
        :param width: Width of the output image (default is 640 pixels).
        :param height: Height of the output image (default is 480 pixels).
        :param max_lines: Maximum number of lines to plot from each path data.
        :param bg_clr: Background color of the plot (default is white).
        :param circle_size: Size of the circle marker at the end of each path (default is 3).
        :param font_size: Font size for displaying animal names (default is 2.0).
        :param font_thickness: Thickness of the font for displaying animal names (default is 2).
        :param line_width: Width of the lines representing paths (default is 2).
        :param animal_names: List of names for the animals corresponding to each path.
        :param clf_attr: Dictionary containing attributes for classification markers.
        :param save_path: Path to save the generated plot image.
        :return: If save_path is None, returns the generated image as a numpy array, otherwise, returns None.


        :example:
        >>> x = np.random.randint(0, 500, (100, 2))
        >>> y = np.random.randint(0, 500, (100, 2))
        >>> position_data = np.random.randint(0, 500, (100, 2))
        >>> clf_data_1 = np.random.randint(0, 2, (100,))
        >>> clf_data_2 = np.random.randint(0, 2, (100,))
        >>> clf_data = {'Attack': {'color': (155, 1, 10), 'size': 30, 'positions': position_data, 'clfs': clf_data_1},  'Sniffing': {'color': (155, 90, 10), 'size': 30, 'positions': position_data, 'clfs': clf_data_2}}
        >>> PlottingMixin.make_path_plot(data=[x, y], colors=[(0, 255, 0), (255, 0, 0)], clf_attr=clf_data)
        """

        check_valid_lst(data=data, source=PlottingMixin.make_path_plot.__name__, valid_dtypes=(np.ndarray,), min_len=1)
        for i in data: check_valid_array(data=i, source=PlottingMixin.make_path_plot.__name__, accepted_ndims=(2,),
                                         accepted_axis_1_shape=(2,),
                                         accepted_dtypes=(int, float, np.int32, np.int64, np.float32, np.float64))
        check_valid_lst(data=colors, source=PlottingMixin.make_path_plot.__name__, valid_dtypes=(tuple,), exact_len=len(data))
        for i in colors: check_if_valid_rgb_tuple(data=i)
        check_instance(source='bg_clr', instance=bg_clr, accepted_types=(np.ndarray, tuple))
        if isinstance(bg_clr, tuple):
            check_if_valid_rgb_tuple(data=bg_clr)
        check_int(name=f'{PlottingMixin.make_path_plot.__name__} height', value=height, min_value=1)
        check_int(name=f'{PlottingMixin.make_path_plot.__name__} height', value=width, min_value=1)
        check_float(name=f'{PlottingMixin.make_path_plot.__name__} font_size', value=font_size)
        check_int(name=f'{PlottingMixin.make_path_plot.__name__} font_thickness', value=font_thickness)
        check_int(name=f'{PlottingMixin.make_path_plot.__name__} line_width', value=line_width)
        timer = SimbaTimer(start=True)
        img = np.zeros((height, width, 3))
        img[:] = bg_clr
        for line_cnt in range(len(data)):
            clr = colors[line_cnt]
            line_data = data[line_cnt]
            if max_lines is not None:
                check_int(name=f'{PlottingMixin.make_path_plot.__name__} max_lines', value=max_lines, min_value=1)
                line_data = line_data[-max_lines:]
            for i in range(1, line_data.shape[0]):
                cv2.line(img, tuple(line_data[i]), tuple(line_data[i - 1]), clr, line_width)
            if circle_size is not None:
                cv2.circle(img, tuple(line_data[-1]), 0, clr, circle_size)
            if animal_names is not None:
                cv2.putText(img, animal_names[line_cnt], tuple(line_data[-1]), cv2.FONT_HERSHEY_COMPLEX, font_size, clr,
                            font_thickness)
        if clf_attr is not None:
            check_instance(source=PlottingMixin.make_path_plot.__name__, instance=clf_attr, accepted_types=(dict,))
            for k, v in clf_attr.items():
                check_if_keys_exist_in_dict(data=v, key=['color', 'size', 'positions', 'clfs'], name='clf_attr')
            for clf_name, clf_data in clf_attr.items():
                clf_positions = clf_data['positions'][np.argwhere(clf_data['clfs'] == 1).flatten()]
                for i in clf_positions:
                    cv2.circle(img, tuple(i), 0, clf_data['color'], clf_data['size'])
        img = cv2.resize(img, (width, height)).astype(np.uint8)
        if save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(save_path))
            timer.stop_timer()
            cv2.imwrite(save_path, img)
            stdout_success(msg=f"Path plot saved at {save_path}", elapsed_time=timer.elapsed_time_str, source=PlottingMixin.make_path_plot.__name__, )
        else:
            return img
