__author__ = "Simon Nilsson"
import io
import os
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
import seaborn as sns
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numba import bool_, njit, uint8
from PIL import Image

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import simba
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict,
                                check_if_valid_rgb_tuple, check_instance,
                                check_int, check_str, check_that_column_exist,
                                check_valid_array, check_valid_dataframe,
                                check_valid_lst)
from simba.utils.enums import Formats, Keys, Options, TextOptions
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
            msg=f"Final probability plot saved at {save_path}",
            elapsed_time=timer.elapsed_time_str,
            source=self.__class__.__name__,
        )

    def make_gantt_plot(self,
                        data_df: pd.DataFrame,
                        bouts_df: pd.DataFrame,
                        clf_names: List[str],
                        fps: int,
                        style_attr: dict,
                        video_name: str,
                        save_path: Optional[str] = None) -> Union[None, np.ndarray]:

        video_timer = SimbaTimer(start=True)
        colours = get_named_colors()
        colour_tuple_x = list(np.arange(3.5, 203.5, 5))
        fig, ax = plt.subplots()
        for i, event in enumerate(bouts_df.groupby("Event")):
            for x in clf_names:
                if event[0] == x:
                    ix = clf_names.index(x)
                    data_event = event[1][["Start_time", "Bout_time"]]
                    ax.broken_barh(data_event.values, (colour_tuple_x[ix], 3), facecolors=colours[ix])

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
        open_cv_image = cv2.resize(open_cv_image, (style_attr["width"], style_attr["height"]))
        frame = np.uint8(open_cv_image)
        buffer_.close()
        plt.close('all')
        if save_path is not None:
            cv2.imwrite(save_path, frame)
            video_timer.stop_timer()
            stdout_success(msg=f"Final gantt frame for video {video_name} saved at {save_path}",elapsed_time=video_timer.elapsed_time_str, source=self.__class__.__name__)
        else:
            return frame

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
    def make_location_heatmap_plot(frm_data: np.array,
                                   max_scale: float,
                                   palette: Literal[Options.PALETTE_OPTIONS],
                                   aspect_ratio: float,
                                   shading: str,
                                   img_size: Tuple[int, int],
                                   file_name: Optional[Union[str, os.PathLike]] = None):

        cum_df = pd.DataFrame(frm_data).reset_index()
        cum_df = cum_df.melt(id_vars="index", value_vars=None, var_name=None, value_name="seconds", col_level=None).rename(columns={"index": "vertical_idx", "variable": "horizontal_idx"})
        cum_df["color"] = ((cum_df["seconds"].astype(float) / float(max_scale)).round(2).clip(upper=100))
        color_array = np.zeros((len(cum_df["vertical_idx"].unique()), len(cum_df["horizontal_idx"].unique())))
        for i in range(color_array.shape[0]):
            for j in range(color_array.shape[1]):
                value = cum_df["color"][(cum_df["horizontal_idx"] == j) & (cum_df["vertical_idx"] == i)].values[0]
                color_array[i, j] = value
        color_array = color_array * max_scale
        matplotlib.font_manager._get_font.cache_clear()
        plt.close("all")
        fig = plt.figure()
        im_ratio = color_array.shape[0] / color_array.shape[1]
        plt.pcolormesh(color_array,
                       shading=shading,
                       cmap=palette,
                       rasterized=True,
                       alpha=1,
                       vmin=0.0,
                       vmax=max_scale)

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
        image = cv2.resize(mat, img_size)
        image = np.uint8(image)
        plt.close("all")
        if file_name is not None:
            cv2.imwrite(file_name, image)
            stdout_success(msg=f"Location heatmap image saved at at {file_name}", source=PlottingMixin.make_location_heatmap_plot.__class__.__name__)
        else:
            return image

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
    @njit([(uint8[:, :, :], bool_)])
    def rotate_img(img: np.ndarray, right: bool) -> np.ndarray:
        """
        Flip a color image 90 degrees to the left or right

        .. image:: _static/img/rotate_img.png
           :width: 600
           :align: center

        :param np.ndarray img: Input image as numpy array in uint8 format.
        :param bool right: If True, flips to the right. If False, flips to the left.
        :returns: The rotated image as a numpy array of uint8 format.

        :example:
        >>> img = cv2.imread('/Users/simon/Desktop/test.png')
        >>> rotated_img = PlottingMixin.rotate_img(img=img, right=False)
        """

        if right:
            img = np.transpose(img[:, ::-1, :], axes=(1, 0, 2))
        else:
            img = np.transpose(img[::-1, :, :], axes=(1, 0, 2))
        return np.ascontiguousarray(img).astype(np.uint8)

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
        """
        Create a 2D scatterplot with a categorical legend.

        .. image:: _static/img/categorical_scatter.png
           :width: 400
           :align: center

        :param Union[np.ndarray, pd.DataFrame] data: Input data, either a NumPy array or a pandas DataFrame.
        :param Optional[List[str]] columns: A list of column names for the x-axis, y-axis, and the categorical variable respectively. Default is ["X", "Y", "Cluster"].
        :param Optional[str] palette: The color palette to be used for the categorical variable. Default is "Set1".
        :param Optional[bool] show_box: Whether to display the plot axis. Default is False.
        :param Optional[int] size: Size of markers in the scatterplot. Default is 10.
        :param Optional[str] title: Title for the plot. Default is None.
        :param Optional[Union[str, os.PathLike]] save_path: The path where the plot will be saved. Default is None which returns the image.
        :returns matplotlib.axes._subplots.AxesSubplot or None: The scatterplot if 'save_path' is not provided, otherwise None.
        """
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
        # pct_x = np.percentile(data[columns[0]].values, 75)
        # pct_y = np.percentile(data[columns[1]].values, 75)
        # plt.xlim(data[columns[0]].min() - pct_x, data[columns[0]].max() + pct_x)
        # plt.ylim(data[columns[1]].min() - pct_y, data[columns[1]].max() + pct_y)

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
        Generate a joint plot.

        Useful when visualizing embedded behavior data latent spaces with dense and overlapping scatters.

        .. image:: _static/img/joint_plot.png
           :width: 700
           :align: center

        :param Union[np.ndarray, pd.DataFrame] data: Input data, either a NumPy array or a pandas DataFrame.
        :param Optional[List[str]] columns: Names of columns if input is dataframe, default is ["X", "Y", "Cluster"].
        :param Optional[str] palette: Palette for the plot, default is "Set1".
        :param Optional[str] kind: Type of plot ("scatter", "kde", "hist", or "reg"), default is "scatter".
        :param Optional[int] size: Size of markers for scatter plot, default is 10.
        :param Optional[str] title: Title of the plot, default is None.
        :param Optional[Union[str, os.PathLike]] save_path: Path to save the plot image, default is None.
        :returns sns.JointGrid or None: JointGrid object if save_path is None, else None.

        :example:
        >>> data, lbls = make_blobs(n_samples=100000, n_features=2, centers=10, random_state=42)
        >>> data = np.hstack((data, lbls.reshape(-1, 1)))
        >>> PlottingMixin.joint_plot(data=data, columns=['X', 'Y', 'Cluster'], title='The plot')
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

        .. image:: _static/img/make_line_plot_plotly.png
           :width: 500
           :align: center

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
        img_bytes = fig.to_image(format="png")
        img = PIL.Image.open(io.BytesIO(img_bytes))
        img = np.array(img)
        if save_path is not None:
            cv2.imwrite(save_path, img)
            stdout_success(msg=f"Line plot saved at {save_path}")
        else:
            return img

    @staticmethod
    def make_path_plot(
        data: List[np.ndarray],
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
        save_path: Optional[Union[str, os.PathLike]] = None,
    ) -> Union[None, np.ndarray]:
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

        check_valid_lst(
            data=data,
            source=PlottingMixin.make_path_plot.__name__,
            valid_dtypes=(np.ndarray,),
            min_len=1,
        )
        for i in data:
            check_valid_array(
                data=i,
                source=PlottingMixin.make_path_plot.__name__,
                accepted_ndims=(2,),
                accepted_axis_1_shape=(2,),
                accepted_dtypes=(
                    int,
                    float,
                    np.int32,
                    np.int64,
                    np.float32,
                    np.float64,
                ),
            )
        check_valid_lst(
            data=colors,
            source=PlottingMixin.make_path_plot.__name__,
            valid_dtypes=(tuple,),
            exact_len=len(data),
        )
        for i in colors:
            check_if_valid_rgb_tuple(data=i)
        check_instance(
            source="bg_clr", instance=bg_clr, accepted_types=(np.ndarray, tuple)
        )
        if isinstance(bg_clr, tuple):
            check_if_valid_rgb_tuple(data=bg_clr)
        check_int(
            name=f"{PlottingMixin.make_path_plot.__name__} height",
            value=height,
            min_value=1,
        )
        check_int(
            name=f"{PlottingMixin.make_path_plot.__name__} height",
            value=width,
            min_value=1,
        )
        check_float(
            name=f"{PlottingMixin.make_path_plot.__name__} font_size", value=font_size
        )
        check_int(
            name=f"{PlottingMixin.make_path_plot.__name__} font_thickness",
            value=font_thickness,
        )
        check_int(
            name=f"{PlottingMixin.make_path_plot.__name__} line_width", value=line_width
        )
        timer = SimbaTimer(start=True)
        img = np.zeros((height, width, 3))
        img[:] = bg_clr
        for line_cnt in range(len(data)):
            clr = colors[line_cnt]
            line_data = data[line_cnt]
            if max_lines is not None:
                check_int(
                    name=f"{PlottingMixin.make_path_plot.__name__} max_lines",
                    value=max_lines,
                    min_value=1,
                )
                line_data = line_data[-max_lines:]
            for i in range(1, line_data.shape[0]):
                cv2.line(
                    img, tuple(line_data[i]), tuple(line_data[i - 1]), clr, line_width
                )
            if circle_size is not None:
                cv2.circle(img, tuple(line_data[-1]), 0, clr, circle_size)
            if animal_names is not None:
                cv2.putText(
                    img,
                    animal_names[line_cnt],
                    tuple(line_data[-1]),
                    cv2.FONT_HERSHEY_COMPLEX,
                    font_size,
                    clr,
                    font_thickness,
                )
        if clf_attr is not None:
            check_instance(
                source=PlottingMixin.make_path_plot.__name__,
                instance=clf_attr,
                accepted_types=(dict,),
            )
            for k, v in clf_attr.items():
                check_if_keys_exist_in_dict(
                    data=v, key=["color", "size", "positions", "clfs"], name="clf_attr"
                )
            for clf_name, clf_data in clf_attr.items():
                clf_positions = clf_data["positions"][
                    np.argwhere(clf_data["clfs"] == 1).flatten()
                ]
                for i in clf_positions:
                    cv2.circle(img, tuple(i), 0, clf_data["color"], clf_data["size"])
        img = cv2.resize(img, (width, height)).astype(np.uint8)
        if save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(save_path))
            timer.stop_timer()
            cv2.imwrite(save_path, img)
            stdout_success(
                msg=f"Path plot saved at {save_path}",
                elapsed_time=timer.elapsed_time_str,
                source=PlottingMixin.make_path_plot.__name__,
            )
        else:
            return img

    @staticmethod
    def rectangles_onto_image(
        img: np.ndarray,
        rectangles: pd.DataFrame,
        show_center: Optional[bool] = False,
        show_tags: Optional[bool] = False,
        circle_size: Optional[int] = 2,
    ) -> np.ndarray:

        check_valid_array(data=img, source=PlottingMixin.rectangles_onto_image.__name__)
        check_valid_dataframe(
            df=rectangles,
            source=PlottingMixin.rectangles_onto_image.__name__,
            required_fields=[
                "topLeftX",
                "topLeftY",
                "Bottom_right_X",
                "Bottom_right_Y",
                "Color BGR",
                "Thickness",
                "Center_X",
                "Center_Y",
                "Tags",
            ],
        )
        check_int(
            name=PlottingMixin.rectangles_onto_image.__name__,
            value=circle_size,
            min_value=1,
        )
        for _, row in rectangles.iterrows():
            img = cv2.rectangle(
                img,
                (int(row["topLeftX"]), int(row["topLeftY"])),
                (int(row["Bottom_right_X"]), int(row["Bottom_right_Y"])),
                row["Color BGR"],
                int(row["Thickness"]),
            )
            if show_center:
                img = cv2.circle(
                    img,
                    (int(row["Center_X"]), int(row["Center_Y"])),
                    circle_size,
                    row["Color BGR"],
                    -1,
                )
            if show_tags:
                for tag_name, tag_data in row["Tags"].items():
                    img = cv2.circle(
                        img, tuple(tag_data), circle_size, row["Color BGR"], -1
                    )
        return img

    @staticmethod
    def circles_onto_image(
        img: np.ndarray,
        circles: pd.DataFrame,
        show_center: Optional[bool] = False,
        show_tags: Optional[bool] = False,
        circle_size: Optional[int] = 2,
    ) -> np.ndarray:

        check_valid_array(data=img, source=PlottingMixin.circles_onto_image.__name__)
        check_valid_dataframe(
            df=circles,
            source=PlottingMixin.circles_onto_image.__name__,
            required_fields=[
                "centerX",
                "centerY",
                "radius",
                "Color BGR",
                "Thickness",
                "Tags",
            ],
        )
        check_int(
            name=PlottingMixin.circles_onto_image.__name__,
            value=circle_size,
            min_value=1,
        )
        for _, row in circles.iterrows():
            img = cv2.circle(
                img,
                (int(row["centerX"]), int(row["centerY"])),
                row["radius"],
                row["Color BGR"],
                int(row["Thickness"]),
            )
            if show_center:
                img = cv2.circle(
                    img,
                    (int(row["Center_X"]), int(row["Center_Y"])),
                    circle_size,
                    row["Color BGR"],
                    -1,
                )
            if show_tags:
                for tag_data in row["Tags"].values():
                    img = cv2.circle(
                        img, tuple(tag_data), circle_size, row["Color BGR"], -1
                    )
        return img

    @staticmethod
    def polygons_onto_image(
        img: np.ndarray,
        polygons: pd.DataFrame,
        show_center: Optional[bool] = False,
        show_tags: Optional[bool] = False,
        circle_size: Optional[int] = 2,
    ) -> np.ndarray:
        """
        Helper to insert polygon overlays onto an image.

        :param np.ndarray img:
        :param polygons:
        :param show_center:
        :param show_tags:
        :param circle_size:
        :return:
        """

        check_valid_array(data=img, source=f"{PlottingMixin.polygons_onto_image.__name__} img")
        check_valid_dataframe(df=polygons, source=f"{PlottingMixin.polygons_onto_image.__name__} polygons", required_fields=["vertices", "Color BGR", "Thickness", "Tags"])
        check_int(name=PlottingMixin.polygons_onto_image.__name__, value=circle_size, min_value=1)
        for _, row in polygons.iterrows():
            img = cv2.polylines(
                img,
                [row["vertices"].astype(int)],
                True,
                row["Color BGR"],
                thickness=int(row["Thickness"]),
            )
            if show_center:
                img = cv2.circle(
                    img,
                    (int(row["Center_X"]), int(row["Center_Y"])),
                    circle_size,
                    row["Color BGR"],
                    -1,
                )
            if show_tags:
                for tag_data in row["vertices"]:
                    img = cv2.circle(
                        img, tuple(tag_data), circle_size, row["Color BGR"], -1
                    )
        return img

    @staticmethod
    def roi_dict_onto_img(
        img: np.ndarray,
        roi_dict: Dict[str, pd.DataFrame],
        circle_size: Optional[int] = 2,
        show_center: Optional[bool] = False,
        show_tags: Optional[bool] = False,
    ) -> np.ndarray:

        check_valid_array(
            data=img, source=f"{PlottingMixin.roi_dict_onto_img.__name__} img"
        )
        check_if_keys_exist_in_dict(
            data=roi_dict,
            key=[
                Keys.ROI_POLYGONS.value,
                Keys.ROI_CIRCLES.value,
                Keys.ROI_RECTANGLES.value,
            ],
            name=PlottingMixin.roi_dict_onto_img.__name__,
        )
        img = PlottingMixin.rectangles_onto_image(
            img=img,
            rectangles=roi_dict[Keys.ROI_RECTANGLES.value],
            circle_size=circle_size,
            show_center=show_center,
            show_tags=show_tags,
        )
        img = PlottingMixin.circles_onto_image(
            img=img,
            circles=roi_dict[Keys.ROI_CIRCLES.value],
            circle_size=circle_size,
            show_center=show_center,
            show_tags=show_tags,
        )
        img = PlottingMixin.polygons_onto_image(
            img=img,
            polygons=roi_dict[Keys.ROI_POLYGONS.value],
            circle_size=circle_size,
            show_center=show_center,
            show_tags=show_tags,
        )
        return img

    @staticmethod
    def insert_directing_line(
        directing_df: pd.DataFrame,
        img: np.ndarray,
        shape_name: str,
        animal_name: str,
        frame_id: int,
        color: Optional[Tuple[int]] = (0, 0, 255),
        thickness: Optional[int] = 2,
        style: Optional[str] = "lines",
    ) -> np.ndarray:
        """
        Helper to insert lines between the actor 'eye' and the ROI centers.

        :param directing_df: Dataframe containing eye and ROI locations. Stored as ``results`` in  instance of ``simba.roi_tools.ROI_directing_analyzer.DirectingROIAnalyzer``.
        :param np.ndarray img: The image to draw the line on.
        :param str shape_name: The name of the shape to draw the line to.
        :param str animal_name: The name of the animal
        :param int frame_id: The frame number in the video
        :param Optional[Tuple[int]] color: The color of the line
        :param Optional[int] thickness: The thickness of the line.
        :param Optional[str] style: The style of the line. "lines" or "funnel".
        :return np.ndarray: The input image with the line.
        """

        check_valid_array(data=img, source=PlottingMixin.insert_directing_line.__name__)
        check_valid_dataframe(
            df=directing_df,
            source=PlottingMixin.rectangles_onto_image.__name__,
            required_fields=[
                "ROI",
                "Animal",
                "Frame",
                "ROI_edge_1_x",
                "ROI_edge_1_y",
                "ROI_edge_2_x",
                "ROI_edge_2_y",
            ],
        )
        r = directing_df.loc[
            (directing_df["ROI"] == shape_name)
            & (directing_df["Animal"] == animal_name)
            & (directing_df["Frame"] == frame_id)
        ].reset_index(drop=True)
        if style == "funnel":
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
            img = cv2.fillPoly(img, [convex_hull_arr], color)
        else:
            img = cv2.line(
                img,
                (int(r["Eye_x"]), int(r["Eye_y"])),
                (int(r["ROI_x"]), int(r["ROI_y"])),
                color,
                thickness,
            )
        return img

    @staticmethod
    def draw_lines_on_img(
        img: np.ndarray,
        start_positions: np.ndarray,
        end_positions: np.ndarray,
        color: Tuple[int, int, int],
        highlight_endpoint: Optional[bool] = False,
        thickness: Optional[int] = 2,
        circle_size: Optional[int] = 2,
    ) -> np.ndarray:
        """
        Helper to draw a set of lines onto an image.

        :param np.ndarray img: The image to draw the lines on.
        :param np.ndarray start_positions: 2D numpy array representing the start positions of the lines in x, y format.
        :param np.ndarray end_positions: 2D numpy array representing the end positions of the lines in x, y format.
        :param Tuple[int, int, int] color: The color of the lines in BGR format.
        :param Optional[bool] highlight_endpoint: If True, highlights the ends of the lines with circles.
        :param Optional[int] thickness: The thickness of the lines.
        :param Optional[int] circle_size: If ``highlight_endpoint`` is True, the size of the highlighted points.

        :return np.ndarray: The image with the lines overlayed.
        """

        check_valid_array(
            data=start_positions,
            source=f"{PlottingMixin.draw_lines_on_img.__name__} img",
        )
        check_valid_array(
            data=start_positions,
            source=f"{PlottingMixin.draw_lines_on_img.__name__} start_positions",
            accepted_ndims=(2,),
            accepted_dtypes=(np.int64,),
            min_axis_0=1,
        )
        check_valid_array(
            data=end_positions,
            source=f"{PlottingMixin.draw_lines_on_img.__name__} end_positions",
            accepted_shapes=[
                (start_positions.shape[0], 2),
            ],
        )
        check_if_valid_rgb_tuple(data=color)
        for i in range(start_positions.shape[0]):
            cv2.line(
                img,
                (start_positions[i][0], start_positions[i][1]),
                (end_positions[i][0], end_positions[i][1]),
                color,
                thickness,
            )
            if highlight_endpoint:
                cv2.circle(
                    img,
                    (end_positions[i][0], end_positions[i][1]),
                    circle_size,
                    color,
                    -1,
                )
        return img


# from sklearn.datasets import make_blobs
# #from sklearn.datasets import ma
#
# x, lbls = make_blobs(n_samples=10000, n_features=2, centers=10, random_state=42)
# x = np.hstack((x, lbls.reshape(-1, 1)))
#
# PlottingMixin.categorical_scatter(data=x, columns=('X', 'Y', 'MY CLUSTERS'), save_path='/Users/simon/Desktop/make_line_plot_plotly.png', show_box=True)
