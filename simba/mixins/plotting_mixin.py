__author__ = "Simon Nilsson"

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import io
import os
import numpy as np
import matplotlib
import shutil
import random
from matplotlib import cm
import imutils
import itertools
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import List, Tuple, Optional, Dict, Any, Union
try:
    from typing import Literal
except:
    from typing_extensions import Literal

import simba
from simba.utils.lookups import get_color_dict, get_named_colors
from simba.utils.read_write import get_fn_ext
from simba.utils.printing import stdout_success, SimbaTimer
from simba.utils.enums import Formats, Options

class PlottingMixin(object):
    """
    Methods for visualizations
    """

    def __init__(self):
        pass

    def create_gantt_img(self,
                         bouts_df: pd.DataFrame,
                         clf_name: str,
                         image_index: int,
                         fps: int,
                         gantt_img_title: str):
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
        relRows = bouts_df.loc[bouts_df['End_frame'] <= image_index]
        for i, event in enumerate(relRows.groupby("Event")):
            data_event = event[1][["Start_time", "Bout_time"]]
            ax.broken_barh(data_event.values, (4, 4), facecolors='red')
        xLength = (round(image_index / fps)) + 1
        if xLength < 10:
            xLength = 10
        ax.set_xlim(0, xLength)
        ax.set_ylim([0, 12])
        plt.ylabel(clf_name, fontsize=12)
        plt.yticks([])
        plt.xlabel('time(s)', fontsize=12)
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

    def create_single_color_lst(self,
                                pallete_name: Literal[Options.PALETTE_OPTIONS],
                                increments: int,
                                as_rgb_ratio: bool = False,
                                as_hex: bool = False) -> List[Union[str, int, float]]:

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

    def remove_a_folder(self,
                        folder_dir: str) -> None:
        """Helper to remove a directory, use for cleaning up smaller multiprocessed videos following concat"""
        shutil.rmtree(folder_dir, ignore_errors=True)

    def split_and_group_df(self,
                           df: pd.DataFrame,
                           splits: int,
                           include_split_order: bool = True) -> (List[pd.DataFrame], int):

        """
        Helper to split a dataframe for multiprocessing. If include_split_order, then include the group number
        in split data as a column. Returns split data and approximations of number of observations per split.
        """

        data_arr = np.array_split(df, splits)
        if include_split_order:
            for df_cnt in range(len(data_arr)):
                data_arr[df_cnt]['group'] = df_cnt
        obs_per_split = len(data_arr[0])

        return data_arr, obs_per_split

    def make_distance_plot(self,
                           data: np.array,
                           line_attr: Dict[int, str],
                           style_attr: Dict[str, Any],
                           fps: int,
                           save_img: bool = False,
                           save_path: Optional[str] = None) -> np.ndarray:
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
            color = (colors[line_attr[j][-1]][::-1])
            color = tuple(x / 255 for x in color)
            plt.plot(data[:, j], color=color, linewidth=style_attr['line width'], alpha=style_attr['opacity'])

        timer = SimbaTimer(start=True)
        max_x = len(data)
        if style_attr['y_max'] == 'auto':
            max_y = np.amax(data)
        else:
            max_y = float(style_attr['y_max'])
        y_ticks_locs = y_lbls = np.round(np.linspace(0, max_y, 10), 2)
        x_ticks_locs = x_lbls = np.linspace(0, max_x, 5)
        x_lbls = np.round((x_lbls / fps), 1)

        plt.xlabel('time (s)')
        plt.ylabel('distance (cm)')
        plt.xticks(x_ticks_locs, x_lbls, rotation='horizontal', fontsize=style_attr['font size'])
        plt.yticks(y_ticks_locs, y_lbls, fontsize=style_attr['font size'])
        plt.ylim(0, max_y)
        plt.suptitle('Animal distances', x=0.5, y=0.92, fontsize=style_attr['font size'] + 4)
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        img = PIL.Image.open(buffer_)
        img = np.uint8(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        buffer_.close()
        plt.close()
        img = cv2.resize(img, (style_attr['width'], style_attr['height']))
        timer.stop_timer()
        if save_img:
            cv2.imwrite(save_path, img)
            stdout_success(f'Final distance plot saved at {save_path}', elapsed_time=timer.elapsed_time_str)
        else:
            return img

    def make_probability_plot(self,
                              data: pd.Series,
                              style_attr: dict,
                              clf_name: str,
                              fps: int,
                              save_path: str) -> np.ndarray:
        """
        Make a single classifier probability plot png image.

        :param pd.Series data: row representing frames and field representing classification probabilities.
        :param dict line_attr: Line color attributes.
        :param dict style_attr: Image attributes (size, font size, line width etc).
        :param int fps: Video frame rate.
        :param str ot
        :param str save_path: Location to store output .png image.



        .. notes::
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
        if style_attr['y_max'] == 'auto':
            max_y = float(data.max().round(2))
        else:
            max_y = float(style_attr['y_max'])
        max_x = len(data)
        plt.plot(list(data), color=style_attr['color'], linewidth=style_attr['line width'])
        plt.plot(len(data), list(data)[-1], "o", markersize=style_attr['circle size'], color=style_attr['color'])
        plt.ylim([0, max_y])
        plt.ylabel(clf_name, fontsize=style_attr['font size'])

        y_ticks_locs = y_lbls = np.round(np.linspace(0, max_y, 10), 2)
        x_ticks_locs = x_lbls = np.linspace(0, max_x, 5)
        x_lbls = np.round((x_lbls / fps), 1)
        plt.xlabel('Time (s)', fontsize=style_attr['font size'] + 4)
        plt.grid()
        plt.xticks(x_ticks_locs, x_lbls, rotation='horizontal', fontsize=style_attr['font size'])
        plt.yticks(y_ticks_locs, y_lbls, fontsize=style_attr['font size'])
        plt.suptitle('{} {}'.format(clf_name, 'probability'), x=0.5, y=0.92, fontsize=style_attr['font size'] + 4)
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)
        ar = np.asarray(image)
        img = cv2.cvtColor(ar, cv2.COLOR_RGB2BGR)
        img = np.uint8(cv2.resize(img, (style_attr['width'], style_attr['height'])))
        buffer_.close()
        plt.close()
        timer.stop_timer()
        cv2.imwrite(save_path, img)
        stdout_success(msg=f'Final distance plot saved at {save_path}', elapsed_time=timer.elapsed_time_str)

    def make_path_plot(self,
                       data_df: pd.DataFrame,
                       video_info: pd.DataFrame,
                       style_attr: dict,
                       deque_dict: dict,
                       clf_attr: dict,
                       save_path: str) -> None:

        """
        Helper to make a path plot.

        :param pd.DataFrame data_df: Dataframe holding body-part coordinates
        :param pd.DataFrame video_info: Video info dataframe (parsed project_folder/logs/video_info.csv)
        :param dict style_attr: Dict holding image style attributes. E.g., {'width': 'As input', 'height': 'As input', 'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': 'White', 'max lines': 100}
        :param dict deque_dict: Dict with deque holsing all paths to visualize
        :param dict clf_attr: Dict holding image classifictaion attributes e.g., {0: ['Attack', 'Black', 'Size: 30'], 1: ['Sniffing', 'Red', 'Size: 30']}
        :param str save_path: Location to save image
        """

        video_timer = SimbaTimer(start=True)
        for frm_cnt in range(len(data_df)):
            for animal_cnt, (animal_name, animal_data) in enumerate(deque_dict.items()):
                bp_x = int(data_df.loc[frm_cnt, '{}_{}'.format(animal_data['bp'], 'x')])
                bp_y = int(data_df.loc[frm_cnt, '{}_{}'.format(animal_data['bp'], 'y')])
                deque_dict[animal_name]['deque'].appendleft((bp_x, bp_y))

        font = cv2.FONT_HERSHEY_COMPLEX
        color_dict = get_color_dict()

        img = np.zeros((int(video_info['Resolution_height'].values[0]), int(video_info['Resolution_width'].values[0]), 3))
        img[:] = style_attr['bg color']

        for animal_name, animal_data in deque_dict.items():
            cv2.circle(img, (deque_dict[animal_name]['deque'][0]), 0, deque_dict[animal_name]['clr'], style_attr['circle size'])
            cv2.putText(img, animal_name, (deque_dict[animal_name]['deque'][0]), font, style_attr['font size'], deque_dict[animal_name]['clr'], style_attr['font thickness'])

        for animal_name, animal_data in deque_dict.items():
            for i in range(len(deque_dict[animal_name]['deque']) - 1):
                line_clr = deque_dict[animal_name]['clr']
                position_1 = deque_dict[animal_name]['deque'][i]
                position_2 = deque_dict[animal_name]['deque'][i + 1]
                cv2.line(img, position_1, position_2, line_clr, style_attr['line width'])

        if clf_attr:
            for clf_cnt, clf_name in enumerate(clf_attr['data'].columns):
                clf_size = int(clf_attr['attr'][clf_cnt][-1].split(': ')[-1])
                clf_clr = color_dict[clf_attr['attr'][clf_cnt][1]]
                clf_sliced_idx = list(clf_attr['data'][clf_attr['data'][clf_name] == 1].index)
                positions = clf_attr['positions'].iloc[clf_sliced_idx].reset_index(drop=True).values

                for i in range(positions.shape[0]):
                    cv2.circle(img, (positions[i][0], positions[i][1]), 0, clf_clr, clf_size)

        video_timer.stop_timer()
        img = cv2.resize(img, (style_attr['width'], style_attr['height']))
        cv2.imwrite(save_path, img)
        stdout_success(msg=f'Final path plot saved at {save_path}', elapsed_time=video_timer.elapsed_time_str)

    def make_gantt_plot(self,
                        data_df: pd.DataFrame,
                        bouts_df: pd.DataFrame,
                        clf_names: List[str],
                        fps: int,
                        style_attr: dict,
                        video_name: str,
                        save_path: str) -> None:

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
        ax.set_yticklabels(clf_names, rotation=style_attr['font rotation'])
        ax.tick_params(axis='both', labelsize=style_attr['font size'])
        plt.xlabel('Session (s)', fontsize=style_attr['font size'] + 3)
        ax.yaxis.grid(True)
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)
        ar = np.asarray(image)
        open_cv_image = cv2.cvtColor(ar, cv2.COLOR_RGB2BGR)
        open_cv_image = cv2.resize(open_cv_image, (style_attr['width'], style_attr['height']))
        frame = np.uint8(open_cv_image)
        buffer_.close()
        plt.close(fig)
        cv2.imwrite(save_path, frame)
        video_timer.stop_timer()
        stdout_success(msg=f'Final gantt frame for video {video_name} saved at {save_path}',
                       elapsed_time=video_timer.elapsed_time_str)

    def make_clf_heatmap_plot(self,
                              frm_data: np.array,
                              max_scale: float,
                              palette: Literal[Options.PALETTE_OPTIONS],
                              aspect_ratio: float,
                              shading: Literal['gouraud', 'flat'],
                              clf_name: str,
                              img_size: Tuple[int, int],
                              file_name: Optional[str] = None,
                              final_img: bool = False):

        cum_df = pd.DataFrame(frm_data).reset_index()
        cum_df = cum_df.melt(id_vars='index', value_vars=None, var_name=None, value_name='seconds',
                             col_level=None).rename(columns={'index': 'vertical_idx', 'variable': 'horizontal_idx'})
        cum_df['color'] = (cum_df['seconds'].astype(float) / float(max_scale)).round(2).clip(upper=100)
        color_array = np.zeros((len(cum_df['vertical_idx'].unique()), len(cum_df['horizontal_idx'].unique())))
        for i in range(color_array.shape[0]):
            for j in range(color_array.shape[1]):
                value = cum_df["color"][(cum_df["horizontal_idx"] == j) & (cum_df["vertical_idx"] == i)].values[0]
                color_array[i, j] = value
        color_array = color_array * max_scale
        fig = plt.figure()
        im_ratio = color_array.shape[0] / color_array.shape[1]
        plt.pcolormesh(color_array, shading=shading, cmap=palette, rasterized=True, alpha=1, vmin=0.0,
                       vmax=float(max_scale))
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.tick_params(axis='both', which='both', length=0)
        cb = plt.colorbar(pad=0.0, fraction=0.023 * im_ratio)
        cb.ax.tick_params(size=0)
        cb.outline.set_visible(False)
        cb.set_label('{} (seconds)'.format(clf_name), rotation=270, labelpad=10)
        plt.tight_layout()
        plt.gca().set_aspect(aspect_ratio)
        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        image = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, img_size)
        image = np.uint8(image)
        plt.close()
        if final_img:
            cv2.imwrite(file_name, image)
            stdout_success(msg=f'Final classifier heatmap image saved at at {file_name}')
        else:
            return image

    def make_location_heatmap_plot(self,
                                   frm_data: np.array,
                                   max_scale: float,
                                   palette: Literal[Options.PALETTE_OPTIONS],
                                   aspect_ratio: float,
                                   shading: str,
                                   img_size: Tuple[int, int],
                                   file_name: str or None = None,
                                   final_img: bool = False):

        cum_df = pd.DataFrame(frm_data).reset_index()
        cum_df = cum_df.melt(id_vars='index', value_vars=None, var_name=None, value_name='seconds',
                             col_level=None).rename(columns={'index': 'vertical_idx', 'variable': 'horizontal_idx'})
        cum_df['color'] = (cum_df['seconds'].astype(float) / float(max_scale)).round(2).clip(upper=100)
        color_array = np.zeros((len(cum_df['vertical_idx'].unique()), len(cum_df['horizontal_idx'].unique())))
        for i in range(color_array.shape[0]):
            for j in range(color_array.shape[1]):
                value = cum_df["color"][(cum_df["horizontal_idx"] == j) & (cum_df["vertical_idx"] == i)].values[0]
                color_array[i, j] = value
        color_array = color_array * max_scale
        fig = plt.figure()
        im_ratio = color_array.shape[0] / color_array.shape[1]
        plt.pcolormesh(color_array, shading=shading, cmap=palette, rasterized=True, alpha=1, vmin=0.0, vmax=max_scale)
        #plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.tick_params(axis='both', which='both', length=0)
        cb = plt.colorbar(pad=0.0, fraction=0.023 * im_ratio)
        cb.ax.tick_params(size=0)
        cb.outline.set_visible(False)
        cb.set_label('location (seconds)', rotation=270, labelpad=10)
        plt.tight_layout()
        plt.gca().set_aspect(aspect_ratio)
        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        image = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, img_size)
        image = np.uint8(image)
        plt.close()
        if final_img:
            cv2.imwrite(file_name, image)
            stdout_success(msg=f'Final location heatmap image saved at at {file_name}')
        else:
            return image

    def get_bouts_for_gantt(self,
                            data_df: pd.DataFrame,
                            clf_name: str,
                            fps: int) -> np.ndarray:
        """
        Helper to detect all behavior bouts for a specific classifier.

        :param pd.DataFrame data_df: Pandas Dataframe with classifier prediction data.
        :param str clf_name: Name of the classifier
        :param int fps: The fps of the input video.
        :return  pd.DataFrame: Holding the start time, end time, end frame, bout time etc of each classified bout.
        """

        boutsList, nameList, startTimeList, endTimeList, endFrameList = [], [], [], [], []
        groupDf = pd.DataFrame()
        v = (data_df[clf_name] != data_df[clf_name].shift()).cumsum()
        u = data_df.groupby(v)[clf_name].agg(['all', 'count'])
        m = u['all'] & u['count'].ge(1)
        groupDf['groups'] = data_df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
        for indexes, rows in groupDf.iterrows():
            currBout = list(rows['groups'])
            boutTime = ((currBout[-1] - currBout[0]) + 1) / fps
            startTime = (currBout[0] + 1) / fps
            endTime = (currBout[1]) / fps
            endFrame = (currBout[1])
            endTimeList.append(endTime)
            startTimeList.append(startTime)
            boutsList.append(boutTime)
            nameList.append(clf_name)
            endFrameList.append(endFrame)

        return pd.DataFrame(list(zip(nameList, startTimeList, endTimeList, endFrameList, boutsList)),
                            columns=['Event', 'Start_time', 'End Time', 'End_frame', 'Bout_time'])

    def resize_gantt(self,
                     gantt_img: np.array,
                     img_height: int) -> np.ndarray:
        """
        Helper to resize image while retaining aspect ratio.
        """

        return imutils.resize(gantt_img, height=img_height)

    @staticmethod
    def roi_feature_visualizer_mp(data: pd.DataFrame,
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
                                  directing_data: pd.DataFrame):

        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        font = cv2.FONT_HERSHEY_COMPLEX

        def __insert_texts(shape_info: dict, img: np.array):
            for shape_name, shape_info in shape_info.items():
                for animal_name in animal_names:
                    shape_color = shape_info['Color BGR']
                    cv2.putText(img, text_locations[animal_name][shape_name]['in_zone_text'],
                                text_locations[animal_name][shape_name]['in_zone_text_loc'], font, scalers['font_size'],
                                shape_color, 1)
                    cv2.putText(img, text_locations[animal_name][shape_name]['distance_text'],
                                text_locations[animal_name][shape_name]['distance_text_loc'], font,
                                scalers['font_size'], shape_color, 1)
                    if directing_viable and style_attr['Directionality']:
                        cv2.putText(img, text_locations[animal_name][shape_name]['directing_text'],
                                    text_locations[animal_name][shape_name]['directing_text_loc'], font,
                                    scalers['font_size'], shape_color, 1)
            return img

        def __insert_shapes(img: np.array, shape_info: dict):
            for shape_name, shape_info in shape_info.items():
                if shape_info['Shape_type'] == 'Rectangle':
                    cv2.rectangle(img, (int(shape_info['topLeftX']), int(shape_info['topLeftY'])),
                                  (int(shape_info['Bottom_right_X']), int(shape_info['Bottom_right_Y'])),
                                  shape_info['Color BGR'], int(shape_info['Thickness']))
                    if style_attr['ROI_centers']:
                        center_cord = ((int(shape_info['topLeftX'] + (shape_info['width'] / 2))),
                                       (int(shape_info['topLeftY'] + (shape_info['height'] / 2))))
                        cv2.circle(img, center_cord, scalers['circle_size'], shape_info['Color BGR'], -1)
                    if style_attr['ROI_ear_tags']:
                        for tag_data in shape_info['Tags'].values():
                            cv2.circle(img, tag_data, scalers['circle_size'], shape_info['Color BGR'], -1)

                if shape_info['Shape_type'] == 'Circle':
                    cv2.circle(img, (int(shape_info['centerX']), int(shape_info['centerY'])), shape_info['radius'],
                               shape_info['Color BGR'], int(shape_info['Thickness']))
                    if style_attr['ROI_centers']:
                        cv2.circle(img, (int(shape_info['centerX']), int(shape_info['centerY'])),
                                   scalers['circle_size'], shape_info['Color BGR'], -1)
                    if style_attr['ROI_ear_tags']:
                        for tag_data in shape_info['Tags'].values():
                            cv2.circle(img, tag_data, scalers['circle_size'], shape_info['Color BGR'], -1)

                if shape_info['Shape_type'] == 'Polygon':
                    cv2.polylines(img, [shape_info['vertices']], True, shape_info['Color BGR'],
                                  thickness=int(shape_info['Thickness']))
                    if style_attr['ROI_centers']:
                        cv2.circle(img, shape_info['Tags']['Center_tag'], scalers['circle_size'],
                                   shape_info['Color BGR'], -1)
                    if style_attr['ROI_ear_tags']:
                        for tag_data in shape_info['Tags'].values():
                            cv2.circle(img, tag_data, scalers['circle_size'], shape_info['Color BGR'], -1)

            return img

        def __insert_directing_line(directing_data: pd.DataFrame,
                                    shape_name: str,
                                    frame_cnt: int,
                                    shape_info: dict,
                                    img: np.array,
                                    video_name: str,
                                    style_attr: dict):

            r = directing_data.loc[(directing_data['Video'] == video_name) & (directing_data['ROI'] == shape_name) & (directing_data['Animal'] == animal_name) & (directing_data['Frame'] == frame_cnt)]
            clr = shape_info[shape_name]['Color BGR']
            thickness = shape_info[shape_name]['Thickness']
            if style_attr['Directionality_style'] == 'Funnel':
                convex_hull_arr = np.array([[r['ROI_edge_1_x'], r['ROI_edge_1_y']],
                                            [r['ROI_edge_2_x'], r['ROI_edge_2_y']],
                                            [r['Eye_x'], r['Eye_y']]]).reshape(-1, 2).astype(int)
                cv2.fillPoly(img, [convex_hull_arr], clr)

            if style_attr['Directionality_style'] == 'Lines':
                cv2.line(img, (int(r['Eye_x']), int(r['Eye_y'])), (int(r['ROI_x']), int(r['ROI_y'])), clr,
                         int(thickness))

            return img

        group_cnt = int(data['group'].values[0])
        start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
        save_path = os.path.join(save_temp_dir, '{}.mp4'.format(str(group_cnt)))
        _, video_name, _ = get_fn_ext(filepath=video_path)
        writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'],
                                 (video_meta_data['width'] * 2, video_meta_data['height']))

        cap = cv2.VideoCapture(video_path)
        cap.set(1, start_frm)

        while current_frm < end_frm:
            ret, img = cap.read()
            img = cv2.copyMakeBorder(img, 0, 0, 0, int(video_meta_data['width']), borderType=cv2.BORDER_CONSTANT,
                                     value=style_attr['Border_color'])
            img = __insert_texts(shape_info=shape_info, img=img)
            if style_attr['Pose_estimation']:
                for animal, animal_bp_name in tracked_bps.items():
                    bp_cords = data.loc[current_frm, animal_bp_name].values
                    cv2.circle(img, (int(bp_cords[0]), int(bp_cords[1])), 0, animal_bps[animal]['colors'][0],
                               scalers['circle_size'])
                    cv2.putText(img, animal, (int(bp_cords[0]), int(bp_cords[1])), font, scalers['font_size'],
                                animal_bps[animal]['colors'][0], 1)

            img = __insert_shapes(img=img, shape_info=shape_info)

            for animal_name, shape_name in itertools.product(animal_names, shape_info):
                in_zone_col_name = '{} {} {}'.format(shape_name, animal_name, 'in zone')
                distance_col_name = '{} {} {}'.format(shape_name, animal_name, 'distance')
                in_zone_value = str(bool(data.loc[current_frm, in_zone_col_name]))
                distance_value = str(round(data.loc[current_frm, distance_col_name], 2))
                cv2.putText(img, in_zone_value, text_locations[animal_name][shape_name]['in_zone_data_loc'], font,
                            scalers['font_size'], shape_info[shape_name]['Color BGR'], 1)
                cv2.putText(img, distance_value, text_locations[animal_name][shape_name]['distance_data_loc'], font,
                            scalers['font_size'], shape_info[shape_name]['Color BGR'], 1)
                if directing_viable and style_attr['Directionality']:
                    facing_col_name = '{} {} {}'.format(shape_name, animal_name, 'facing')
                    facing_value = bool(data.loc[current_frm, facing_col_name])
                    cv2.putText(img, str(facing_value), text_locations[animal_name][shape_name]['directing_data_loc'],
                                font, scalers['font_size'], shape_info[shape_name]['Color BGR'], 1)
                    if facing_value:
                        img = __insert_directing_line(directing_data=directing_data,
                                                      shape_name=shape_name,
                                                      frame_cnt=current_frm,
                                                      shape_info=shape_info,
                                                      img=img,
                                                      video_name=video_name,
                                                      style_attr=style_attr)
            writer.write(img)
            current_frm += 1
            print('Multi-processing video frame {} on core {}...'.format(str(current_frm), str(group_cnt)))
        cap.release()
        writer.release()

        return group_cnt

    @staticmethod
    def directing_animals_mp(data: pd.DataFrame,
                             directionality_data: pd.DataFrame,
                             bp_names: dict,
                             style_attr: dict,
                             save_temp_dir: str,
                             video_path: str,
                             video_meta_data: dict,
                             colors: list):

        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        group_cnt = int(data.iloc[0]['group'])
        start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
        save_path = os.path.join(save_temp_dir, '{}.mp4'.format(str(group_cnt)))
        _, video_name, _ = get_fn_ext(filepath=video_path)
        writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'],
                                 (video_meta_data['width'], video_meta_data['height']))
        cap = cv2.VideoCapture(video_path)
        cap.set(1, start_frm)
        color = colors[0]

        def __draw_individual_lines(animal_img_data: pd.DataFrame,
                                    img: np.array):
            color = colors[0]
            for cnt, (i, r) in enumerate(animal_img_data.iterrows()):
                if style_attr['Direction_color'] == 'Random':
                    color = random.sample(colors[0], 1)[0]
                cv2.line(img, (int(r['Eye_x']), int(r['Eye_y'])),
                         (int(r['Animal_2_bodypart_x']), int(r['Animal_2_bodypart_y'])), color,
                         style_attr['Direction_thickness'])
                if style_attr['Highlight_endpoints']:
                    cv2.circle(img, (int(r['Eye_x']), int(r['Eye_y'])), style_attr['Pose_circle_size'] + 2, color,
                               style_attr['Pose_circle_size'])
                    cv2.circle(img, (int(r['Animal_2_bodypart_x']), int(r['Animal_2_bodypart_y'])),
                               style_attr['Pose_circle_size'] + 1, color, style_attr['Pose_circle_size'])

            return img

        while current_frm < end_frm:
            ret, img = cap.read()
            try:
                if ret:
                    if style_attr['Show_pose']:
                        bp_data = data.loc[current_frm]
                        for cnt, (animal_name, animal_bps) in enumerate(bp_names.items()):
                            for bp in zip(animal_bps['X_bps'], animal_bps['Y_bps']):
                                x_bp, y_bp = bp_data[bp[0]], bp_data[bp[1]]
                                cv2.circle(img, (int(x_bp), int(y_bp)), style_attr['Pose_circle_size'],
                                           bp_names[animal_name]['colors'][cnt], style_attr['Direction_thickness'])

                    if current_frm in list(directionality_data['Frame_#'].unique()):
                        img_data = directionality_data[directionality_data['Frame_#'] == current_frm]
                        unique_animals = img_data['Animal_1'].unique()
                        for animal in unique_animals:
                            animal_img_data = img_data[img_data['Animal_1'] == animal].reset_index(drop=True)
                            if style_attr['Polyfill']:
                                convex_hull_arr = animal_img_data.loc[0, ['Eye_x', 'Eye_y']].values.reshape(-1, 2)
                                for animal_2 in animal_img_data['Animal_2'].unique():
                                    convex_hull_arr = np.vstack((convex_hull_arr, animal_img_data[
                                        ['Animal_2_bodypart_x', 'Animal_2_bodypart_y']][
                                        animal_img_data['Animal_2'] == animal_2].values)).astype('int')
                                    convex_hull_arr = np.unique(convex_hull_arr, axis=0)
                                    if convex_hull_arr.shape[0] >= 3:
                                        if style_attr['Direction_color'] == 'Random':
                                            color = random.sample(colors[0], 1)[0]
                                        cv2.fillPoly(img, [convex_hull_arr], color)
                                    else:
                                        img = __draw_individual_lines(animal_img_data=animal_img_data, img=img)

                            else:
                                img = __draw_individual_lines(animal_img_data=animal_img_data, img=img)

                    img = np.uint8(img)

                    current_frm += 1
                    writer.write(img)
                    print('Multi-processing video frame {} on core {}...'.format(str(current_frm), str(group_cnt)))

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
    def distance_plotter_mp(data: np.array,
                            video_setting: bool,
                            frame_setting: bool,
                            video_name: str,
                            video_save_dir: str,
                            frame_folder_dir: str,
                            style_attr: dict,
                            line_attr: dict,
                            fps: int):

        group = int(data[0][0])
        line_data = data[:, 2:]
        color_dict = get_color_dict()
        video_writer = None
        if video_setting:
            fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
            video_save_path = os.path.join(video_save_dir, '{}.mp4'.format(str(group)))
            video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (style_attr['width'], style_attr['height']))

        for i in range(line_data.shape[0]):
            frame_id = int(data[i][1])
            for j in range(line_data.shape[1]):
                color = (color_dict[line_attr[j][-1]][::-1])
                color = tuple(x / 255 for x in color)
                plt.plot(line_data[0:i, j], color=color, linewidth=style_attr['line width'],
                         alpha=style_attr['opacity'])

            x_ticks_locs = x_lbls = np.round(np.linspace(0, i, 5))
            x_lbls = np.round((x_lbls / fps), 1)
            plt.ylim(0, style_attr['max_y'])
            plt.xlabel('time (s)')
            plt.ylabel('distance (cm)')
            plt.xticks(x_ticks_locs, x_lbls, rotation='horizontal', fontsize=style_attr['font size'])
            plt.yticks(style_attr['y_ticks_locs'], style_attr['y_ticks_lbls'], fontsize=style_attr['font size'])
            plt.suptitle('Animal distances', x=0.5, y=0.92, fontsize=style_attr['font size'] + 4)

            buffer_ = io.BytesIO()
            plt.savefig(buffer_, format="png")
            buffer_.seek(0)
            img = PIL.Image.open(buffer_)
            img = np.uint8(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
            buffer_.close()
            plt.close()

            img = cv2.resize(img, (style_attr['width'], style_attr['height']))
            if video_setting:
                video_writer.write(np.uint8(img))
            if frame_setting:
                frm_name = os.path.join(frame_folder_dir, str(frame_id) + '.png')
                cv2.imwrite(frm_name, np.uint8(img))

            print('Distance frame created: {}, Video: {}, Processing core: {}'.format(str(frame_id + 1), video_name, str(group + 1)))

        return group

    @staticmethod
    def gantt_creator_mp(data: np.array,
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
                         video_name: str):

        group, frame_rng = data[0], data[1:]
        start_frm, end_frm, current_frm = frame_rng[0], frame_rng[-1], frame_rng[0]
        video_writer = None
        if video_setting:
            fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
            video_save_path = os.path.join(video_save_dir, '{}.mp4'.format(str(group)))
            video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

        while current_frm < end_frm:
            fig, ax = plt.subplots()
            bout_rows = bouts_df.loc[bouts_df['End_frame'] <= current_frm]
            for i, event in enumerate(bout_rows.groupby("Event")):
                for x in clf_names:
                    if event[0] == x:
                        ix = clf_names.index(x)
                        data_event = event[1][["Start_time", "Bout_time"]]
                        ax.broken_barh(data_event.values, (color_tuple[ix], 3), facecolors=colors[ix])

            x_ticks_locs = x_lbls = np.round(np.linspace(0, round((current_frm / fps), 3), 6))
            ax.set_xticks(x_ticks_locs)
            ax.set_xticklabels(x_lbls)
            ax.set_ylim(0, color_tuple[len(clf_names)])
            ax.set_yticks(np.arange(5, 5 * len(clf_names) + 1, 5))
            ax.tick_params(axis='both', labelsize=font_size)
            ax.set_yticklabels(clf_names, rotation=rotation)
            ax.set_xlabel('Session (s)', fontsize=font_size)
            ax.yaxis.grid(True)
            canvas = FigureCanvas(fig)
            canvas.draw()
            mat = np.array(canvas.renderer._renderer)
            img = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            img = np.uint8(cv2.resize(img, (width, height)))
            if video_setting:
                video_writer.write(img)
            if frame_setting:
                frame_save_name = os.path.join(frame_folder_dir, '{}.png'.format(str(current_frm)))
                cv2.imwrite(frame_save_name, img)
            plt.close(fig)
            current_frm += 1

            print('Gantt frame created: {}, Video: {}, Processing core: {}'.format(str(current_frm + 1), video_name,
                                                                                   str(group + 1)))

        if video_setting:
            video_writer.release()

        return group


    @staticmethod
    def roi_plotter_mp(data: pd.DataFrame,
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
                       style_attr: dict):

        def __insert_texts(shape_df):
            for animal_name in roi_analyzer_data.multi_animal_id_list:
                for _, shape in shape_df.iterrows():
                    shape_name, shape_color = shape['Name'], shape['Color BGR']
                    cv2.putText(border_img, loc_dict[animal_name][shape_name]['timer_text'],
                                loc_dict[animal_name][shape_name]['timer_text_loc'], font, scalers['font_size'],
                                shape_color, 1)
                    cv2.putText(border_img, loc_dict[animal_name][shape_name]['entries_text'],
                                loc_dict[animal_name][shape_name]['entries_text_loc'], font, scalers['font_size'],
                                shape_color, 1)

            return border_img

        font = cv2.FONT_HERSHEY_TRIPLEX
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        group_cnt = int(data['group'].values[0])
        start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
        save_path = os.path.join(save_temp_directory, '{}.mp4'.format(str(group_cnt)))
        writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'],
                                 (video_meta_data['width'] * 2, video_meta_data['height']))
        cap = cv2.VideoCapture(input_video_path)
        cap.set(1, start_frm)

        while current_frm < end_frm:
            ret, img = cap.read()
            border_img = cv2.copyMakeBorder(img, 0, 0, 0, int(video_meta_data['width']), borderType=cv2.BORDER_CONSTANT,
                                            value=[0, 0, 0])
            border_img = __insert_texts(roi_analyzer_data.video_recs)
            border_img = __insert_texts(roi_analyzer_data.video_circs)
            border_img = __insert_texts(roi_analyzer_data.video_polys)

            for _, row in roi_analyzer_data.video_recs.iterrows():
                top_left_x, top_left_y, shape_name = row['topLeftX'], row['topLeftY'], row['Name']
                bottom_right_x, bottom_right_y = row['Bottom_right_X'], row['Bottom_right_Y']
                thickness, color = row['Thickness'], row['Color BGR']
                cv2.rectangle(border_img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)

            for _, row in roi_analyzer_data.video_circs.iterrows():
                center_x, center_y, radius, shape_name = row['centerX'], row['centerY'], row['radius'], row['Name']
                thickness, color = row['Thickness'], row['Color BGR']
                cv2.circle(border_img, (center_x, center_y), radius, color, thickness)

            for _, row in roi_analyzer_data.video_polys.iterrows():
                vertices, shape_name = row['vertices'], row['Name']
                thickness, color = row['Thickness'], row['Color BGR']
                cv2.polylines(border_img, [vertices], True, color, thickness=thickness)

            for animal_cnt, animal_name in enumerate(roi_analyzer_data.multi_animal_id_list):
                if style_attr['Show_body_part'] or style_attr['Show_animal_name']:
                    bp_data = data.loc[current_frm, body_part_dict[animal_name]].values
                    if roi_analyzer_data.settings['threshold'] < bp_data[2]:
                        if style_attr['Show_body_part']:
                            cv2.circle(border_img, (int(bp_data[0]), int(bp_data[1])), scalers['circle_size'],
                                       colors[animal_cnt], -1)
                        if style_attr['Show_animal_name']:
                            cv2.putText(border_img, animal_name, (int(bp_data[0]), int(bp_data[1])), font,
                                        scalers['font_size'], colors[animal_cnt], 1)

                for shape_name in video_shape_names:
                    timer = round(data.loc[current_frm, '{}_{}_cum_sum_time'.format(animal_name, shape_name)], 2)
                    entries = data.loc[current_frm, '{}_{}_cum_sum_entries'.format(animal_name, shape_name)]
                    cv2.putText(border_img, str(timer), loc_dict[animal_name][shape_name]['timer_data_loc'], font,
                                scalers['font_size'], shape_meta_data[shape_name]['Color BGR'], 1)
                    cv2.putText(border_img, str(entries), loc_dict[animal_name][shape_name]['entries_data_loc'], font,
                                scalers['font_size'], shape_meta_data[shape_name]['Color BGR'], 1)

            writer.write(border_img)
            current_frm += 1
            print('Multi-processing video frame {} on core {}...'.format(str(current_frm), str(group_cnt)))

        cap.release()
        writer.release()

        return group_cnt


    @staticmethod
    def validation_video_mp(data: pd.DataFrame,
                            bp_dict: dict,
                            video_save_dir: str,
                            settings: dict,
                            video_path: str,
                            video_meta_data: dict,
                            gantt_setting: Optional[Literal['Gantt chart: final frame only (slightly faster)', 'Gantt chart: video']],
                            final_gantt: Optional[np.ndarray],
                            clf_data: np.ndarray,
                            clrs: List[List],
                            clf_name: str,
                            bouts_df: pd.DataFrame):

        dpi = plt.rcParams['figure.dpi']

        def create_gantt(bouts_df: pd.DataFrame,
                         clf_name: str,
                         image_index: int,
                         fps: int):

            fig, ax = plt.subplots(figsize=(final_gantt.shape[1] / dpi, final_gantt.shape[0] / dpi))
            matplotlib.font_manager._get_font.cache_clear()
            relRows = bouts_df.loc[bouts_df['End_frame'] <= image_index]

            for i, event in enumerate(relRows.groupby("Event")):
                data_event = event[1][["Start_time", "Bout_time"]]
                ax.broken_barh(data_event.values, (4, 4), facecolors='red')
            xLength = (round(image_index / fps)) + 1
            if xLength < 10: xLength = 10

            ax.set_xlim(0, xLength)
            ax.set_ylim([0, 12])
            ax.set_xlabel('Session (s)', fontsize=12)
            ax.set_ylabel(clf_name, fontsize=12)
            ax.set_title(f'{clf_name} GANTT CHART', fontsize=12)
            ax.set_yticks([])
            ax.yaxis.set_ticklabels([])
            ax.yaxis.grid(True)
            canvas = FigureCanvas(fig)
            canvas.draw()
            img = np.array(np.uint8(np.array(canvas.renderer._renderer)))[:, :, :3]
            plt.close(fig)
            return img

        fourcc, font = cv2.VideoWriter_fourcc(*'mp4v'), cv2.FONT_HERSHEY_COMPLEX
        cap = cv2.VideoCapture(video_path)
        group = data['group'].iloc[0]
        start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
        video_save_path = os.path.join(video_save_dir, '{}.mp4'.format(str(group)))
        if gantt_setting is not None:
            writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data['fps'], (int(video_meta_data['width'] + final_gantt.shape[1]), int(video_meta_data['height'])))
        else:
            writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data['fps'], (video_meta_data['width'], video_meta_data['height']))

        cap.set(1, start_frm)
        while current_frm < end_frm:
            clf_frm_cnt = np.sum(clf_data[0:current_frm])
            ret, img = cap.read()
            if settings['pose']:
                for animal_cnt, (animal_name, animal_data) in enumerate(bp_dict.items()):
                    for bp_cnt, bp in enumerate(range(len(animal_data['X_bps']))):
                        x_header, y_header = animal_data['X_bps'][bp], animal_data['Y_bps'][bp]
                        animal_cords = tuple(data.loc[current_frm, [x_header, y_header]])
                        cv2.circle(img, (int(animal_cords[0]), int(animal_cords[1])), 0, clrs[animal_cnt][bp_cnt],
                                   settings['styles']['circle size'])
            if settings['animal_names']:
                for animal_cnt, (animal_name, animal_data) in enumerate(bp_dict.items()):
                    x_header, y_header = animal_data['X_bps'][0], animal_data['Y_bps'][0]
                    animal_cords = tuple(data.loc[current_frm, [x_header, y_header]])
                    cv2.putText(img, animal_name, (int(animal_cords[0]), int(animal_cords[1])), font,
                                settings['styles']['font size'], clrs[animal_cnt][0], 1)
            target_timer = round((1 / video_meta_data['fps']) * clf_frm_cnt, 2)
            cv2.putText(img, 'Timer', (10, settings['styles']['space_scale']), font, settings['styles']['font size'],
                        (0, 255, 0), 2)
            addSpacer = 2
            cv2.putText(img, (f'{clf_name} {target_timer}s'), (10, settings['styles']['space_scale'] * addSpacer), font,
                        settings['styles']['font size'], (0, 0, 255), 2)
            addSpacer += 1
            cv2.putText(img, 'Ensemble prediction', (10, settings['styles']['space_scale'] * addSpacer), font,
                        settings['styles']['font size'], (0, 255, 0), 2)
            addSpacer += 2
            if clf_data[current_frm] == 1:
                cv2.putText(img, clf_name, (10, + settings['styles']['space_scale'] * addSpacer), font,
                            settings['styles']['font size'], (2, 166, 249), 2)
                addSpacer += 1
            if gantt_setting == 'Gantt chart: final frame only (slightly faster)':
                img = np.concatenate((img, final_gantt), axis=1)
            elif gantt_setting == 'Gantt chart: video':
                gantt_img = create_gantt(bouts_df, clf_name, current_frm, video_meta_data['fps'])
                img = np.concatenate((img, gantt_img), axis=1)

            writer.write(np.uint8(img))
            current_frm += 1
            print('Multi-processing video frame {} on core {}...'.format(str(current_frm), str(group)))

        cap.release()
        writer.release()

        return group

    @staticmethod
    def bbox_mp(frm_range: list,
                polygon_data: dict,
                animal_bp_dict: dict,
                data_df: Optional[pd.DataFrame],
                intersection_data_df: Optional[pd.DataFrame],
                roi_attributes: dict,
                video_path: str,
                key_points: bool,
                greyscale: bool):

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
                for animal_cnt, (animal, animal_data) in enumerate(animal_bp_dict.items()):
                    if key_points:
                        for bp_cnt, (x_col, y_col) in enumerate(zip(animal_data['X_bps'], animal_data['Y_bps'])):
                            cv2.circle(frame, (frm_data[x_col], frm_data[y_col]), 0, roi_attributes[animal]['bbox_clr'],
                                       roi_attributes[animal]['keypoint_size'])
                    animal_polygon = np.array(
                        list(polygon_data[animal][current_frame].convex_hull.exterior.coords)).astype(int)
                    if intersection_data_df is not None:
                        intersect = intersection_data_df.loc[
                            current_frame, intersection_data_df.columns.str.startswith(animal)].sum()
                        if intersect > 0:
                            cv2.polylines(frame, [animal_polygon], 1, roi_attributes[animal]['highlight_clr'],
                                          roi_attributes[animal]['highlight_clr_thickness'])
                    cv2.polylines(frame, [animal_polygon], 1, roi_attributes[animal]['bbox_clr'],
                                  roi_attributes[animal]['bbox_thickness'])
                img_lst.append(frame)
                current_frame += 1
            else:
                print(
                    'SIMBA WARNING: SimBA tried to grab frame number {} from video {}, but could not find it. The video has {} frames.'.format(
                        str(current_frame), video_path, str(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        return img_lst

    @staticmethod
    def path_plot_mp(data: np.array,
                       video_setting: bool,
                       frame_setting: bool,
                       video_save_dir: str,
                       video_name: str,
                       frame_folder_dir: str,
                       style_attr: dict,
                       animal_attr: dict,
                       fps: int,
                       video_info: pd.DataFrame,
                       clf_attr: dict):

        group = int(data[0][0][0])
        color_dict = get_color_dict()
        if video_setting:
            fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
            video_save_path = os.path.join(video_save_dir, '{}.mp4'.format(str(group)))
            video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (style_attr['width'], style_attr['height']))

        for i in range(data.shape[0]):
            frame_id = int(data[i, -1, 1] + 1)
            frame_data = data[i, :, 2:].astype(int)
            frame_data = np.split(frame_data, len(list(animal_attr.keys())), axis=1)

            img = np.zeros(
                (int(video_info['Resolution_height'].values[0]), int(video_info['Resolution_width'].values[0]), 3))
            img[:] = style_attr['bg color']
            for animal_cnt, animal_data in enumerate(frame_data):
                animal_clr = style_attr['animal clrs'][animal_cnt]
                for j in range(animal_data.shape[0] - 1):
                    cv2.line(img, tuple(animal_data[j]), tuple(animal_data[j + 1]), animal_clr,
                             style_attr['line width'])
                cv2.circle(img, tuple(animal_data[-1]), 0, animal_clr, style_attr['circle size'])
                cv2.putText(img, style_attr['animal names'][animal_cnt], tuple(animal_data[-1]),
                            cv2.FONT_HERSHEY_COMPLEX, style_attr['font size'], animal_clr, style_attr['font thickness'])

            if clf_attr:
                for clf_cnt, clf_name in enumerate(clf_attr['data'].columns):
                    clf_size = int(clf_attr['attr'][clf_cnt][-1].split(': ')[-1])
                    clf_clr = color_dict[clf_attr['attr'][clf_cnt][1]]
                    clf_sliced = clf_attr['data'][clf_name].loc[0: frame_id]
                    clf_sliced_idx = list(clf_sliced[clf_sliced == 1].index)
                    locations = clf_attr['positions'][clf_sliced_idx, :]
                    for i in range(locations.shape[0]):
                        cv2.circle(img, (locations[i][0], locations[i][1]), 0, clf_clr, clf_size)

            img = cv2.resize(img, (style_attr['width'], style_attr['height']))
            if video_setting:
                video_writer.write(np.uint8(img))
            if frame_setting:
                frm_name = os.path.join(frame_folder_dir, str(frame_id) + '.png')
                cv2.imwrite(frm_name, np.uint8(img))

            print('Path frame created: {}, Video: {}, Processing core: {}'.format(str(frame_id + 1), video_name,
                                                                                  str(group + 1)))

        return group

    @staticmethod
    def probability_plot_mp(data: list,
                            probability_lst: list,
                            clf_name: str,
                            video_setting: bool,
                            frame_setting: bool,
                            video_dir: str,
                            frame_dir: str,
                            highest_p: float,
                            fps: int,
                            style_attr: dict,
                            video_name: str):

        group, data = data[0], data[1:]
        start_frm, end_frm, current_frm = data[0], data[-1], data[0]

        if video_setting:
            fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
            video_save_path = os.path.join(video_dir, '{}.mp4'.format(str(group)))
            video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (style_attr['width'], style_attr['height']))

        while current_frm < end_frm:
            fig, ax = plt.subplots()
            current_lst = probability_lst[0:current_frm + 1]
            ax.plot(current_lst, color=style_attr['color'], linewidth=style_attr['line width'])
            ax.plot(current_frm, current_lst[-1], "o", markersize=style_attr['circle size'], color=style_attr['color'])
            ax.set_ylim([0, highest_p])
            x_ticks_locs = x_lbls = np.linspace(0, current_frm, 5)
            x_lbls = np.round((x_lbls / fps), 1)
            ax.xaxis.set_ticks(x_ticks_locs)
            ax.set_xticklabels(x_lbls, fontsize=style_attr['font size'])
            ax.set_xlabel('Time (s)', fontsize=style_attr['font size'])
            ax.set_ylabel('{} {}'.format(clf_name, 'probability'), fontsize=style_attr['font size'])
            plt.suptitle(clf_name, x=0.5, y=0.92, fontsize=style_attr['font size'] + 4)
            canvas = FigureCanvas(fig)
            canvas.draw()
            mat = np.array(canvas.renderer._renderer)
            image = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            image = np.uint8(cv2.resize(image, (style_attr['width'], style_attr['height'])))
            if video_setting:
                video_writer.write(image)
            if frame_setting:
                frame_save_name = os.path.join(frame_dir, '{}.png'.format(str(current_frm)))
                cv2.imwrite(frame_save_name, image)
            plt.close()
            current_frm += 1

            print(
                'Probability frame created: {}, Video: {}, Processing core: {}'.format(str(current_frm + 1), video_name,
                                                                                       str(group + 1)))

        return group
