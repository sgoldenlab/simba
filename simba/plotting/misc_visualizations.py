__author__ = "Simon Nilsson"

import numpy as np
import matplotlib.pyplot as plt
import io
import PIL
import cv2
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from simba.misc_tools import (get_color_dict,
                              get_named_colors,
                              SimbaTimer)



def make_distance_plot(data: np.array,
                       line_attr: dict,
                       style_attr: dict,
                       fps: int,
                       save_path: str or None=None,
                       save_img: bool=False):
    """
    Helper to make a single line plot .png image with N lines.

    Parameters
    ----------
    data: np.array
        Two-dimensional array where rows represent frames and columns represent values.
    line_attr: dict
        Line color attributes.
    style_attr: dict
        Image attributes (size, font size, line width etc).
    fps: int
        Video frame rate.
    save_path:
        Location to store output .png image.


    Notes
    -----
    `GitHub tutorial/documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-distance-plots>`__.

    Examples
    -----
    >>> fps = 10
    >>> data = np.random.random((100,2))
    >>> line_attr = {0: ['Blue'], 1: ['Red']}
    >>> save_path = '/_tests/final_frm.png'
    >>> style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'y_max': 'auto'}
    >>> make_distance_plot(fps=fps, data=data, line_attr=line_attr, style_attr=style_attr, save_path=save_path)
    """
    colors = get_color_dict()
    for j in range(data.shape[1]):
        color = (colors[line_attr[j][-1]][::-1])
        color = tuple(x / 255 for x in color)
        plt.plot(data[:, j], color=color, linewidth=style_attr['line width'], alpha=style_attr['opacity'])

    timer = SimbaTimer()
    timer.start_timer()
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
        print('SIMBA COMPLETE: Final distance plot saved at {} (elapsed time: {}s)'.format(save_path, timer.elapsed_time_str))
    else:
        return img

def make_probability_plot(data: pd.Series,
                          style_attr: dict,
                          clf_name: str,
                          fps,
                          save_path: str):
    """
    Helper to make a single classifier probability plot png image.

    Parameters
    ----------
    data: pd.Series
        With row representing frames and field representing classification probabilities.
    line_attr: dict
        Line color attributes.
    style_attr: dict
        Image attributes (size, font size, line width etc).
    fps: int
        Video frame rate.
    save_path:
        Location to store output .png image.


    Notes
    -----
    `GitHub tutorial/documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-classification-probabilities>`__.

    Examples
    -----
    >>> data = pd.Series(np.random.random((100, 1)).flatten())
    >>> style_attr = {'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'blue', 'circle size': 20}
    >>> clf_name='Attack'
    >>> fps=10
    >>> save_path = '/_test/frames/output/probability_plots/Together_1_final_frame.png'

    >>> _ = make_probability_plot(data=data, style_attr=style_attr, clf_name=clf_name, fps=fps, save_path=save_path)
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
    print('SIMBA COMPLETE: Final distance plot saved at {} (elapsed time: {}s)'.format(save_path, timer.elapsed_time_str))


def make_path_plot(data_df: pd.DataFrame,
                   video_info: pd.DataFrame,
                   style_attr: dict,
                   deque_dict: dict,
                   clf_attr: dict,
                   save_path: str):

    video_timer = SimbaTimer()
    video_timer.start_timer()
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
        for i in range(len(deque_dict[animal_name]['deque'])-1):
            line_clr = deque_dict[animal_name]['clr']
            position_1 = deque_dict[animal_name]['deque'][i]
            position_2 = deque_dict[animal_name]['deque'][i+1]
            cv2.line(img, position_1, position_2, line_clr, style_attr['line width'])

    if clf_attr:
        animal_1_name = list(deque_dict.keys())[0]
        animal_bp_x, animal_bp_y = deque_dict[animal_1_name]['bp'] + '_x', deque_dict[animal_1_name]['bp'] + '_y'
        for clf_cnt, clf_data in clf_attr.items():
            clf_size = int(clf_data[-1].split(': ')[-1])
            clf_clr = color_dict[clf_data[1]]
            sliced_df_idx = data_df[data_df[clf_data[0]] == 1].index
            locations = data_df.loc[sliced_df_idx, [animal_bp_x, animal_bp_y]].astype(int).values
            for i in range(locations.shape[0]):
                cv2.circle(img, (locations[i][0], locations[i][1]), 0, clf_clr, clf_size)

    video_timer.stop_timer()
    img = cv2.resize(img, (style_attr['width'], style_attr['height']))
    cv2.imwrite(save_path, img)
    print('SIMBA COMPLETE: Final path plot saved at {} (elapsed time: {}s)'.format(save_path, video_timer.elapsed_time_str))

def make_gantt_plot(data_df: pd.DataFrame,
                    bouts_df: pd.DataFrame,
                    clf_names: list,
                    fps: int,
                    style_attr: dict,
                    video_name: str,
                    save_path: str):

        video_timer = SimbaTimer()
        video_timer.start_timer()
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
        print('SIMBA COMPLETE: Final gantt frame for video {} saved at {} (elapsed time: {}s) ...'.format(video_name, save_path, video_timer.elapsed_time_str))


def make_clf_heatmap_plot(frm_data: np.array,
                      max_scale: float,
                      palette: str,
                      aspect_ratio: float,
                      shading: str,
                      clf_name: str,
                      img_size: tuple,
                      file_name: str or None=None,
                      final_img: bool=False):

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
    plt.pcolormesh(color_array, shading=shading, cmap=palette, rasterized=True, alpha=1, vmin=0.0, vmax=float(max_scale))
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
        print('SIMBA COMPLETE: Final classifier heatmap image saved at at {}...'.format(file_name))
    else:
        return image

def make_location_heatmap_plot(frm_data: np.array,
                               max_scale: float,
                               palette: str,
                               aspect_ratio: float,
                               shading: str,
                               img_size: tuple,
                               file_name: str or None=None,
                               final_img: bool=False):

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
    plt.gca().invert_yaxis()
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
        print(f'SIMBA COMPLETE: Final location heatmap image saved at at {file_name}...')
    else:
        return image




# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 5,
#               'font size': 5,
#               'font thickness': 2,
#               'circle size': 5,
#               'bg color': 'White',
#               'max lines': 100}
# animal_attr = {0: ['Ear_right_1', 'Red']}
# clf_attr = {0: ['Attack', 'Black', 'Size: 30'], 1: ['Sniffing', 'Red', 'Size: 30']}
#







# fps = 10
# data = np.random.random((100,2))
# line_attr = {0: ['Green'], 1: ['Red']}
# save_path = '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/frames/output/line_plot/final_frm.png'
# style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8}
# make_distance_plot(fps=fps, data=data, line_attr=line_attr, style_attr=style_attr, save_path=save_path)

# data = pd.Series(np.random.random((100, 1)).flatten())
# style_attr = {'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'blue', 'circle size': 20}
# clf_name='Attack'
# fps=10
# save_path = '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/frames/output/probability_plots/Together_1_final_frame.png'
#
#
# _ = make_probability_plot(data=data,
#                           style_attr=style_attr,
#                           clf_name=clf_name,
#                           fps=fps,
#                           save_path=save_path)