import io
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import PIL
import plotly.express as px

from simba.utils.checks import (check_float, check_instance, check_int,
                                check_valid_dataframe)
from simba.utils.data import create_color_palettes, detect_bouts
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import read_df, seconds_to_timestamp


def gantt_plotly(bouts_df: pd.DataFrame,
                 img_size: Optional[Tuple[int, int]] = (640, 480),
                 bg_clr: Optional[str] = 'white',
                 title: Optional[str] = None,
                 font_size: Optional[int] = 12,
                 y_lbl: Optional[str] = 'Event',
                 x_lbl: Optional[str] = 'Session time (HH:MM:SS)',
                 show_grid: Optional[bool] = True,
                 color_palette: Optional[str] = 'Set1',
                 x_length: Optional[int] = None,
                 bar_height: Optional[float] = 0.4,
                 x_tick_spacing: Optional[Union[int, float]] = 30,
                 marker_line_color: Optional[str] = 'black',
                 marker_line_width: Optional[float] = 0.1,
                 time_format: Optional[str] = 'HH:MM:SS',
                 tick_angle: Optional[int] = 45,
                 font: Optional[str] = 'Georgia') -> np.ndarray:

    """
    Generates a Gantt chart using Plotly to visualize bout events over time.

    Creates a horizontal bar chart where each row represents an event (e.g., animal behavior) over time, with estensive customization options.

    .. image:: _static/img/gantt_plotly.webp
       :width: 600
       :align: center

    :param pd.DataFrame bouts_df: A pandas DataFrame containing the bout events data. Can be created by :func:`~simba.utils.data.detect_bouts`.
    :param Optional[Tuple[int, int]] img_size: The size of the output image as (width, height). Default (640, 480).
    :param Optional[str] bg_clr: The color of the background as a string. Deafult: white.
    :param Optional[str] title: The title of image. Deafult: None.
    :param Optional[int] font_size: The font size for the title, labels, ticks. deafult 12.
    :param Optional[str] y_lbl: The label on the y-axis. Deafult: `Event`.
    :param Optional[str] x_lbl: The label on the x-axis. Deafult: 'Session time (s)'.
    :param Optional[bool] show_grid: Whether to show the grid on the plot. Default is True.
    :param Optional[str] color_palette: The name of the color palette to use for event bars. Default: ``Set``.
    :param Optional[int] x_length: The maximum value of the x-axis. If None, the x-axis is determined based on the data.
    :param Optional[str] bar_height: The height of each bar in the Gantt chart. Default is 0.4.
    :param Optional[Union[int, float]] x_tick_spacing: The spacing between x-axis ticks. Can be either an integer or float. If float, it is treated as a fraction of the x-axis range (between 0 and 1). If integer, then interpreted as seconds. Default is 30.
    :param Optional[str] marker_line_color: Color of the bar borders. Default is 'black'.
    :param Optional[float] marker_line_width: Width of the bar borders. Default is 0.5.
    :param Optional[str] time_format: Format for x-axis tick labels. Supported formats are: - 'HH:MM:SS': Converts seconds to hours:minutes:seconds format. - 'seconds': Displays tick labels as raw seconds. Default is 'HH:MM:SS'.
    :param Optional[int] tick_angle: Angle for rotating x-axis tick labels. Default is 45 degrees.
    :param Optional[str] font: Font name. E.g., "Arial", "Verdana", "Helvetica", "Tahoma", "Trebuchet MS", "Times New Roman", "Georgia", "Courier New", "Lucida Console". Default is None, which uses the default Plotly font.
    :return: A Gantt chart image as a NumPy array (dtype=np.uint8).
    :rtype: np.ndarray

    :example:
    >>> FILE_PATH = r"D:\troubleshooting\mitra\project_folder\logs\all\592_MA147_CNO1_0515.csv"
    >>> data = read_df(file_path=FILE_PATH, file_type='csv')
    >>> bouts_df = detect_bouts(data_df=data, target_lst=list(data.columns), fps=30)
    >>> img = gantt_plotly(bouts_df=bouts_df, x_tick_spacing=120, color_palette='Set1', img_size=(1200, 700), font_size=32, show_grid=True)
    """

    check_valid_dataframe(df=bouts_df, source=f'{gantt_plotly.__name__} bouts_df', required_fields=['Start_time', 'Bout_time', 'Event'])
    check_instance(source=f'{gantt_plotly.__name__} x_tick_spacing', instance=x_tick_spacing, accepted_types=(int, float,))
    if isinstance(x_tick_spacing, float):
        check_float(name=f'{gantt_plotly.__name__} x_tick_spacing', value=x_tick_spacing, min_value=10e-6, max_value=1.0, raise_error=True)
        if x_length is not None:
            x_tick_spacing = int(x_length * x_tick_spacing)
        else:
            x_tick_spacing = int(bouts_df['End Time'].max()  * x_tick_spacing)
    elif isinstance(x_tick_spacing, int):
        check_int(name=f'{gantt_plotly.__name__} x_tick_spacing', value=x_tick_spacing, min_value=1, raise_error=True)
    check_int(name=f'{gantt_plotly.__name__} tick_angle', value=tick_angle, min_value=0, max_value=360)
    last_bout_time = bouts_df['Start_time'].max() + bouts_df['Bout_time'].max()
    if x_length is not None:
        last_bout_time = max(x_length, last_bout_time)
    tickvals = np.arange(0, last_bout_time, x_tick_spacing)
    if time_format == 'seconds':
        ticktext = [str(x) for x in tickvals]
    elif time_format == 'HH:MM:SS':
        ticktext = []
        for val in tickvals: ticktext.append(seconds_to_timestamp(val))
    else:
        raise InvalidInputError(msg=f'{time_format} is not a valid time_format', source=gantt_plotly.__name__)
    width, height = img_size
    unique_event_cnt = len(list(bouts_df['Event'].unique()))
    clrs = create_color_palettes(no_animals=1, map_size=unique_event_cnt, cmaps=[color_palette])[0]
    clrs = dict(zip(bouts_df["Event"].unique(), clrs))
    clrs = {k: f'rgb{tuple(v)}' for k, v in clrs.items()}
    fig = px.bar(bouts_df, base="Start_time", x="Bout_time", y="Event", color="Event", color_discrete_map=clrs, orientation="h")
    fig.update_layout(width=width, height=height, title=title, yaxis_type="category", showlegend=False)
    fig.update_traces(width=bar_height, marker_line_color=marker_line_color, marker_line_width=marker_line_width)

    fig.update_layout(width=width,
                      height=height,
                      title=title,
                      font=dict(
                          family=font,
                      ),
                      xaxis=dict(
                          title=dict(
                              text=x_lbl,
                              font=dict(size=font_size, family=font),
                          ),
                          showline=True,
                          linecolor="black",
                          tickvals=tickvals,
                          ticktext=ticktext,
                          tickangle=tick_angle,
                          tickfont=dict(size=font_size),
                          showgrid=show_grid,
                          range=[0, last_bout_time]),
                      yaxis=dict(
                          title=dict(
                              text=y_lbl,
                              font=dict(size=font_size)
                          ),
                          tickfont=dict(size=font_size),
                          showline=True,
                          linecolor="black",
                          showgrid=show_grid,
                      ),
                      showlegend=False)

    if bg_clr is not None:
        fig.update_layout(plot_bgcolor=bg_clr)
    img_bytes = fig.to_image(format="png")
    img = PIL.Image.open(io.BytesIO(img_bytes))
    fig = None
    return np.array(img).astype(np.uint8)


# FILE_PATH = r"D:\troubleshooting\mitra\project_folder\logs\all\592_MA147_CNO1_0515.csv"
#
# data = read_df(file_path=FILE_PATH, file_type='csv')
# bouts_df = detect_bouts(data_df=data, target_lst=list(data.columns), fps=30)
# img = gantt_plotly(bouts_df=bouts_df, x_tick_spacing=120, color_palette='Set1', img_size=(1200, 700), font_size=32, show_grid=True, bg_clr='white')
# cv2.imshow('asdasd', img)
# cv2.waitKey(1)