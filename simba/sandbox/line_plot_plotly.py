from typing import List, Optional, Union
import os

import cv2
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from PIL import Image
import io

from simba.utils.lookups import get_color_dict
from simba.utils.printing import stdout_success


def make_line_plot_plotly(data: List[np.ndarray],
                          colors: List[str],
                          show_box: Optional[bool] = True,
                          show_grid: Optional[bool] = True,
                          width: Optional[int] = 640,
                          height: Optional[int] = 480,
                          line_width: Optional[int] = 6,
                          font_size: Optional[int] = 8,
                          bg_clr: Optional[str] = 'white',
                          x_lbl_divisor: Optional[float] = None,
                          title: Optional[str] = None,
                          y_lbl: Optional[str] = None,
                          x_lbl: Optional[str] = None,
                          y_max: Optional[int] = -1,
                          line_opacity: Optional[int] = 0.5,
                          save_path: Optional[Union[str, os.PathLike]] = None):

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
    >>> x = np.random.randint(0, 50, (100,))
    >>> y = np.random.randint(0, 50, (200,))
    >>> data = [x, y]
    >>> img = make_line_plot_plotly(data=data, show_box=False, font_size=20, bg_clr='white', show_grid=False, x_lbl_divisor=30, colors=['Red', 'Green'], save_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/line_plot/Trial     3_final_img.png')


    """

    def tick_formatter(x):
        if x_lbl_divisor is not None:
            return x / x_lbl_divisor
        else:
            return x

    fig = go.Figure()
    clr_dict = get_color_dict()
    if y_max == -1: y_max = max([np.max(i) for i in data])
    for i in range(len(data)):
        line_clr = clr_dict[colors[i]][::-1]
        line_clr = f'rgba({line_clr[0]}, {line_clr[1]}, {line_clr[2]}, {line_opacity})'
        fig.add_trace(go.Scatter(y=data[i], mode='lines', line=dict(color=line_clr, width=line_width)))

    if not show_box:
        fig.update_layout(width=width, height=height, title=title, xaxis_visible=False, yaxis_visible=False, showlegend=False)
    else:
        if fig['layout']['xaxis']['tickvals'] is None:
            tickvals = [i for i in range(len(data))]
        else:
            tickvals = fig['layout']['xaxis']['tickvals']
        ticktext = [tick_formatter(x) for x in tickvals]
        fig.update_layout(
            width=width,
            height=height,
            title=title,
            xaxis=dict(
                title=x_lbl,
                tickmode='linear' if x_lbl_divisor is None else 'auto',
                tickvals=tickvals,
                ticktext=ticktext,
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
            showlegend=False
        )

    if bg_clr is not None:
        fig.update_layout(plot_bgcolor=bg_clr)
    if save_path is not None:
        pio.write_image(fig, save_path)
        stdout_success(msg=f'Line plot saved at {save_path}')
    else:
        img_bytes = fig.to_image(format="png")
        img = Image.open(io.BytesIO(img_bytes))
        fig.purge()
        return np.array(img).astype(np.uint8)



x = np.random.randint(0, 50, (100,))
y = np.random.randint(0, 50, (200,))
data = [x, y]
img = make_line_plot_plotly(data=data, show_box=True, font_size=20, bg_clr='white', show_grid=False, x_lbl_divisor=30.2, colors=['Red', 'Green'], save_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/line_plot/Trial     3_final_img.png')

#
# img = make_line_plot(data=data, show_box=False, font_size=20, bg_clr='white', show_grid=False, x_lbl_divisor=30, colors=['Red', 'Green'], save_path=None)
#
# import cv2
# cv2.imshow('img', img)
# cv2.waitKey(5000)



