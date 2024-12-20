
import pandas as pd
import numpy as np
import plotly.express as px
import PIL
import io
import cv2
from typing import Optional


def gantt_plotly(bouts_df: pd.DataFrame,
                 width: Optional[int] = 640,
                 height: Optional[int] = 480,
                 bg_clr: Optional[str] = 'white',
                 title: Optional[str] = None,
                 font_size: Optional[int] = 12,
                 y_lbl: Optional[str] = 'Event',
                 x_lbl: Optional[str] = 'Session time (s)',
                 show_grid: Optional[bool] = False,
                 x_length: Optional[int] = None):

    last_bout_time = bouts_df['Start_time'].max() + bouts_df['Bout_time'].max()
    if x_length is not None:
        last_bout_time = max(x_length, last_bout_time)

    fig = px.bar(bouts_df,
                 base="Start_time",
                 x="Bout_time",
                 y="Event",
                 color=bouts_df.Event.astype(str),
                 orientation="h")

    fig.update_layout(
        width=width,
        height=height,
        title=title,
        yaxis_type="category",
        showlegend=False,
    )

    fig.update_layout(
        width=width,
        height=height,
        title=title,
        xaxis=dict(
            title=x_lbl,
            tickfont=dict(size=font_size),
            showgrid=show_grid,
            range=[0, last_bout_time]
        ),
        yaxis=dict(
            title=y_lbl,
            tickfont=dict(size=font_size),
            showgrid=show_grid,
        ),
        showlegend=False,
    )

    if bg_clr is not None:
        fig.update_layout(plot_bgcolor=bg_clr)
    img_bytes = fig.to_image(format="png")
    img = PIL.Image.open(io.BytesIO(img_bytes))
    fig = None
    return np.array(img).astype(np.uint8)


bouts_df = pd.read_csv('/Users/simon/Desktop/envs/simba/simba/simba/sandbox/bouts_df')
img = gantt_plotly(bouts_df=bouts_df)
cv2.imshow('img', img)
cv2.waitKey(5000)


