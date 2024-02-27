import math
from typing import Union

import circle_fit
import numpy as np

from simba.mixins.circular_statistics import CircularStatisticsMixin


def calculate_weighted_avg_1(x, p=None, threshold=0.2):
    if p is not None and len(x) != len(p):
        raise ValueError("Got x and p with different lengths")

    selected_x, selected_p = [], []
    if p is not None:
        p = [0 if val is None else val for val in p]
        for i in range(len(x)):
            if p[i] > threshold:
                selected_x.append(x[i])
                selected_p.append(p[i])

    if len(selected_x) > 0:
        return np.ma.average(selected_x, weights=selected_p)
    else:
        return np.ma.average(x)


def calculate_weighted_avg_2(
    x: np.ndarray, p: Union[np.ndarray, None], threshold: float = 0.2
):
    results = np.full((x.shape[0]), -1.0)
    n = x.shape[0]
    for i in range(n):
        if p is not None:
            p_thresh_idx = np.argwhere(p[i] > threshold).flatten()
            if p_thresh_idx.shape[0] > 0:
                p_vals = p[i][p_thresh_idx]
                bp_vals = x[i][p_thresh_idx]
                weighted_sum = 0
                for x in range(p_vals.shape[0]):
                    weighted_sum += bp_vals[x] * p_vals[x]
                frm_result = weighted_sum / np.sum(p_vals)
            else:
                frm_result = np.mean(x[i])
        else:
            frm_result = np.mean(x[i])

        results[i] = frm_result
    return results


def get_circle_fit_angle_1(x, y, p, threshold=0.5):
    if np.average(p) < threshold:
        return 0

    xc, yc, r, sigma = circle_fit.least_squares_circle(list(zip(x, y)))
    print("first", xc, yc)

    angle = math.degrees(
        math.atan2(y[-1] - yc, x[-1] - xc) - math.atan2(y[0] - yc, x[0] - xc)
    )

    return angle + 360 if angle < 0 else angle


def get_circle_fit_angle_2(x: np.ndarray, y: np.ndarray, p: np.ndarray, threshold=0.5):
    combined_arr = np.stack((x, y), axis=2)
    circles = CircularStatisticsMixin.fit_circle(data=combined_arr)
    print(circles)
    diff_x_last = x[:, -1] - circles[:, 0]
    diff_y_last = y[:, -1] - circles[:, 1]
    diff_x_first = x[:, 0] - circles[:, 0]
    diff_y_first = y[:, 0] - circles[:, 1]
    # angle = math.degrees(math.atan2(y[-1] - yc, x[-1] - xc) - math.atan2(y[0] - yc, x[0] - xc))

    angles = np.degrees(
        np.arctan2(diff_y_last, diff_x_last) - np.arctan2(diff_y_first, diff_x_first)
    )
    angles = angles + 360 * (angles < 0)
    below_thresh_idx = np.argwhere(np.average(p, axis=1) < threshold)
    angles[below_thresh_idx] = 0.0
    return angles


x = np.random.randint(0, 50, (10,))
y = np.random.randint(0, 50, (10,))
p = np.random.random((50,))


print("Original circle fit", get_circle_fit_angle_1(x=x, y=y, p=p))
print(
    "New circle fit",
    get_circle_fit_angle_2(
        x=x.reshape(1, x.shape[0]),
        y=y.reshape(1, y.shape[0]),
        p=p.reshape(1, p.shape[0]),
    ),
)


# print('Original calculate_weighted_avg', calculate_weighted_avg_1(x=x, p=p))
# print('Speeded up calculate_weighted_avg', calculate_weighted_avg_2(x=x.reshape(1, x.shape[0]), p=p.reshape(1, p.shape[0]))[0])
