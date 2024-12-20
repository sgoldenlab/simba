from typing import List, Optional, Tuple, Union, Dict, Any
from simba.utils.printing import stdout_success, SimbaTimer
import os
import cv2
import numpy as np
from simba.utils.checks import check_int, check_instance, check_valid_lst, check_if_valid_rgb_tuple, check_if_dir_exists, check_valid_array, check_if_keys_exist_in_dict, check_float

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
    >>> make_path_plot(data=[x, y], colors=[(0, 255, 0), (255, 0, 0)], clf_attr=clf_data)
    """

    check_valid_lst(data=data, source=make_path_plot.__name__, valid_dtypes=(np.ndarray,), min_len=1)
    for i in data: check_valid_array(data=i, source=make_path_plot.__name__, accepted_ndims=(2,), accepted_axis_1_shape=(2,), accepted_dtypes=(int, float, np.int32, np.int64, np.float32, np.float64))
    check_valid_lst(data=colors, source=make_path_plot.__name__, valid_dtypes=(tuple,), exact_len=len(data))
    for i in colors: check_if_valid_rgb_tuple(data=i)
    check_instance(source='bg_clr', instance=bg_clr, accepted_types=(np.ndarray, tuple))
    if isinstance(bg_clr, tuple):
        check_if_valid_rgb_tuple(data=bg_clr)
    check_int(name=f'{make_path_plot.__name__} height', value=height, min_value=1)
    check_int(name=f'{make_path_plot.__name__} height', value=width, min_value=1)
    check_float(name=f'{make_path_plot.__name__} font_size', value=font_size)
    check_int(name=f'{make_path_plot.__name__} font_thickness', value=font_thickness)
    check_int(name=f'{make_path_plot.__name__} line_width', value=line_width)
    timer = SimbaTimer(start=True)
    img = np.zeros((height, width, 3))
    img[:] = bg_clr
    for line_cnt in range(len(data)):
        clr = colors[line_cnt]
        line_data = data[line_cnt]
        if max_lines is not None:
            check_int(name=f'{make_path_plot.__name__} max_lines', value=max_lines, min_value=1)
            line_data = line_data[-max_lines:]
        for i in range(1, line_data.shape[0]):
            cv2.line(img, tuple(line_data[i]), tuple(line_data[i-1]), clr, line_width)
        if circle_size is not None:
            cv2.circle(img,  tuple(line_data[-1]),  0, clr, circle_size)
        if animal_names is not None:
            cv2.putText(img, animal_names[line_cnt], tuple(line_data[-1]), cv2.FONT_HERSHEY_COMPLEX, font_size, clr, font_thickness)
    if clf_attr is not None:
        check_instance(source=make_path_plot.__name__, instance=clf_attr, accepted_types=(dict,))
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
        stdout_success(msg=f"Path plot saved at {save_path}", elapsed_time=timer.elapsed_time_str, source=make_path_plot.__name__,)
    else:
        return img

# x = np.random.randint(0, 500, (100, 2))
# y = np.random.randint(0, 500, (100, 2))
# position_data = np.random.randint(0, 500, (100, 2))
# clf_data_1 = np.random.randint(0, 2, (100,))
# clf_data_2 = np.random.randint(0, 2, (100,))
# clf_data = {'Attack': {'color': (155, 1, 10), 'size': 30, 'positions': position_data, 'clfs': clf_data_1},  'Sniffing': {'color': (155, 90, 10), 'size': 30, 'positions': position_data, 'clfs': clf_data_2}}
# plot = make_path_plot(data=[x, y], colors=[(0, 255, 0), (255, 0, 0)], clf_attr=clf_data)
#
