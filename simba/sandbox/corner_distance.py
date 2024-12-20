import numpy as np

from simba.utils.read_write import read_df


def img_edge_distances(data: np.ndarray,
                   pixels_per_mm: float,
                   img_resolution: np.ndarray,
                   time_window: float,
                   fps: int) -> np.ndarray:

    """
    Calculate the distances from a set of points to the edges of an image over a specified time window.

    This function computes the average distances from given coordinates to the four edges (top, right, bottom, left)
    of an image. The distances are calculated for points within a specified time window, and the results are adjusted
    based on the pixel-to-mm conversion.

    :param np.ndarray data: 3d array of size len(frames) x N x 2 with body-part coordinates.
    :param np.ndarray img_resolution: Resolution of video in WxH format.
    :param float pixels_per_mm: Pixels per millimeter of recorded video.
    :param int fps: FPS of the recorded video
    :param float time_windows: Rolling time-window as floats in seconds. E.g., ``0.2``
    :returns np.ndarray: Size data.shape[0] x 4 array with millimeter distances from TOP LEFT, TOP RIGH, BOTTOM RIGHT, BOTTOM LEFT.
    :rtype: np.ndarray

    :example I:
    >>> data = np.array([[0, 0], [758, 540], [0, 540], [748, 540]])
    >>> img_edge_distances(data=data, pixels_per_mm=2.13, img_resolution=np.array([748, 540]), time_window=1.0, fps=1)

    :example II:
    >>> data = read_df(file_path=FILE_PATH, file_type='csv', usecols=['Nose_x', 'Nose_y', 'Tail_base_x', 'Tail_base_y'])
    >>> data = data.values.reshape(len(data), 2, 2)
    >>> img_edge_distances(data=data, pixels_per_mm=2.13, img_resolution=np.array([748, 540]), time_window=1.0, fps=1)

    """

    results = np.full((data.shape[0], 4), np.nan)
    window_size = int(time_window * fps)
    for r in range(window_size, data.shape[0]+1):
        l = r - window_size
        w_data = data[l:r].reshape(-1, 2)
        w_distances = np.full((4, w_data.shape[0]), np.nan)
        for idx in range(w_data.shape[0]):
            w_distances[0, idx] = np.linalg.norm(w_data[idx] - np.array([0, 0]))
            w_distances[1, idx] = np.linalg.norm(w_data[idx] - np.array([img_resolution[0], 0]))
            w_distances[2, idx] = np.linalg.norm(w_data[idx] - np.array([img_resolution[0], img_resolution[1]]))
            w_distances[3, idx] = np.linalg.norm(w_data[idx] - np.array([0, img_resolution[1]]))
        for i in range(4):
            results[r-1][i] = np.mean(w_distances[i]) / pixels_per_mm

    return results.astype(np.float32)


FILE_PATH = r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\501_MA142_Gi_CNO_0514.csv"



data = read_df(file_path=FILE_PATH, file_type='csv', usecols=['Nose_x', 'Nose_y', 'Tail_base_x', 'Tail_base_y'])
data = data.values.reshape(len(data), 2, 2)

data = np.array([[0, 0], [758, 540], [0, 540], [748, 540]])

img_edge_distances(data=data, pixels_per_mm=2.13, img_resolution=np.array([748, 540]), time_window=1.0, fps=1)


