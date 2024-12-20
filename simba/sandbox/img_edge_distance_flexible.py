import numpy as np


def img_edge_distances_flexible(data: np.ndarray,
                                edge_distances: np.ndarray,
                                pixels_per_mm: float,
                                img_resolution: np.ndarray,
                                time_window: float,
                                fps: int) -> np.ndarray:

    """
    """

    results = np.full((data.shape[0], 4), np.nan)
    left = img_resolution[1] * edge_distances[0]
    top = img_resolution[0] * edge_distances[1]
    right = img_resolution[1] * edge_distances[2]
    bottom = img_resolution[0] * edge_distances[3]
    print(edge_distances[0])
    edge_distances = np.array([int(left), int(top), int(right), int(bottom)])
    window_size = int(time_window * fps)
    for r in range(window_size, data.shape[0] + 1):
        l = r - window_size
        w_data = data[l:r].reshape(-1, 2)
        w_distances = np.full((4, w_data.shape[0]), np.nan)
        for idx in range(w_data.shape[0]):
            w_distances[0, idx] = np.linalg.norm(w_data[idx] - edge_distances[0])
            w_distances[1, idx] = np.linalg.norm(w_data[idx] - edge_distances[1])
            w_distances[2, idx] = np.linalg.norm(w_data[idx] - edge_distances[2])
            w_distances[3, idx] = np.linalg.norm(w_data[idx] - edge_distances[3])
        for i in range(4):
            results[r - 1][i] = np.mean(w_distances[i]) / pixels_per_mm

    return results.astype(np.float32)

data = np.array([[0, 0], [758, 540], [0, 540], [748, 540]])
img_edge_distances_flexible(data=data,
                            edge_distances=np.array([0.0, 0.0, 0.0, 0.0]),
                            pixels_per_mm=2.13,
                            img_resolution=np.array([748, 540]),
                            time_window=1.0,
                            fps=1)
