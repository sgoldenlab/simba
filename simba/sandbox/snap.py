import numpy as np

from simba.utils.read_write import read_df


def snap(x: np.ndarray, sample_rate: float, pixels_per_mm: float):
    """
    :example:
    >>> x = read_df(file_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/outlier_corrected_movement_location/FRR_gq_CNO_0625.csv', file_type='csv', usecols=['Nose_x', 'Nose_y']).values
    >>> snap(x=x, sample_rate=30, pixels_per_mm=2.15)
    """
    dt = 1 / sample_rate
    velocity = np.gradient(x, axis=0) / dt
    acceleration = np.gradient(velocity, axis=0) / dt
    jerk = np.gradient(acceleration, axis=0) / dt
    snap = np.gradient(jerk, axis=0) / dt
    snap_magnitude = np.linalg.norm(snap, axis=1) / pixels_per_mm
    return snap_magnitude


x = read_df(file_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/outlier_corrected_movement_location/FRR_gq_CNO_0625.csv', file_type='csv', usecols=['Nose_x', 'Nose_y']).values
snap(x=x, sample_rate=30, pixels_per_mm=2.15)

#
# x = np.random.randint(0, 500, (200, 2))
# sample_rate = 10
# pixels_per_mm = 3.4
# snap(x=x, sample_rate=sample_rate, pixels_per_mm=pixels_per_mm)
# time = np.linspace(0, 10, 100)







# # Example data
# time = np.linspace(0, 10, 100)  # Time array (N,)
# x = np.sin(time)               # Example x positions
# y = np.cos(time)               # Example y positions
#
# # Combine into a 2D array of shape (N, 2)
# positions = np.column_stack((x, y))
#
# # Compute derivatives
# velocity = np.gradient(positions, axis=0, edge_order=2) / np.gradient(time)      # First derivative
# acceleration = np.gradient(velocity, axis=0, edge_order=2) / np.gradient(time)  # Second derivative
# jerk = np.gradient(acceleration, axis=0, edge_order=2) / np.gradient(time)      # Third derivative
# snap = np.gradient(jerk, axis=0, edge_order=2) / np.gradient(time)              # Fourth derivative
#
# # Print some results
# print("Velocity (first 5):", velocity[:5])
# print("Acceleration (first 5):", acceleration[:5])
# print("Jerk (first 5):", jerk[:5])
# print("Snap (first 5):", snap[:5])
