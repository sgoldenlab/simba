import numpy as np


def avg_kinetic_energy(x: np.ndarray, mass: float, sample_rate: float) -> float:
    """
    Calculate the average kinetic energy of an object based on its velocity.

    :param np.ndarray x: A 2D NumPy array of shape (n, 2), where each row contains the x and y  position coordinates of the object at each time step.
    :param float mass: The mass of the object.
    :param float sample_rate: The sampling rate (Hz), i.e., the number of data points per second.
    :return: The average kinetic energy of the object.
    :rtype: float: The mean kinetic energy calculated from the velocity data.
    """
    delta_t = np.round(1 / sample_rate, 2)
    vx, vy = np.gradient(x[:, 0], delta_t), np.gradient(x[:, 1], delta_t)
    speed = np.sqrt(vx ** 2 + vy ** 2)
    kinetic_energy = 0.5 * mass * speed ** 2

    return np.mean(kinetic_energy).astype(np.float32)





import pandas as pd

x = pd.read_csv(r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement\501_MA142_Gi_CNO_0514.csv", usecols=['Nose_x', 'Nose_y']).values.astype(np.int32)

#x = np.random.randint(0, 500, (10, 2))
mass = 1
avg_kinetic_energy(x=x, mass=mass,sample_rate=30)








