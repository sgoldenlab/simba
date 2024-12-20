import numpy as np
import pandas as pd


class KalmanFilter(object):
    def __init__(self, H: np.ndarray, fps: float):

        dt = 1.0 / fps
        print(dt)
        self.F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
        self.H = H
        self.B = 0
        self.n = self.F.shape[1]
        self.m = self.H.shape[1]
        self.Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
        self.R = np.array([0.5]).reshape(1, 1)
        self.P = np.eye(self.n)
        self.x = np.zeros((self.n, 1))


    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)




H= np.array([1, 0, 0]).reshape(1, 3)
kf = KalmanFilter(fps=60, H=H)

df = pd.read_csv('/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Together_1.csv')
for i in df.columns:
    x = df[i].values
    predictions = np.full((x.shape[0]), np.nan)
    for c, z in enumerate(x):
        val = np.dot(H, kf.predict())[0]
        predictions[c] = np.dot(H, kf.predict())[0]
        #predictions.append(np.dot(H, kf.predict())[0])
        kf.update(z)