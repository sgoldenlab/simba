import numpy as np
from numba import njit, prange
import time

@njit('(float64[:], float64[:], float64[:], float64, boolean, float64)')
def sliding_two_signal_crosscorrelation(x: np.ndarray,
                                        y: np.ndarray,
                                        windows: np.ndarray,
                                        sample_rate: float,
                                        normalize: bool,
                                        lag: float) -> np.ndarray:
    """
    Calculate sliding (lagged) cross-correlation between two signals, e.g., the movement and velocity of two animals.

    .. note::
        If no lag needed, pass lag 0.0.

    :param np.ndarray x: The first input signal.
    :param np.ndarray y: The second input signal.
    :param np.ndarray windows: Array of window lengths in seconds.
    :param float sample_rate: Sampling rate of the signals (in Hz or FPS).
    :param bool normalize: If True, normalize the signals before computing the correlation.
    :param float lag: Time lag between the signals in seconds.

    :return: 2D array of sliding cross-correlation values. Each row corresponds to a time index, and each column corresponds to a window size specified in the `windows` parameter.

    :example:
    >>> x = np.random.randint(0, 10, size=(20,))
    >>> y = np.random.randint(0, 10, size=(20,))
    >>> sliding_two_signal_crosscorrelation(x=x, y=y, windows=np.array([1.0, 1.2]), sample_rate=10, normalize=True, lag=0.0)
    """


    results = np.full((x.shape[0], windows.shape[0]), 0.0)
    lag = int(sample_rate * lag)
    for i in prange(windows.shape[0]):
        W_s = int(windows[i] * sample_rate)
        for cnt, (l1, r1) in enumerate(zip(range(0, x.shape[0] + 1), range(W_s, x.shape[0] + 1))):
            l2 = l1 - lag
            if l2 < 0: l2 = 0
            r2 = r1 - lag
            if r2 - l2 < W_s: r2 = l2 + W_s
            X_w = x[l1:r1]
            Y_w = y[l2:r2]
            if normalize:
                X_w = (X_w - np.mean(X_w)) / (np.std(X_w) * X_w.shape[0])
                Y_w = (Y_w - np.mean(Y_w)) / np.std(Y_w)
            v = np.correlate(a=X_w, v=Y_w)[0]
            if np.isnan(v):
                results[r1 - 1, i] = 0.0
            else:
                results[int(r1 - 1), i] = v
    return results.astype(np.float32)

start = time.time()
x = np.random.randint(0, 10, size=(4000000,)).astype(np.float64)
y = np.random.randint(0, 10, size=(4000000,)).astype(np.float64)
p = sliding_two_signal_crosscorrelation(x=x, y=y, windows=np.array([1.0, 1.2]), sample_rate=10.0, normalize=True, lag=0.0)
print(time.time() - start)






