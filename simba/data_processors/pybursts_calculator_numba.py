### CODE MODIFID FROM PYBURST PACKAGE  https://pypi.org/project/pybursts/
import math
from typing import Union

import numpy as np
from numba import float64, int64, njit

from simba.utils.checks import check_float
from simba.utils.errors import ArrayError, FloatError, InvalidInputError


@njit("float64[:](float64[:], int64, float64[:], float64)", cache=True)
def _kleinberg_viterbi(gaps: np.ndarray, k: int, alpha: np.ndarray, gamma_log_n: float) -> np.ndarray:
    """Numba-accelerated Viterbi forward pass with backpointer traceback."""
    n = gaps.shape[0]
    log_alpha = np.empty(k)
    for j in range(k):
        log_alpha[j] = math.log(alpha[j])
    C = np.full(k, np.inf)
    C[0] = 0.0
    C_prime = np.empty(k)
    backptr = np.empty((n, k), dtype=np.int64)

    for t in range(n):
        for j in range(k):
            best_cost = np.inf
            best_i = 0
            for i in range(k):
                tc = 0.0 if i >= j else (j - i) * gamma_log_n
                cost = C[i] + tc
                if cost < best_cost:
                    best_cost = cost
                    best_i = i
            C_prime[j] = best_cost - log_alpha[j] + alpha[j] * gaps[t]
            backptr[t, j] = best_i
        for j in range(k):
            C[j] = C_prime[j]

    best_j = 0
    best_val = C[0]
    for j in range(1, k):
        if C[j] < best_val:
            best_val = C[j]
            best_j = j

    states = np.empty(n, dtype=np.int64)
    states[n - 1] = best_j
    for t in range(n - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]

    q = np.empty(n, dtype=np.float64)
    for t in range(n):
        q[t] = float(states[t] + 1)
    return q


def kleinberg_burst_detection(offsets: np.ndarray, s: float, gamma: float) -> np.ndarray:
    """
    Detect hierarchical bursts in a 1D sequence of event times using Kleinberg's two-state infinite-state automaton (modified from `pybursts <https://pypi.org/project/pybursts/>`_).

    Bursts are intervals where events arrive at a higher-than-baseline rate. The algorithm assigns each inter-event gap to a discrete *level* ``q``: level ``0`` is baseline, higher levels (1, 2, …) are progressively faster (higher-rate) bursts. Each level transition opens or closes a burst at that level, producing a hierarchy of nested bursts.

    .. note::
       Private helper used by :class:`~simba.data_processors.kleinberg_calculator.KleinbergCalculator`.
       For an end-to-end pipeline (frame indices → bouts → bursts), use that class.

    :param np.ndarray offsets: 1D numeric array of event times (seconds, frame indices, or any monotonically meaningful unit). May be unsorted; sorted internally. Must have **strictly positive gaps** (no two events at the same time).
    :param float s: Base of the rate scale (``> 1``). The candidate rate at level ``j`` is ``s**j / mean_gap``, so larger ``s`` means levels grow farther apart and bursts must be more pronounced to reach higher levels. Common choice: ``s = 2``.
    :param float gamma: Cost of moving up one level (``>= 0``). Higher ``gamma`` penalizes rising into a burst, producing fewer / shorter bursts. Lower ``gamma`` makes the detector more sensitive. Common choice: ``gamma = 1``.

    :return: 2D ``np.ndarray`` of shape ``(N, 3)`` and ``dtype=object`` with one row per detected burst, columns ``[level, start_offset, end_offset]``:

        * ``level`` — integer burst level (``0`` is the baseline level, higher levels are nested faster bursts).
        * ``start_offset`` — value from ``offsets`` where this burst opens.
        * ``end_offset`` — value from ``offsets`` where this burst closes (inclusive of the last event in the run).

        For a single-event input, returns a single row ``[0, offsets[0], offsets[0]]``.
    :rtype: np.ndarray

    :raises FloatError: If ``s`` is not a valid float or is <= 1.0.
    :raises FloatError: If ``gamma`` is not a valid float or is < 0.
    :raises ArrayError: If ``offsets`` is not a valid numpy array.
    :raises InvalidInputError: If ``offsets`` has fewer than 1 element.
    :raises InvalidInputError: If ``offsets`` contains duplicate values (zero gaps).

    .. seealso::
       :func:`~simba.data_processors.pybursts_calculator.kleinberg_burst_detection` for non-Numba version.

    .. csv-table:: EXPECTED RUNTIMES
       :file: ../../docs/tables/kleinberg_burst_detection.csv
       :widths: 20, 15, 20, 20, 15
       :align: center
       :header-rows: 1

    :example:
    >>> import numpy as np
    >>> from simba.data_processors.pybursts_calculator_numba import kleinberg_burst_detection
    >>> offsets = np.array([1.0, 1.1, 1.2, 5.0, 9.0, 9.05, 9.1])
    >>> bursts = kleinberg_burst_detection(offsets=offsets, s=2.0, gamma=1.0)
    >>> bursts.shape[1]
    3
    """

    check_float(name=f'{kleinberg_burst_detection.__name__} s', value=s, min_value=1.01)
    check_float(name=f'{kleinberg_burst_detection.__name__} gamma', value=gamma, min_value=0.0)
    if not isinstance(offsets, (np.ndarray, list, tuple)):
        raise ArrayError(msg=f'offsets should be a numpy array, list, or tuple, but got {type(offsets)}', source=kleinberg_burst_detection.__name__)
    offsets = np.array(offsets, dtype=np.float64)
    if offsets.size < 1:
        raise InvalidInputError(msg=f'offsets must have at least 1 element, but got {offsets.size}', source=kleinberg_burst_detection.__name__)

    if offsets.size == 1:
        bursts = np.array([0, offsets[0], offsets[0]], ndmin=2, dtype=object)
        return bursts

    offsets = np.sort(offsets)
    gaps = np.diff(offsets)

    if not np.all(gaps):
        raise InvalidInputError(msg=f'offsets contains duplicate values (zero-length gaps between events). All offsets must be unique.', source=kleinberg_burst_detection.__name__)

    T = float(np.sum(gaps))
    n = gaps.size
    g_hat = T / n
    k = int(math.ceil(1.0 + math.log(T, s) + math.log(1.0 / float(np.amin(gaps)), s)))
    gamma_log_n = gamma * math.log(n)
    alpha = np.array([s ** j / g_hat for j in range(k)], dtype=np.float64)

    q = _kleinberg_viterbi(gaps, k, alpha, gamma_log_n)

    prev_q = 0.0
    N = 0
    for t in range(n):
        if q[t] > prev_q:
            N += int(q[t] - prev_q)
        prev_q = q[t]

    bursts = np.empty((N, 3), dtype=object)
    burst_counter = -1
    prev_q = 0.0
    stack = np.empty(N, dtype=np.int64)
    stack_counter = -1
    for t in range(n):
        if q[t] > prev_q:
            num_levels_opened = int(q[t] - prev_q)
            for i in range(num_levels_opened):
                burst_counter += 1
                bursts[burst_counter, 0] = int(prev_q + i)
                bursts[burst_counter, 1] = int(offsets[t])
                stack_counter += 1
                stack[stack_counter] = burst_counter
        elif q[t] < prev_q:
            num_levels_closed = int(prev_q - q[t])
            for i in range(num_levels_closed):
                bursts[stack[stack_counter], 2] = int(offsets[t])
                stack_counter -= 1
        prev_q = q[t]

    while stack_counter >= 0:
        bursts[stack[stack_counter], 2] = int(offsets[n])
        stack_counter -= 1

    return bursts
#
#
# if __name__ == "__main__":
#     import time
#
#     np.random.seed(42)
#     sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
#     s, gamma = 2.0, 0.3
#
#     # warm up numba JIT
#     _ = kleinberg_burst_detection(offsets=np.arange(10, dtype=np.float64), s=s, gamma=gamma)
#
#     for n in sizes:
#         offsets = np.sort(np.cumsum(np.random.exponential(scale=1.0, size=n)))
#         start = time.perf_counter()
#         result = kleinberg_burst_detection(offsets=offsets, s=s, gamma=gamma)
#         elapsed = time.perf_counter() - start
#         print(f"n={n:>10,}python   bursts={result.shape[0]:>6,}  time={elapsed:.4f}s")
