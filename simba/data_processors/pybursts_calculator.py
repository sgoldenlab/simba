### CODE MODIFID FROM PYBURST PACKAGE  https://pypi.org/project/pybursts/
import math

import numpy as np


def kleinberg_burst_detection(offsets: np.ndarray, s: float, gamma: float):
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

    :raises ValueError: If ``offsets`` contains two or more events at the same time (zero gap).

    .. seealso::
       :func:`~simba.data_processors.pybursts_calculator_numba.kleinberg_burst_detection` for Numba JIT-accelerated version.

    .. csv-table:: EXPECTED RUNTIMES
       :file: ../../docs/tables/kleinberg_burst_detection.csv
       :widths: 20, 15, 20, 20, 15
       :align: center
       :header-rows: 1

    :example:
    >>> import numpy as np
    >>> from simba.data_processors.pybursts_calculator import kleinberg_burst_detection
    >>> offsets = np.array([1.0, 1.1, 1.2, 5.0, 9.0, 9.05, 9.1])
    >>> bursts = kleinberg_burst_detection(offsets=offsets, s=2.0, gamma=1.0)
    >>> bursts.shape[1]
    3
    """

    offsets = np.array(offsets, dtype=object)

    if offsets.size == 1:
        bursts = np.array([0, offsets[0], offsets[0]], ndmin=2, dtype=object)
        return bursts

    offsets = np.sort(offsets)
    gaps = np.diff(offsets)

    if not np.all(gaps):
        raise ValueError("Input cannot contain events with zero time between!")

    T = np.sum(gaps)
    n = np.size(gaps)

    g_hat = T / n

    k = int(math.ceil(float(1 + math.log(T, s) + math.log(1 / np.amin(gaps), s))))

    gamma_log_n = gamma * math.log(n)

    def tau(i, j):
        if i >= j:
            return 0
        else:
            return (j - i) * gamma_log_n

    alpha_function = np.vectorize(lambda x: s**x / g_hat)
    alpha = alpha_function(np.arange(k))

    log_alpha = np.log(alpha)

    C = np.repeat(float("inf"), k)
    C[0] = 0

    q = np.empty((k, 0))
    for t in range(n):
        C_prime = np.repeat(float("inf"), k)
        q_prime = np.empty((k, t + 1))
        q_prime.fill(np.nan)

        for j in range(k):
            cost_function = np.vectorize(lambda x: C[x] + tau(x, j))
            cost = cost_function(np.arange(0, k))

            el = np.argmin(cost)

            C_prime[j] = cost[el] - log_alpha[j] + alpha[j] * gaps[t]

            if t > 0:
                q_prime[j, :t] = q[el, :]

            q_prime[j, t] = j + 1

        C = C_prime
        q = q_prime

    j = np.argmin(C)
    q = q[j, :]

    prev_q = 0

    N = 0
    for t in range(n):
        if q[t] > prev_q:
            N = N + q[t] - prev_q
        prev_q = q[t]

    bursts = np.array(
        [np.repeat(np.nan, N), np.repeat(offsets[0], N), np.repeat(offsets[0], N)],
        ndmin=2,
        dtype=object,
    ).transpose()
    burst_counter = -1
    prev_q = 0
    stack = np.repeat(np.nan, N)
    stack_counter = -1
    for t in range(n):
        if q[t] > prev_q:
            num_levels_opened = q[t] - prev_q
            for i in range(int(num_levels_opened)):
                burst_counter += 1
                bursts[burst_counter, 0] = prev_q + i
                bursts[burst_counter, 1] = offsets[t]
                stack_counter += 1
                stack[stack_counter] = burst_counter
        elif q[t] < prev_q:
            num_levels_closed = prev_q - q[t]
            for i in range(int(num_levels_closed)):
                bursts[int(stack[stack_counter]), 2] = offsets[t]
                stack_counter -= 1
        prev_q = q[t]

    while stack_counter >= 0:
        bursts[int(stack[stack_counter]), 2] = offsets[n]
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
#     for n in sizes:
#         offsets = np.sort(np.cumsum(np.random.exponential(scale=1.0, size=n)))
#         start = time.perf_counter()
#         result = kleinberg_burst_detection(offsets=offsets, s=s, gamma=gamma)
#         elapsed = time.perf_counter() - start
#         print(f"n={n:>10,}  bursts={result.shape[0]:>6,}  time={elapsed:.4f}s")
