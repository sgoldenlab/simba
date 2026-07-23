import math

import numpy as np
from numba import cuda

from simba.utils.checks import check_valid_array
from simba.utils.enums import Formats

THREADS_PER_BLOCK = 256


@cuda.jit()
def _cdist_3d_kernel(data, results):
    """One thread per (frame i, flattened pair jk). results[i,j,k] = euclidean(data[i,j], data[i,k])."""
    i, jk = cuda.grid(2)
    n, m, d = data.shape[0], data.shape[1], data.shape[2]
    if i >= n or jk >= m * m:
        return
    j = jk // m
    k = jk % m
    s = 0.0
    for c in range(d):
        diff = data[i, j, c] - data[i, k, c]
        s += diff * diff
    results[i, j, k] = math.sqrt(s)


def cdist_3d_cuda(data: np.ndarray) -> np.ndarray:
    """
    Compute the per-frame pairwise Euclidean distance matrix of a set of coordinates against itself, on the GPU.

    For each frame the (m, m) matrix of distances between all m points is computed (a batched ``scipy.cdist``).

    .. note::
       Output is a dense (n_frames, m, m) float64 array, so memory scales with n * m^2 (the binding constraint;
       e.g. m=8 ~ 512 bytes/frame, so ~15,000,000 frames on a 12 GB card). Matches the CPU to floating-point
       rounding (~1e-13).

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/cdist_3d_cuda.csv
       :widths: 20, 20, 20, 20, 20
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.cdist_3d`.

    :param np.ndarray data: 3D array (n_frames, n_points, n_features) of coordinates (n_features is typically 2).
    :return: (n_frames, n_points, n_points) float64 array of per-frame pairwise Euclidean distances.
    :rtype: np.ndarray

    :example:

    >>> data = np.random.randint(0, 500, (10000, 8, 2)).astype(np.float32)
    >>> cdist_3d_cuda(data=data)
    """
    check_valid_array(data=data, source=f'{cdist_3d_cuda.__name__} data', accepted_ndims=(3,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    n, m = data.shape[0], data.shape[1]
    data_dev = cuda.to_device(np.ascontiguousarray(data).astype(np.float64))
    results = cuda.device_array((n, m, m), dtype=np.float64)
    tpb = (16, 16)
    bpg = (math.ceil(n / tpb[0]), math.ceil((m * m) / tpb[1]))
    _cdist_3d_kernel[bpg, tpb](data_dev, results)
    return results.copy_to_host()


@cuda.jit()
def _cosine_similarity_kernel(data, norms, results):
    """One thread per (i, j). results[i,j] = dot(row_i, row_j) / (norm_i * norm_j)."""
    i, j = cuda.grid(2)
    n, fdim = data.shape[0], data.shape[1]
    if i >= n or j >= n:
        return
    dot = 0.0
    for c in range(fdim):
        dot += data[i, c] * data[j, c]
    results[i, j] = dot / (norms[i] * norms[j])


def cosine_similarity_cuda(data: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise cosine similarity between all rows of a 2D array, on the GPU.

    The cosine similarity is the cosine of the angle between two vectors (magnitude ignored); values range from 1
    (same direction) through 0 (orthogonal) to -1 (opposite). Zero-vectors yield a similarity of 0.

    .. note::
       Output is a dense (n, n) float32 matrix, so memory scales with n^2 (n=50,000 ~ 10 GB, the in-VRAM ceiling
       on a 12 GB card). Matches the CPU version to ~1e-6 (float32 output).

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/cosine_similarity_cuda.csv
       :widths: 20, 20, 20, 20, 20
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numpy) version: :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.cosine_similarity`.

    :param np.ndarray data: 2D array (n_observations, n_features).
    :return: (n, n) float32 pairwise cosine-similarity matrix.
    :rtype: np.ndarray

    :example:

    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    >>> cosine_similarity_cuda(data=data)
    """
    check_valid_array(data=data, source=f'{cosine_similarity_cuda.__name__} data', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    x = np.ascontiguousarray(data).astype(np.float64)
    norms = np.linalg.norm(x, axis=1)
    norms[norms == 0] = 1.0
    n = x.shape[0]
    data_dev = cuda.to_device(x)
    norms_dev = cuda.to_device(np.ascontiguousarray(norms))
    results = cuda.device_array((n, n), dtype=np.float32)
    tpb = (16, 16)
    bpg = (math.ceil(n / tpb[0]), math.ceil(n / tpb[1]))
    _cosine_similarity_kernel[bpg, tpb](data_dev, norms_dev, results)
    return results.copy_to_host()
