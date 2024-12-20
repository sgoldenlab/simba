import time

import numba as nb
import numpy as np
from numba import cuda, guvectorize

a = np.random.rand(1024 * 1024, 32).astype('float32')
b = np.random.rand(1024 * 1024, 32).astype('float32')
dist = np.zeros(a.shape[0]).astype('float32')


@guvectorize(['void(float32[:], float32[:], float32[:])'], '(n),(n)->()',
             target='cuda')
def numba_dist_cuda(a, b, dist):
    len = a.shape[0]
    x = 0
    for i in range(len):
        x += a[i] + b[i]
    dist[0] = x


nb.cuda.detect()
print(nb.cuda.is_available())


d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_dist = cuda.to_device(dist)

t = time.time()
numba_dist_cuda(d_a, d_b, d_dist)
cuda.synchronize()
elapsed = time.time() - t

print(elapsed)