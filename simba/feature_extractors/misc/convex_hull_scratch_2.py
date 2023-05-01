import numba
import numpy as np
from numba import njit, types, typeof, prange
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError
from numba.typed import List
from scipy.spatial import ConvexHull
import time
from functools import cmp_to_key


@overload(ConvexHull)
def impl_convex_hull(data):


    def func(data):
        def brute_hull(data):
            s = []
            mid = [0, 0]
            for i in range(data.shape[0]):
                for j in range(i+1, data.shape[0]):
                    x1, x2 = data[i][0], data[j][0]
                    y1, y2 = data[i][1], data[j][1]
                    a1, b1, c1 = y1 - y2, x2 - x1, x1 * y2 - y1 * x2
                    pos, neg = 0, 0
                    for k in range(data.shape[0]):
                        if a1 * data[k][0] + b1 * data[k][1] + c1 <= 0:
                            neg += 1
                        if a1 * data[k][0] + b1 * data[k][1] + c1 >= 0:
                            pos += 1
                    if pos == data.shape[0] or neg == data.shape[0]:
                        s.append(list(data[i]))
                        s.append(list(data[j]))

            s_unique = []
            for i in s:
                if i not in s_unique: s_unique.append(i)

            n = len(s_unique)
            for i in range(n):
                mid[0] += s_unique[i][0]
                mid[1] += s_unique[i][1]
                s_unique[i][0] *= n
                s_unique[i][1] *= n
            


            #s = np.unique(np.array(s), axis=0)




        def division(frm_data):
            if frm_data.shape[0] <= 5:
                brute_hull(frm_data)



        for frm_cnt in range(data.shape[0]):
            frm_data = data[frm_cnt, :, :]
            frm_data = frm_data[frm_data[:, 0].argsort()]
            _ = division(frm_data)

    return func

data = np.random.randint(1, 10, size=(10, 4, 2))
@njit
def test_find_left(data):
    return ConvexHull(data)

start = time.time()
test_find_left(data)
end = time.time()
print(end-start)