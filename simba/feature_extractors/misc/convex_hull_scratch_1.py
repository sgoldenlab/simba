# import numba
# import numpy as np
# from numba import njit, types, typeof, prange
# from numba.extending import overload, register_jitable
# from numba.core.errors import TypingError
# from numba.typed import List
# from scipy.spatial import ConvexHull
# import time
# from functools import cmp_to_key
#
#
# @overload(ConvexHull)
# def impl_convex_hull(data):
#     def func(data):
#
#         def find_left_idx(data):
#             idx = 0
#             for i in range(data.shape[0]):
#                 if data[i][0] < data[idx][0]:
#                     idx = i
#                 elif data[i][0] == data[idx][0]:
#                     if data[i][1] > data[idx][1]:
#                         idx = i
#             return idx
#
#         def orientation(p, q, r):
#             val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
#
#             if val == 0:
#                 return 0
#             elif val > 0:
#                 return 1
#             else:
#                 return 2
#
#         for i in range(data.shape[0]):
#             frm_data = data[i, :, :]
#             left_idx = find_left_idx(frm_data)
#             p, q, break_flag = left_idx, 0, False
#             hull_idx = []
#
#             while not break_flag:
#                 hull_idx.append(p)
#                 q = (p + 1) % frm_data.shape[0]
#
#                 for j in range(frm_data.shape[0]):
#
#
#                 break_flag = True
#
#
#
#
#             #     hull_idx.append(p)
#             #     q = (p + 1) % frm_data.shape[0]
#             #     print(hull_idx)
#                 # for j in range(frm_data.shape[0]):
#                 #     if (orientation(frm_data[p], frm_data[j], frm_data[q]) == 2):
#                 #         print(orientation)
#             #             q = j
#             #     p = q
#             #     print(p, left_idx)
#             #     if (p == left_idx):
#             #         break_flag = True
#             #
#             # hull_vals = np.full((len(hull_idx), 2), np.nan)
#             # print(hull_idx)
#             # for idx, val in enumerate(hull_idx):
#             #     hull_vals[idx] = frm_data[val]
#
#
#     return func
#
#
# data = np.random.randint(1, 10, size=(1, 5, 2))
#
# @njit
# def test_find_left(data):
#     return ConvexHull(data)
#
# start = time.time()
# test_find_left(data)
# end = time.time()
# print(end-start)
#
#
#
#
#
#
#
# #
# #
# # def convex_hull_calculator_mp(arr: np.array, px_per_mm: float) -> float:
# #     arr = np.unique(arr, axis=0).astype(int)
# #     if arr.shape[0] < 3:
# #         return 0
# #     for i in range(1, arr.shape[0]):
# #         if (arr[i] != arr[0]).all():
# #             try:
# #                 return ConvexHull(arr, qhull_options='En').area / px_per_mm
# #             except QhullError:
# #                 return 0
# #         else:
# #             pass
# #     return 0
#
# # @overload(ConvexHull)
# # def impl_convex_hull(data):
# #     def func(data):
# #
# #         def find_left(data):
# #             start = data[0]
# #             min_x = start[0]
# #
# #             for p in data[1:]:
# #                 if p[0] < min_x:
# #                     min_x = p[0]
# #                     start = p
# #             print(data, start)
# #             return start
# #
# #         def get_orientation(origin, p1, p2):
# #             #print(origin, p1, p2)
# #             difference = (((p2[0] - origin[0]) * (p1[1] - origin[1])) - ((p1[0] - origin[0]) * (p2[1] - origin[1])))
# #
# #             return difference
# #
# #         def check_similarity(p1, p2):
# #             for i in range(0, p1.shape[0]):
# #                 if p1[i] == p2[i]:
# #                     pass
# #                 else:
# #                     return 0
# #             return 1
# #
# #         for frm_cnt in range(data.shape[0]):
# #             hull_points = []
# #             far_point = None
# #             frm_data = data[frm_cnt, :, :]
# #             frm_data = frm_data[frm_data[:, 0].argsort()]
# #             point = find_left(frm_data)
# #             hull_points.append(start)
# #
# #             while far_point is not start:
# #                 p1 = None
# #                 for p in frm_data:
# #                     if np.array_equal(p, point):
# #                         continue
# #                     else:
# #                         p1 = p
# #                         break
# #                 far_point = p1
# #
# #                 for p2 in frm_data:
# #                     if np.array_equal(p2, point):
# #                         continue
# #                     if check_similarity(p1, p2):
# #                         continue
# #                     else:
# #
# #                         direction = get_orientation(point, far_point, p2)
# #                         #print(point, far_point, p2, direction)
# #                         #print(direction)
# #             #             if direction > 0:
# #             #                 far_point = p2
# #             #
# #             #     hull_points.append(far_point)
# #             #     point = far_point
# #             #
# #             # print(hull_points)
# #
# #                     # if np.array_equal(p2, p1):
# #                     #     continue
# #
# #                 #     else:
# #                 #         print('s')
# #
# #                 #     else:
# #                 #         direction = get_orientation(point, far_point, p2)
# #                 #         if direction > 0:
# #                 #             far_point = p2
# #                 # print(far_point)
# #
# #
# #
# #
# #
# #         #sorted_data = data[data[:, 0].argsort()]
# #         #print(data[:, 0].argsort())
# #
# #
# #         #return sorted_data
# #
# #
# #     return func
#
