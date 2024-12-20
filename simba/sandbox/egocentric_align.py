import cv2
import numpy as np
import pandas as pd

from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.read_write import read_df


def egocentric_alignment(df: pd.DataFrame,
                         anchor_1: int,
                         anchor_2: int):

    anchors_idx = [anchor_1, anchor_2]
    data = df.values.reshape(len(df), int((len(list(df.columns))/2)), 2).astype(np.int32)

    results = np.zeros_like(data, dtype=np.int64)

    for frm in range(data.shape[0]):
        x_corr, y_corr = data[frm][anchors_idx[0]][0], data[frm][anchors_idx[0]][1]
        print(x_corr)



        # anchor_data = data[frm][anchors_idx]
        # print(anchor_data)


    #     ref_bp = data[frm][anchors_idx[1]]
    #     centered_anchors = anchor_data - ref_bp
    #     avg_vector = centered_anchors[0] - centered_anchors[1]
    #     line_body3_body2 = centered_anchors[2] - centered_anchors[1]
    #    # avg_vector = (line_body1_body2 + line_body3_body2) / 2
    #     angle = np.arctan2(avg_vector[1], avg_vector[0])
    #     print(frm)
    #
    #     rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    #     translated_points = data[frm] - ref_bp  # Translate to the origin
    #     rotated_points = np.dot(translated_points, rotation_matrix.T)  # Apply rotation
    #     results[frm] = rotated_points + ref_bp  # Translate back
    #
    # return results






data_path = r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\501_MA142_Gi_CNO_0516.csv"
df = read_df(file_path=data_path, file_type='csv')
df = df[[x for x in df.columns if not '_p' in x]]
results = egocentric_alignment(df=df.head(1), anchor_1=2, anchor_2=6)
# points = GeometryMixin.multiframe_bodypart_to_point(data=results, core_cnt=10)
# for i in range(len(points)):
#     img = GeometryMixin.view_shapes(shapes=points[i])
#     cv2.imshow('asdasd', img)
#     cv2.waitKey(33)














