import os
from typing import Union

import cv2
import numpy as np
import pandas as pd

from simba.utils.read_write import read_frm_of_video


def stabalize_video(data_path: Union[str, os.PathLike], video_path: Union[str, os.PathLike]):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    df = pd.read_csv(data_path, index_col=0).head(500)
    tail_base_points = df[['Tail_base_x', 'Tail_base_y']].values
    nose_points = df[['Nose_x', 'Nose_y']].values

    for img_idx in range(len(df)):
        print(img_idx)
        img = read_frm_of_video(video_path=cap, frame_index=img_idx)
        point1_current = tail_base_points[img_idx]
        point2_current = nose_points[img_idx]

        dist = np.linalg.norm(point1_current-point2_current)

        point1_fixed = (300, 300)  # Fixed location for Point 1
        point2_fixed = (int(point1_fixed[1]-dist), 300)  # Fixed location for Point 2

        translation_x1 = point1_fixed[0] - point1_current[0]
        translation_y1 = point1_fixed[1] - point1_current[1]
        #
        translation_x2 = point2_fixed[0] - point2_current[0]
        translation_y2 = point2_fixed[1] - point2_current[1]

        # Average translation (for simplicity, can also be calculated separately)
        avg_translation_x = (translation_x1 + translation_x2) / 2
        avg_translation_y = (translation_y1 + translation_y2) / 2

        # Create a translation matrix
        translation_matrix = np.array([[1, 0, avg_translation_x], [0, 1, avg_translation_y]])
        #
        #     # Apply the translation transformation to the frame
        stabilized_frame = cv2.warpAffine(img, translation_matrix, (frame_width, frame_height))
        cv2.imshow('asdasd', stabilized_frame)
        cv2.waitKey(33)





stabalize_video(data_path=r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\FL_gq_Saline_0626.csv", video_path=r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\test\bg_temp\geometry_bg.mp4")




#
#
# # Input video path and output video path
# input_video_path = r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\test\bg_temp\geometry_bg.mp4"
# output_video_path = r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\test\bg_temp\geometry_bg_test.mp4"
# data_path = r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\FL_gq_Saline_0626.csv"
#
# df = pd.read_csv(data_path, index_col=0)
#
#
# # Load the video
# cap = cv2.VideoCapture(input_video_path)
#
# # Get video properties
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#
# # Create a VideoWriter object to save the stabilized video
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
#
#
# # Define the two pixels to stabilize (current locations)
# point1_current = (182, 250)  # Current location of Point 1 (x, y)
# point2_current = (400, 300)   # Current location of Point 2 (x, y)
#
# # Define the fixed locations for these points (where you want them to be)
# point1_fixed = (300, 300)  # Fixed location for Point 1
# point2_fixed = (500, 300)  # Fixed location for Point 2
#
# cnt = 0
# # Loop through the video frames
# while cap.isOpened():
#     ret, frame = cap.read()
#     print(cnt)
#     if not ret:
#         break
#
#     # Calculate the translation needed for each point
#     translation_x1 = point1_fixed[0] - point1_current[0]
#     translation_y1 = point1_fixed[1] - point1_current[1]
#
#     translation_x2 = point2_fixed[0] - point2_current[0]
#     translation_y2 = point2_fixed[1] - point2_current[1]
#
#     # Average translation (for simplicity, can also be calculated separately)
#     avg_translation_x = (translation_x1 + translation_x2) / 2
#     avg_translation_y = (translation_y1 + translation_y2) / 2
#
#     # Create a translation matrix
#     translation_matrix = np.array([[1, 0, avg_translation_x],
#                                     [0, 1, avg_translation_y]])
#
#     # Apply the translation transformation to the frame
#     stabilized_frame = cv2.warpAffine(frame, translation_matrix, (frame_width, frame_height))
#
#     # Write the stabilized frame to the output video
#     out.write(stabilized_frame)
#     cnt += 1
#     if cnt == 5000:
#         break
#
#
# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()
