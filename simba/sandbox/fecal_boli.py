import cv2
import numpy as np

from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.image_mixin import ImageMixin
from simba.utils.read_write import read_frm_of_video

video_path = r"C:\troubleshooting\mitra\project_folder\videos\bg_removed\503_MA109_Gi_Saline_0513.mp4"
img = read_frm_of_video(video_path=video_path, frame_index=0, greyscale=True)

# Set up the parameters for blob detection
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 0  # Detect bright blobs on a dark background
params.filterByArea = True
params.minArea = 2  # Minimum blob area
params.maxArea = 5000  # Maximum blob area

# Initialize the blob detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs in the image
keypoints = detector.detect(img)
keypoints_array = np.array([[kp.pt[0], kp.pt[1], kp.size] for kp in keypoints])




def count_fecal_boli(video_path: str):
    img = read_frm_of_video(video_path=video_path, frame_index=0)
    contours = ImageMixin.find_contours(img=img, mode='all', method='simple')
    contour_lst = []
    for i in contours:
        contour_lst.append(i.reshape(1, -1, 2))

    geometries = GeometryMixin.contours_to_geometries(contours=contour_lst, force_rectangles=False)
    print(geometries)
    img = GeometryMixin.view_shapes(shapes=geometries, bg_img=img, thickness=5)
   #
   #
   # # print(contours)
   #
   #
    cv2.imshow('azsdasdas', img)
    cv2.waitKey(0)




count_fecal_boli(video_path=r"C:\troubleshooting\mitra\project_folder\videos\503_MA109_Gi_Saline_0513.mp4")