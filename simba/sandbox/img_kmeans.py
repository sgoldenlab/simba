import cv2
import numpy as np
from sklearn.cluster import KMeans

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.image_mixin import ImageMixin
from simba.utils.checks import check_valid_array
from simba.utils.data import savgol_smoother
from simba.utils.read_write import (get_video_meta_data, read_df,
                                    read_frm_of_video)


def img_kmeans(data: np.ndarray,
               greyscale: bool = True,
               k: int = 3,
               n_init: int = 5):

    check_valid_array(data=data, accepted_ndims=(4, 3,), accepted_dtypes=(np.uint8,))
    print('s')




    # N, H, W, C = data.shape
    # kmeans = KMeans(n_clusters=k,  n_init=n_init).fit(imgs)
    #
    #
    #
    pass


CONFIG_PATH = r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini"
VIDEO_PATH = r"C:\troubleshooting\RAT_NOR\project_folder\videos\2022-06-20_NOB_DOT_4_clipped.mp4"

video_meta = get_video_meta_data(video_path=VIDEO_PATH)

config = ConfigReader(config_path=CONFIG_PATH, read_video_info=False)
config.read_roi_data()
shapes, _ = GeometryMixin.simba_roi_to_geometries(rectangles_df=config.rectangles_df, circles_df=config.circles_df, polygons_df=config.polygon_df, color=True)
shape = shapes['2022-06-20_NOB_DOT_4']['Rectangle']
shapes = []
for i in range(video_meta['frame_count']):
    shapes.append(shape)

imgs = ImageMixin().slice_shapes_in_imgs(imgs=VIDEO_PATH, shapes=shapes)
imgs = ImageMixin.pad_img_stack(image_dict=imgs, pad_value=0)
imgs = np.stack(imgs.values(), axis=0)

img_kmeans(data=imgs)




# cv2.imshow('asdasdasd', imgs[60])
# cv2.waitKey(0)








