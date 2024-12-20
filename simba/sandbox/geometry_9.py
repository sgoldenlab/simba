import pandas as pd

from simba.video_processors.video_processing import video_bg_subtraction_mp, create_average_frm, read_frm_of_video
import matplotlib
import matplotlib.pyplot as plt
dpi = matplotlib.rcParams["figure.dpi"]
import cv2

df = pd.read_parquet('/Users/simon/Downloads/from-ds-team_test_embedding.parquet')

VIDEO_PATH = cv2.VideoCapture('/Users/simon/Downloads/webm_20240715111159/Ant Test.webm')


#VIDEO_PATH = '/Users/simon/Downloads/Ant Test.mp4'
input_frm = read_frm_of_video(video_path=VIDEO_PATH, frame_index=1)

height, width, depth = input_frm.shape
figsize = width / float(dpi), height / float(dpi)
plt.figure(figsize=figsize)
plt.axis("off")
plt.imshow(input_frm)
plt.show()

avg_frm = create_average_frm(video_path=VIDEO_PATH, verbose=False)
#height, width, depth = avg_frm.shape
figsize = width / float(dpi), height / float(dpi)



plt.figure(figsize=figsize)
plt.axis("off")
plt.imshow(avg_frm)
plt.show()





video_bg_subtraction_mp(video_path=VIDEO_PATH,
                        save_path='/Users/simon/Desktop/1_LH_clipped_cropped_bg_removed.mp4',
                        verbose=False)














#video_bg_substraction_mp(video_path=VIDEO_PATH, save_path=)

