import cv2
from tqdm import tqdm
import os
from configparser import ConfigParser

configFile = r"Y:\DeepLabCut\DLC_extract\New_062719\project_folder\project_config.ini"
config = ConfigParser()
config.read(configFile)
frames_dir_out = config.get('Frame settings', 'frames_dir_out')
allDirs = [f.path for f in os.scandir(frames_dir_out) if f.is_dir() ]
fps = config.getint('Frame settings', 'fps')
fileformat = config.get('Create movie settings', 'file_format')
bitrate = config.getint('Create movie settings', 'bitrate')

relDirs = [s for s in allDirs if "_merged" in s]

for dir in relDirs:
    currentFrameFolder = dir
    video_name = os.path.basename(currentFrameFolder)
    video_name = str(video_name) + str(fileformat)
    images = [img for img in os.listdir(currentFrameFolder) if img.endswith(".png")]
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    frame = cv2.imread(os.path.join(currentFrameFolder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 25, (width,height))
    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(currentFrameFolder, image)))
cv2.destroyAllWindows()
video.release()