import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import cv2
import numpy as np
from deepposekit.models import load_model
from deepposekit.io import DataGenerator, VideoReader, VideoWriter
import os
from configparser import ConfigParser
import glob
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

dpkini = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Hazel\project_folder\logs\measures\dpk\Hazel_1\dpk_config.ini"
videofolder = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Hazel\project_folder\logs\measures\dpk\Hazel_1\videos\input"

configFile = str(dpkini)
config = ConfigParser()
config.read(configFile)
project_folder = config.get('general DPK settings', 'project_folder')
modelPath = config.get('predict settings', 'modelPath')
videoFolderPath = videofolder
print(videoFolderPath)
batchSize = config.getint('predict settings', 'batch_size')
outputfolder = os.path.join(project_folder, 'predictions')
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

bodyPartColumnNames = []

skeletonPath = os.path.join(project_folder, 'skeleton.csv')
skeletonDf = pd.read_csv(skeletonPath)
skeletonList = list(skeletonDf['name'])

for i in skeletonList:
    x_col, y_col, p_col = (str(i) + '_x', str(i) + '_y', str(i) + '_p')
    bodyPartColumnNames.append(x_col)
    bodyPartColumnNames.append(y_col)
    bodyPartColumnNames.append(p_col)

filesFound = glob.glob(videoFolderPath + '/*.mp4')


#Check if videos are greyscale
cap = cv2.VideoCapture(filesFound[0])
cap.set(1, 0)
ret, frame = cap.read()
fileName = str(0) + str('.bmp')
filePath = os.path.join(videoFolderPath, fileName)
cv2.imwrite(filePath, frame)
img = cv2.imread(filePath)
imgDepth = img.shape[2]
if imgDepth == 3:
    greyscaleStatus = False
else:
    greyscaleStatus = True
os.remove(filePath)

# This loads the trained model into memory for making predictions
model = load_model(modelPath)
for video in filesFound:
    print('Analyzing file: ' + str(os.path.basename(video)))
    reader = VideoReader(video, batch_size=batchSize, gray=greyscaleStatus)
    predictions = model.predict(reader, verbose=1)
    reader.close()
    outputFilename = os.path.join(outputfolder, os.path.basename(video).replace('.mp4', '.csv'))
    x, y, confidence = np.split(predictions, 3, -1)

confidence_diff = np.abs(np.diff(confidence.mean(-1).mean(-1)))
confidence_outlier_peaks = find_peaks(confidence_diff, height=0.1)[0]
time_diff = np.diff(predictions[..., :2], axis=0)
time_diff = np.abs(time_diff.reshape(time_diff.shape[0], -1))
time_diff = time_diff.mean(-1)

time_diff_outlier_peaks = find_peaks(time_diff, height=10)[0]
outlier_index = np.concatenate((confidence_outlier_peaks, time_diff_outlier_peaks))
outlier_index = np.unique(outlier_index) # make sure there are no repeats

reader = VideoReader(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Hazel\project_folder\logs\measures\dpk\Hazel_1\videos\input\Hazel_2_fps_20_downsampled.mp4", batch_size=1, gray=False)
outlier_images = []
outlier_keypoints = []
for idx in outlier_index:
    outlier_images.append(reader[idx])
    outlier_keypoints.append(predictions[idx])

outlier_images = np.concatenate(outlier_images)
outlier_keypoints = np.stack(outlier_keypoints)

reader.close()

data_generator = DataGenerator(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Hazel\project_folder\logs\measures\dpk\Hazel_1\annotation_sets\Annotation_Hazel.h5")
from deepposekit.io.utils import merge_new_images


merge_new_images(
    datapath=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Hazel\project_folder\logs\measures\dpk\Hazel_1\annotation_sets\Annotation_Hazel.h5",
    merged_datapath=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Hazel\project_folder\logs\measures\dpk\Hazel_1\annotation_sets\Annotation_Hazel_merged_2.h5",
    images=outlier_images,
    keypoints=outlier_keypoints,
    # overwrite=True # This overwrites the merged dataset if it already exists
)

