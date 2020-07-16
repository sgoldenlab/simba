import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np
from deepposekit.io import VideoReader, DataGenerator, initialize_dataset
from deepposekit.annotate import KMeansSampler
import tqdm
import pandas as pd
import glob
from configparser import ConfigParser
import os
import cv2

def createAnnotationSet(dpkini):
    config = ConfigParser()
    configFile = str(dpkini)
    config.read(configFile)
    randomly_sampled_frames = []
    project_folder = config.get('general DPK settings', 'project_folder')
    videoFolder = os.path.join(project_folder, 'videos', 'input')
    bodyPartsListPath = os.path.join(project_folder, 'skeleton.csv')
    annotationSaveName = config.get('create annotation settings', 'annotation_output_name')
    annotationSaveName = annotationSaveName + '.h5'
    annotationSavePath = os.path.join(project_folder, 'annotation_sets', annotationSaveName)
    read_batch_size = config.getint('create annotation settings', 'read_batch_size')
    k_means_batch_size = config.getint('create annotation settings', 'k_means_batch_size')
    k_means_n_custers = config.getint('create annotation settings', 'k_means_n_custers')
    k_means_max_iterations = config.getint('create annotation settings', 'k_means_max_iterations')
    k_means_n_init = config.getint('create annotation settings', 'k_means_n_init')
    videos = glob.glob(videoFolder + '/*.mp4')


    #check if video are greyscale
    cap = cv2.VideoCapture(videos[0])
    cap.set(1, 0)
    ret, frame = cap.read()
    fileName = str(0) + str('.bmp')
    filePath = os.path.join(videoFolder, fileName)
    cv2.imwrite(filePath, frame)
    img = cv2.imread(filePath)
    imgDepth = img.shape[2]
    if imgDepth == 3:
        greyscaleStatus = False
    else:
        greyscaleStatus = True
    os.remove(filePath)

    print('Generating DeepPoseKit annotation set...')

    for i in range(len(videos)):
        reader = VideoReader(videos[i], batch_size=read_batch_size, gray=greyscaleStatus)
        print('Sampling frames: ' + 'video ' + str(os.path.basename(videos[i])))
        for idx in tqdm.tqdm(range(len(reader)-1)):
            batch = reader[idx]
            random_sample = batch[np.random.choice(batch.shape[0], 10, replace=False)]
            randomly_sampled_frames.append(random_sample)
        reader.close()
    randomly_sampled_frames = np.concatenate(randomly_sampled_frames)
    print('Performing K-means...')
    kmeans = KMeansSampler(n_clusters=k_means_n_custers, max_iter=k_means_max_iterations, n_init=k_means_n_init, batch_size=k_means_batch_size, verbose=True)
    kmeans.fit(randomly_sampled_frames)
    kmeans_sampled_frames, kmeans_cluster_labels = kmeans.sample_data(randomly_sampled_frames, n_samples_per_label=k_means_n_custers)
    initialize_dataset(images=kmeans_sampled_frames, datapath=annotationSavePath, skeleton=bodyPartsListPath, overwrite=True)
    print('DeepPoseKit annotations saved in: ' + str(annotationSavePath))
