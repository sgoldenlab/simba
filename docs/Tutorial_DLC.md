# Create tracking model using [DeepLabCut](http://www.mousemotorlab.org/deeplabcut)

## Pipeline breakdown

<img src="https://github.com/sgoldenlab/simba/blob/master/images/dlcworkflow.PNG" width="591" height="294" />

### [Part 1: Create DLC Project](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#part-1-create-dlc-project-1)

### [Part 2: Load DLC Project](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#part-2-load-dlc-project-1)
- [Extract Frames](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#step-4-extract-frames-dlc-more-details-here)
- [Label Frames](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#step-5-label-frames)
- [Generate Training Set](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#step-7-generate-training-set)
- [Download Weights]() and [Train Model](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#step-8-train-network)
- [Evaluate Model](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#step-9-evaluate-network)
- [Analyze Videos](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#step-10-video-analysis)

### [Part 3: Improve the Model](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#part-3-improve-the-model-1)
- [Extract more frames](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#extract-more-frames) 
- [Outlier correction](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#outlier-correction)

## Part 1: Create DLC Project
This section creates a new project for your DLC tracking analysis.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/createdlcmodel.png" width="400" height="200" />

### Step 1: Generate a new DLC project
This step generates a new DeepLabCut project.

1. Go to `Tracking` --> `DeepLabCut` --> `Create DLC Model`. The following window will pop up.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/createdlcmodel2.png" width="400" height="336" />

2. Enter the name of your project in `Project Name`.

3. Enter your name in the `Experimenter Name`.

> **Note:** Spaces are not allowed in your project name and experimenter name.

4. If you are only tracking one video, you can click on <img src="https://github.com/sgoldenlab/simba/blob/master/images/importsinglevideo.PNG" width="120" height="27" /> and the <img src="https://github.com/sgoldenlab/simba/blob/master/images/videofolder.PNG" width="274" height="27" /> in **green** should change to <img src="https://github.com/sgoldenlab/simba/blob/master/images/videopath.PNG" width="274" height="27" /> in **blue**. Then click on `Browse File` to select a video.

5. If you are tracking multiple videos, click on <img src="https://github.com/sgoldenlab/simba/blob/master/images/importmultivideo.PNG" width="120" height="27" /> and the **green** <img src="https://github.com/sgoldenlab/simba/blob/master/images/videofolder.PNG" width="274" height="27" /> will appear. Click on `Browse Folder` to choose a folder with the videos. 

>**Note:** The default settings is always to **import multiple videos**.

6. Next, select the main directory that your project will be in. In `Working Directory` click `Browse Folder` and choose a directory.

7. If you wish to use the Golden Lab's settings, check the `Apply Golden Aggression Config` box.

8. You have a choice of copying all the videos to your DLC project folder by checking the `Copy Videos` checkbox.

9. Click `Create Project` to create your DLC project. The project will be located in the **working directory** that was chosen.


## Part 2: Load DLC Project
This section allows you to load your DeepLabCut project.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/loaddlcmodel.png" width="402" height="206" />

### Step 1: Check the settings in the config.yaml file

1. Go to the project folder.

2. Double click on the config.yaml file and open it in Notepad.

3. Change the settings.

### Step 2: Load the config.yaml file

1. Go to `Tracking` --> `DeepLabCut` --> `Load DLC Model`. The following window will pop up.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/loaddlcmodel2.PNG" width="569" height="353" />

2. Under **Load Model**, click on `Browse File` and load the *config.yaml* file from the project folder.

### Step 3: Add additonal videos into the project (optional)

<img src="https://github.com/sgoldenlab/simba/blob/master/images/additionalvideo.PNG" width="350" height="200" />

#### Single Video

1. Under **Single Video**, click on `Browse File` and select the video.

2. Click `Add single video`.

#### Multiple Videos

1. Under **Multiple Videos**, click on `Browse Folder` and select the folder with only the videos.

2. Click `Add multiple videos`

### Step 4: Extract Frames using DLC (more details [here](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/functionDetails.md#c-data-selection-extract-frames))

<img src="https://github.com/sgoldenlab/simba/blob/master/images/extractframesdlc.PNG" width="300" height="300" />

1. Enter the number of frames you wish to extract in the `numframes2pick` entry box.

2. Select the **Mode** of selecting the frames. 

- `Automatic` selects the frames for you automatically. 

- `Manual` allows you to select the frames.

3. Select the **Algorithm** to pick the frames. 

- `Uniform` selects the frames uniformly in a series format. 

- `KMeans` uses k-means clustering algorithm to select the frames randomly based on cluster. 

- `Cluster Resize Width` is the setting for KMeans. The default is set to 30.

- `Cluster Step` is the setting for KMeans. The default is set to 1.

4. To use color information for clustering, check the `True` box under **Cluster Color**

5. If you wish to use **OpenCV** to extract frames, check the `True` box. If you wish to use **ffmpeg**, keep it unchecked.

6. Click `Extract Frames` to extract frames.

### Step 5: Label Frames

<img src="https://github.com/sgoldenlab/simba/blob/master/images/labelframes.PNG" width="100" height="50" />

1. Under **Label Frames**, click on `Label Frames` and *DeepLabCut- Labelling ToolBox* will pop up.

2. Click on `Load Frames` located in the bottom left-hand corner.

3. Choose the folder with the video name where your extracted frames are saved and click `Select Folder`. *They should be located at workingdirectory/yourproject/labeled-data/*

4. Now you can start labelling frames. (more details [here](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/functionDetails.md#d-label-frames)) 

5. Once all the frames are labelled, move on to **Step 6**.

### Step 6: Check Labels 

1. Under **Check Labels**, click on `Check Labelled Frames` to check the labelled frames.

### Step 7: Generate Training Set

1. Under **Generate Training Set**, click on the ` Generate training set` button to generate the training data set.

### Step 8: Train Network

`iteration` is the number of iterations.

`init_weight` is the initial weight to train the model.

#### Training a new model

1. If you are training a new model and want to start from scratch, click the `Train Network` button and do not change the settings.

2. The `iteration` will be set to 0 by default and the `init_weight` will be set to resnet-50 by default

#### Training an existing model

1. If you want to retrain a model, you can select the iteration in the `iteration` entry box and click `Update iteration`.

2. Click `Browse File` to find the model that you wish to retrain and click `Update init_weight` 

3. Click `Train Network` to start training your model.

### Step 9: Evaluate Network

1. Click on `Evaluate Network` to evaluate the trained model.

### Step 10: Video Analysis

#### Single Video Analysis

1. Under **Single Video Analysis**, click on `Browse File` and select one of the videos.

2. Click on `Single Video Analysis`.

#### Multiple Videos Analysis

1. Under **Multiple Videos Analysis**, click on `Browse Folder` and select the video folder.

2. Enter the video format in the `Video type` entry box (eg: mp4, avi). *Do **not** include any " . " dot in the entrybox (eg: .mp4, .flv, .avi, etc.)*

3. Click on `Multi Videos Analysis`.

### Step 11: Plot Video Graph

1. Click on `Browse File` to select the video.

2. Click on `Plot Results` .

### Step 12: Create Video
This step will generate the video with labelled tracking points.

1. Click on `Browse File` to select the video.

2. You can choose to save the frames with tracking points on the video by checking the `Save Frames` box. This process takes longer.

3. Click on `Create Video`.

## Part 3: Improve the Model
There are two ways that you can improve the current model.

1. Extract more frames to train the model.

2. Correct the tracking points of the existing frames.

## Extract more frames
This step creates a temp.yaml file to automate the **Extract Frames** process. It will copy the settings from the *config.yaml* and remove all the videos in the *config.yaml* file. It will then add the new videos that the user specifies into the yaml file and will generate a *temp.yaml*

### Step 1: Generate temp.yaml from the original config.yaml

#### Single Video

1. Under **Load Model**, click `Browse File` and select the *config.yaml* file. 

2. Under **Single Video**, click on `Browse File` and select the video.

3. Click `Add single video`

4. *temp.yaml* will be generated in the same folder as the project folder.

#### Multiple Videos

1. Under **Load Model**, click `Browse File` and select the *config.yaml* file. 

2. Under **Multiple Videos**, click on `Browse Folder` and select the folder containing only the videos.

3. Click `Add multiple videos`.

4. *temp.yaml* will be generated in the same folder as the project folder.

### Step 2: Load temp.yaml 

1. Under **Load Model**, click `Browse File` and select the *temp.yaml* that you have just created.

2. Now you can extract frames of the videos that you have just added.

## Outlier Correction

### Step 1: Load config.yaml

1. Go to `Tracking` --> `DeepLabCut` --> `Load DLC Model`.

2. Under **Load Model**, click on `Browse File` and load the *config.yaml* file from the project folder.

### Step 2: Extract Outliers

1. Under **Extract Outliers**, click `Browse File` to select the videos to be corrected.

2. Click `Extract Outliers`.

### Step 3: Label Outliers

1. Under **Label Outliers**, click on `Refine Outliers`. The *DeepLabCut - Refinement ToolBox* will pop up.

2. In the bottom left-hand corner, click on `Load labels` and select the machinelabels file to correct the tracking points.

### Step 4: Merge Labelled Outliers

1. Under **Merge Labeled Outliers**, click on `Merge Labelled Outliers`.




