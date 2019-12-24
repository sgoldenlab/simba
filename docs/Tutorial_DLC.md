# Using SimBA to create and access [DeepLabCut](http://www.mousemotorlab.org/deeplabcut) tracking models
*Note*: This part of SimBA was written early on - when [DeepLabCut](http://www.mousemotorlab.org/deeplabcut) was accessed by the command line - and did not come with the very nice GUI it comes with today. At that early point we added this part of SimBA to make DeepLabCut accessable for all members of the Sam Golden lab. If users prefer to work in the DeepLabCut interface to create and access pose estimation models, the tracking data can instead be generated elsewhere and this step can be skipped. The tracking data can instead be imported directly in csv file format when creating the [behavioral classifiers](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md).     

## Pipeline breakdown
For detailed information on the DeepLabCut workflow, see the [DeepLabCut repository](https://github.com/AlexEMG/DeepLabCut).

![alt-text-1](/images/dlcworkflow.PNG "dlcworkflow.PNG")

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

## Part 1: Create DLC Model
Here you can create a new project for your DLC pose-estimation tracking analysis.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/createdlcmodel.png" width="400" height="200" />

### Step 1: Generate a new DLC model. 
This step generates a new DeepLabCut project.

1. Go to `Tracking` --> `DeepLabCut` --> `Create DLC Model`. The following window will pop up.

![alt-text-1](/images/createdlcmodel2.png "createdlcmodel2")

2. Enter the name of your project in the `Project name` entry box.

3. Enter your name in the experimenter in the `Experimenter name` entry box.

> **Note:** Spaces are not allowed in your project name and experimenter name.

4. Next, import your videos into your project. If you are only tracking one video, you can click on <img src="https://github.com/sgoldenlab/simba/blob/master/images/importsinglevideo.PNG" width="120" height="27" /> and the <img src="https://github.com/sgoldenlab/simba/blob/master/images/videofolder.PNG" width="274" height="27" /> in **green** should change to <img src="https://github.com/sgoldenlab/simba/blob/master/images/videopath.PNG" width="274" height="27" /> in **blue**. Then click on `Browse File` to select a video file.

5. If you are importing multiple videos into your project, click on <img src="https://github.com/sgoldenlab/simba/blob/master/images/importmultivideo.PNG" width="120" height="27" /> and the **green** <img src="https://github.com/sgoldenlab/simba/blob/master/images/videofolder.PNG" width="274" height="27" /> will appear. Click on `Browse Folder` to choose a folder containing the videos you want to import into your project. 

>**Note:** The default settings is to **import multiple videos**.

6. Next, select the main directory that your project will be located in. Next to `Project directory`, click on `Browse Folder` and choose a directory.

7. If you wish to use the Golden Lab's settings, check the `Apply Golden Aggression Config` box. This setting is used to track two mice, and eight body parts on each of the two mice. For more information about these bosy parts and how they are labeled, please see. If you Wish to generate your on DeepLabCut tracking config, please leave this box un-ticked.

8. You can either copy all the videos to your DLC project folder, or create shortcuts to the videos. By checking the `Copy Videos` checkbox, the videos will be copied to your project folder. If you leave this box un-ticked, shortcuts to your videos will be created. 

9. Click on `Create Project` to create your DLC project. The project will be located in the chosen **working directory**.

## Part 2: Load DLC Project
These menues are used to load created DeepLabCut projects.

![alt-text-1](/images/loaddlcmodel.png "loaddlcmodel")

### Step 1: Check the settings in the DeepLabCut config.yaml file

1. Go to the project folder.

2. Double click on the config.yaml file and open it in Notepad.

3. Change the settings if necessery. 

### Step 2: Load the DeepLabCut config.yaml file

1. In the main SimBA window, click on `Tracking` --> `DeepLabCut` --> `Load DLC Model`. The following window will pop up.

![alt-text-1](/images/loaddlcmodel2.PNG "loaddlcmodel2")

2. Under the **Load Model** tab, click on `Browse File` and load the *config.yaml* file from your project folder.

### Step 3: Add additonal videos to the project (optional)

![alt-text-1](/images/additionalvideo.PNG "additionalvideo")

#### Single Video

1. Under the **Add videos into project** tab and **Single Video** heading, click on `Browse File` and select the video you wish to add to the project.

2. Click on `Add single video`.

#### Multiple Videos

1. Under the **Add videos into project** tab and **Multiple Videos** heading, click on `Browse Folder` and select the folder containing the videos you wish to add to the project.

2. Click on `Add multiple videos`.

### Step 4: Extract frames for labeling body parts using DLC. For more details, see the [DeepLabcut repository](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/functionDetails.md#c-data-selection-extract-frames))

![alt-text-1](/images/extractframesdlc.PNG "extractframesdlc")

1. Under the **Extract/label frames tab** tab and **Extract Frames DLC** heading, enter the number of frames you wish to extract from the videos in the `numframes2pick` entry box.

2. Select the **Mode** of extracting the frames. 

- `Automatic` selects the frames to extract automatically. 

- `Manual` allows you to select the frames to extract.

3. Select the **Algorithm** to pick the extracted frames. 

- `Uniform` selects the frames uniformly in a series format. 

- `KMeans` uses k-means clustering algorithm to select the frames randomly based on cluster. 

- `Cluster Resize Width` is the setting for KMeans. The default is set to 30.

- `Cluster Step` is the setting for KMeans. The default is set to 1.

4. To use color information for clustering, check the `True` box under **Cluster Color**.

5. If you wish to use **OpenCV** to extract frames, check the `True` box. If you wish to use **ffmpeg**, keep it unchecked.

6. Click `Extract Frames` to extract frames.

### Step 5: Label Frames

![alt-text-1](/images/labelframes.PNG "labelframes")

1. Under **Label Frames** heading in the **Extract/label frames tab** tab, click on `Label Frames` and the *DeepLabCut- Labelling ToolBox* will pop up.

2. Click on `Load Frames` located in the bottom left-hand corner.

3. Choose the folder with the video name where your extracted frames are saved and click `Select Folder`. *They should be located at workingdirectory/yourproject/labeled-data/*

4. Now you can start labelling frames. For more details, see the [DeepLabCut repository](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/functionDetails.md#d-label-frames).

5. Once all the frames are labelled, move on to **Step 6**.

### Step 6: Check Labels 

1. Under **Check Labels**, click on `Check Labelled Frames` to check the labelled frames.

### Step 7: Generate training set

1. Under **Generate Training Set**, click on the `Generate training set` button to generate the training data set.

### Step 8: Train Network
Train the model using a generated training set.

1. In the `iteration` entry box, fill in an integer representing the model iteration number. Once done, click on **Update iteration**.  If left blank, then the iteration number will be the most recently used, or 0. 

2. In the `init_weight` box, specify the path to the initial weights. If this is left blank it will default to resnet-50. If you want to use other weights, click on `Browse File`. Pre-trained weights for mouse and rat resident-intruder protocols using 16 body-parts, as well as other pre-trained weights, can be downloaded [here](https://osf.io/5t4y9/). 

### Step 9: Evaluate Network

1. Click on `Evaluate Network` to evaluate the trained model. For more details, see the [DeepLabCut repository](https://alexemg.github.io/DeepLabCut/docs/functionDetails.html#h-evaluate-the-trained-network).

### Step 10: Video Analysis

#### Single Video Analysis

1. Under the **Video Analysis** tab and the **Single Video Analysis** header, click on `Browse File` and select one a video.

2. Click on `Single Video Analysis`.

#### Multiple Videos Analysis

1. Under the **Video Analysis** tab and the **Multiple Videos Analysis** header, click on `Browse Folder` and select a folder containing the videos.

2. Enter the video format in the `Video type` entry box (eg: mp4, avi). *Do **not** include any " . " dot in the entrybox (e.g,:  do not enter .mp4, .flv, .avi, etc.)*

3. Click on `Multi Videos Analysis`.

### Step 11: Plot Video Graph

1. Click on `Browse File` to select the video.

2. Click on `Plot Results`. For more details, see the [DeepLabCut documentation](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/UseOverviewGuide.md)

### Step 12: Create Video
This step will generate a video with labelled tracking points.

1. Click on `Browse File` to select the video.

2. You can choose to save the frames with tracking points on the video by checking the `Save Frames` box. This process takes longer.

3. Click on `Create Video`.

## Part 3: Improve the Model
There are two ways that you can improve the current model.

1. Extract more frames to train the model.

2. Correct the tracking points of frames with predictions made by the model.

## Extract more frames
This step automates the **Extract Frames** process, to only extract frames from select videos. This function will copy the settings from the DLC *config.yaml* and remove all the videos in the *config.yaml* file. It will then add the new videos that the user specifies into a new, temporary yaml file, and extract frames from only these videos. 

### Step 1: Generate temp.yaml from the original config.yaml

#### Single Video

1. Under **Load Model**, click `Browse File` and select the *config.yaml* file. 

2. Under `[Generate temp yaml]` tab --> `Single video`, and click on `Browse File` and select the video.

3. Click `Add single video`

4. *temp.yaml* will be generated in the same folder as the project folder.

#### Multiple Videos

1. Under **Load Model**, click `Browse File` and select the *config.yaml* file. 

2. Under `[Generate temp yaml]` tab --> `Multiple videos`, click on `Browse Folder` and select the folder containing only the videos.

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




