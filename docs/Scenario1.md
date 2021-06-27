# SimBA Tutorial:

To faciliate the initial use of SimBA, we provide several use scenarios. We have created these scenarios around a hypothetical experiment that take a user from initial use (completely new start) all the way through analyzing a complete experiment and then adding additional experimental datasets to an initial project.

All scenarios assume that the videos have been [pre-processed](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md) and that DeepLabCut CSV pose-estimation tracking files have been created.

# **Hypothetical data set**:
Three days of resident-intruder testing between aggressive CD-1 mice and subordinante C57 intruders. Each day of testing has 10 pairs of mice, for a total of 30 videos recorded across 3 days. Recordings are 3 minutes in duration, in color, at 30fps.

Also, so that we do not overfit the predictive classifiers to the experimental data, a different set of pilot videos have been recorded of resident-inturder pairings using identical video acquisition parameters. A total of 20 pilot videos were recorded.

# **Scenario 1**: From scratch...
In this series of Scenarios, you have the above pilot videos and DLC tracking data, the experimental videos and DLC tracking data, and SimBA. All videos have been pre-processed and DLC tracked. You now would like to generate a new predictive classifier for "Behavior that Will Get a Nature Paper (Behavior BtWGaNP)". In this first Scenario, we will use the pilot data to generate the classifier for behavior BtWGaNP. In later scenario tutorials, we will use this predictive classifier to classify BtWGaNP behaviours for Day 1 of the experiment ([Scenario 2](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md), add behavioural data with the goal of improving the behavioral classifier ([Scenario 3](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario3.md), and use the predictive classifier to classify behaviour BtWGaNP for Day 2 of the experiment ([Scenario 4](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario4.md)). 

### Pipeline breakdown:
For processing datasets, the pipeline is split into a few sections. These sections are listed below along with their corresponding functions:

![alt-text-1](/images/Vis_build_2.JPG "simbaworkflow")


### Part 1: [Create a new project](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#part-1-create-a-new-project-1)
- [Generate project config](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config) (create new classifiers)
- [Import videos into project folder](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-2-import-videos-into-project-folder)
- [Import DLC Tracking Data](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-3-import-dlc-tracking-data) 
- [Extract Frames into project folder](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-4-extract-frames-into-project-folder)

### Part 2: [Load project](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-load-project-config)
- [Load the project.ini](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-load-project-config)
- [Set video parameters](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-3-set-video-parameters)
- [Outlier correction](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-4-outlier-correction)
- [Extract Features](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features)
- [Label Behavior](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-6-label-behavior)
- [Train Machine Model](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-7-train-machine-model)
- [Run Machine Model](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model)
- [Analyze Machine Results](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-9-analyze-machine-results)
- [Visualization](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations)
- [Plot Graphs](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#gantt-plot)
- [Merge Frames](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-12-merge-frames)
- [Create Video](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-13-create-videos)

## Part 1: Create a new project
This section describes how to create a new project for behavioral classifier predictions.

![](/images/createproject2.PNG "createproject2")

### Step 1: Generate Project Config

In this step you create your main project folder, which will then auto-populate with all the required sub-directories.

1. In the main SimBA window, click on `File` and and `Create a new project`. The following windows will pop up.

![](/images/Create_project_1.PNG "createproject")

2. Navigate to the `[ Generate project config ]` tab. Under **General Settings**, specify a `Project Path` which is the directory that will contain your main project folder.

3. `Project Name` is the name of your project. 
*Keep in mind that the project name cannot contain spaces. We suggest to instead use underscore "_"* 

4. In the `SML Settings` sub-menu, put in the number of predictive classifiers that you wish to create. For an example, in Scenario 1 we would like to create a single classifier. We will enter the number 1. Note that in the real world you would probably want to create multiple classifiers concurrently as this would decrease the number of times a video would need to be manually annotated. For simplicity, we will here create a single classifier.

5. Click <img src="https://github.com/sgoldenlab/simba/blob/master/images/addclassifier.PNG" width="153" height="27" /> a single time, and it creates a row as shown in the following image. In each entry box, fill in the name of the behavior (BtWGaNP) that you want to classify. If you click too many times, as long as you leave the extra boxes empty, all is well.

<p align="center">
  <img width="385" height="106" src="https://github.com/sgoldenlab/simba/blob/master/images/classifier1.PNG">
</p>

6. The sub-menu `Animal Settings - # config` is used to specify the number of animals and body parts that that the pose estimation tracking data contains. The default for **SimBA** is 2 animals and 16 body parts (`2 animals, 16bp`). There are a few other - **yet not validaded** - options, accessible in the dropdown menu. This selection is the annotation configuration you should have previously used when labelling images in DeepLabCut or DeepPoseKit - see the tutorial for **[Pose estimation body-part labelling](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#pose-estimation-body-part-labelling)** for more information.

The second drop-down in the `Animal Settings` menu is the `Type of Tracking` menu. In the current scenario we will leave this at the default `Classical tracking`. `Classical tracking` is used when your videos contain one animal, or two animals that are clearly (by eye) discriminable, such as a white and a black coated animal. A second option in the `Type of Tracking` dropdown menu is `Multi tracking`. Select the `Multi tracking` option from the `Type of Tracking` dropdown menu *only when* the animals in your protocol are not discriminabl by eye and you have used newer pose-estimation tools to perform your animal tracking. These newer tools include multi-animal DeepLabCut [DLC version >=2.2b5](https://github.com/DeepLabCut/DeepLabCut/releases/tag/v2.2b5) and [SLEAP](https://sleap.ai/). In this tutorial we will use `Classical tracking`. If you have multi-animal tracking data from [DeepLabCut >=2.2b5](https://github.com/DeepLabCut/DeepLabCut/releases/tag/v2.2b5) or [SLEAP](https://sleap.ai/), then head over to the seperate tutorial on *Import multi-animal data* to learn how to import this data into SimBA. Come back to this tutorial after you have completed the *Import multi-animal data* tutorial and your multi-animal data has been imported into SimBA. 

*>Note:* If you want to use a different body-part configuration that is not shown in the drop-down menu - go tho the tutorial for creating [user-defined pose-configurations](https://github.com/sgoldenlab/simba/blob/simba_JJ_branch/docs/Pose_config.md) in SimBA. 

7. Click on `Generate Project Config` to generate your project. The project folder will be located in the specified `Project Path`. The Project will remain open, and the main console will report that the project has been created.

### Step 2: Import videos into project folder

In general, you can choose to import either one or multiple videos. The imported videos are used for visualizing predictions, standardizing distances across videos by calculating metric distances from pixel distances, and annotating behaviours for supervised machine learning classifiers.  

In this current Scenario 1, we now want to import the 20 pilot videos that we will use to train the classifier for Behavior BtWGaNP. 

>*Note*: There are several considerations when selecting the number of videos for training classifers. Most importantly, the number of videos is not as important as the total number of behavioral events present within the videos. Using something that is frequent, such as attacks, may only require a handfull of videos; using something that is less frrequent, such as lateral threat displays, will require more videos for the same number of events. Further, the expression of these behaviors should be representative of the overall experimental data (i.e., attack behaviors exhibited should generalize to how attacks normally look, and not only be extremely robust or extemely weak). Lastly, you need enough videos that some can be left over to validate the generated predicitive classifiers against. We do not want to test the classifiers on videos that have been used to generate the classifiers. **Here we import 20 videos, having identified that 19 videos contain enough events of Behavior BtWGaNP to get an acceptable sample, and that the remaining video will be used for validation**. As a sanity check, we can also create rendered visualizations of the data that is processed in the subsequent days and phases of Scenarios [2-4](https://github.com/sgoldenlab/simba/tree/master/docs). We will also evaluate the accuracy of the predictive classifier using various [evaluation tools](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-7-train-machine-model) built into SimBA. 

![](/images/Import_videos.PNG "Import_videos")

#### To import multiple videos
1. Navigate to the `[ Import videos into project folder ]` tab.
2. Under the `Import multiple videos` heading, click on `Browse Folder` to select a folder that contains **all the pilot videos** to import into your project.
3. Enter the file type of your videos. (e.g., *mp4*, *avi*, *mov*, etc) in the `Video type` entry box.
4. Click on `Import multiple videos`. 
>*Note*: If you have a lot of videos, or exceptionally high resolution/frame rate/duration videos, it might take a few minutes before all the videos are imported. The main SimBA terminal window will report when the process is complete.

### Step 3: Import DLC Tracking Data
In this step, you will import your pose-estimation tracking data from DeepLabCut in CSV file format. For the pilot videos, this means that you should have 20 individual CSV files corresponding to each of the individual videos.

>*Note*: DLC outputs CSV files with exceptionally long file names. We have included a function that will automatically copy the DLC CSV files, remove the excess DLC notations from the filename, and paste the new CSV files within the SimBA project. The new CSV will now have the same name as the corresponding video. The original DLC output CSV will still be present in its original folder.

![](/images/Create_project_2_new.PNG "importcsv")

#### To import multiple DLC csv files

1. Navigate to the `[ Import tracking data ]` tab. The first dropdown menu specify your file-type. In this tutorial we are importing CSV files from DLC and we will leave this at the default (**CSV (DLC/DeepPoseKit)**). If you have pose-estimation data in alternative file-formats, for example from multi-animal DLC or SLEAP, then please see the seperate SimBA  multi-animal tutorial for how to import this data. 

2. Under the `Interpolate missing pose-estimation data` heading, select how SimBA should handle missing data in your pose-estimation input files. For best classifier performance, it is important that the animals are present in all the videos throughout the recording. We recommend that you use the [SimBA video clipping tools](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#shorten-videos) to remove time-segments of the video where the animals are absent from the recorded arena. However - if clipping the videos is not an option - then we can tell SimBA to interpolate the missing values. SimBA offers three methods of interpolation - linear, nearest, and quadratic. Figures showing the differences between interpolation methods can be found [HERE](https://github.com/sgoldenlab/simba/blob/master/images/Interpolation_comparison.png) or [HERE](https://gist.github.com/gyassine/b47b90e8de935cc06ef856401f7582b0).

>*Note*: We recommend that you make sure that the animals are not absent from the video recording and that you pre-process the videos by removing video segments where the animals are not present, before performing pose-estimation and importing the data into SimBA. Interpolations will result in feeding inaccurate data into SimBA and should be avoided where possible.

3. Under the `Import multiple csv files` heading, click on `Browse Folder` to select the folder that contains the CSV files that you wish to import into your project. Click on `Import csv to project folder`. 



>*Note*: SimBA also provides the ability to import single videos and their corressponding CSV files. For more information, click [here](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#part-1-create-a-new-project-1). This method is not used in Scenario 1.

### Step 4: Extract frames into project folder
This step will extract all the frames from every pilot video that is imported into the project following [**Step 2**](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-import-videos-into-project-folder). The frames are used for behavioral labeling when creating classifiers.

>*Note*: Splitting videos up into frames can take time if you have a lot of high resolution videos at high frame rates. The frames can also take up a lot of hard drive space. This is a good point to grab a coffee, or perhaps the latest edition of Machine Intelligence Daily. Once the step is completed, close the `Project Configuration` window. The main console will remain open and report when the process is complete. 

![](/images/extractframeintop.PNG "extractframeintop")

## Part 2: Load project
In Part 1, we created a project. To continue working with this project, we **must** load it. This section describes how to load and work with SimBA projects.

### Step 1: Load Project Config
In this step you will load the *project_config.ini* file that was created.
> *Note:* A project_config.ini should **always** be loaded before any other process.

1. In the main SimBA window, click on `File` and `Load project`. The following windows will pop up.

![](https://github.com/sgoldenlab/simba/blob/master/images/Load_project_SimBA.PNG "importdlc")

2. Click on `Browse File`. Then, navigate to the directory that you created your project in and click on your *project folder*. Locate the *project_config.ini* file and select it. Once this step is completed, the `File Selected` entry box should look like the following, and you should no longer see the text *No file selected*:

![](https://github.com/sgoldenlab/simba/blob/master/images/Load_project2_SimBA.PNG "importdlc")

In this image, you can see the `Desktop` is my selected working directory, `tutorial` is my project name, and the last two sections of the folder path is always going to be `project_folder/project_config.ini`.

### Step 2 (Optional step) : Import more DLC Tracking Data or videos

In this step, you can choose to import more pose estimation data and/or more videos. You can only archive analysed files, define new classifiers, and remove previously defined classifiers.  If this isn't relevant then you can skip this step. This is however relevant for [Scenario 4](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario4.md#part-2-load-the-project-and-import-your-new-data) and you can read more about these options in the [Scenario 4 tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario4.md#part-2-load-the-project-and-import-your-new-data). However, this is not relevant for the current Scenaro 1, so please skip to Step 3 (but remain aware you can do this if needed later).

![](/images/Create_project_3.PNG "importdlc")

1. Click on the `[ Further imports (data/video/frames) ]` tab. From here you can import more data or videos into the project folder. The imported .csv files will be placed in the `project_folder/csv/input` directory, and the imported videos will be placed in the `project_folder/videos` directory. 

2. Once the videos are imported, you can extract frames from the additional videos by clicking on `Extract frames` under the **Extract further frames into project folder** heading. 

3. If you already have existing frames of the videos in the project folder, you can import the folder that contains the frames into the project. Under the **Import frame folders** heading, click on `Browse Folder` to choose the folder thar contains the frames, and click on `Import frames`. The frames will be imported into the `project_folder/frames/input` folder. 

4. If you would like to *add* a new classifier to the current project, type the name of the new classifier in the `Classifier` entry box in the `Add new classifier(s)` sub-menu. 

5. If you would like to *remove* a previously defined classifier from the current project, click on `Choose a classifier to remove` in the `Remove existing classifiers(s)` submenu. Once clicked, a window will appear with a dropdown menu where you select the classifier you wish to remove from the current project. 

6. Once you have analyzed your videos, and/or used videos to create classifiers in SimBA  (see [Scenario 4](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario4.md#part-2-load-the-project-and-import-your-new-data)), you may want to archive your analyzed files to prevent them from being analyzed by SimBA again. Use the `Archive processed files` menu to enter a folder name for your analyzed files and click on `Archive` to move your analyzed files into this folder. 

### Step 3: Set video parameters
In this step, you can customize the meta parameters for each of your videos (fps, resolution, metric distances) and provide additional custom video information (Animal ID, group etc). This can be very helpful when analyzing data later on. Are there any groups, conditions, days, treatments, etc, that will make your analysis easier? Note that if youd like to change any of these parameters, you can do so with the SimBA [video pre-processing tools](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md). Note that the video pre-processing should be performed prior to analyzing the videos through DeepLabCut, SLEAP, or DeepPoseKit. 

Importantly, in these menues, you also set the **pixels per millimeter** for all of your videos. You will be using a tool that requires the known distance between two points (e.g., the cage width or the cage height) in order to calculate **pixels per millimeter**. We calculate this measure in each video so that we can standardize variables in metric units, and no longer become bound by working with pixel location data. This means that it does not matter if your camera moved slighly across different video recording sessions. 

1. Under **Set video parameters (distances,resolution,etc.)**, the entry box named `Distance in mm` is the known distance between two points in the videos in **millimeters**. If the known distance is the same in all the videos in the project, then enter the millimeter value *(e.g,: 245)* and click on `Auto populate Distance in mm in tables`. This will auto-populate the table in the next step (see below). If you leave the `Distance in mm` entry box empty, the known distance will default to zero and you will fill in the value for each video individually. 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/setvidparameter.PNG" width="391" height="96" />

2. Click on `Set Video Parameters` and the following windows will pop up.

![](/images/videoinfo_table_20vid.PNG "videoinfo_table_20vid.PNG")

3. In the above example we imported the 20 pilot videos and their names are listed the left most `Video` column. SimbA auto-populated the known distance to 245 millimeter in the previous step, and this is now displayed in the `Distance in mm` column. 

4. You can click on the values in the entry boxes and change them until you are satisfied. By default, the entry boxes are populated with the meta data from the video files. 

5. Regardless if you updated the values in the table or not, click on `Update distance_in_mm` at the top of the window: this will save the information displayed in the table into a CSV file. The CSV file is saved as *video_info.csv* and is stored in the `project_folder\logs` folder. After clicking on the `Update distance_in_mm` button, you can proceed to the next step. 

6. Next, to get the `Pixels/mm` for the first video, click on `Video1` and the following window will pop up. The window that pops up displays the first frame of `Video1`.

>*Note*: When the frame is displayed, it may not be shown at the correct aspect ratio. To fix this, drag the window corner to the correct aspect ratio. 

<p align="center">
  <img width="300" height="400" src="https://github.com/sgoldenlab/simba/blob/master/images/getcoord1.PNG">
</p>

7. Now, double **left** click to select two points that defines the known distance in real life. In this case, we know that the two **pink connected dots** represent a distance of 245 millimeter in real life.
<p align="center">
  <img width="300" height="400" src="https://github.com/sgoldenlab/simba/blob/master/images/getcoord2.PNG">
</p>

8. If you misplaced one or both of the dots, you can double click on either of the dots to place them somewhere else in the image. Once you are done, hit `Esc`.

<p align="center">
  <img width="400" height="500" src="https://github.com/sgoldenlab/simba/blob/master/images/getcoord.gif">
</p>

9. If every step is done correctly, the `Pixels/mm` column in the table should populate with the number of pixels that represent one millimeter, 
<p align="center">
  <img width="700" height="350" src="https://github.com/sgoldenlab/simba/blob/master/images/videoinfo_table2.PNG">
</p>

10. Repeat the steps for every video in the table, and once it is done, click on `Save Data`. This will update the CSV file named **video_info.csv** in `/project_folder/log` directory that contains your video meta data. 

> *Note:* If you have a a very large amount of videos then this process (Step 10 above) can become time-consuming. If you are confident that your camera moved **minimally** across recordings, and that the `Pixels/mm` therefore is near-identical across videos, then SimBA has the option of applying the `Pixels/mm` created for the first video in the table, on all other videos in the table. To proceed with this, click on the red `Duplicate index 1 pixel/mm (CAUTION!)` button. 

![](/images/Create_project_4.PNG "project4_dup.PNG")

11. You can also chose to add further columns to the meta data file (e.g., AnimalID or experimental group) by clicking on the `Add Column` button. This information will be saved in additional columns to your **video_info.csv** file.

### Step 4: Outlier Correction

Outlier correction is used to correct gross tracking inaccuracies by detecting outliers based on movements and locations of body parts in relation to the animal body length. For more details, click [here](https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf). The annotation options are based upon the yaml.config settings, and we suggest that defaults be kept. The only values that must be manually entered in the Settings menu are the *Location Criterion* and the *Movement Criterion*, explained below.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/outliercorrection.PNG" width="156" height="109" />

1. Click on `Settings` and the following window will pop up.

<p align="center">
  <img width="300" height="400" src="https://github.com/sgoldenlab/simba/blob/master/images/outliercorrection2.PNG">
</p>

2. Select the body parts for Animal 1 and Animal 2 that you want to use to calculate a reference value. The reference value will be the mean or median Euclidian distance in millimeters between the two body parts of the two animals in all frames. Again, we suggest you keep the default settings.

3. Enter values for the `Movement criterion` and the `Location criterion`. 

- `Movement criterion`. A body part coordinate will be flagged and corrected as a "movement outlier" if the body part moves the *reference value multiplied by the criterion value* across two sequential frames. The reference value is the mean or median length of the animal between the selected two body parts. The criteron value is a multiplier of the reference value. For example, the distance between Nose_1 and Tail_base_1 is 100mm and the movement criterion is 0.5, any body part that moves 50mm across two sequential frames will be corrected. **We suggest a movement criterion value between 0.7 to 1.5**.

- `Location criterion`. A body part coordinate will be flagged and correct as a "location outlier" if the distance between the body part and at least two other body parts belonging to the same animal are longer than the *reference value multiplied by the criterion value* within a single frame. The reference value is the mean or median length of the animal between the selected two body parts. The criteron value is a multiplier of the reference value. For example, the distance between Nose_1 and Tail_base_1 is 100mm and the location criterion is 1.5, any body part located greater than 150 mm from two other body parts will be corrected. **We suggest a location criterion value of 1.5 or greater**.

>*Note:* Movement outliers are corrected prior to location outliers. Specifically, (i) movement outliers are first corrected, and (ii) location outliers are then corrected on the dataframe that contains the movement corrected data. That means that there is likely to be *fewer* location outliers than movement outliers corrected.   

Body parts flagged as movement or location outliers will be re-placed in their last reliable coordinate. 

4. Chose to calculate the *median or mean* Euclidian distance in millimeters between the two body parts at the bottom of the `Settings` window and click on `Confirm Config`. 

5. Click to run the outlier correction. You can follow the progress in the main SimBA window. Once complete, two new CSV log files will appear in the `/project_folder/log` folder. These two files contain the number of body parts corrected following the two outlier correction methods for each video in the project. The files will look similar to the worksheets in [this Microsoft Excel file](https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_corrections.xlsx). 

> *Note:* In some scenarios, SimBA users are confident that the pose-estimation is perfected, no gross innacuracies are present, and the user may want to completely skip the use of SimBAs outlier correction tools. To do this, click on the red `Skip outlier correction (CAUTION)` button. Clicking on this button will format your CSV files and make them compatible with subsequent procedures (see below) without removing any outliers from you tracking data. 

>*Note:* Some users are are primarily intrested in using SimBA for other things than predictive classification. This may include using SimBA for analyzing animal movements, the animals directionality or "gaze", animal velocities / distance travelled, or time spent / entries made into user-defined regions of interest. If you are inrested in using these functions in SimBA, you now have all the data you need and do not need to proceed with the current tutorial. Instead, head to the [SimBA ROI tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md).          

### Step 5: Extract Features
Based on the coordinates of body parts in each frame - and the frame rate and the pixels per millimeter values - the feature extraction step calculates a larger set of features used for behavioral classification. Features are values such as metric distances between body parts, angles, areas, movement, paths, and their deviations and rank in individual frames and across rolling windows. This feature set extracted from the tracking data is what we are going to use to predict behavior BtWGaNP, using the relationships between features, rather than the pose estimation data itself. 

This set of features will depend on the body-parts tracked during pose-estimation (which is defined when creating the project). Click [here](https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv) for an example list of features when tracking 2 mice and 16 body parts. 

>*Note I*: If you use a different SimBA pre-packaged DeepLabCut body-part annotation configuration, such as 3 body parts with 2 mice, then the feature list will be significantly reduced and may subsequently negatively influence prediction performance. For a reminder of the different SimBA pre-packaged DeepLabCut body-part annotation configurations, click [here](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md). 

>*Note II*: If have created your own [*user-defined* body-part configuration]9https://github.com/sgoldenlab/simba/blob/simba_JJ_branch/docs/Pose_config.md), then SimBA will generate a set of generic features based on the distances and movements of all body-parts and their relation to each other. To see a list of these generic features, please click [here](https://github.com/sgoldenlab/simba/blob/master/misc/features_user_defined_pose_config.csv). This generic feature list is likely to have significicantly reduced predictive power and generate weaker predictive classifiers relative to the feature list calculated from [16 body-parts and 2 mice](https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv).

1. Click on the tab `Extract Features`, and then the button `Extract Features`. 

![](https://github.com/sgoldenlab/simba/blob/master/images/Extract_features_SimBA.PNG "extract_features")

New CSV files, that contain the feature data and the pose-estimation data, will be generated and saved in the `project_folder\csv\features_extracted` directory, with one file for every DLC tracking file imported to the project. This tab also displays a second button - `Append ROI data to features`. We explain the function of this button in the [ROI tutorial](https://github.com/sgoldenlab/simba/blob/simba_JJ_branch/docs/ROI_tutorial.md), and how it can be used to create features from spatial locations/objects in the videos.  

2. We want to validate the classifier for behavior BtWGaNP on a separate video, and that video should not be used in the training and testing steps. In the current Scenario 1, we have generated 20 files containing features and they are stored in the `project_folder\csv\features_extracted` folder. To store away one of these files for later validation, navigate to the `project_folder\csv\features_extracted` folder, and cut one file out of the `project_folder\csv\features_extracted` folder, and paste it somewhere else outside of your `project_folder`. This way, SimBA won't see the file in later steps, and it will be omitted from inclusion for when creating the model. We will later define the directory path to this file, and try to predict behavior BtWGaNP in this file from the classifer generated using data in the other 19 data files.  

### Step 6: Label Behavior (i.e, create annotations for predictive classifiers)
This step is used to label the behaviors in each frames of a video. This data will be concatenated with the extracted features and used for the creating behavioral classifier. 

>*Note:* SimBA performs similar functions such as the open-source JWatcher or commercial Noldus Observer systems, with the exception that SimBA automates the backend integration of behavioral annotation with creating predictive classifiers. If you already have such annotations stored in alterantive file formats, like JWatcher or Noldus Observer, they can be appended directly to the tracking data and no behavioral annotations needs to be done in SimBA. For example, the [CRIM13](http://www.vision.caltech.edu/Video_Datasets/CRIM13/CRIM13/Main.html) dataset was annotated using ['Piotr’s Matlab Toolbox'](https://pdollar.github.io/toolbox/) and we appended these annotations to the tracking data using a version of [this script](https://github.com/sgoldenlab/simba/blob/master/misc/Caltech_2_DLC.py). 

If you already have annotation videos created with these alternative tools, or any other behavioral annotator, and would like to use them to create predictive classifiers, please let us know as we would like to write scripts that could process these data for SimBA. If you have created such scripts yourself, please consider contributing them to the community!

**Important**: The behavioral labelling is a very critical step. A computer will only learn what you teach it, so if your annotations are not correct then the classifiers will fail. **SimBA uses every single frame**, and therefore the "start" and "stop" frames for a given behavior are very important. Please take the time to create clear and well-defined operationalized definitions of the complete behavior, including start and stop frames. As an example, here are machine learning operationalized definitions of [mouse](https://github.com/sgoldenlab/simba/blob/master/misc/Operational_definitions_mouse_resident_intruder.pdf) and [rat](https://github.com/sgoldenlab/simba/blob/master/misc/Operational_definitions_rat_resident_intruder.pdf) predictive classifiers for aggressive behaviors in resident-intruder protocols.

1. Click on the `Label Behavior` tab in the `Load Project` window and you should see the following menu:

![](/images/Create_project_11.jpg "project11_dup.PNG")

Use the first button (i) in the image above if you are starting to create behavior labels for a video from scratch. Use the second button (ii) to continoue annotating a video that you have made behavior labels for previously in the current project. In the third menu (iii) we have button-click access to functions that append behavior labels generated in third-party apps to your SimBA dataset.   

1. In the current scenario, we want to click on the first button - `Select folder with frames (Create new video annotation)`. In your project folder navigate to the `/project_folder/frames/input/` folder, and you should see folders that are named after your videos. These folders contain the extracted video frames that we created in [Step 4](https://github.com/sgoldenlab/simba/blob/SimBA_no_TF/docs/Scenario_1_new.md#step-4-outlier-correction) of this current tutorial. Select one of the folder and the following window should pop up. To revisit the tutorial on how to extract the frames for your project, click [here](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-extract-frames-into-project-folder). 

<p align="center">
  <img width="720" height="720" src="https://github.com/sgoldenlab/simba/blob/master/images/labelbe.PNG">
</p>

2. Please click [here](/docs/labelling_aggression_tutorial.md) to learn how to use the behavior annotation interface.

3. Once finished, click on `Generate/Save` and it will generate a new CSV file in */csv/targets_inserted* folder. Repeat this step for all the files you wish you use to generate the predictive classifier for behavior BtWGaNP.

>*Note:* If you used SimBAs behavior label annotation interface to create behavior labels for the first half of your video, and now you want create labels for the second half of a video, then you can do so by clicking on the second button in the `Label Behavior` menu titled `Select folder with frames (Continue with existing video annotation)`. After clicking on this button, follow the same steps as when creating a new annotations. The difference with clicking on this button will be that when the SimBA behavior label interface shows up, it will show up from the last saved frame with your previous annotations for this file in memory. 

### Step 7: Train Machine Model
This step is used for training new machine models for behavioral classifications. There are a large number of machine learning parameters, called Hyperparameters, that influence Random Forest models. We have currated a list of Hyperparameters and made it easy to tweak, and validate, their values. For more in-depth technical explanations, please see [sklearn.ensemble.RandomForestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) or join the discussion on our [Gitter page](https://gitter.im/SimBA-Resource/community). We have also made it possible to import these settings, in order to make the setting of these Hyperparameters as painless as possible (see below). This is a lot to look at, please read this whole section before starting anything.

**Note**: SimBA allows you to generate predictive classifiers in two different *modes*. You can either (i) specify a single set of Hyperparameters, and train a single predictive classifiers using that defined set of hyperparameters, or alternatively, you can (ii) specify many different Hyperparameter settings, and batch train multiple different models, each using a different set of Hyperparameters. The second option is relevant for the current Scenario. For example, here we may want to generate five different classifiers that predict the behavior BtWGaNP, evaluate each one, and proceed to the Experimental data with the classifier that best captures behavior BtWGaNP in the pilot data and validation video. Thus, the first section of this part of the tutorial describes the different Hyperparameter settings, and what you can do to avoid setting them manually (*HINT*: you can load them all from a *metadata* file at the top of the window), while the second part of the tutorial section describes how to proceed with either of the two *modes* for generating classifiers.       

#### Train predictive classifier(s): settings

1. Click on `Settings` and the following, slightly indimidating (but I promise: easy to use!) window will pop up. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Settings1.png" />
</p>

>*Note I:* If you have a CSV file containing hyperparameter metadata, you can import this file by clicking on `Browse File` and then click on `Load`. This will autofill all the Hyperparameter entry boxes and model evaluation settings. For the Scenario 1, we [provide](https://github.com/sgoldenlab/simba/blob/master/misc/BtWGaNP_meta.csv) a Metadata file that will populate the Hyperparameter entry boxes and evaluation settings with some default values. Please save this file to disk and load it. If you downloaded SimBA through our github page, the Metadata file should be in the *simba/misc* directory. 

>*Note II:* If you open the *BtWGaNP_meta.csv* file, you'll see that the left-most column - with the heading *Classifier_name* - contains one entry: *BtWGaNP*. This *Classifier_name* entry is only for reference, and does not affect the classfier you are generating. If you are generating a classifier using a different classifier name, let's say *attack*, you can leave the entries in the *BtWGaNP_meta.csv* file as is without causing any errors. 

2. Under **Machine Model**, choose a machine model from the drop down menu: `RF` ,`GBC`,`Xboost`. For this Scenario, choose RF (*Note:*: GBC and Xgboost options are still under development). 

- `RF`: Random forest

- `GBC`: Gradient boost classifier

- `Xgboost`: eXtreme Gradient boost

3. Under the **Model** heading, use the dropdown menu to select the behavioral classifier you wish to define the hyper-parameters for. In this Scenario, only one *Model Name* should be seen in the drop-down menu (BtWGaNP). If you are generating multiple classifiers, they should all be seen in the drop-down menu. 

4. Under the **Hyperparameters** heading, select the Hyperparameters settings for your model. For more details, please click [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). Alternatively, import settings from a meta data file (see above). 

Here is a brief description of the different Hyperparameter settings, together with some further links to where you can read more about them. 

- `RF N estimators`: Number of decision trees in the decision ensemble (e.g., 2000).

- `RF Max features`: Number of features to consider when looking for the best split (e.g., enter *sqrt* to take the square root of the total number of features in your dataset when evaluating a split).

- `RF Criterion`: The metric used to measure the quality of each split (e.g., *gini* or *entropy*).

- `Train Test Size`: The ratio of the dataset withheld for testing the model (e.g., 0.20).

- `RF Min sample leaf`: The minimum number of samples required to be at a leaf node (e.g., *1*, or increase to [prevent over-fitting](https://elitedatascience.com/overfitting-in-machine-learning)). 

- `Under sample setting`: "Random undersample" or "None". If "Random undersample", a random sample of the majority class will be used in the train set. The size of this sample will be taken as a ratio of the minority class and should be specified in the "under sample ratio" box below. For more information, click [here](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html). This setting address issues that arise from "imbalanced" data sets, where the behavior that is predicted is very sparse. **Note:** *Class* means the classification that a video frame belongs to. In this Scenario it is either (i) Not BtWGaNP, or (ii) BtWGaNP. The majority class is the class that contains the most examples, which - most certaily - in any use case of SimBA,will be *Not BtWGaNP*. Conversely, the minority class is the class that contains the least examples, which will be *BtWGaNP*.    

- `Under sample ratio`: The ratio of samples of the majority class to the minority class in the training data set. Applied only if `Under sample ratio` is set to "Random undersample". Ignored if "Under sample setting" is set to "None" or NaN.

- `Over sample setting`: "SMOTE", "SMOTEEN" or "None". If "SMOTE" or "SMOTEEN", synthetic data will be generated in the minority class based on k-mean distances to balance the two classes. For more details on SMOTE, click [here](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html). Alternatively, import recommended settings from a meta data file (see **Step 1**). For more information on *minority/majority classes* see the documentation for `Under sample setting` above. 

- `Over sample ratio`: The desired ratio of the number of samples in the minority class over the number of samples in the majority class after over sampling.Applied only if `Over sample setting` is set to "SMOTE" or "SMOTEEN". Ignored if "Under sample setting" is set to "None" or NaN.

5. Under **Model Evaluation Settings**.

Here is a brief description of the different Model evaluation settings available in SimBA, together with links to where you can read more about them. These model evaluation tools generate graphs, images, and CSV files that contain different metrics on how well your classifiers performs, how the classifiers reaches its decisions, and how you could improve the predictive classifier.   

- `Generate RF model meta data file`: Generates a CSV file listing the hyper-parameter settings used when creating the model. The generated meta file can be used to create further models by importing it in the `Load Settings` menu (see above, **Step 1**).

- `Generate Example Decision Tree - graphviz`: Saves a visualization of a random decision tree in PDF and .DOT formats. Requires [graphviz](https://graphviz.gitlab.io/). For more information on this visualization, click [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html). For information on how to install on Windows, click [here](https://bobswift.atlassian.net/wiki/spaces/GVIZ/pages/20971549/How+to+install+Graphviz+software). For an example of a graphviz decision tree visualization generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_tree.pdf). *Note:*  The trees can be very large depending on the Hyperparameter settings. Rather than using a dedicated PDF viewer, try opening the generated PDF by dragging it into a web browser to get a better view. 

- `Generate Fancy Example Decision Tree - dtreeviz`: Saves a nice looking visualization of a random decision tree in SVG format. Requires [dtreeviz](https://github.com/parrt/dtreeviz). For an example of a dtreeviz decision tree visualization generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/dtreeviz_SimBA.png). *Note:* These SVG example decision trees are very large. To be able to view them on standard computers, SimBA limits the depth of the example tree to 3 levels.

- `Generate Classification Report`: Saves a classification report truth table in PNG format displaying precision, recall, f1, and support values. For more information, click [here](http://www.scikit-yb.org/zh/latest/api/classifier/classification_report.html). For an example of a classification report generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_classificationReport.png).  

- `Generate Features Importance Log`: Creates a CSV file that lists the importance's [(gini importances)](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) of all features used to generate the classifier. The CSV file is saved in the `project_folder\logs` directory, with the file name 'BtWGaNP_feature_importance_log.csv'. For an example of a feature importance list generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_feature_importance_log.csv). *Note:* Although [gini importances](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) gives an indication of the most important features for predicting the behavior, there are known [flaws when assessing well correlated features](https://explained.ai/rf-importance/). As SimBA uses a very large set of well correlated features to enable flexible usage, consider evaluating features through permutation importance calculatations instead (see below). 

- `Generate Features Importance Bar Graph`: Creates a bar chart of the top *N* features based on gini importances (see above). Specify *N* in the `N feature importance bars` entry box below (see blow). **The creation of this bar graph requires that the `Generate Features Importance Log` box is ticked**. For an example of a bar chart depicting the top *N* features generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_feature_bars.png).   

- `N feature importance bars`: Integer defining the number of top features to be included in the bar graph (e.g., 15). 

- `Compute Feature Permutation Importance's`: Creates a CSV file listing the importance's (permutation importance's) of all features for the classifier. For more details, please click [here](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html)). Note that calculating permutation importance's is computationally expensive and takes a long time. For an example CSV file that list featue permutation importances, click [here] (https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_prediction_permutations_importances.csv). 

- `Generate Sklearn Learning Curve`: Creates a CSV file listing the f1 score at different test data sizes. For more details on learning curves, please click [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)). For more information on the f1 performance score, click [here](https://en.wikipedia.org/wiki/F1_score). The learning curve is useful for estimating the benefit of annotating further data. For an example CSV file of the learning curve generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_prediction_learning_curve.csv). 

- `LearningCurve shuffle K splits`: Number of cross validations applied at each test data size in the learning curve. 

- `LearningCurve shuffle Data splits`: Number of test data sizes in the learning curve.  

- `Generate Precision Recall Curves`: Creates a CSV file listing precision at different recall values. This is useful for titration of the false positive vs. false negative classifications of the models by manipulating the `Discrimination threshold` (see below). For an example CSV file of of a precision-recall curve generated through SimBA, click [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_prediction_precision_recall.csv). This may be informative if your aiming to titrate your classifier to (i) predict all occurances of behavior BtWGaNP, and can live with a few false-positive classifications, or (ii) conversely, you want all predictions of behavior BtWGaNP to be accurate, and can live with a few false-negative classifications, or (iii) you are aiming for a balance between false-negative and false-positive classifications. 

- `Calculate SHAP scores`: Creates a CSV file listing the contribution of each individual feature to the classification probability of each frame. For more information, see the [SimBA SHAP tutorial](https://github.com/sgoldenlab/simba/edit/master/docs/SHAP.md) and the [SHAP GitHub repository](https://github.com/slundberg/shap). SHAP calculations are an computationally expensive process, so we most likely need to take a smeller random subset of our video frames, and calculate SHAP scores for this random subset:

  - `# target present`: The number of frames (integer - e.g., `100`) with the behavioral target **present** (with the behavioral target being the behavior selected in the **Model** drop-down menu in Step 3 above) to calculate SHAP values for. 
 
   - `# target absent`: The number of frames (integer - e.g., `100`) with the behavioral target **absent** (with the behavioral target being the behavior selected in the **Model** drop-down menu in Step 3 above) to calculate SHAP values for.

 
#### Train predictive classifier(s): start the machine training

Once all the entry boxes have been filled in with the desired Hyperparameters and Model Evaluation Settings, the user can either click on `Save settings into global environment` or `Save settings for specific model`. If you click on `Save settings into global environment`, the settings will be saved in to your *project_config.ini* file located in your `project_folder`. These settings can subsequently be retreived and executed to generate a predictive classifier (**Mode 1**). 

However, if you click on `Save settings for specific model`, a config file will be generated in the background (in the `/project_folder/configs` folder) that contains the defined settings. To generate 5 different config files with different settings, for example, simply update the Hyperparameters and Model Evaluation Settings five different times, representing the five different predictive classifiers you want to generate, and after each time you have updated the values, click on `Save settings for specific model` (**Mode 2**). See below for more information.

**(i) Mode 1**: To save the settings into the global environment, and generate a single predictive classifier using these settings, click on `Save settings into global environment`, and then **exit the machine model settings window by closing it**. Next, click on the blue button: `Train single model from global environment`. While the model is training, text is printed in the main SimBA window indicating its progress. A message saying that training is complete will also be printed in the terminal window, together with information on where you can find the generated files.  The model, in *.sav file format*, will be saved in the `project_folder\models\generated_models` folder. If you have chosen to generate model evaluation files, then they will be be saved in the `project_folder\models\generated_models\model_evaluations` folder. 

**(ii) Mode 2**: Alternatively, click on the `Save settings for specific model` button to save the settings for one model. To generate multiple models - for multiple different Hyperparameters used to predict behavior BtWGaNP - redefine the Machine model settings and click on `Save settings for specific model` again. Each time the `Save settings for specific model` is clicked, a new config file is generated in the */project_folder/configs* folder. Next, **exit the machine model settings window by closing it** and click on the *green* button: `Train multiple models, one for each saved setting`. This will generate one model for each of the config files in the */project_folder/configs* folder. The models, in *.sav file format*, will be saved in the `project_folder\models\validations\model_files` folder. Model evaluation files will be saved in the `project_folder\models\validations\model_evaluations` folder. Model meta files, will be saved in the `project_folder\models\validations\meta_data` directory. 

### Critical validation step before running machine model on new data

If you have chosen to generate classification reports or other metrics of classifier performance, it is worth studying them to ensure that the model(s) performance is acceptable. However, a classifiers performance is perhaps most readily validated by visualizing its predictions and prediction probabilities on a new video, which have not been used for training or testing. This step is critical for (i) visualizing and choosing the ideal classification probability thresholds which captures all of your BtWGaNP behaviors (for more information on what classification threshold is - see Step 4 below), and (i) visual confirmation that model performance is sufficent for running it on experimental data.

You can validate each model *(saved in SAV format)* file. This should be done in a "gold-standard" video that has been fully manually annotated for your behavior of interest, but has not been included in the training dataset. If you followed the tutorial, you may remember that we stored away one CSV file away in a safe place earlier, a [file which we had extracted the features](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-5-extract-features) for but we did not use this file for training or testing of the classifier. Now is the time to use this file. 

In this validation step the user specifies the path to a previously created model in SAV file format, and the path to a CSV file [that contain the features extracted from a video (Step 5 above)](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-5-extract-features). The first process will run the predictions on the video, and visualize the probabilities in a user-interactable line chart, that together with a user-interactable visualization of the video can be used to gauge the ideal discrimination threshold. The second process will create a video with the predictions overlaid together with a gantt plot showing predicted behavioral bouts. Click [here](https://youtu.be/UOLSj7DGKRo) for an expected output validation video for predicting the behavior *copulation*. 

This process allows you to rapidly access the results of the Hyperparameters you have selected on a "gold-standard" behavioral video. If the predictions are not good, you can go back to tweak the appropriate parameters without first running through numerous other videos or adding and/or refining your annotations.

For more details, please [click here](/docs/validation_tutorial.md#validate-model-on-single-video)

![](/images/validatemodel_graph1.PNG)

**Note: If you check the 'Generate Gantt plot' box, the final validation will include two seperate videos - 1 of the behavior and 1 of the gantt plot. However, this takes longer and we suggest leaving it unchecked unless you really want the gantt plot. This may or may not be combined to a single video in the future!**

1. Under the **Run machine model tab** and *Validate Model on Single Video* heading, click on `Browse File` next to `Select features file`. In this current Scenario 1, select the CSV file you have stored [in a safe place outside of the project folder](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-5-extract-features). 

2. Next to `Select model file`, click on `Browse File` to select a model (*.sav file*). Your *.sav file(s)* should be saved in the `project_folder\models\generated_models` or the `project_folder\models\validations\model_files` folder, depending on which **mode** of model generatation you [used above](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-start-the-machine-training). However, note that SimBA does not care where the model *.sav file* is located, and could be - for example - in the *Downloads* folder if you have downloaded it from our [OSF repository](https://osf.io/d69jt/). When done, click on `Run model`. There will be a message in the main SimBA terminal window when the process is complete.   

3. Next, click on `Generate plot`. Two windows should pop open, a *Prediction probability* line graph for your chosen behaviour (left) and an interactive display of your video (right):

- `Graph window`: model prediction probability versus frame numbers will be plot. The graph is interactive, click on the graph and the frame window will display the selected frames.

- `Frame window`: Frames of the chosen video with controls.

![](/images/validategraph1.PNG)


Clicking on the graph on the line displays the corresponding frame in the video on the right. Use the buttons below the image on the right to scan through the images around the peaks on the line graph in order to identity the *classification probability* that separates all the BtWGaNP behaviors from non-BtWGaNP behaviors. Once complete, make a note of the ideal threshold and close the two windows. 

4. Click on the points on the graph and picture displayed on the other window will jump to the corresponding frame. There will be a red line to show the points that you have clicked.

![](/images/validategraph2.PNG)

5. Once it jumps to the desired frame, you can navigate through the frames to determine if the behavior is present. This step is to find the optimal threshold to validate your model.

![](/images/validategraph.gif)

6. Set the identified `Discrimination threshold` and `Minimum behavior bout length (ms)`:

- `Discrimination threshold`: This value represents the level of probability required to define that the frame belongs to the target class (i.e., the BtWGaNP class) and it accepts a float value between 0.0 and 1.0.  In other words, how certain does the computer have to be that behavior BtWGaNP occurs in a frame, in order for the frame to be classified as containing behavior BtWGaNP? For example, if set to 0.50, then all frames with a probability of containing the behavior of 0.5 or above will be classified as containing the behavior. For further information on classification theshold, click [here](https://www.scikit-yb.org/en/latest/api/classifier/threshold.html). In this Scenario. Go ahead and enter the classification threshold identified in the previous *Step 6 of this sub-section*. 

- `Minimum behavior bout length (ms)`: This value represents the minimum length of a classified behavioral bout. **Example**: The random forest makes the following predictions for behavior BtWGaNP over 9 consecutive frames in a 50 fps video: 1,1,1,1,0,1,1,1,1. This would mean, if we don't have a minimum bout length, that the animals enganged in behavior BtWGaNP for 80ms (4 frames), took a brake for 20ms (1 frame), then again enganged in behavior BtWGaNP for another 80ms (4 frames). You may want to classify this as a single 180ms behavior BtWGaNP bout, rather than two separate 80ms BtWGaNP bouts. If the minimum behavior bout length is set to 20, any interruption in the behavior that is 20ms or shorter will be removed and the example behavioral sequence above will be re-classified as: 1,1,1,1,1,1,1,1,1 - and instead classified as a single 180ms BtWGaNP bout. 

7. Click on `Validate` to run the validation of the selected model on you specified video. This will apply the selected model to the feature data of the specified file and generate a video with behavioral predictions overlaid on the frames together with a gantt plot depicting predicted behavioral bouts - like the below gif. Click [here](https://youtu.be/UOLSj7DGKRo) for an expected full output example validation video. The video will be stored in the `\project_folder\frames\validation` folder. Note that rendering frames and videos can take a while if your video is long, high-frame rate, or high resolution. 

![](https://github.com/sgoldenlab/simba/blob/master/images/validation_example_2.gif "validation2")

>*Note I*: SimBA does offer a further form of [`classifier validation` visualization tool](https://github.com/sgoldenlab/simba/blob/master/docs/classifier_validation.md). This further tool was designed to visualize the possible presence and extent of false-positives classifications. The tool displays the bouts classified as containing the target behavior, with a printed overlay displaying the total number of frames contained each bout, the frames conatined within the bout, and the probability that each specific bout contains the behavior. For more infomration on the `Classifier Validation` tool, see the tutoral on the [Post-classification Validation (detecting false-positives)](https://github.com/sgoldenlab/simba/blob/master/docs/classifier_validation.md). 

Congrats! You have now generated RF models that can be run on new, experimental data, and we hope those models are doing a good job at recognizing the behaviors of intrest. Proceed to [Scenario 2](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario2.md) to use the BtWGaNP classifier to evaluate new experimental data. If you already have further data that has been processed (outliers are corrected, and the featureshave been calculated), and you now want to run the classifier on this data, proceed to [Scenario 2 - Part 3: Run the classifier on new data](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data).


