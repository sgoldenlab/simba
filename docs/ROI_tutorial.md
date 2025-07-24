# Regions of Interest (ROIs) in SimBA

The SimBA region of interest (ROI) interface allows users to define and draw ROIs on videos. ROI data can be used to calculate basic descriptive statistics based on animals movements and locations (such as how much time the animals have spent in different ROIs, or how many times the animals have entered different ROIs). Furthermore, the ROI data can  be used to build potentially valuable, additional, features for random forest predictive classifiers. Such features can be used to generate a machine model that classify behaviors that depend on the spatial location of body parts in relation to the ROIs. **CAUTION**: If spatial locations are irrelevant for the behaviour being classified, then such features should *not* be included in the machine model generation as they just only introduce noise.   

# Before analyzing ROIs in SimBA

To analyze ROI data in SimBA (for descriptive statistics, machine learning features, or both descriptive statistics and machine learning features) the tracking data **first** has to be processed the **up-to and including the *Outlier correction* step described in [Part 2 - Step 4 - Correcting outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction)**. Thus, before proceeding to calculate ROI based measures, you should have one CSV file for each of the videos in your project located within the `project_folder\csv\outlier_corrected_movement_location` sub-directory of your SimBA project. 

Specifically, for working with ROIs in SimBA, begin by (i) [Importing your videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-import-videos-into-project-folder), (ii) [Import the tracking data and relevant videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-import-dlc-tracking-data), (iii) [Set the video parameters](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters), and lastly (iv) [Correct outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction).

>**A short explanation on what is meant by "using ROI data as features for random forest classifiers"** When ROIs have been drawn in SimBA, then we can calculate several metrics that goes beyond the coordinates of the different body-parts from pose-estimation. For example - for each frame of the video - we can calculate the distance between the different body-parts and the different ROIs, or if the animal is inside or outside the ROIs. In some instances (depending on which body parts are tracked), we can also calculate if the animal is [directing towards the ROIs](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data)(see below for more information). These measures can be used to calculate several further features such as the percentage of the session, up to and including the any frame, the animal has spent within a specific ROI. These and other ROI-based features could be useful additions for classifying behaviors in certain scenarios.

# Part 1. Defining ROIs in SimBA

See [THIS](https://github.com/sgoldenlab/simba/edit/master/docs/roi_tutorial_new_2025.md) tutorial for how to draw ROIs in SimBA.

# Part 2. Analyzing ROI data.

Once we have [drawn the ROIs](https://github.com/sgoldenlab/simba/blob/master/docs/roi_tutorial_new_2025.md), we can compute descriptive statistics based on movement of the animal(s) in relation to the ROIs 
(for example, how much time the animals spends in the ROIs, or how many times the animals enter the ROIs etc). To compute ROI statistics, click in the <kbd>ANALYZE ROI DATA: AGGREGATES</kbd> button in the `[ ROI ]` tab and you should see the 
following pop up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/roi_analyze_tutorial_1.webp" />
</p>

1) In the `# OF ANIMALS` dropdown, select the number of animals you want to compute ROI statistics for.
   
2) In the `PROBABILITY THRESHOLD` entry-box, select the minimum pose-estimation probability score (between 0 and 1) that should be considered when performing ROI analysis. Any frame body-part probability score above the entered value will be filtered out.  
> [!CAUTION]
> If possible, we recommend having reliable pose-estimation data in every frame. This includes pre-process all videos, and remove any segments of the videos where animals are not present in the video as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md),
> and performing pose-estimation interpolation of missing data at data import as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#to-import-multiple-dlc-csv-file).

3). In the `SELECT BODY-PART(S)` frame, use the dropdown menus to select the body-parts that you wish to use to infer the locations of the animals. 

4). In the `DATA OPTIONS` frame, select the data that you wish to compute:

 * `TOTAL ROI TIME (S)`: The total time, in seconds, that each animal spends in each defined ROI in each video.
 * `ROI ENTRIES (COUNT): The total number of times, that each animal enters each defined ROI in each video.
 * `FIRST ROI ENTRY TIME (S)`: The video time, in seconds, when each animal first enters each defined ROI in each video (NOTE: will be `None` if an animal never enters a defined ROI).   
 * `LAST ROI ENTRY TIME (S)`: The video time, in seconds, when each animal last enters each defined ROI in each video (NOTE: will be `None` if an animal never enters a defined ROI).
 * `MEAN ROI BOUT TIME (S)`: The mean length, in seconds, of each sequence the animal spends in each defined ROI in each video (NOTE: will be `None` if an animal never enters a defined ROI).
 * `DETAILED ROI BOUT DATA (SEQUENCES)`: If checked, the SimBA ROI analysis generates a CSV file within the `project_folder/logs` directory named something like *Detailed_ROI_bout_data_20231031141848.csv*. This file contains the exact frame numbers, time stamps, and duration of each seqence when animals enter and exits each user-drawn ROIs. (NOTE: no file will be created if no animal in any video never enters an ROI)
 * `ROI MOVEMENT (VELOCITY AND DISTANCES)`: The total distance moved, and the average velocity, of each animal in each defined ROI in each video.
 * `OUTSIDE ROI ZONES DATA`: If checked, SimBA will treat **all areas NOT covered by a ROI drawing** as a single additional ROI and compute the chosen metrics for this, single, ROI. For example of the expected output when this checkbox is ticked, see [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/outside_roi_ex.csv) file.

5). In the `FORMAT OPTION` frame, select how the output data should be formatted, and any addtional video meta data that should be included in the output which could be helpful for sanity checks. 

* `TRANSPOSE OUTPUT TABLE`: If checked, one row in the output data will represent a video. If unchecked, one row in the output data will represent a data measurment.
* `INCLUDE FPS DATA`: If checked, the FPS used to compute the metrics of each video will be included in the output table. 
* `INCLUDE VIDEO LENGTH DATA`: If checked, the length of each video in seconds will be included in the output table.
* `INCLUDE INCLUDE PIXEL PER MILLIMETER DATA`: If checked, the pixel per millimeter conversion factor (used to compute distances and velocity) of each video will be included in the output table.

6. Once the above is filled in, click the <kbd>RUN</kbd> button. You can foillow the progress in the main SimBA terminal.

Once complete, a file will be stored in the logs folder of your SimBA project named something like `ROI_descriptive statistics_20250306162014.csv`. If you did **not** check the
`TRANSPOSE OUTPUT TABLE` checkbox, the file will look something like [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/ROI_descriptive%20statistics_non_transposed.csv), where each row represent a specific video, ROI, animal, and measurement. If you **did** check the  `TRANSPOSE OUTPUT TABLE` checkbox, the file will look something like [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/ROI_descriptive%20statistics_transposed.csv), where each row represent a specific video
and each column represents a specific measurment.

If you did check the `DETAILED ROI BOUT DATA (SEQUENCES)` checkbox, an additional file will be crated in the SimBA project logs folder named something like `Detailed_ROI_data_20250306162014.csv` that can be expected to look something like [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/Detailed_ROI_data_20250307101923.csv). This file contains a row of information (entry and exit time, frame and entry duration) for every ROI, for every animal, and every video. It also contains columns for the animal name and body-part name for reference. 

# Part 2. Analyzing ROI data by time-bin

Once we have [drawn the ROIs](https://github.com/sgoldenlab/simba/blob/master/docs/roi_tutorial_new_2025.md), we can compute descriptive statistics based on movement of the animal(s) in relation to the ROIs stratisfied by time-bins. 
For example, we can anser how much time the animals spends in the ROIs, or how many times the animals enter the ROIs, within each N second bin in each video. 

To compute ROI statistics by time-bin, click in the <kbd>ANALYZE ROI DATA: TIME-BINS</kbd> button in the `[ ROI ]` tab and you should see the 
following pop up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/roi_analyze_tutorial_2.webp" />
</p>

1) In the `# OF ANIMALS` dropdown, select the number of animals you want to compute ROI statistics for.

2) In the `TIME BIN (S)` entry-box, enter the size of the time-bin in seconds. E.g., (60 for a minute, or a 15.5 for 15.5s time-bins)

3) In the `PROBABILITY THRESHOLD` entry-box, select the minimum pose-estimation probability score (between 0 and 1) that should be considered when performing ROI analysis. Any frame body-part probability score above the entered value will be filtered out. Enter 0.0 to use all frames regardless of pose-estimation probability score. 
> [!CAUTION]
> If possible, we recommend having reliable pose-estimation data in every frame. This includes pre-process all videos, and remove any segments of the videos where animals are not present in the video as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md),
> and performing pose-estimation interpolation of missing data at data import as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#to-import-multiple-dlc-csv-file).

4). In the `SELECT BODY-PART(S)` frame, use the dropdown menus to select the body-parts that you wish to use to infer the locations of the animals. 

5). In the `DATA OPTIONS` frame, select the data that you wish to compute:

 * `TOTAL ROI TIME (S)`: The total time, in seconds, that each animal spends in each defined ROI in each video in each time-bin. 
 * `ROI ENTRIES (COUNT): The total number of times, that each animal enters each defined ROI in each video in each time-bin. 
 * `FIRST ROI ENTRY TIME (S)`: The video time, in seconds, when each animal first enters each defined ROI in each time-bin in each video (NOTE: will be `None` if an animal never enters a defined ROI).   
 * `LAST ROI ENTRY TIME (S)`: The video time, in seconds, when each animal last enters each defined ROI  in each time-bin in each video (NOTE: will be `None` if an animal never enters a defined ROI).
 * `DETAILED ROI BOUT DATA (SEQUENCES)`: If checked, the SimBA ROI analysis generates a CSV file within the `project_folder/logs` directory named something like *Detailed_ROI_bout_data_20231031141848.csv*. This file contains the exact frame numbers, time stamps, and duration of each seqence when animals enter and exits each user-drawn ROIs. (NOTE: no file will be created if no animal in any video never enters an ROI)
 * `ROI MOVEMENT (VELOCITY AND DISTANCES)`: The total distance moved, and the average velocity, of each animal in each defined ROI  in each time-bin in each video.

6). In the `FORMAT OPTION` frame, select how the output data should be formatted, and any addtional video meta data that should be included in the output which could be helpful for sanity checks. 

* `TRANSPOSE OUTPUT TABLE`: If checked, one row in the output data will represent a video. If unchecked, one row in the output data will represent a data measurment.
* `INCLUDE FPS DATA`: If checked, the FPS used to compute the metrics of each video will be included in the output table. 
* `INCLUDE VIDEO LENGTH DATA`: If checked, the length of each video in seconds will be included in the output table.
* `INCLUDE PIXEL PER MILLIMETER DATA`: If checked, the pixel per millimeter conversion factor (used to compute distances and velocity) of each video will be included in the output table.
* `INCLUDE TIME-BIN TIME STAMPS`:  If checked, the start time, start frame, end time, and end time of each time-bin will be included in the output table. If unchecked, the time-bins will only be represeted by single integer values running from 0 -> last time bin.

7. Once the above is filled in, click the <kbd>RUN</kbd> button. You can foillow the progress in the main SimBA terminal.

Once complete, a file will be stored in the logs folder of your SimBA project named something like `ROI_time_bins_140.0s_data_20250308104641.csv`. If you did **not** check the
`TRANSPOSE OUTPUT TABLE` checkbox, the file will look something like [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/ROI_time_bins_120.0s_data_20250308110536_non_transposed.csv), where each row represent a specific video, ROI, animal, and measurement. If you **did** check the `TRANSPOSE OUTPUT TABLE` checkbox, the file will look something like [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/ROI_time_bins_120.0s_data_20250308110349_transposed.csv), where each row represent a specific video
and each column represents a specific measurment.

If you did check the `DETAILED ROI BOUT DATA (SEQUENCES)` checkbox, an additional file will be crated in the SimBA project logs folder named something like `Detailed_ROI_data_20250306162014.csv` that can be expected to look something like [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/Detailed_ROI_data_20250307101923.csv). This file contains a row of information (entry and exit time, frame and entry duration) for every ROI, for every animal, and every video. It also contains columns for the animal name and body-part name for reference. 

# Part 3. Generating features from ROI data. 

1. With the ROI information at hand, we can generate several further features that might be useful for predicting behaviors, or be handy within other third-party applications. For each frame of the video, the following features can be added to a previously calculated battery of features:

* Boolean (TRUE or FALSE) value noting if body-parts are located within or outside each of the ROIs.
* The millimeter distance between body-parts and the center of ROIs.
* The cumulative time spent in each of the ROIs.
* The cumulative percentage of the total session time spent in each of the ROIs.

Furthermore, if your project uses one of the [recommended body part pose settings](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#pose-estimation-body-part-labelling) (i.e., your project includes nose, left ear, and right ear tracking), SimBA will calculate if the animal is directing towards each of the ROIs:

<img src="https://github.com/sgoldenlab/simba/blob/master/images/Directionality_ROI.PNG" width="425"/> <img src="https://github.com/sgoldenlab/simba/blob/master/images/ROI_directionality.gif" width="425"/>

Using the directionality measure described in the image above, we can calculate further potentially informative features from each frame for downstream machine learning models:

* Boolean (TRUE or FALSE) value noting if the animal is directing towards the center each of the ROIs or not. 
* The cumulative time spent directing towards each of the ROIs.
* The cumulative percentage of the total session time spent directing towards each of the ROIs.

2. To generate these features and add them to the battery of features already calculated, begin by navigating to the  `Extract features` tab. 

>*Note:* To calculate ROI based features it is first necessery to [extract the non-ROI based features](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features). If you haven't done so already, go ahead and first click on `Extract features` under the **Extract features** tab. A message will be printed in the main SimBA terminal window when the process is complete. 



3. You should see two different buttons (i) a red `APPEND ROI DATA TO FEATURES: BY ANIMAL` and (ii) a orange `APPEND ROI DATA TO FEATURES: BY BODY-PARTS`. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/append_roi_features_032023_1.png" />
</p>

To append ROI features based on the location of **a single body-part per animal**, use the red `APPEND ROI DATA TO FEATURES: BY ANIMAL` button. 

> Note: This method has been available in SimBA for a couple of years. It is **not** possible to create multiple ROI-based features based on several within-animal body-parts using this method. 

To append ROI features based on the location of **several (or a single) body-part**, regardless of animal identities, use the red `APPEND ROI DATA TO FEATURES: BY BODY-PART` button. 

> Note: This method has been available in SimBA since 03/2023. Using this method, you can create multiple ROI-based features from multiple within-animal body-parts. E.g.,, use this method if you need features such as the distance between the center of a user-defined ROI and `Animal_1` nose as well as `Animal_2` tail-base. 

First, begin by selecting the number of animals (or body-parts) you wish to use to calculate the ROI features for, and then click `Confirm`. A second sub-menu will appear below, named `Choose bodyparts`. This menu will contain as many dropdown menus as the number of animals (or body-parts) selected in the `Select number of animals/body-parts` menu. Each of these drop-down menus will contain the body-parts used to track the animals. The body-part(s) choosen in this menu will determine which body-parts is used to calculate ROI-based features. For example, if you select the `Nose` body-part, the cumulative time spent in each of the ROIs, as well as all other features, will be calculated based on the `Nose` body-part. Once you have chosen your body-parts, then click on `Run`. 

3. Once complete, a statement will be printed in the main SimBA terminal window noting that the process is complete. New CSV files will be generated for each of the videos in the project, located in the `project_folder/csv/features_extracted` folder. If you open these files using Microsoft Excel or OpenOffice Calc (watch out, they might be fairly large and therefore difficult to navigate manually) you will see several new columns at the end of the CSV: these columns contain the ROI-based features.

> Note I: As of 04/05/2021, SimBA will also generate a summary file of the ROI-based features - the summary file can be located in the `project_folder/logs` directory. This file will be a date-time stamped CSV file named something like `ROI_features_summary_20210401144559.csv`. The file contains the mean distance to the center of each ROI for each animal in each video. If SimBA calculates *directionality* based features, then this file will also contain the sum of time (in seconds) that each animal spent directed towards each ROI in each video.   

> Note II: For a description of the different features as named in the output files, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/roi_features_examples.md)

To proceed and generate machine learning classifiers that use these features, continue reading the [Label behavior](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-6-label-behavior) step, followed by the [Train machine models](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-7-train-machine-model) parts of the SimBA tutorials. 


### Removing ROI data from the feature sets

We may have created ROI features, which have been appended to our files inside the `project_folder/csv/features_extracted` directory, and now we feel like we don't have any use for them and want to remove them. To do this click the `REMOVE ROI FEATURES FROM FEATURE SET` button. 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/remove_roi_features_features_extracted.png" width="600"/>

When clicking this button, you will be prompt to confirm that you do want to remove your ROI features. 

If clicking **YES**, SimBA will identify the ROI-based features columns inside each file within the project_folder/csv/features_extracted directory. SimBA will then remove thses columns fropm each file, and place them within the `project_folder/logs/ROI_data_(datetime)` directory. This means that can't be used within the project, and you can go ahead and delete this folder manually.  

> Note: To identfy the ROI based columns, SimBA looks inside your projects ROI definitions. Thus, if you have created ROI features, and subsequently renamed your ROI shapes, then SimBA won't be able to find and remove the ROI-based columns using this method. 

# Part 4. Visualizing ROI data

You can create visualizations of the ROIs together with counters displaying how much time time animals have spent in, and how many times the animals have entered, each ROI, throughout the videos. You can use this tool to generate visualizations of the ROI and tracking data, like in these gifs below,

<img src="https://github.com/sgoldenlab/simba/blob/master/images/ROI_visualization_1.gif" width="425"/> <img src="https://github.com/sgoldenlab/simba/blob/master/images/ROI_OF.gif" width="625"/>

To to this, click the <kbd>VISUALIZE ROI TRACKING</kbd> in the `[ ROI ]` tab and you should see this pop-up window:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/roi_visualize_popup.webp" />
</p>

1) In the ` BODY-PART PROBABILITY THRESHOLD` entry-box, select the minimum pose-estimation probability score (between 0 and 1) that should be considered when performing ROI analysis. Any frame body-part probability score above the entered value will be filtered out. Enter 0.0 to use all frames regardless of pose-estimation probability score. 
> [!CAUTION]
> If possible, we recommend having reliable pose-estimation data in every frame. This includes pre-process all videos, and remove any segments of the videos where animals are not present in the video as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md),
> and performing pose-estimation interpolation of missing data at data import as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#to-import-multiple-dlc-csv-file).

2) If you want to show the pose-estimated location(s) of the animals in the video, set the `SHOW POSE ESTIMATION LOCATIONS` dropdown to True.
> [!NOTE]
> Setting this option to False will grey out the `KEYPOINT SIZE` and `COLOR` in the body-part menu (see below)

3) `NUMBER OF CPU CORES`: This dropdown has values running from 1 to the number of CPU cores available on your computer. Higher values will produce the videos faster (but may require higher RAM). Select the number of CPU cores you wish to use to create the videos. 

4) In the `SELECT NUMBER OF ANIMAL(S)` frame, select the number of animals (and body-parts) that you want to visualize ROI data for. Note: The maximum number you can choose in this menu is the number of animals specified in your SimBA project.

5) `SELECT BODY-PARTS` frame: Select the body-parts that you wish to act as proxies for the location of the animal. If you have set the `SHOW POSE ESTIMATION LOCATIONS` dropdown to True, you can also select the color, and the size, of the circles denoting the location of the animals in each frame. Note: if KEY-POINT SIZE is set to "AUTO", then SimBA will try to auto-compute the optimal size of the bosy-part location using the resolution of your video.

6). To create a single ROI visualization video, select which video you want to create an ROI video for in the *select Video* drop-down menu* and click the <kbd>CREATE SINGLE VIDEO</kbd> button. 

7) To create ROI visualizations for all available videos in your project, click the the <kbd>CREATE ALL ROI VIDEOS</kbd> button.

You can follow the progress in the main SimBA terminal and the main opertaing system terminal from where SimBA was launched. The ROI videos are saved in the `project_folder/frames/output/ROI_analysis` directory. 

# Part 5. Visualizing ROI features

Rather than visualizing the traditional ROI data (i.e., how many times the animals have entered and how much time that has been spent inside the ROIs etc), we may want to visualize other continous and boolean ROI-features that SimBA caluclates. This includes the distances to the ROIs, if the is directing towards the ROIs, and if the animals are inside the ROIs. 

To visualize these types of data, click the `Visualize ROI features` button in the [ROI] tab, and you should see the following pop-up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/roi_feature_viz_0123_.png" />
</p>

* The first submenu is titled **SETTINGS** and allows you to specify how the videos are created and how they look, most options are ticked by default.
  - If you want to show the pose-estimated location of the animals body-parts, tick `Show pose`. 
  - If you want to mark the center of each ROI with a circle, tick `Show ROI centers`.
  - If you want to see all the [ear tags](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md#shape-attributes) of the ROIs, tick `Show ROI ear tags`. 
  - If you want to visualize the ["directionality"](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data) between animals and ROIs, tick `Show directionlity`.
  - Direction type: Chhose if you want to view "directionality" as "Funnels" or "Lines. For example, see the gifs below where the top represents ["Funnel"](https://github.com/sgoldenlab/simba/blob/master/images/Funnel_directionality.gif) and the bottom represents ["Lines"](https://github.com/sgoldenlab/simba/blob/master/images/Lines_directionality.gif).
  - Multiprocessing video (faster): Creating videos is computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this in part by using multiprocessing over the multiple cores on your computer. To use multi-processing, tick the Multiprocess videos (faster) checkbox. Once ticked, the CPU cores dropdown becomes enabled. This dropdown contains values between 2 and the number of cores available on your computer with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your visualizations.

SINGLE VIDEO: Use this menu to create a single visualization. The Video drop-down will contain all the videos in your project. Choose which video you want to create a visualization for. Once choosen, click the Create single video button. You can follow the progress in the main SimBA terminal window. Once complete, a new video will be saved in the project_folder/frames/output/ROI_features directory.
MULTIPLE VIDEO: Use this menu to create a visualization for every video in your project. After clicking the Create multiple videos button. You can follow the progress in the main SimBA terminal window. Once complete, one new video for every input video will be saved in the project_folder/frames/output/ROI_features directory.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Funnel_directionality.gif" />
</p>



<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/Lines_directionality.gif" />
</p>


# Part 6. Miscellaneous ROI tools

## Heatmaps

Users can generate heatmaps - in video or image formats - that represents the time spend in different regions of the arena. Note, this tool does **not** depend on *user-drawn* ROIs: instead this tool is for visualizing time spent in all regions of the arena according to user defined region sizes and user-defined color-scales. As an example, this tool is used for generating images, or videos, resembling this image below. For an example of the heatmap tool output in video format, check the [SimBA YouTube playlist](https://youtu.be/O41x96kXUHE). 

![](https://github.com/sgoldenlab/simba/blob/master/images/heat_1.png)

These heatmap images, or videos, are colour-coded according to the time spent in different parts of the areana. To generate such heatmaps, begin by clicking on the `Create heatmaps` button in the ROI tab:

![](https://github.com/sgoldenlab/simba/blob/master/images/Button4.PNG)

When clicking on `Create heatmaps`, the following menu pops open which accepts several required user-defined parameters:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/location_heat_maps_2023.png" />
</p>

`1. Palette`. Use the drop-down to specify the palette of the heatmaps. For reference of how the palette looks like, see the the [SimBA visualization tutorials - Step 9](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#heatmap) 

`2. Shading`. Pick the shading/smoothing (Gouraud or Flat). For a comparison of Gouraud vs Flat, see the [classifier heatmap documentation](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-classification-heatmaps).

`3. Bodypart:` SimBA can, currently, only generate heatmaps representing the time spent in different regions of the image based on a single body-part. Use the drop-down menu to select the body-part you wish to use to represent location when creating the heatmaps. 

`4. Max time scale (s) ('auto-compute' or integer):` The created heatmaps have adjoining colorbars. In this entry box, define the number of seconds that should represent the strongest color and max value in your heatmap. Users can also insert the string `auto-compute` in this entry-box. If `auto`, the SimBa will calculate the max value in the video (max time spent in a single zone) and use this value as max (I'm not entirly sure how these heatmaps are created in commercial tools but I was inspired by the [MATLAB Pathfinder tool](https://matthewbcooke.github.io/Pathfinder/).

`5. Bin size (mm):` To generate heatmaps, SimBA needs to divide the image into different square regions. Use this entry-box to define the size of each square region. To get a better sense of what a `Bin size (mm)` is, and how SimBA goes about generating the heatmaps, see the image below (1). For example, select *80x80* in this dropdown to create heatmaps from 8 by 8 cm squares.

`6. Create frames:` If ticked, will create one heatmap .png for every frame in each video. (Warning: for many large videos, you'll end up with A LOT of files) 

`7. Create videos:` If ticked, will create a heatmap video for every input video. 

`8. Crate last frame:` If ticked, will generate single images that represents the cumulative time spent in each region across the entire video, with one image per video.  

`9. Multi-process:` Creating heat-map videos and/or images is computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this using multiprocessing over multiple cores on your computer. To use multi-processing, tick the `Multiprocess videos (faster)` checkbox. Once ticked, the `CPU cores` dropdown becomes enabled. This dropdown contains values between `2` and the number of cores available on your computer, with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your probability charts. 

`10. Multi-process:` Creating heat-map videos and/or images is computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this using multiprocessing over multiple cores on your computer. To use multi-processing, tick the `Multiprocess videos (faster)` checkbox. Once ticked, the `CPU cores` dropdown becomes enabled. This dropdown contains values between `2` and the number of cores available on your computer, with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your probability charts. 

`11. SINGLE VIDEO`: Use this menu to create a single heatmap visualization video. The Video drop-down will contain all the videos in your project_folder/outlier_corrected_movement_location directory. Choose which video you want to create a heatmap visualization for. Once choosen, click the Create single video button. You can follow the progress in the main SimBA terminal window. Once complete, a new video and/or frames will be saved in the project_folder/frames/output/heatmap_location directory.

`12. MULTIPLE VIDEOS`: Use this menu to create a heatmap visualizations for every video in your project. After clicking the Create multiple videos button. You can follow the progress in the main SimBA terminal window. Once complete, one new video and/or frames folder for every input video will be saved in the project_folder/frames/output/heatmap_location directory.

Once this information is entered, click `Run`. Generated heatmap videos/frames are saved in the `project_folder/frames/output/heatmap_location` directory. 


![](https://github.com/sgoldenlab/simba/blob/master/images/Heatmap_parameters.PNG)

## 'Directionality' between animals

To analyze and visualize how animals are directing towards each other, rather than user-defined ROIs, head to [THIS TUTORIAL](https://github.com/sgoldenlab/simba/edit/master/docs/directionality_between_animals.md)

<p align="center">
  <img src="https://github.com/sgoldenlab/simba/blob/master/images/Testing_Video_3_short.gif" width="425"/>
</p>


## Compute aggregate conditional statistics from boolean fields

Sometimes we may want to compute conditional aggregate statistics from boolean fields (fields which contains `1` and `0` values). Boolean fields created from ROI analysis include fields such as if animals are inside (`1`) or outside (`0`) user-defined ROIs, and if animals are directing towards (`1`) or away from (`0`) user-defined ROIs. These computations allows us to answer questions such as:

* **For how much time, and for how many frames, in each video, was Animal 1 inside my shape called Rectangle_1 while also facing my shape names Polygon_1, while at the same time Animal_2 was outside the shape names Rectangle_1, and directed towards my shape called Polygon_1**

To compute these statistics, begin by extrating ROI features as described [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data). 

Once done, click the <kbd>AGGREGATE BOOLEN CONDITIONAL STATISTICS</kbd> button in the `ROI` tab, shown on the right in the screengrab below:

>Note: Your SimBA project needs to have files inside the `project_folder/csv/features_extracted` directory **AND** these files needs to contain at least 2 fields with Boolean variables for this to work. If these two conditions arre not fullfilled, you will be shown an error when clicking the <kbd>AGGREGATE BOOLEN CONDITIONAL STATISTICS</kbd> button.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/aggregate_boolean_conditional_stats.png" />
</p>

Once clicked, you should see the pop up menu to the right in the screengrab above. 

* In the `CONDITINAL RULES` window and #RULES drop-down, select the number of rules that you wish to use to calculate your doncitional statistics. In my example above, I have selected 4 rules. Once you have selected the number of rules, the lower frame is populated with the number of rows that you selected in the `# RULES` dropdown.

* Each row contains one dropdown menu in the `BEHAVIOR` column, and a second dropdown in the `STATUS` column. The dropdowns in the  `BEHAVIOR` column contains options for all the Boolean fields present in your data in the `project_folder/csv/features_extracted` directory. The dropdowns in the `STATUS` column contains the options `TRUE` and `FALSE`.

* In the `BEHAVIOR` column, select the behaviors that you wish to use in your conditional statistics. In the `STATUS` column, select if the behavior should be present (TRUE) or absent (TRUE). Note that a behavior selected in the dropdown in the `BEHAVIOR` column can not feature in several rows. In my screengrab above, for example, I will calculate when an animal called Simon is inside the shape called Rectangle_1 while Simon also faces the shape Polygon_1, while at the same time an animal called JJ is outside the shape Rectangle_1 and directing towards the shape Polygon_1.

* Once completed, click the run button. You can follow the progress in the main SimBA terminal window. When complete, SimBA will save a CSV file in the logs directory of your SimBA project names something like `Conditional_aggregate_statistics_{self.datetime}.csv`. This file contains one row per video in your project, where the first column represents the video name and the following columns represents the conditional rules you applied (with values TRUE or FALSE). The last two columns represents the TIME and the number of frames your conditional rule was found to be happening. See the screengrab below or click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/Conditional_aggregate_statistics_20231004130314.csv) for an example of expected output for two videos.  

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/aggregate_boolean_stats_table.png" />
</p>


##
Author [Simon N](https://github.com/sronilsson), [JJ Choong](https://github.com/inoejj)








