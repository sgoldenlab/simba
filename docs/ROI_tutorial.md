# Regions of Interest (ROIs) in SimBA

The SimBA region of interest (ROI) interface allows users to define and draw ROIs on videos. ROI data can be used to calculate basic descriptive statistics based on animals movements and locations (such as how much time the animals have spent in different ROIs, or how many times the animals have entered different ROIs). Furthermore, the ROI data can  be used to build potentially valuable, additional, features for random forest predictive classifiers. Such features can be used to generate a machine model that classify behaviors that depend on the spatial location of body parts in relation to the ROIs. **CAUTION**: If spatial locations are irrelevant for the behaviour being classified, then such features should *not* be included in the machine model generation as they just only introduce noise.   

# Before analyzing ROIs in SimBA

To analyze ROI data in SimBA (for descriptive statistics, machine learning features, or both descriptive statistics and machine learning features) the tracking data **first** has to be processed the **up-to and including the *Outlier correction* step described in [Part 2 - Step 4 - Correcting outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction)**. Thus, before proceeding to calculate ROI based measures, you should have one CSV file for each of the videos in your project located within the `project_folder\csv\outlier_corrected_movement_location` sub-directory of your SimBA project. 

Specifically, for working with ROIs in SimBA, begin by (i) [Importing your videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-import-videos-into-project-folder), (ii) [Import the tracking data and relevant videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-import-dlc-tracking-data), (iii) [Set the video parameters](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters), and lastly (iv) [Correct outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction).

>**A short explanation on what is meant by "using ROI data as features for random forest classifiers"** When ROIs have been drawn in SimBA, then we can calculate several metrics that goes beyond the coordinates of the different body-parts from pose-estimation. For example - for each frame of the video - we can calculate the distance between the different body-parts and the different ROIs, or if the animal is inside or outside the ROIs. In some instances (depending on which body parts are tracked), we can also calculate if the animal is [directing towards the ROIs](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data)(see below for more information). These measures can be used to calculate several further features such as the percentage of the session, up to and including the any frame, the animal has spent within a specific ROI. These and other ROI-based features could be useful additions for classifying behaviors in certain scenarios.

# Part 1. Defining ROIs in SimBA

1. In the main SimBA console window, begin loading your project by clicking on `File` and `Load project`. In the **[Load Project]** tab, click on `Browse File` and select the `project_config.ini` that belongs to your project. 

2. Navigate to the **ROI** tab, which should look like this:

![](https://github.com/sgoldenlab/simba/blob/master/images/ROI_11.PNG)

3. Begin at the left hand side of the **ROI** tab: using the sub-menu **Define ROI**. This menu lets you define **the number of rectangles, circles, and polygons** that you which to draw on the videos in the project. 

>*Note:* All videos in the project needs to contain the same number of shapes. For example, if you wish to analyze four ROIs in the first video of your peoject, but only three ROIs in the second video of your project, then you still need to draw four ROIs for both the first and second video in your project. 

In the entry boxes associated with each shape, enter the number of each shape that you wish to draw. If you do *not* want to draw a specific shape, then leave the entry box empty.

4. Once you've entered the number of shapes that you wish to draw, then click on the `Show Shape Definitions Table` button and the following table will pop up.

![](/images/roidef1.PNG)

In this table, enter a name for each of your shapes in the Rectangle Name, Circle Name, and Polygon Name entry boxes. Make sure that the names of each of the shapes are different from each other - e.g., there cannot be teo shapes that are both named `Area_1`. When all the shape names have been entered, click on the `Set Shape Definitions` button which is located at the bottom of the `Shape Definitions Table`.

>*Note:* Each *Circle* and *Rectangle* row has further size-related columns associated with them: Width, Height, or Radius. These columns are set to 0 and should, for now, **remain set to 0**. These settings will - **in future version of SimBA** - be used to draw shapes that are of precise metric sizes (e.g., a 40x40mm rectangle).   

5. After clicking on the `Set Shape Definitions` button, the following table will pop open.

![](https://github.com/sgoldenlab/simba/blob/master/images/ROI_menu.JPG)

This table contain one row for each of the videos in the project. Each video in the project has three buttons associated with it: **Draw, Reset,** and **Apply to all**. The functions associated with each button is described in detail below. However, in brief, the **Draw** button allows you to start to draw the defined shapes on the specific video. If drawings already exist for the specific video, then the **Draw** buttons opens an interface where you can move the shapes that has previously been drawn on the video. The **Reset** button deletes any drawing made on the specific video and allows you to restart the drawings from scratch by next clicking on the **Draw** button. The **Apply to all** buttons copies the drawing made on the specific video and replicates it on all other videos in the project. If a drawing has been applied to all videos videos in the project by clicking on the **Apply to all** button, then the shapes for any specific video can be moved by clicking on the **Draw** button.

6. To begin to draw your shapes, click on the **Draw** button for the first video in the table. Once clicked, a window will first pop up that contains instructions for how to draw your shapes. The instructanctions for drawing rectangles, circles, and polygons are slightly different and are descriped below. 

Any user-defined regles will be processed first. The instructions for drawing rectangles will look like the image below on the left, and the process of drawing the rectangle should look like the gif image below on the right:

<img src="https://github.com/sgoldenlab/simba/blob/master/images/ROI_rectangle.JPG" width="425"/> <img src="https://github.com/sgoldenlab/simba/blob/master/images/Rectangle_draw.gif" width="425"/>

In this example, to begin to draw the rectangle named "TopLeft" on Video1, press `Esc`. When `Esc` is pressed, the first frame of Video1 is displayed. Click and hold the left mouse button at the top left corner of rectangle "TopLeft" and drag the mouse to the bottom right corner of the rectangle. If you're unhappy with your rectangle, start to draw the rectangle again by holding the left mouse button at the top left corner of your, new, revised, rectangle. The previous rectangle will be automatically discarded. When you are happy with your rectangle, press `Esc` twice to save your rectangle and move to your next shape. 

>*Note*: If the image of the first frame of the video looks distorted, and the aspect ratio appears to be off, then grab the side of the window that contains your frame and correct the aspect ratio. If the first frame of your video contains the experimenter hand, or no animal(s) are visible (because the camera was turned on before the animal(s) was placed in the arena), then consider going back and [pre-process your videos appropriately](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md#tutorial-for-processing-videos-in-batch). 

Once all user-defined rectangles have been drawn, any user-defined circles will be processed. The instructions for drawing circles will look like the image below on the left, and the process of drawing the a circle should look like the gif image below on the right:

<img src="https://github.com/sgoldenlab/simba/blob/master/images/ROI_circles.JPG" width="425"/> <img src="https://github.com/sgoldenlab/simba/blob/master/images/Circle_draw.gif" width="425"/>

In this example, to begin to draw the circle named "Cage" on Video1, press on `Esc`. When `Esc` is pressed, the first frame of Video1 is displayed. If you have drawn any rectangles previously, then these rectangles will also be displayed on the frame. Begin by double left mouse clicking on the image where you would like the center of the circle to be. Next, double left click on the image where you would like the outer bound of the circle to be. If you are unhappy with the center of the circle, then double left mouse click on the center of the circle, and double left mouse click again at the new circle center location. Likewise, if you are unhappy with the outer bounds and the diameter of the circle, then double left mouse click on outer bounds of the circle, and then double left click again at the new circle outer bounds location. When you are happy with your circle, press `Esc` to save your circle and move to the next user-defined  shape. 

Once all the circles have been drawn, any user-defined polygons will be processed. The instructions for drawing polygons will look like the image below on the left, and the process of drawing the polygon should look like the gif image below on the right:

<img src="https://github.com/sgoldenlab/simba/blob/master/images/ROI_polygons.JPG" width="425"/> <img src="https://github.com/sgoldenlab/simba/blob/master/images/Polygon_draw.gif" width="425"/>

In this example, to begin to draw the polygon named "Polygon1" on Video1, press `Esc`. When `Esc` is pressed, the first frame of Video1 is displayed. If you have previously drawn any rectangles or circles, then these rectangles and/or circles will also be displayed on the frame. Click on at least three different locations in the image that defines the outer bounds of "Polygon1". SimBA does currently not allow re-definitions of the bound locations of the polygon once they have been specified. If you are unhappy with your drawn polygon, you can chose to either (i) move the entire polygon in a later step (see Step 6, just below), or **Reset** the drawings and start again (again see Step 6, just below). When you have finished your polygon, press `Esc` to save your polygon and move to the next user-defined shape (or to finish, if all user-defined shapes have been drawn).

6. If you have finished the drawings for Video1, and you are not entirely happy with the drawings, you can either: (i) click on the **Reset** button next to Video1: this will delete the drawings you have made on Video1, and you can now again click on **Draw** button next to Video1 to restart your drawing as described in Step 5 above. Alternatively, you can click on **Draw** again **without** first clicking on the **Reset** button. This allows you to move the shapes in Video1 that you have previously drawn.

When you click on **Draw** again, **without** first clicking on the **Reset** button, then the following instruction window will pop open (below left), and moving the shapes should look like this (below right):

<img src="https://github.com/sgoldenlab/simba/blob/master/images/ROI_move.JPG" width="425"/> <img src="https://github.com/sgoldenlab/simba/blob/master/images/Move_shapes.gif" width="425"/>

To start moving the shapes that has previously been drawn, first click on the centroid belonging to the shape that you wish to move. This will initially cause the shape to disappear. Next, double left click on the new centroid location of the shape you are moving. This will display the shape in its new location. When you are happy with the locations of all of your shapes, then press on `Esc` to save the new shape locations.  

7. Once all the shapes have been drawn on a single video there are two possible routes to getting the shapes drawn on all of the videos in the project. The first alternative, which is likely the more time-consuming option, is to click on each of the **Draw** buttons associated with each video, and replicate **Step 5** for all of the videos in your project. The second route, and likely the faster and more standardized option, is to replicate the shapes drawn on Video1 in all of the videos of the project, and then nudge them to the correct location in each of the videos (if the camera/arena moved *slightly* across recordings).

>*Note:* If the camera moved **significantly** across the different recordings in the project, the first alternative route described above is the only option to generate accurate all the ROI shapes in each video. 

To replicate the ROI shapes drawn in Video1 to all of the videos in the project, click on the **Apply to all** button next to Video1. Once done, click on **Draw** for Video2. When you click on **Draw** for Video2, a window will pop open with Instructions for how to move shapes - press `Esc` to proceed. When pressing `Esc` the first frame of Video2 will be displayed. Now move the shapes so that they are in the correct locations (as described in Step 6 above), and press `Esc` to save the locations of the shapes.

Repeat this step for all of the videos in the project. Once complete, close the **ROI Table** window, and navigate back to the **ROI** tab in the **Load project** window.

>*Note*: If you at any point want to look at, or modify, the defined ROI shapes for each of the videos, click on the `Load defined ROI table` button in the **ROI** tab under the `Load project` menu. 

![](https://github.com/sgoldenlab/simba/blob/master/images/Load_ROI_table.JPG)

Likewise, if you'd like to delete all the shapes you have drawn for all of the videos and start fresh with new shapes, go back and repeat the above [Part 1 - Step 3](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-1-defining-rois-in-simba).

# Part 2. Analyzing ROI data.

1. You can now generate CSV files containing descriptive statistics for the the number entries into each ROI shape, and the time spent within each ROI. SimBA allows you to generate these measures for one body-part per animal in your project. If you would like to get these measures for more than one body-part per animal in your project, then we suggest that you simply run the analysis multiple times. 

2. To analyze your ROI data, first click on `Analyze ROI data` in the `Analyze ROI` sub-menu. The following menu will pop open:

![](https://github.com/sgoldenlab/simba/blob/master/images/ROI_analyze1.jpg)

Here, begin by selecting the number of animals (or body-parts) you wish to produce the ROI descriptive statistics for, and then click confirm. A second sub-menu will appear below named `Choose bodyparts`. This menu will contain as many dropdown menus as the number of animals (or body-parts) selected in the  `Select number of animals` menu. Each of these drop-down menus will contain the body-parts used to track the animals in pose-estimation. Select the body-parts you wish to use when calculating the ROI entry, time, and movement data.

At the bottom of this window, you'll see an entry box named `Bp probability threshold` (the first arrow in the image below) and a tick box option named `calculate distance moved within ROI` (the second arrow in the image below).

![](https://github.com/sgoldenlab/simba/blob/master/images/ROI_0320_1.png)

**`Bp probability threshold`**: If you can, we **strongly** recommend to pre-process all videos, and remove any segments of the videos where animals are not present in the video as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md) prior to perfroming pose-estimation and importing videos into SimBA. However, if this is not possible, we can filter out out frames where probability for the pose-estimation accuracy is low, and not use these frames when we calculate the location of the animal in relation to the ROI (i.e., useful if you have an animal leaving the frame and the pose-estimation predictions are jumping all over the place). To filter out these frames, increase the probability threshold (DEFAULT = 0.0. - i.e., all frames will be treated as the animal is present). 

**`calculate distance moved within ROI`**: This tick-box option allows you to calculate the distance (in centimeters) that each animal has moved within each of your ROIs. If you run the analysis with this box ticked, SimBA will generate a time-stamped CSV file inside your project log folder named something akin to this: `Movement_within_ROIs_20210320143608.csv`. This file contains one row per analyzed video, animal, and ROI, and one . If you open this CSV up after analysis, you can expect it to look something like this (click on the image to enlarge):

![](https://github.com/sgoldenlab/simba/blob/master/images/movement_inside_ROI_0622.png)

This CSV file tells me - for example - that in `Video10`, the animal named `Animal_1` moved 58 cm within the ROI shape named `My_rectangle` (looking in cell E0). 

Click `Run` to perform the ROI analysis. 

Once complete, a statement will be printed in the main SimBA terminal window noting that the process is complete. The ROI descriptive statistics can be found in the `Project_folder/logs` directory in two time-stamped files. 

* One file will be named `ROI_entry_data` and contain the number of entries into the different ROIs. For an expected output ROI entry CSV file, click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/ROI_entry_data_example.csv).

* A second file, named `ROI_time_data`, contain the time spent in seconds the different ROIs. It also contains the percent of the session time spent in each of the ROIs. For an expected output ROI time CSV file, click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/ROI_time_data_example.csv).

*Note*: As of June-27 2022, the SimBA ROI analysis also generates a CSV file within the `project_folder/logs` directory named **`Detailed_ROI_data`**. This file contains additional information, with exact frame numbers of when animals enter and exits user-drawn ROIs. Every row in this file represents and entry and exit by an animal into an ROI. Within this file, each row is indexed by the (i) video name, (ii) animal name, and (iii) ROI name. The last two columns are name `Entry_frame`, and `Exit_frame`, and represents the frame number that the animal eneterd and exited the ROI. The file looks something like this:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/detailed_roi_data_0622.png" />
</p>

This CSV file tells me - for example - that in `Video10`, the animal named `Animal_1` entered the ROI shape named `My_rectangle` at frame 97 and exited the ROI shape named `My_rectangle` at frame 102. For an example of of how this file looks like, check out [THIS CSV FILE](https://github.com/sgoldenlab/simba/blob/master/misc/Detailed_ROI_data_20220627045851.csv).

Consult these files if you want to know the exact frame numbers when animals entered and exited your drawn ROIs. If you want to convert the frame numbers to time-stamps, then divide these numbers with the fps of the video. 

3. If you want to further calculate time-spent and the number of entries made into the user-defined ROIs within **user-specified time-bins**, go ahead and click on **Time bins: Analyze ROIs** located in the *Analyze ROI* sub-menu. Clicking on this button brings up this menu:

![](https://github.com/sgoldenlab/simba/blob/master/images/Roi_time_3.png)

Select (i) how many animals you want to calculate time-bin ROI data for, (ii) which body-parts you want to use to infer the location of the animals, (iii) and the size of each time bin in seconds. Once the information is filled in click on `Analyze`. You can follow the analyis progress in the SimBA main terminal window. Once the analysis is complete, the data will be saved in two date-time stamped CSV file the `project_folder/log` directory with names containing the size of the time-bins. The file names could be something like `ROI_time_bins_60s_entry_data_20210713164401.csv` and `ROI_time_bins_60s_entry_data_20210713162124.csv`, with one file containing the entry data and the other file containing the latency data. The file should contain 7 columns, and the top of the time file look similar to this:


![](https://github.com/sgoldenlab/simba/blob/master/images/roi_time_9.png)

The first column `Video` gives the name of the video. The second column `Time bin #` gives you the ordinal value of the time bin. The third column `Time bin start` gives you the start time of the time bin in seconds. The fourth column `Time bin end` gives you the end of the time bin in seconds. The fifth column `Animal` gives you the animal ID. The sixth column `ROI` gives you the ROI name. The seventh and last column `Time` gives you the time latency count. For example, if you want to know how much time `Animal_1` in `Video_1` spend in `Circle_1` in `Time bin 1`, look at row 10 (4.57s) in the CSV above. To download the full example file shown in the image above, click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/ROI_time_bins_60s_time_data_20210713164401.csv).

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

You can now generate visualizations of the ROI data for each of the videos in your project. To generate the visualizations, click on `Visualize ROI tracking` in the `Visualize ROI` submenu of the **ROI** tab. You can use this tool to generate visualizations of the ROI and tracking data, like in these gifs:

<img src="https://github.com/sgoldenlab/simba/blob/master/images/ROI_visualization_1.gif" width="425"/> <img src="https://github.com/sgoldenlab/simba/blob/master/images/ROI_OF.gif" width="425"/>

Clicking on this button brings up the below menu: 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/roi_viz_012023_2.png" />
</p>

The first sub-menu is called **SETTINGS**. In the *Body-part probability threshold* entry-box, fill in the minimum pose-estimation confidence required for a body-part to be plotted (a value between 0.0 and 1.0). E.g., set this value to `0.0` to plot all body-parts in all frames. Alternatively, set this value to `1.0` to plot only the body-parts that your pose-estimation tool is most condident about. 

There are two checkboxes that are auto-ticked when opening the menu: (i) `Show pose-estimated location`, and (ii) `Show animal names`. To remove the "circles" representing the location of the body-parts in the video, untick the the `Show pose-estimated location` check-box. To remove the names of the animals in the video, untick the the `Show animal names` check-box. Four example outputs are shown in the GIF below. (TOP LEFT: `Show pose-estimated location` and `Show animal names` are both **ticked**. TOP RIGHT: `Show pose-estimated location` and `Show animal names` are both **un-ticked**. BOTTOM LEFT: `Show pose-estimated location` is **ticked** and `Show animal names` is **un-ticked**. BOTTOM RIGHT: `Show pose-estimated location` is **un-ticked** and  `Show animal names` is **ticked**.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/merged_video_20230125154753.gif" />
</p>


Next, there is a `Multi-process (faster)` check-box. If you are on a computer with a high CPU count, you may want to take advantage of several or all of to cores to generate your videos in parallel. This way you could make hefty saves on run-time. To take advantage of many or all cores, check the `Multi-process (faster)` check-box and choose how many cores you want to use in the `CPU cores drop-down` menu. 

To create a single ROI visualization video, click the **Create SINGLE ROI Video** button. Select which video you want to create in the *select Video* drop-down menu*. 

To create many ROI visualization videos, click the **Create ALL ROI Videos** button.

You can follow the progress in the main SimBA terminal. The ROI videos are saved in the `project_folder/frames/output/ROI_analysis` directory. 


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








