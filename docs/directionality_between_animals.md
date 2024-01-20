# Who is looking at who: calculate "directionality" between animals in SimBA 

In videos containing multiple animals, it may be valuable to know when, and how much time, each animal spend directing towards each other or towards stationary objects. We can calculate this with appropriate pose-estimation data in SimBA. For a better understanding of this kind of data, and if it is relevant for your specific application, see the videos and gifs below and the rendered visualization examples at the end of this tutorial. Here we use SimBA to get measures such as:

* **Detailed boolean data**: For example, for each frame and animal in the video, which body-parts belonging to other animals are within the observing animals  *"line of sight"* and which body-parts belonging to other animals are **not** within the observing animals  *"line of sight"*.

* **Detailed summary data**: For example, when a body-part of another animal is within *"line of sight"* for an observing animal, what is the approximate in pixel coordinates of the observing animals eye, and where is the observed animals body-part in pixel coordinates.

* **Aggregate data**: For example, how many seconds is animal A observing animal B in the video, and how many seconds is animal B observing animal A in the video?

We can also use SimBA to generate visualizations of these metrics, so that we can be comfortable with that the "directionality" data that is summarized in the created CSV files and dataframes are accurate. 

>Note 1: Firstly, and importantly, SimBA does not calculate actual *gaze* (this is *not* possible using **only** pose-estimation data). Instead, SimBA use a proxy calculation for estimating if the animal is directing toward another animal, or [a user-defined region of interest](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data), using the nose coordinate and the two ear-coordinates (or the equivalent coordinate on any other animal species). For more information on how SimBA estimates the location of the animals *eyes*, and how SimBA estimates if an object, or another animal, is within the *line of sight* of the specific animal, check out the [SimBA ROI tutorial - Part 3](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data).

>Note 2: The pose-estimation tracking of the two *ears*, and the *nose*, of each animal, or the equivalent in other species, is required for this to work. See more information on this pre-requisite below. Furthermore, see the below image and its legend for more information on how the code in SimBA estimates the relationships between animal body-parts and/or user-defined ROIs. 

<div align="center">
  
[Click here for better-quality rendered MP4](https://www.youtube.com/watch?v=d6pAatreb1E&list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl&index=20&t=0s)

</div>

[![Video example: SimBA YouTube playlist](https://github.com/sgoldenlab/simba/blob/master/images/Direct_10.JPG)](https://youtu.be/d6pAatreb1E "Video example: SimBA YouTube playlist")


<p align="center">
  <img src="https://github.com/sgoldenlab/simba/blob/master/images/Testing_Video_3_short.gif" width="425"/>
</p>

<p align="center">
  <img src="https://github.com/sgoldenlab/simba/blob/master/images/Directionality_ROI.PNG" width="425"/>
</p>



# Before analyzing directionality in SimBA

To analyze directionality data in SimBA (for descriptive statistics, machine learning features, or both descriptive statistics and machine learning features) the tracking data **first** has to be processed the **up-to and including the *Outlier correction* step described in [Part 2 - Step 4 - Correcting outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction)**. Thus, before proceeding to calculate directionality-based measurements between animals in SimBA, you should have one CSV file for each of the videos in your project located within the `project_folder\csv\outlier_corrected_movement_location` sub-directory of your SimBA project. 

Specifically, for working with directionality between animal in SimBA, begin by (i) [Importing your videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-import-videos-into-project-folder), (ii) [Import the tracking data and relevant videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-import-dlc-tracking-data), (iii) [Set the video parameters](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters), and lastly (iv) [Correct outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction)

# Calculate what other body-parts and what other animals, are in the line of sight for specific animal.   

1. In the main SimBA console window, begin loading your project by clicking on `File` and `Load project`. In the **[Load Project]** tab, click on `Browse File` and select the `project_config.ini` that belongs to your project. 

2. Navigate to the ROI tab. On the right hand side, within the `Analyze distances/velocity` sub-menu, there are two buttons (circled in red in the image below) - (i) `Analyze directionality between animals` and (ii) `Visualize directionality between animals`.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/directionality_tutorial_1.png" />
</p>

3. We will begin clicking on the `ANALYZE DIRECTIONALITY BETWEEN ANIMALS` button. This will bring up a pop-up menu with three ouput options. Select which data output format you would like.
  
* **CREATE BOOLEAN TABLES**: Creates one CSV file per video in your project, with one row per frame in the video, and where each column represents a **OBSERVING ANIMALS -> OTHER ANIMAL BODY-PART** relationship. The columns are populated with `0` and `1`s. A `1` means that the observing animals has the other animals body-part in "line of sight", a `0` means that is is **not** in "line of sight". For an example of expected output boolean table, see [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/boolean_directionaly_example.csv) CSV file. 

* **CREATE DETAILED SUMMARY TABLES**: Creates one CSV file per video in your project, with one row for every "line of sight" observation detected in the video, as in the screengrab below. The `Video` columns denotes the video the data was computed from. The `Frame_#` columns denotes the frame number the "line of sight" observation. The `Animal_1` columns represents the observing animal. The `Animal_2` column represents the animal beeing observed. The `Animal_2_body_part` represents the body-part of `Animal_2` that is beeing observed. The `Eye_x` and	`Eye_y` columns denotes the approximate pixel location of the `Animal_1` eye that observed the `Animal_2_body_part`. The `Animal_2_bodypart_x` and `Animal_2_bodypart_y` columns represents the location of the `Animal_2_body_part` that is observed by `Animal_1`.  For a full example of expected output of the detailed summary tables, see [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/detailed_summary_directionality_example.csv) CSV file. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/directionality_tutorial_2.png" />
</p>

* **CREATE AGGREGATE STATISTICS TABLES**: Creates one CSV file with aggregate statistics, with each row representing a relationship between two animals within a video and the aggregate seconds that one animal was within the "line of sight" of another animal. The aggregate seconds is computed by summing all the frames that where one or more body-part of the observed animals was in the "line of sight" of the observing animal. For an example of expected output of the aggregate summary table, see [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/direction_data_aggregates_example.csv) CSV file. 

>Note: These aggregate summary statistics are calculated based on all body-parts for each animal. If an animal is directing **any** body-part belonging to a specific other animal, then the animal is directing towards that other specific animal. For example, even if the animal with the ID *Simon* only has the tail-base of the animal with the ID JJ in the *line of sight* in one frame, the animal with the ID *JJ* is still counted as being in the *line of sight* in of the animal with the ID *Simon* in the aggregate statistics table. 

* **APPEND BOOLEAN TABLES TO FEATURES**: At times, we may want to append the [boolean directionality tables](https://github.com/sgoldenlab/simba/blob/master/misc/boolean_directionaly_example.csv) to our feature inside the `project_folder/csv/features_extracted` directory. When we do this, SimBA will identify them as features and use these fields when it comes to creating supervides machine learning classifiers. To do this, check the `APPEND BOOLEAN TABLES TO FEATURES` checkbox.

>Note 1: When this checkbox is checked, the `CREATE BOOLEAN TABLES` checkbox is automatically checked. Thus, you **cannot** append the boolean tables to your features without computing the boolean tables.

>Note 2: For this to work, all files inside the `project_folder/csv/outlier_corrected_movement_location` directory has to have an equivalently named file inside the `project_folder/csv/features_extracted` directory with the same number of rows. For examples, he directionality statistics is compututed using the file `project_folder/csv/outlier_corrected_movement_location/Video_1.csv`. After it has been computed, it will be appended to the `project_folder/csv/features_extracted/Video_1.csv`. If this file does not exist, or contains more or less rows than the `project_folder/csv/outlier_corrected_movement_location/Video_1.csv` file, it cannot be appended. 

Once you filled in your selected output formats, click `RUN`. You can follow the progress in the main SimBA terminal. The results will be saved in the `project_folder/logs` directory of your SimBA project. 

4. Next, we may to visualize these directionality data - for peace of mind - that the metrics seen in the output desriptive statistics and CSV files outlined above are accurate and plausable. To do this, go ahead and click on the second button described above - `VISUALIZE DIRECTIONALITY BETWEEN ANIMALS` and you should see this pop-up allowing control over how the output vides are created:


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/directionality_tutorial_3.png" />
</p>

* `Shows pose-estimated body-parts`. If checked, the ouput video will include circles denoting the predicted location of the animal body-parts. If not checked, no pose-estimated body-part locations are shown.

* `Highlight direction end-points`: If checked, SimBA will emphasize that a body-part is observed by showing the observed body-part using a salient circle color and size. 

* `Polyfill direction lines`: If checked, SimBA will highlight the relationship between the "eye" and the observed body-parts with a polyfill "funnel-style" visualization. If unchecked, simpler lines will be shown.

* `Direction color`: The color of the lines or polyfill representing observating relationships. If `Random`, then a random color will be selected.

* `Pose circle size`: The size of the circles denoting the pose-estimated body-part locations.

* `Line thickness`: The size of the lines denoting observating relationships.

* `Multi-process (faster)`: Ceating videos can be computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this using multiprocessing over multiple cores on your computer. To use multi-processing, tick the Multiprocess videos (faster) checkbox. Once ticked, the CPU cores dropdown becomes enabled. This dropdown contains values between 2 and the number of cores available on your computer with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your visualizations.

Once complete, you can find your final rendered videos in your `project_folder\frames\output\ROI_directionality_visualize` folder. For higher-quality examples of the expected final output videos for experimental scenarious containing five animals, or two animals, see the SimBA [SimBA YouTube playlist](https://www.youtube.com/playlist?list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl). 



[Example 1 - 5 mice](https://www.youtube.com/watch?v=d6pAatreb1E&list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl&index=20&t=0s)
![](https://github.com/sgoldenlab/simba/blob/master/images/Testing_Video_3_short.gif)


[Example 2 - 2 mice on YouTube](https://www.youtube.com/watch?v=tsOJCOYZRAA&list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl&index=19&t=0s)
![](https://github.com/sgoldenlab/simba/blob/master/images/Together_2.gif)


Author [Simon N](https://github.com/sronilsson)




















 



