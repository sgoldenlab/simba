# Who is looking at who: calculate "directionality" between animals in SimBA 

In videos containing multiple animals, it may be valuable to know when, and how much time, each animal spend directing towards each other or objects. We can calculate this with appropriate pose-estimation data in SimBA. For a better understanding of this kind of data, and if it is relevant for your specific application, see the videos and gifs below and the  rendered visualization examples at the end of this tutorial. Here we use SimBA to get measures, in seconds, of much time *Animal 1* spend directing towards *Animal 2*, and how much time *Animal 2* spend directing towards *Animal 1* (... and so on for all the relationships for all the animals tracked in each video). We can also use SimBA to generate visualizations of these metrics, so that we can be comfortable with that the "directionality" data that is summarized in the created CSV files and dataframes are accurate. 

>Note 1: Firstly, and importantly, SimBA does not calculate actual *gaze* (this is *not* possible using **only** pose-estimation data). Instead, SimBA use a proxy calculation for estimating if the animal is directing toward another animal, or [a user-defined region of interest](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data), using the nose coordinate and the two ear-coordinates (or the equivalent coordinate on any other animal species). For more information on how SimBA estimates the location of the animals *eyes*, and how SimBA estimates if an object, or other animal, is within the *line of sight* of the specific animal, check out the [SimBA ROI tutorial - Part 3](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data).

>Note 2: The pose-estimation tracking of the two *ears*, and the *nose*, of each animal, or the equivalent in other species, is required for this to work. See more information on this pre-requisite below. Furthermore, see the below image and its legend more information on how the code in SimBA estimates the relationships between animal body-parts and/or user-defined ROIs. 

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

![](https://github.com/sgoldenlab/simba/blob/master/images/Directionality_99.PNG)

3. We will begin clicking on the `Analyze directionality between animals` button. Clicking on this button will calculate how much time each animal spend directing towards each other within all of your videos represented as CSV files within in your `project_folder/csv/outlier_corrected_movement_location` directory. Again, for more information on how SimBA calculates directionality, check out [this schematic](https://github.com/sgoldenlab/simba/raw/master/images/Directionality_ROI.PNG). This schematic depicts a user-defined ROI, but exactly the same methods is applied when using body-parts rather than a user-defined ROI regions. 

Once you have clicked the `Analyze directionality between animals` button, you can follow the calculation progress in the main SimBA terminal window. 

Once the calculations are complete, several new files will be generated in your SimBA project directory. First, a new folder has been created in your project directory. If you look within the `project_folder/csv/directionality_dataframes` directory, you should see one new file for each of the videos in your project, each file should have the name of the video that was analyzed. If you open up one of these files, then you should see one row for each frame in your video, together with a bunch of columns representing the different animal relationships. For each animal (and by *animal* I mean the animal "looking" at something) and target (and by *target* I mean the body-part being "looked" at or directed towards), there will be a total of 5 columns. If we look at the **first five** column in example image just below, then the **first of these five columns** has the header *Simon_directing_JJ_Nose* and contains boolean 0 or 1 values. If the specific animal (in this case Simon) directed towards the specific body-part (in this case JJs's nose) in that frame (i.e., the animal called Simon had the animal JJs nose within the line of sight) then the column will read `1`. Conversely, if the animal (Simon) did **not** direct towards that body-part (JJ's nose) in that frame (i.e., JJ's nose was **not** within Simon's line of sight), then the column reads a `0`. Thus, in this example image below, animal JJs nose **was** within the line of sight of the animal Simon in frame 23-40, but was **not** within the line of sight of Simon in frame 0-22. 

Next, the preceeding four columns are primarily saved for visualization purposes, should the user want to verify the data by genering rendered videos of the different "line of sight" paths. The **second** and **third** column (*Simon_directing_JJ_Nose_eye_x* and *Simon_directing_JJ_Nose_eye_y*) stores the coordinates of animal *Simon's* directing eye when the specific body-part was within the line of sight. If the specific body-part was **not** within the line of sight, then the specific row for these columns will read `0`. Lastly, the **fourth** and **fifth** column (*Simon_directing_JJ_Nose_bp_x* and *Simon_directing_JJ_Nose_bp_y*) stores the coordinates of the body-part being observed by animal Simon. If the specific body-part was **not** within the line of sight, then the specific row for these columns will also read `0`. 

![](https://github.com/sgoldenlab/simba/blob/master/images/Directionality_98.PNG)

Perhaps more importantly, SimBA generates a CSV log file that contains summary statistics. This includes how much time each animal spent directing towards all other animals present in the video. This file can be found in the `project_folder/logs` directory. This filename of this file is time-stamped, and have a name that may look something like this: `Direction_data_20200822151424.csv`. The content of this file should look like the image just below (if you are tracking 5 animals. if you are tracking fewer animals, then the file will contain fewer columns):

![](https://github.com/sgoldenlab/simba/blob/master/images/Directionality_97.PNG)
**(click on this image for enlarged view)**

Thi log file will contain one column for each animal relationship, and one row for each analysed video. In the example screenshot above, the first column header reads *JJ_directing_Simon_s* and contains the time, in seconds, that the animal with the ID *JJ* spent directing towards the animal with the ID *Simon* (92.2s in the video named Video1). In column F (or column 5), with the header *Simon_directing_JJ_s*, we can read the time, in seconds, that the animal with the ID *Simon* spent directing towards the animal with the ID *JJ* (107.3s in Video1). Indeed, studying the directionality results in all of the four videos in the example screenshot above, the animal *Simon* does seems a little more intrested in directing towards the animal named *JJ*, than the animal *JJ* is interested in directing towards the animal named *Simon*. 

>Note: These summary statistics are calculated based on all body-parts for each animal. If an animal is directing **any** body-part belonging to a specific other animal, then the animal is directing towards that other specific animal. For example, even if the animal with the ID *Simon* only has the tail-base of the animal with the ID JJ in the *line of sight* in one frame, the animal with the ID *JJ* is still counted as being in the *line of sight* in of the animal with the ID *Simon*. 

4. Next, we may to visualize these directionality data - for peace of mind - that the metrics seen in the output desriptive statistics and CSV files outlined above are accurate and plausable. To do this, ga ahead and click on the second button described above - `Visualize directionality between animals`.

>Note: Rendering videos are time-consuming and computaionally expensive. If you have many videos, many animals, and high fps/resolution, it might take some time. You can follow the progress in the main SimBA terminal window. 

Once complete, you can find your final rendered videos in your `project_folder\frames\output\ROI_directionality_visualize` folder. For higher-quality examples of the expected final output videos for experimental scenarious containing five animals, or two animals, see the SimBA [SimBA YouTube playlist](https://www.youtube.com/playlist?list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl). 



[Example 1 - 5 mice](https://www.youtube.com/watch?v=d6pAatreb1E&list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl&index=20&t=0s)
![](https://github.com/sgoldenlab/simba/blob/master/images/Testing_Video_3_short.gif)


[Example 2 - 2 mice on YouTube](https://www.youtube.com/watch?v=tsOJCOYZRAA&list=PLi5Vwf0hhy1R6NDQJ3U28MOUJPfl2YWYl&index=19&t=0s)
![](https://github.com/sgoldenlab/simba/blob/master/images/Together_2.gif)



























