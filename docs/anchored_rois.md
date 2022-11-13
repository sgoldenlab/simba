# <p align="center"> Animal-anchored (ROIs) in SimBA </p>

The animal-anchored region-of-interest (ROI) interface allows users to define bounding boxes (or circles) around pose-estimated animal key-points. Once defined, we can calulate how often and when the different bounding boxes and key-points intersect which each other, in order to infer when and how animals interact with each other. 

# Before analyzing ROIs in SimBA

To analyze anchored-roi data in SimBA (for descriptive statistics, machine learning features, or both descriptive statistics and 
machine learning features) the tracking data **first** has to be processed the **up-to and including the 
*Outlier correction* step described in [Part 2 - Step 4 - Correcting outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction)**. 
Thus, before proceeding to calculate anchored-ROI based measures, you should have one file for each of the videos in your 
project located within the `project_folder\csv\outlier_corrected_movement_location` sub-directory of your SimBA project.

Specifically, for working with anchored-ROI in SimBA, begin by 
(i) [Importing your videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-import-videos-into-project-folder), 
(ii) [Import the tracking data and relevant videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-import-dlc-tracking-data), 
(iii) [Set the video parameters](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters), 
and lastly (iv) [Correct outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction) (or click to indicate that you want to *Skip outlier correction* as detailed in the Correct outliers tutorial)


# ANALYZING ANIMAL-ANCHORED ROIs IN SIMBA

## FINDING THE ANIMAL-ANCHORED ROI'S

Before analyzing animal-anchored ROI data in SimBA, we need to define the shape of, and find, the bounding boxes that define each animal in each frame. To do this, click on the `Find animal boundaries` button in the `SIMBA ANCHORED ROI` pop-up window displayed in the screengrab above. 

This brings up a further pop-up window named `FIND ANIMAL BOUNDARIES` with a single drop-down menu named `SELECT SHAPE TYPE`. The dropdown menu has **three** options, (i) Entire animal, (ii) Single body-part square, and (iii) Single body-part circle. More information on each option follows below, but in brief: 
  
  * When choosing  *Entire animal*, all of the animal body-parts will be inside the animal-anchored ROI.
  * When choosing *Single body-part square*, a single user-defined body-part will be inside a square animal-anchored ROI.  
  * When choosing *Single body-part circle*, a single user-defined body-part will be inside a circular animal-anchored ROI. 

If choosing **Entire animal** in the `SELECT SHAPE TYPE` dropdown the settings menu on the **left** in the screen-grab below will be show up. If choosing **Single body-part circle** or **Single body-part square**  in the `SELECT SHAPE TYPE` dropdown the settings menu on the **right** in the screen-grab below will be show up. We will first go through the settings for **Entire animal** based bounding boxes, followed by body-part anchored bounding boxes. 

### ANCHORED-ROI SHAPE TYPES - ENTIRE ANIMAL BASED BOUNDING BOXES

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/bounding_boxes_example_2.png" />
</p>  

CLICK [HERE](https://github.com/sgoldenlab/simba/blob/master/images/bounding_boxes_example_1.png) FOR A SIMILAR SHAPE-TYPE REFERENCE TABLE FOR NON-SHAPE SHIFTERS (MICE) 

Entire animal-based bounding boxes has two user-defined parameters:

1. FORCE RECTANGLE: Rather then defining each animals ROI through polygons, we can force the polygon to its minimim bounding rectangle. If you want each ROI to be a rectangle rather than a polygon, then check the `FORCE RECTANGLE` checkbox. If you would want the animals ROi to be defined as a polygon, then un-check the `FORCE RECTANGLE` checkbox. 

2. PARALLEL OFFSET: Sometimes we may not want to draw our animal-anchored ROIs exacactly by the outer-bounds of the animal hull key-points. Instead, we may want to introduce a little extra wriggle room that is included in the animals personal space. If you want to introduce a little extra room inside the animals ROI, then enter the size of that space in the `PARALLEL OFFSET` entry box in millimeter. 

### SINGLE BODY-PART BASED BOUNDING BOXES.

Single body-part based bounding also has two user-defined parameters:

1. BODY-PARTS: For each animal, we need to define which body-part the ROI should be anchored to. When selecting **Single body-part circle** or **Single body-part square** in the `SELECT SHAPE TYPE` dropdown, you should see a row for each animal in your SimBA project. Each row has a drop-down menu named `BODY-PART`. For each animal, select the body-part you wish to anchor the ROI too.  

2. PARALLEL OFFSET: Just as *Entire animal* based ROIs, there is a parallel offset entry box when working with single body-part based ROIs. This entry box defines the size of the ROI from the animal body-part (see the example image below). As opposed to when working with *Entire animal* based ROIs, this entry box **cannot be zero or empty** when working with ingle body-part based ROIs. 

### FINDING ANIMAL-ANCHORED ROIs

Once you have filled in your parameters above, click the `RUN` button. You should be able to follow the progress in the mian SimBA main terminal window and the OS terminal. 

Once complete, SimBA saves all information of all the anchored ROIs for all the animals in all frames and videos in a *pickled dictionary with shapely shapes* in the `project_folder/log` directory. I know - a *pickled dictionary with shapely shapes* will be nonsense to many and difficult to work with. However, this file containes all the information we need to compute all the statistics we need. We save it in this format as we need to **compress** it as much as we possibly can, because it contains a potentially very large about of data (depending on the number of videos, individuals, and frame rate of your videos). 

## VISUALIZING ANIMAL-ANCHORED ROIs

Once the animal anchored-ROIs have been computed, we may want to visualize them to confirm they look as expected. To visualize the boundaries, click on the `VISUALIZE BOUNDARIES` button which should bring up the following pop-up window allowing user-defined settings:

* In the `SELECT VIDEO` drop-down menu, select the video you wish to visualize the boundaries for. 
* Tick the `INCLUDE KEY-POINTS` checkbox if you want to visualize the body-part pose-estimated key-points **in addition** to the animal-anchored ROIs. 
* Sometimes the animal-anchored ROIs (and key-points) are more easily visable of the rest of the images are in greyscale. To create greyscale images (except the ROIs/key-points), tick the `GREYSCALE` checkbox. 

To create the boundary videos, click the `RUN` button. You can follow the progress in the main SimBA  terminal window and the OS terminal. Once complete, a new file representing your video is created in the `project_folder/frames/output/anchored_rois` directory of your SimBA project, and may look something like these examples:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/bounding_boxes_entire_animal_termite.gif" />
</p>
<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/bounding_boxes_head_termite.gif" />
</p>

## CALCULATING BOUNDARY STATISTICS

Next, we want to calculate statistics based on on each animal-anchored ROI. For example, for each frame and each animal-anchored ROI, we may want to know:

* Which other animal-anchored ROIs it intersects with
* Which pose-estimated key-points belonging to other animals intersects with the animal-anchored ROIs

Thus, an animal-anchored ROi can intersect with other animal-anchored ROIs, or intersect with other animal key-points. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/intersecting_examples.png" />
</p>

To calculate these statistics, click on `CALCULATE BOUNDARY STATISTICS` which brings up the following pop-up settings window:

To calculate ROI-ROI intersection data (as in the left image above), tick the `ROI-ROI INTERSECTIONS` checkbox. To calculate ROI-keypoints intersection data (as in the right image above), tick the `ROI-KEYPOINT INTERSECTIONS` checkbox. 

Next, we want to choose the output file-format on how to store our data. If your data is relatively small (e.g.,  <100k-ish frames per video, <5 animals per video, and you have a good amount of storage space), consider ticking the `.csv` radio-button `OUTPUT FILE TYPE` sub-menu. This is the easiest file-type to work with (you can open and play with it in any spreadsheet-viewer) but comes at the cost of the files being very large and time-consuming to read and write. If you have longer videos and less storage available, then you may be forced to tick either the `.pickle` or `.parquet` radio-buttons. 

Once you've made your selections, click the `RUN` button. You can follow the progress in the main SimBA terminal window. Once complete, a data-file for each of your videos is generated in the `project_folder/csv/anchored_roi_data` directory of your SimBA project. These files can be rather big *truth tables* (containing only 0 and 1s) with rows representing frames, and columns representing the different possible interactions/intersections. 

From these truth tables we can calculate all necessery aggregate statistics of such as latencies and event count. But, to better enable flexibility and user-defined custom metrics, we will go through the structure of the file in detail.


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/data_table_intersection.png" />
</p>

There are two types of columns in this file, representing (i) ROI-ROI intersections, and (ii) ROI keypoint intersections. The ROI-ROI intersections are represented by the columns to the **left** in the image above. These column headers contain two strings separated by **:** characters. The ROI-keypoint intersections are represented by the columns to the **right** in the image above. These column headers contain three strings separated by **:** characters. 

If you saved the data in CSV file format, and open the file in a spreadsheet viewer, you might see something like this when viewing the first two columns and first 27 frames:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/intersaction_tables.png" />
</p>

The first column is named `Animal_1:Animal_2`. This column contains data for the intersections of the **Animal 1 anchored ROI** and the **Animal 2 anchored ROI**. The `1` in rows 14-20 indicate that the Animal 1 anchored ROI and Animal 2 anchored ROI **where** overlapping in those frames. 
The value `0` in the cells representing frames 0-13 and frames 21-27 shows that the Animal 1 anchored ROI and Animal 2 anchored ROI **where not** overlapping in those frames.

As opposed to the **first** column header, the **second** column header contains three `:` characters and is named `Animal 1:Animal 2:Head`. This column contains data for the intersections of the **Animal 1 anchored ROI** and the **Animal 2 head body-part**. The `1` in rows 21-27 indicate that the Animal 1 anchored ROI and Animal 2 head **where** overlapping in those frames. The value `0` in the cells representing frames 0-20 shows that the Animal 1 anchored ROI and Animal 2 head **where not** overlapping in those frames.

## CALCULATING SUMMARY AGGREGATE BOUNDARY STATISTICS

With this information, we can now compute aggregate statistics proxying how much each animal interacted with each other. To compute aggregate statistics, we click the `CALCULATE AGGREGATE BOUNDARY STATISTICS` button which brings up the following pop-up window:


The first `Settings` 















