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

### ENTIRE ANIMAL BASED BOUNDING BOXES

Entire animal-based bounding boxes has two user-defined parameters:

1. FORCE RECTANGLE: Rather then defining each animals ROI through polygons, we can force the polygon to its minimim bounding rectangle. If you want each ROI to be a rectangle rather than a polygon, then check the `FORCE RECTANGLE` checkbox. If you would want the animals ROi to be defined as a polygon, then un-check the `FORCE RECTANGLE` checkbox. 

2. PARALLEL OFFSET: Sometimes we may not want to draw our animal-anchored ROIs exacactly by the outer-bounds of the animal hull key-points. Instead, we may want to introduce a little extra wriggle room that is included in the animals personal space. If you want to introduce a little extra room inside the animals ROI, then enter the size of that space in the `PARALLEL OFFSET` entry box in millimeter. 

### SINGLE BODY-PART BASED BOUNDING BOXES.

Single body-part based bounding also has two user-defined parameters:

1. BODY-PARTS: For each animal, we need to define which body-part the ROI should be anchored to. When selecting **Single body-part circle** or **Single body-part square** in the `SELECT SHAPE TYPE` dropdown, you should see a row for each animal in your SimBA project. Each row has a drop-down menu named `BODY-PART`. For each animal, select the body-part you wish to anchor the ROI too.  

2. PARALLEL OFFSET: Just as *Entire animal* based ROIs, there is a parallel offset entry box when working with single body-part based ROIs. This entry box defines the size of the ROI from the animal body-part (see the example image below). As opposed to when working with *Entire animal* based ROIs, this entry box **cannot be zero or empty** when working with ingle body-part based ROIs. 

### FINDING ANIMAL-ANCHORED ROIs

Once you have filled in your parameters above, click the `RUN` button. 



