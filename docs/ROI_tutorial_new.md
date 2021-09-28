# Regions of Interest (ROIs) in SimBA

The SimBA region of interest (ROI) interface allows users to define and 
draw ROIs on videos. ROI data can be used to calculate basic descriptive 
statistics based on animals movements and locations such as:

* How much time the animals have spent in different ROIs.
* How many times the animals have entered different ROIs.
* The distance animals have moved in the different ROIs
* How many behavior bouts and / or how much time the animals have engaged in 
different classified behaviors in each ROI.

Furthermore, the ROI data can  be used to build potentially valuable,
additional, features for random forest predictive classifiers. 
Such features can be used to generate a machine model that classify behaviors that depend on the 
spatial location of body parts in relation to the ROIs. **CAUTION**: If spatial locations are irrelevant for the 
behaviour being classified, then such features should *not* be included in the machine model generation as they just 
only introduce noise.

# Before analyzing ROIs in SimBA

To analyze ROI data in SimBA (for descriptive statistics, machine learning features, or both descriptive statistics and 
machine learning features) the tracking data **first** has to be processed the **up-to and including the 
*Outlier correction* step described in [Part 2 - Step 4 - Correcting outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction)**. 
Thus, before proceeding to calculate ROI based measures, you should have one CSV file for each of the videos in your 
project located within the `project_folder\csv\outlier_corrected_movement_location` sub-directory of your SimBA project.

Specifically, for working with ROIs in SimBA, begin by 
(i) [Importing your videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-import-videos-into-project-folder), 
(ii) [Import the tracking data and relevant videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-import-dlc-tracking-data), 
(iii) [Set the video parameters](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters), 
and lastly (iv) [Correct outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction).

>**A short explanation on what is meant by 
> "using ROI data as features for random forest classifiers"** When ROIs have been drawn in SimBA, 
> then we can calculate several metrics that goes beyond the coordinates of the different body-parts from pose-estimation. 
> For example - for each frame of the video - we can calculate the distance between the different body-parts and the different ROIs, 
> or if the animal is inside or outside the ROIs. In some instances (depending on which body parts are tracked), we can also calculate if the animal is 
> [directing towards the ROIs](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data)
> (see below for more information). These measures can be used to calculate several further features such as the percentage of the session, 
> up to and including the any frame, the animal has spent within a specific ROI. These and other ROI-based features could be useful additions 
> for classifying behaviors in certain scenarios.

# Part 1. Defining ROIs in SimBA

1. In the main SimBA console window, begin loading your project by clicking on `File` and `Load project`. In the **[Load Project]** tab, click on `Browse File` and select the `project_config.ini` that belongs to your project. 

2. Navigate to the **ROI** tab, which should look like this:

![](https://github.com/sgoldenlab/simba/blob/master/images/ROI_11.PNG)

3. Next, click on "Create ROI shape definitions", and the following table will pop open.

![](https://github.com/sgoldenlab/simba/blob/master/images/ROI_menu.JPG)

This table contain one row for each of the videos in the project. 
Each video in the project has three buttons associated with it: **Draw, Reset,** and **Apply to all**. 
The functions associated with each button is described in detail below. However, in brief, the **Draw** button allows you to start to draw the defined shapes on the specific video. 
If drawings already exist for the specific video, then the **Draw** buttons opens an interface where you can move the shapes that has previously been drawn on the video. The **Reset** button deletes any drawing made on the specific video and allows you to restart the drawings from scratch by next clicking on the **Draw** button. 
The **Apply to all** buttons copies the drawing made on the specific video and replicates it on all other videos in the project. If a drawing has been applied to all videos videos in the project by clicking on the **Apply to all** button, then the shapes for any specific video can be moved by clicking on the **Draw** button.

6. To begin to draw your shapes, click on the **Draw** button for the first video in the table. 
Once clicked, two windows will pop up that look like this:

The left window will display the first frame of the video that you indicated you wanted to draw ROIs on. 
The right-most window will contain buttons and entry-boxes for creating and manpulating your ROIs, and we will go through the function of each part below.  


## The "Region of Interest Settings" menu

### Video Information

The first part of the "Region of Interest Settings" menu is titled *Video Information* and is useful for general troubleshooting. This menu displays the current video being processed, 
the format of this video, its frame rate, and the frame number and the time of the frame that is being displayed in the left window. 

### Change image

The second part of the "Region of Interest Settings" menu is titled *Change image*. Use this menu if you want to use a different frame of the video when defining your ROIs. Clicking on `+1s` will display a frame one second later in the video than the frame currently displayed.
Clicking on `-1s` will display a frame one second earlier in the video than the frame currently displayed. To move a custom distance forward in the video, enter the number of seconds you wish to move in the `Seconds forward` entry box and click on the `Move` button. 

### New shape
The second part of the "Region of Interest Settings" menu is titled *New shape* and contains three sub-menus (`Shape type`, `Shape attributes`, and `Shape name`) that are used for creating new ROI shapes.

#### Shape type
To create a new ROI, begin by selecting its type. SimBA supports three shape types - rectangles, circles, and polygons. Select the shape type you wish to draw by clicking the appropriate button. 

#### Shape attributes 
Users drawing ROIs in SimBA are often working in very different resolutions and are sometimes drawing relatively complex geometries involving many shapes. It is therefore important to keep shapes visible, distinguable and aligned while drawing regardless of the resolution of your video.
To facilitate this, SimBA allowes the user to set three different *shape attributes* (if you want to change these later, after drawing your shape, you can - more info below):

* **Shape thickness**: this dropdown controls the thickness of the lines in the ROIs. If you select a higher number the lines of the ROI will be thicker. 
* **Ear tag size**: Each shape that you draw will have *ear tags*. These tags can be clicked on to move shapes, align shapes, or manipulate the dimensions of the shapes (see beloe for more information). In this dropdown menu, select the size that the ear-tags of your ROI should have. If you select a higher number the ear-tags of the ROI will be bigger.  
* **Shape color**: Each shape that you draw will have a color. From the dropdown, select the color that your ROI should have. 

#### Shape name
Each shape in SimBA has a nam





