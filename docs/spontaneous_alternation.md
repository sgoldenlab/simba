# <p align="center"> Spontaneous Alternation in SimBA </p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/docs/_static/img/spontaneous_alternations.png" />
</p>

The SimBA spontaneous alternation interface allows users to compute detailed alternations data from pose-estimation data and user-defined ROIs. 

# Before analyzing Spontaneous Alternation in SimBA

i) To analyze spontaneous alternation data in SimBA, we need to work with a project which contains a **single** animal. Also, the tracking data **first** has to be processed the **up-to and including the 
*Outlier correction* step described in [Part 2 - Step 4 - Correcting outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction)**. 
Thus, before proceeding to calculate ROI based measures, you should have one CSV file for each of the videos in your 
project located within the `project_folder\csv\outlier_corrected_movement_location` sub-directory of your SimBA project.

ii) Use the [SimBA ROI](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md) tools to define four different ROIs in each video: One ROI which 
encompasses the center of the maze, and three or more arms that defines the arms of the maze. E.g., in this example image below, I have drawn 4 ROI rectangles that encompasses the three arms
and the center of the maze:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/spontaneous_alternation_1.png" />
</p>

# Analyzing Spontaneous Alternation in SimBA

Onces the ROIs have been defined, click on the `SPONTANEOUS ALTERNATION` button in the ROI tab, and you should see the following pop-up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/spontaneous_alternation_2.png" />
</p>

1) In the `ARM DEFINITIONS` frame, use the `MAZE ARM 1-3:` dropdown menus select the names of your ROIs that represents your three maze arms. 

2) In the `ARM DEFINITIONS` frame, use the `MAZE CENTER` dropdown menu to select the name of your ROI that represents the center of your maze.

3) The `ANIMAL SETTINGS` frame contains settings that helps us control how an animal and entries/exits are defined:

  - `POSE ESTIMATION THRESHOLD`: Each body-part tracked through pose-estimation data has a detetion probability/confidence score associated with it in each frame. Use this dropdown menu to filter out body-parts detected below a certain confidence.
    Setting this value to `0` means that all body-parts in all frames will be used to detect the animal.

  - `ANIMAL AREA (%)`: When detecting the location of the animal within the maze, SimBA will draw a "convex hull" or polygon around the animal that encapsulates all of the tracked body-parts, similar to how propriatory tools like [AnyMaze](https://www.any-maze.com/applications/y-maze/) compute spontaneous alternations.

    <p align="center"> <img src="https://github.com/sgoldenlab/simba/blob/master/images/spontaneous_alternation_2.png" /> </p>

    The `ANIMAL AREA (%)` represents the percent of this area that have to enter, or exit, the maze arms or the maze center for it to count as an entry or an exit. Smaller `ANIMAL AREA (%)` values are likely to results in more arm entries and exits being counted.
    The below image may not be the most suited as it doesnt show a 3-arm maze, but it can help to visualize how `ANIMAL AREA (%)` represents a threshold value:

    <p align="center"> <img src="https://github.com/sgoldenlab/simba/blob/master/images/spontaneous_alternation_3.png" /> </p>
    
  - `ANIMAL BUFFER (MM)`: As mentioned above, SimBA will draw a "convex hull" or polygon around the animal that encapsulates all of the tracked body-parts. Sometimes, you may want to extend the
    size if this polygon, and give it a little wriggle room, especially if you don't pose-estimate the outer boundaries of your animals. For example, in this image below, the original detected animal polygon is shown at the top, and the buffered polygon with N % is shown at the bottom:

      <p align="center"> <img src="https://github.com/sgoldenlab/simba/blob/master/images/spontaneous_alternation_4.png" /> </p>

  - `SAVE DETAILED DATA`: Beyond the standard metrics (alternation rate, alternation count, same-arm return errors, alternate arm return errors), we may want data for specific arm combinations and exact frame count for when  

    

     







