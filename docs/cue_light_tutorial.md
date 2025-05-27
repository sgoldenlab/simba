# <p align="center"> Cue Light Analysis in SimBA </p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_lights_89.png" />
</p>

# <p align="center">  </p>


Behavioral experiments can involve conditioned stimuli (such as cue lights) where experimenters assess behaviors around the time periods of such conditioned stimuli.

For example, we may want to know:

* How many times, and for how long, is behavior X expressed during the cue light (and in user-specified time-windows preceding and proceding the cue light)?
 
* How much does the animal move during the cue light (and in user-specified time-windows preceding and proceding the cue light)? 

* Where in the environment does the animal spend time during the cue light (and in user-specified time-windows preceding and proceding the cue light)? 

In this tutorial we will use SimBA to infer when the cue light is on, and analyze movement, velocities, and behavioral classifications in relationship with the cue light status. This tutorial involves a scenario with a single video, single classifier, single animal and single cue light. However, note that SimBA methods supports several cue lights, multiple animals and classifiers, and different cue lights in different locations across multiple videos. 

# Before analyzing cue light data in SimBA

To analyze cue light data in SimBA, the tracking data **first** has to be processed the **up-to and including the *Outlier correction* step described in [Part 2 - Step 4 - Correcting outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction)**. Thus, before proceeding to analyze cue light based measures, you should have one file for each of the videos in your project located within the `project_folder\csv\outlier_corrected_movement_location` sub-directory of your SimBA project. 

Specifically, when working cue lights in SimBA, begin by (i) [Importing your videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-import-videos-into-project-folder), (ii) [Import the tracking data and relevant videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-import-dlc-tracking-data), (iii) [Set the video parameters](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters), and lastly (iv) [Correct outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction).

# Part 1. Defining cue light ROIs in SimBA.

1) SimBA uses ROIs to locate your cue light(s) in the image and to compute their states. To define the locations of the cue lights, [load your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-2-load-project-1) and head to the [ROI] tab. Next, click on `Define ROIs`. A pop up will be displayed listing all the videos in your project. In this tutorial, I am working with a SimBA project which contains a single video. As in the video below, I click on `Draw` next to the video to define my ROI cue lights, I draw a single rectangular ROI called `MY_CUE_LIGHT` in coral color, and click to save my ROI data.  

https://github.com/user-attachments/assets/8ac1d59a-910e-4126-a6c6-655ff885aa79

>[!NOTE]
>There are no restrictions for which shape an ROI region (including the cue lights) can have. In the example above I drew a rectangle, but it could also be a circle or a polygon.
>
>Also, there are no restrictions for how many cue lights you have. Draw however many ROIs as there are cue lights.
>
>For a detailed tutorial for how to use the ROI interface, see [THIS](https://github.com/sgoldenlab/simba/blob/master/docs/roi_tutorial_new_2025.md) documentation. 

# Part 2. Analyzing cue light(s) states. 

1. To begin analyzing your cue light data, head to the right-most tab in the SimBA interface called [Add-ons] and click on `CUE LIGHT ANALYSIS`, button, which brings up the following pop-up:

![cue_light_2](https://github.com/user-attachments/assets/9155e4c9-915e-4147-a007-85e0d9e6b662)


2. In this interface, we begin, on the left-hand side, to select how many cue-lights we have. Once we have selected how many cue lights we have, we select the names of cue lights in the appropriate dropdown:

In my case, I have only drawn one ROI (named `MY_CUE_LIGHT`) so I don't have many options. However, should you have more ROIs, then those options will be displayed in the appropriate dropdowns:

![cue_light_3](https://github.com/user-attachments/assets/978fd9f6-6236-4127-9420-33b67dcb57e5)


3. Next, I go ahead and click the `ANALYZE CUE LIGHT DATA` button. We will need to do this step before doing any other process (e.g., visualization, movement, or classification analysis). This process will look at each of your video files and find when the cue-light is ON or OFF and save this data inside your SimBA project. Once a click this button, I should see one more pop-up giving me some option on how I want to calculate these values:

![cue_light_4](https://github.com/user-attachments/assets/ceb8de21-5bdf-4c8b-a518-2edef26d0792)

`COMPUTE DETAILED CUE LIGHT BOUT DATA`: If set to `TRUE`, SimBA will create a CSV inside the `porject/folder/logs` directory named something like `cue_light_details_20250525112722.csv` that lists each time-stamp, in each video, where each cue light is ON (including duration in seconds, start time, end time, start frame, end frame). 

`CPU CORE COUNT`: How many CPU cores you want to use to compute the cue light ON and OFF times. The more CPU cores the faster the processing will be, but it will require more memory RAM available. If you hit errors related to memory, try decreasing the number of CPU cores. The maximal number is the number of CPU cores available on your machine. 

`VERBOSE`: If True, then prints out progress (such frame is currently being analyzed etc) in the terminal window from where you launched SimBA. 

Once set, click the <kbd>RUN</kbd> button. You can follow the progress in the main SimBA terminal and the terminal window from where you launched SimBA. Once complete, SimBA will save once CSV file for each of your CSV files in the `project_folder/csv/outlier_corrected_movement_location` directory inside the `project_folder\csv\cue_lights` directory. 

Each of these files will have two additional columns for each of your cue lights. One column will be named after your cue-light, and populated with either `0` or `1`, representing if the cue light is OFF (`0`) or ON (`1`) on each of the frames of the video. A second collumn will be named after your cue light with the `_INTENSITY` suffix. This is te luminosity of the cue light in each of the frames of the video. For an example of the expected output, see [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/cue_light_example.csv) CSV file.  

Finally, if you selected `COMPUTE DETAILED CUE LIGHT BOUT DATA`, there will be a CSV file inside the `porject/folder/logs` directory that list each episode that each cue light was on in each video. In my case (a single short video with one cue light), it can be expected to look like [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/cue_light_details_20250525112722.csv)

# Part 3. Visualizing cue light states.

**Once Part 2 is completed**, we'd want to visualize the results of the preceding step to confirm that SimBA has accurately captured the onsets and offsets of the cue lights. We go ahead and click on <kbd>VISUALIZE CUE LIGHT DATA</kbd> and the following pop-up - showing options for how to generate the visualizations - pop up: 

![cue_light_5](https://github.com/user-attachments/assets/86666782-6f4b-41ca-bfe3-29f9273482d6)

`SHOW POSE`: If set to True, SimBA will show the pose-estimated body-part locations of the animal(s) together with data on if the cue light(s) are ON or OFF. 

`CPU CORE COUNT`: How many CPU cores you want to use to create the visualization. The more CPU cores the faster the processing will be, but it will require more memory RAM available. If you hit errors related to memory, try decreasing the number of CPU cores. The maximal number is the number of CPU cores available on your machine. 

`CREATE VIDEO`: If True, SimBA will create a video displaying when the cue light(s) are ON, OFF, how long the cue light(s) have ON, and how many times the cue light(s) have turned ON. Videos will be saved in the `project_folder\frames\output\cue_lights` directory of your SimBA project. 

`CREATE INDIVIDUAL FRAMES`: If True, SimBA will create a directory for the video with each frame saved as a seperate `.png` file. Frames will be saved in a subdirectory of the `project_folder\frames\output\cue_lights` directory named according to your video. 

`VERBOSE`: If True, then prints out progress (such frame is currently being analyzed etc) in the terminal window from where you launched SimBA. 

Finally, in the `SELECT VIDEO` frame, use the dropdown to select the video which you want to visualize. Once selected, click the <kbd>RUN</kbd> button. You can follow the progress in the main SimBA terminal and the terminal window from where you launched SimBA. See below for an example of expected output:

https://github.com/user-attachments/assets/441989a3-300c-407d-9582-cf5900e04cdd

>[!NOTE]
>To visualize the ROI data, SimBA uses the ROI analysis data computet in Part 1. If your cue light ROIs changes, please re-do Part 1 before visualizing the cue-light data. 


# Part 4. Analyzing movement at cue light states. 

Next, we may want to analyze movements (distances moved and velocities) in and around the times the cue light(s) are on. To do this we click on <kbd>ANALYZE CUE LIGHT MOVEMENT</kbd> which will bring up a pop-up menu looking like below:

![cue_light_6](https://github.com/user-attachments/assets/f1cf4e35-bd80-4362-97f3-d233b80f2121)

`SELECT BODY-PART`: Select the body-part that you want to use to proxy the animals location to calculate the movement and velocity. 

`PRE CUE LIGHT WINDOW (S)`: We may not just want to calculate the movement and velocity when the cue-lights(s) are ON, but we may also want to calculate the movement and velocity in a control period immediately **BEFORE** each cue-light come ON. To do this, set the pre-cue light period in seconds here (.e.g., enter `10` to calculate the movement and velocity in each 10s window preceeding each cue light onset). If you do not want this data, enter `0` in this entry box or leave this entry box blank. 

`POST CUE LIGHT WINDOW (S)`: We may not just want to calculate the movement and velocity when the cue-lights(s) are ON, but we may also want to calculate the movement and velocity in a control period immediately **AFTER** each cue-light turns OFF. To do this, set the post-cue light period in seconds here (.e.g., enter `10` to calculate the movement and velocity in each 10s window proceeding each cue light offset). If you do not want this data, enter `0` in this entry box or leave this entry box blank. 

Once selected, click the <kbd>RUN</kbd> button. You can follow the progress in the main SimBA terminal and the terminal window from where you launched SimBA. Once complete, a file will be created in the `project_folder/logs/` directory named something like `Cue_lights_movement_statistics_20250523144123.csv`. This file will contain a row for each episode that each cue light was on in each video. It will have additional columns containing the movement (in centimeters), and the velocity as CM/S, during each episode the cue light was ON. It will also have values representing the movement (in centimeters), and the velocity as CM/S, for the pre- and post- cue light bout period (if selected). For example, in my project the my single cue light named `MY_CUE_LIGHT` came on twice, and [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/Cue_lights_movement_statistics_20250523144123.csv) is the expected output. 

>[!NOTE]
>If the cue-light post- or pre-period in the video is shorter than the specified pre- and post period in seconds (e.g., the light comes on five seconds into the video and I have specified a 10s `PRE CUE LIGHT WINDOW (S)` time), then SimBA will use a pre-period of 5s for that episode).

# Part 5. Analyzing classifications at cue light states. 

Next, we want to analyze behavioral classifications in and around the times the cue light(s) are on. To do this we click on <kbd>ANALYZE CUE LIGHT CLASSIFICATIONS</kbd>. This will bring up a up pop-up menu looking like the window below accepting some user-defined settings:

![cue_light_7](https://github.com/user-attachments/assets/548e779b-b0e4-4f5b-ba39-f384ccc122b1)

>[!NOTE]
>To analyze classifications at cue light states, we need to have analyzed our videos using the classifiers of interest. That means that **before** we analyze classifications at cue light states, we need a file representing each video of interest in the project_folder/csv/machine_results` directory. Click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model) for documentation on how to analyze videos using previously created classifiers. 

`CLASSIFIERS`: Thsi frame contains a checkbox for each of the classifiers in the SimBA project. tick teh checkboxes for the classifiers you want to analyze in relation to the cue light(s) ON and OFF states. 

`PRE CUE LIGHT WINDOW (S)`: We may not just want to calculate the duration and event counts of classifiers during the cue-lights(s), but we may also want to calculate duration and event counts of classifiers in a control period immediately **BEFORE** each cue-light come ON. To do this, set the pre-cue light period in seconds here (.e.g., enter `10` to calculate behavior classifier events 10s window preceeding each cue light onset). If you do not want this data, enter `0` in this entry box or leave this entry box blank. 

`POST CUE LIGHT WINDOW (S)`: We may not just want  to calculate the duration and event counts of classifiers during the cue-lights(s) are ON, but we may also want to calculate duration and event counts of classifiers in a control period immediately **AFTER** each cue-light turns OFF. To do this, set the post-cue light period in seconds here (.e.g., enter `10` to calculate the movement and velocity in each 10s window proceeding each cue light offset). If you do not want this data, enter `0` in this entry box or leave this entry box blank. 

Once filled in, hit the <kbd>RUN</kbd> button. You can follow the progress in the main SimBA terminal and the terminal window from where you launched SimBA. Once complete, a file will be saved in the `project_folder/logs/ directory named something like `Cue_lights_movement_statistics_20250523144123.csv` which will contain the duration of each classifier during, before, and after each cue light ON event. For an example output file with expected data, see [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/Cue_lights_clf_statistics_20250524113810.csv) file.  


> [!NOTE]
> If the cue-light post- or pre-period in the video is shorter than the specified pre- and post period in seconds (e.g., the light comes on five seconds into the video and I have specified a 10s `PRE CUE LIGHT WINDOW (S)` time), then SimBA will use a pre-period of 5s for that episode).

>[!IMPORTANT]
>Cue light analyses involve  new and un-tested functions in SimBA. Should you encounter errors that are difficult to interpret, or you need additional functionality, please reach out to us by opening a [GitHub issue](https://github.com/sgoldenlab/simba/issues) or write to us on [Gitter](https://gitter.im/SimBA-Resource/community). Also, 


Author [Simon N](https://github.com/sronilsson)































