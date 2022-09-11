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

1) SimBA uses ROIs to locate your cue light(s) in the image and to compute their states. To define the locations of the cue lights, [load your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-2-load-project-1) and head to the [ROI] tab. Next, click on `Define ROIs`. A pop up will be displayed listing all the videos in your project. In this tutorial, I am working with a SimBA project which contains a single video. I click on `Draw` next to the video to define my ROIs:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_light_1.png" />
</p>

2) A new window pops up that allows you to define your regions of interest. In my video, I have a single cue light (an operant house-light) . I am also intrested in where in the environment my animal spends time in relation the the cue light state. I therefore draw **three** rectangular regions using the SimBA ROI interface, representing (i) the left side of the box, (ii) the right side of the box, and (iii) the houselight. To learn how to use the SimBA ROI interface, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md).

Note I: There are no restrictions for which shape an ROI region (including the cue lights) can have. I choosed three rectangles, but they can also be defined as circles or polygons. 

Note II: There are no restrictions for how many cue lights you have. Define an ROI for each cue light of interest.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_light_2.png" />
</p>

# Part 2. Analyzing cue light(s) states. 

To begin analyzing your cue light data, head to the right-most tab in the SimBA interface called [Add-ons] and click on `Cue-light analysis`. 

To begin analysing 
<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_light_4.png" />
</p>

Once clicked, a window will pop open looking something like the below gif. Using this interface, we will tell SimBA which of our ROIs are cue lights, and analyse animal movements and classifications around cue lights onsets and offsets. We will also visualize the cue light data to validate and a sanity check that the onsets and offsets of the cue light(s) are accurately captured by SimBA. 

To begin, we use the `Define cue lights` menu and select `1` in the `# Cue lights` drop-down to indicate that we have one cue light in our video(s). Once done, a single dropdown menu will appear asking us for the name of the single cue light. I use this dropdown menu to say that the cue light is names `Cue light`. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_light_1.gif" />
</p>

Next, I go ahead and click the `Analyze cue light data` button, which will start the analysis and extract the times when the cue light(s) are on and off in all frames of all videos in the project. You can follow the progress in the SimBA main terminal window. Once complete, a new file is creates inside the `project_folder/csv/cue_lights` directory within your SimBA project. If you open up one of these files, you will see columns named after your cue lights right towards the end of the file, filled with ones and zeros. Each row represents a frame of your data, a `1` indicated that the cue light is ON, and a `0` indicated that the cue light is OFF. 

> Note: To analyze the cue-light states on standard computers at acceptable runtimes, SimBA uses [kmeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and python [multi-processing](https://docs.python.org/3/library/multiprocessing.html). Should you happen to see any `MemoryErrors`, or analyses that are *not* completed in an acceptable time, please reach out to us by opening a [GitHib issue](https://github.com/sgoldenlab/simba/issues) or chat to us on [Gitter](https://gitter.im/SimBA-Resource/community). 

# Part 3. Visualizing cue light states.

As a further sanity check before analysing larger video batches, we'd want to visualize the results of the preceding step to confirm that SimBA has accurately captured the onsets and offsets of the cue lights. We have to option to generate compressed video files or individual images. I choose to generate a compressed video, and click `Visualize cue light data`. Once clicked, you can follow the progress in the main SimBA terminal.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_light_5.png" />
</p>

> Note: Reading and writing images and videos to disc is computationally costly on standard computers. I recommend generating video confirmations on a sub-set of videos to gauge accuracy and troubleshooting. 

Once complete, one video file for each of your input data files will be saved inside the `project_folder/frames/output/cue_lighs` directory of your SimBA project. In the videos, the numbers to the right of the video will tell you (i) the current status of each hoselight (ON vs OFF), (ii) the onset count, (iii) the total ON time in seconds, and the total OFF time in seconds, as in the gif example below:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/example_cue_light.gif" />
</p>

# Part 4. Analyzing movement at cue light states. 

Next, we want to analyze movements (distances moved, velocities, and time spend in different parts of the environment) in and around the times the cue light(s) are on. To do this we click on `Analyze cue light movements` which will bring up a up pop-up menu looking like the window to the right (indicated with a red rectangle) accepting different user-defined settings: 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_light_6.png" />
</p>

**(1)** Begin by selecting how many animals you want to compute movements for in the `# Animals` dropdown menu. In my SimBA project I only have one animal, so I use the drop-down to select `1` (which is the only option available, as the project only has data for 1 animal). 

**(2)** Select which body-part you want to use to determine the location of each animal in the `Animal 1 bp` dropdown menu. If you have more than one animal, there will be a dropdown shown for each animal chosen in step (1) above. 

**(3)** Select a pre-cue window size (in milliseconds) in the `Pre-cue window (ms)` entry box. This is the time period **before** the onset of each each cue light which we want to compute movement statistics for. In the screengrab above, I have typed `15000`, and SimBA will therefore calculate movement statistics in the 15s preceeding each cue light onset. 

**(4)** Select a post-cue window size (in milliseconds) in the `Post-cue window (ms)` entry box. This is the time period **after** the onset of each each cue light which we want to compute movement statistics for. In the screengrab above, I have typed `15000`, and SimBA will therefore calculate movement statistics in the 15s proceding each cue light onset. 

**(5)** Select a body-part threshold value in the `Threshold (0.00 - 1.00)` entry box. This is the minimum pose-estimation confidence probability value for the present of the body-part SimBA should use. For example, if you want SimBA to calculate movements when the confidence is high, set this value to close to `1.00`. If you want SimBA to use all pose-estimation body-part locations to calculate movement statistics, set this value to `0.00`. 

**(6)** If you have additional ROIs defined in SimBA (i.e., more ROIs than there are cue lights), we may want to calculate how much time the animal(s) are spending in these ROIs when the cue light is on, as well as the `pre-cue` and `post-cue` periods. If we want to perform these calculations, we go ahead and tick the `Analyze ROI data` checkbox. 

**(7)** Once all the fields have been filled in, we click the `Analyze movement data` button. You can follow the progress in the main SimBA terminal window. Once complete, navigate to the `project_folder/logs` directory in your SimBA project. You should see a datetimed CSV file named something like this: `Cue_lights_movement_statistics_20220909103957.csv`. Opening this file you should see the different movement statistics for the different animals within the different cue lights periods, e.g., somthing akin to this:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_light_7.png" />
</p>

As we ticked the checkbox during step **(6)**, SimBA will also generate a second output file inside the `project_folder/logs` directory named something like `Cue_lights_roi_statistics_20220911112453.csv`. This file contains statistics on where in relation to the non-cue light ROIs the animals spent time during the different cue lights period:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_light_8.png" />
</p>

# Part 5. Analyzing classifications at cue light states. 

Next, we want to analyze behavioral classifications in and around the times the cue light(s) are on. To do this we click on `Analyze cue light classifications`. Which will bring up a up pop-up menu looking like the window to the right (indicated with a red rectangle) accepting different user-defined settings: 

>Note: To analyze classifications at cue light states, we need to have analyzed our videos using the classifiers of interest. That means that **before** we analyze classifications at cue light states, we need a file representing each video of interest in the `project_folder/csv/machine_results` directory. Click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model) for documentation on how to analyze videos using previously created classifiers. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_light_10.png" />
</p>

**(1)** Begin by selecting shich behavior you want statistics for in relation to the onsets and offsets of the cue lights(s). In this example, I only have one classifier named `Freezing`. I go ahead and tick the checkbox next to `Freezing`. 

**(2)** Select a pre-cue window size (in milliseconds) in the `Pre-cue window (ms)` entry box. This is the time period **before** the onset of each each cue light which we want to compute behavior statistics for. In the screengrab above, I have typed `1500`, and SimBA will therefore calculate movement statistics in the 1.5s preceding each cue light onset.

**(3)** Select a post-cue window size (in milliseconds) in the `Post-cue window (ms)` entry box. This is the time period **after** the onset of each each cue light which we want to compute behavior statistics for. In the screengrab above, I have typed `1500`, and SimBA will therefore calculate movement statistics in the 1.5s proceding each cue light onset.

**(7)** Once all the fields have been filled in, we click the `Analyze classifier data` button. You can follow the progress in the main SimBA terminal window. Once complete, navigate to the `project_folder/logs` directory in your SimBA project. You should see a datetimed CSV file named something like this: `Cue_lights_clf_statistics_20220911120047.csv`. Opening this file you should see the different statistics for the different classifications within the different cue lights periods, e.g., something like the screengrab below. The last row for each classifier will tell you how much the animals engaged in the classified behavior outside the cue light time periods. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_light_12.png" />
</p>


## Notes

Cue light analyses involve  new and un-tested functions in SimBA. Should you encounter errors that are difficult to interpret, please reach out to us by opening a [GitHub issue](https://github.com/sgoldenlab/simba/issues) or write to us on [Gitter](https://gitter.im/SimBA-Resource/community). 


































