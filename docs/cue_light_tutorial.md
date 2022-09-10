# <p align="center"> Cue Light Analysis in SimBA </p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_lights.png" />
</p>

# <p align="center">  </p>


Behavioral experiments can involve conditioned stimuli (such as cue lights) where experimenters assess behaviors in and around the time periods of such conditioned stimuli.
For example, we may want to know:

* How many times, and for how long, is behavior X expressed during the cue light (and in user-specified time-windows preceding and proceding the cue light)?
 
* How much does the animal move during the cue light (and in user-specified time-windows preceding and proceding the cue light)? 

* Where in the environment does the animal spend time during the cue light (and in user-specified time-windows preceding and proceding the cue light)? 


In this tutorial we will use SimBA to infer when the cue light is on, and analyze movement, velocities, and behavioral classifications in relationship with the cue light status.
This tutorial involves a scenario with a single video, single classifier, single animal and single cue light. However, note that SimBA methods supports several cue lights, multiple and classifiers, and different cue lights in different locations across videos. 

# Before analyzing cue light data in SimBA

To analyze cue light data in SimBA, the tracking data **first** has to be processed the **up-to and including the *Outlier correction* step described in [Part 2 - Step 4 - Correcting outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction)**. Thus, before proceeding to analyze cue light based measures, you should have one file for each of the videos in your project located within the `project_folder\csv\outlier_corrected_movement_location` sub-directory of your SimBA project. 

Specifically, for working cue lights in SimBA, begin by (i) [Importing your videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-import-videos-into-project-folder), (ii) [Import the tracking data and relevant videos to your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-import-dlc-tracking-data), (iii) [Set the video parameters](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters), and lastly (iv) [Correct outliers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction).

# Part 1. Defining cue light ROIs in SimBA.

1) SimBA uses ROIs to locate your cue lights in the image and to compute their states. To define the locations of the cue lights, [load your project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-2-load-project-1) and head to the [ROI] tab. Next, click on `Define ROIs`. A pop up will be displayed listing all the videos in your project. In this tutorial, I am working with a SimBA project which contains just a single video. I click on `Draw` next to the video to define my ROIs:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_light_1.png" />
</p>

2) A new window pops up that allows you to define your regions of interest. In my video, I have a cue light (an operant house-light) . I am also intrested in where in the environment my animal spends time in relation the the cue light state. I therefore draw three rectangular regions using the SimBA ROI interface, representing (i) the left side of the box, (ii) the right side of the box, and (iii) the houselight. To learn how to use the SimBA ROI interface, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md).

Note I: There are no restrictions for which shape an ROI region (including the cue lights) can have. I chosed rectangles, but they can also be defined as circles or polygons. 

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

Once clicked, a window will pop open looking something like the below gif. Using this interface, we will tell SimBA with of our ROIs are cue lights, and analyse animal movements and classifications around cue lights onsets and offsets. We will also visualize the cue light data to validatation and sanity check that the onsets and offsets of the cue light(s) are accurately captured by SimBA. 

To begin, we use the `Define cue lights` menu and select `1` in the `# Cue lights` drop-down to indicate that we have one cue light in our video(s). Once done, a single dropdown menu will appear asking us for the name of the single cue light. I use this dropdown menu to say that the cue light is names `Cue light`. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/cue_light_1.gif" />
</p>

Next, I go ahead and click the `Analyze cue light data` button, which will start the analysis and extract when the cue light(s) are on and off in all frames of all videos in the project. You can follow the progress in the SimBA main terminal window. Once complete, a new file is creates inside the `project_folder/csv/cue_lights` directory within your SimBA project. If you open up one of these files, you will see columns named after your cue lights right towards the end of the file, filled with ones and zeros. Each row represents a frame of your data, a `1` in the 

> Note: To analyze the cue-light states on standard computers at acceptable runtimes, SimBA uses [kmeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and python [multi-processing](https://docs.python.org/3/library/multiprocessing.html). Should you happen to see any `MemoryErrors`, or analysis that are *not* completed in an acceptable time, please reach out to us by opening a [GitHib issue](https://github.com/sgoldenlab/simba/issues) or chat to us on [Gitter](https://gitter.im/SimBA-Resource/community). 

# Part 3. Analyzing cue light(s) states.

















