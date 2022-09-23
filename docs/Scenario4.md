# SimBA Tutorial:

To faciliate the initial use of SimBA, we provide several use scenarios. We have created these scenarios around a hypothetical experiment that take a user from initial use (completely new start) all the way through analyzing a complete experiment and then adding additional experimental datasets to an initial project.

All scenarios assume that the videos have been [pre-processed](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md) and that [DLC behavioral tracking CSV files](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md) have been created.

# **Hypothetical Experiment**:
Three days of resident-intruder testing between aggressive CD-1 mice and subordinante C57 intruders. Each day of testing has 10 pairs of mice, for a total of 30 videos recorded across 3 days. Recordings are 3 minutes in duration, in color, at 30fps.

Also, so that we do not overfit the predictive classifiers to the experimental data, a different set of pilot videos have been recorded of resident-inturder pairings using identical video acquisition parameters. A total of 20 pilot videos were recorded.

# **Scenario 4**: Analyzing and adding new Experimental data to a previously started project. 
In this current Scenario, you have already generated a predictive classifier for "Behavior that Will Get a Nature Paper (Behavior BtWGaNP)" (see [Scenario 1](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md)), and you have used the classifier to analyze Day 1 of your Experimental data (see [Scenario 2](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md)). You may have also updated your classifier to make it better in [Scenario 3](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario3.md). You now wish to use the BtWGaNP classifier to analyse Days 2 and 3 of your Experimental data. 

>**Note:** This Scenario is very similar to [Scenario 2](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md), however, we will take a slightly different approach to analyzing additional data in SimBA (see *Part 1* below). 

## Part 1: 'Clean up your previous project' (.. or alternatively create a new project). 

We will need start with a project directory tree that does not contain any other CSV data than the data we want to analyze. If you are coming along from [Scenario 1](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario1.md) or [Scenario 2](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario2.md), you will have one or several project trees already. However, the project trees may contain the files used to create the BtWGaNP classifier ([Scenario 1](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario1.md)), or data from Day 1 of the experiments ([Scenario 2](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario1.md)). If you look in the different sub-directories of the `project_folder/csv` folder in your previous projects, you will see the CSV files we used to generate the project and/or have analyzed previously. If we continue using one of these project as is, SimBA will see these CSV files and analyze these files in addition to your new as yet un-analyzed experimental data from Day 2 and 3 of the experiment. 

As these files have been analyzed before, we shouldn't have to analyze them again. In [Scenario 2](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md), we created a new project to avoid mixing the files that were used to create the predictive classifier for behavior BtWGaNP with the new experimental files we were to analyze. 

Another option is to "remove" these files from the immediate sub-directories of our `project_folder/csv` folder. This way SimBA cannot see them, and won't analyze the already analyzed files again. In this current Scenario, we will hide the already analyzed files from the project created in [Scenario 2](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md), by ***manually** putting the files in a different directory. To hide the Day 1 files from SimBA, one option is to move the files to a sub-directory called Day 1, as in *Step 1* in the below image:

![](/images/dir_info.JPG "dir_info")

**Important**: In this example we hide the files during *Step 1* in a sub-directory of `project_folder/csv/input` folder. You will need to repeat the process for the (i) `project_folder/csv/outlier_corrected_movement`, (ii) `project_folder/csv/outlier_corrected_movement_location`, (iii) `project_folder/csv/features_extracted`, and (iv) `project_folder/csv/machine_results` folders. This way you will keep the Day 1 analysis files stored safely and they won't interfere with the analysis of Day 2 and Day 3.

>*Note*: In [Scenario 1](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario1.md), the data analysis also generated additional [CSV files indicating how many outliers were corrected](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction), and further [CSV files containing descriptive statistics of the classified behaviors](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results). These files are stored in the  `project_folder/csv/logs` folder and you may also want to cut/paste them to a new sub-directory to have better control over the outputted data. Also note that the sub-directory containing the Day 1 analysis files does not have to be in the `project_folder`, but can be *anywhere* except directly within the `project_folder/csv` sub-directories. 

## Part 2: Load the project and import your new data.

1. After cleaning your project folders, it is time to load your project in SimBA. Follow the instructions in Scenario 1 - [Step 1: Load Project Config](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-1-load-project-config) to load your project. 

2. Once the project is loaded, follow the instructions in Scenario 1 - [Step 2 (Optional step) : Import more DLC Tracking Data or videos](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-optional-step--import-more-dlc-tracking-data-or-videos) to import further DLC tracking data, and further videos, into your project using the following menu:

![](/images/importdlc.PNG "importdlc")

>*Note:* If you wish to visualize the data from imported videos, as decribed in Scenario 2 - [Part 5: Visualizing machine predictions](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-machine-predictions), you will need to create the frames for the newly added videos. Go ahead and click on the `Extract frames` button. If the frames are stored somewhere else, they do not have to be generated again and you can instead copy them in to the project by using the `Import frame folders` menu. 

Once the data is imported, you should see the new imported data for Day 2 in the `project_folder/csv/input` directory, similar to after *Step 2* in the below image:

![](/images/dir_info.JPG "dir_info")

## Part 3: Process the data for Day 2-3 of the experiment. 

Next, we will need to process the data for Day 2-3 of the experiment. This process includes (i) [correcting outliers in the tracking](https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf), and (ii) [extracting features](https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv). To process the newly added data imported during **Part 2** of this current Scenario,  follow the instructions for **Step 3 to 5** in [Part 2 of Scenario 1](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters).

## Part 4: Run the predictive classifier on the data for Day 2-3. 

At this point we have the Day 2-3 data within the project, and the data has been corrected for outliers and the features have been extracted. We now want to predict behavior BtWGaNP in these videos - just like we did in [Scenario 2](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md). The process to do this is documented in [Part 3 of tutorial for Scenario 2](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data), and is repeated here below.

1. In the Load Project menu, navigate to the **Run Machine Model** tab and you should see the following window. 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/runrfmodel.PNG" width="343" height="132" />

2. **The `Model Selection` window** .If you click on `Model Selection`. The following window, containing the classifier names that were defined when you created the project, will pop up.

<p align="center">
  <img width="312" height="256" src="https://github.com/sgoldenlab/simba/blob/master/images/rfmodelsettings.PNG">
</p>

>**Note**: You should not have to re-define the paths to the model files in this Scenario, and therefore you should not have to open the  `Model Selection` window. However, if you do decide to however open the `Model Selection` window , you'll see that the paths are empty. It does not matter. The paths have been saved in the background into your *project_config.ini* file located in your `project_folder` the first time you defined the paths (i.e., when you analyzed the data for Day 1 during [Scenario 2](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md)). If you open the *project_config.ini* you can see the previously defined paths to your predictive classifier(s) near the top of the text file.  

3. **Fill in the `Discrimination threshold` and the `Minimum behavior bout length` entry boxes.** For a reminder of the functions of these entry boxes, click [here](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data).

Click on Run RF Model to run the machine model on the data for Day 2 of the experiment. 

## Part 4: Analyze Machine Results
See [Scenario 2 - Part 4: Analyze Machine Results](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results) for how to generate descriptive statistics for the behavioral classification for Day 2 of the experiment. 

## Part 5: Visualizing machine predictions
See [Scenario 2 - Part 5 - Visualizing machine predictions](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-machine-predictions) for how to generate visualizations of the features and machine learning classification results for Day 2 of the experiment. 

# PART 6: Post-classification Validation (detecting false-positives)

Now, when you likely have generated a large number of predictions, you may want to visualize them in videos that only display the classified events. This *post-classification* validation step generates a video for each .CSV in the project that contains the concatenated clips of all the events of the target behavior that the predictive classifier identifies.

![](/images/classifiervalidation1.PNG)

- `Seconds` is the duration to add in seconds to the start of an event and to the end of the event. Let's say there was a event of **2 seconds of an attack**, entering 1 in the **Seconds** entry box will add 1 second before the 2 second attack and 1 second after.

- `Target` is the target behavior to implement into this step.

## How to use *post-classification Validation*

1. Enter 1 or 2 in the `Seconds` entry box. *Note: the larger the seconds, the longer the duration of the video.**

2. Select the target behavior from the `Target` dropdown box.

3. Click `Validate` button and the videos will be generated in `/project_folder/frames/output/classifier_validation`. The name of the video will be formated in the following manner: **videoname** + **target behavior** + **number of bouts** + .mp4

![](/images/classifiervalidation.gif)

**PLEASE HELP BY REPORTING BUGS VIA GITHUB, OR JOIN THE SIMBA [GITTER](https://gitter.im/SimBA-Resource/community) FOR DISCUSSION**



##
Author [Simon N](https://github.com/sronilsson)
