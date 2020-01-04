# SimBA Tutorial:

To faciliate the initial use of SimBA, we provide several use scenarios. We have created these scenarios around a hypothetical experiment that take a user from initial use (completely new start) all the way through analyzing a complete experiment and then adding additional experimental datasets to an initial project.

All scenarios assume that the videos have been [pre-processed](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md) and that [DLC behavioral tracking .CSV dataframes](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md) have been created.

# **Hypothetical Experiment**:
Three days of resident-intruder testing between aggressive CD-1 mice and subordinante C57 intruders. Each day of testing has 10 pairs of mice, for a total of 30 videos recorded across 3 days. Recordings are 3 minutes in duration, in color, at 30fps.

Also, so that we do not overfit the predictive classifiers to the experimental data, a different set of pilot videos have been recorded of resident-inturder pairings using identical video acquisition parameters. A total of 20 pilot videos were recorded.

# **Scenario 4**: Analyzing further data using created classifiers. 
In this Scenario you have already generated a predictive classifier for "Behavior that Will Get a Nature Paper (Behavior BtWGaNP)" (see [Scenario 1](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md)), and you have used the classifier to analyze Day 1 of your Experimental data (see [Scenario 2](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md)). You now wish to analyse Day 2 of your Experimental data.

## Part 1: 'Clean up your project' (.. or alternatively create a new project). 

We will need start with a project directory tree that does not contain any other CSV data than the data we want to analyze. If you are coming along from [Scenario 1](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario1.md) or [Scenario 2](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario2.md), you will have one or several project trees already. However, the project tree(s) contains the files used to create the BtWGaNP classifier, or other files from Day 1 of the experiments. If you look in the differen subdirectories of the `project_folder/csv/input` directory in your previous projects, you will see the  csv files we used to generate the project and/or have analyzed previously. If we continue using one of these project as is, SimBA will see these csv files and analyze these files in addition to your Experimental data. As these files have been analyzed before, we shouldn't have to analyze them again. In [Scenario 2](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md), we created a new project to avoid mixing the files that were used to create the predictive classifier for behavior BtWGaNP with the new experimental files we were to analyze. 

Another option is to "remove" these files from the direct subdirectories of our `project_folder/csv` folder. This way SimBA cannot see them, and wont analyze them again. In this Scenario we will hide the already analyzed files from SimBA by putting them in a different directory. To hide the files from Day1 in SimBA, move them to a subdirectory called Day 1, as in Step 1 below:

![](/images/dir_info.JPG "dir_info")

**Important**: In this example we hide the files during Step 1 the `project_folder/csv/input` directory. You will need to repeat the process for the (i) `project_folder/csv/outlier_corrected_movement`, (ii) `project_folder/csv/outlier_corrected_movement_location`, (iii) `project_folder/csv/features_extracted`, and (iv) `project_folder/csv/machine_results` folders. This way you will keep the Day 1 analysis files stored and they won't interfere with the analysis of Day 2. *Note*: The analysis of Day 1 also generated [CSV files indicating how many outliers were corrected](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction), and further [CSV files containing descriptive statistics of the classified behaviors](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results). These files are stored in the  `project_folder/csv/logs` folder and should also be copy/pasted to a new subdirectory. 

## Part 2: Load the project and import your new data.

1. After cleaning up your folder, it is time to load your project in SimBA. Follow the instructions in Scenario 1 - [Step 1: Load Project Config](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-1-load-project-config) to load your project. 

2. Once the project is loaded, follow the instructions in Scenario 1 - [Step 2 (Optional step) : Import more DLC Tracking Data or videos](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-optional-step--import-more-dlc-tracking-data-or-videos) to import further DLC tracking data and further videos into your project using the following menu:

![](/images/importdlc.PNG "importdlc")

>*Note:* If you wish yo visualize the data from Day2, as decribed in Scenario 2 - [Part 5: Visualizing machine predictions](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-machine-predictions), you will need to extract frames from the newly added videos. Go ahead and click on the `Extract frames` button. If the frames are stored somewhere else, they do not have to be generated again and you can instead copy them in to the project by using the `Import frame folders` menu. 

Once the data is imported, you should see the new data for Day 2 in the `project_folder/csv/input` directory, similar to this example following *Step 2*:

![](/images/dir_info.JPG "dir_info")

## Part 3: Process the data for Day 2 of the experiment. 

Next, we need to process the data for Day 2 of the experiment. 

we will need to [correcting outliers in the tracking](https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf) f



However, **Step 1-5**, which we need to compplete,  deals with [correcting outliers in the tracking](https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf), and [extracting features](https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv) from our experimental data.









