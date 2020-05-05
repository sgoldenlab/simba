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
