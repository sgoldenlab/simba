# SimBA Tutorial:

To faciliate the initial use of SimBA, we provide several use scenarios. We have created these scenarios around a hypothetical experiment that take a user from initial use (completely new start) all the way through analyzing a complete experiment and then adding additional experimental datasets to an initial project.

All scenarios assume that the videos have been [pre-processed](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md) and that [DLC behavioral tracking .CSV dataframes](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md) have been created.

# **Hypothetical Experiment**:
Three days of resident-intruder testing between aggressive CD-1 mice and subordinante C57 intruders. Each day of testing has 10 pairs of mice, for a total of 30 videos recorded across 3 days. Recordings are 3 minutes in duration, in color, at 30fps.

# **Scenario 2**: Using a classifier on new experimental data...
In this scenario you have either (i) generated a classifier yourself which performance you are happy with. For example, you have followed [Scenario 1](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario1.md), and generated the classifier for "Behavior that Will Get a Nature Paper (Behavior BtWGaNP)" and its working well. Or, you have (ii) got a behavioral classifier generated somewhere else, and now you want to use the classifier to score Behavior BtWGaNP on your experimental videos. For example, you have downloaded the classifier from [this](https://osf.io/d69jt/) website.

## Part 1: Create a new project

We will need start with a project directory tree that does not contain any other data than the data we want to analyze. If you are coming along from [Scenario 1](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario1.md), you will have a project tree already. However, this project tree contains the files used to create the BtWGaNP classifier: if you look in the subdirectories of the `project_folder/csv/input` directory, you will see the 19 csv files we used to generate the project. If we continue using this project, SimBA will see these csv files and analyze these files in addition to your Experimental data. Thus, the option is to remove these files from the subdirectories of our project, or alternatively, to [create a new](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project) project that only contains the data from our Experiment. In this Scenario, we will create a [new project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project). 

To create a new project with your experimental data, follow the instructions for creating a new project in either of these tutorials: [1](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project-1), [2](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config]. Instead of using your pilot data as indicated in the tutorial from [Scenario 1]((https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project-1), use the data, videos for the Experiment you want to analyze.

## Part 2: Load project and process your tracking data

In Part 1, we created a project that contains your experimental data. To continue working with this project, we **must** load it. To load the project and process your experimental data, follow the instructions for **Step 1-5** in either of these tutorials: [1](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project-1), [2](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config]. In this Scenario you can **ignore Steps 6-7 of the Load Project tutorials**. **Step 1-5**, which we need to compplete, deals with [correcting outliers in the tracking](https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf), and [extracting features](https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv) from our data.

## Part 3: Run the classifier on new data

At this point we have Experimental data that has been corrected for outliers and with features extracted and we want to predict behavior BtWGaNP in these videos. If you have downloaded the predictive classifier, we will also include information on how you can use those files to predict behavior BtWGaNP in these videos. 

1. In the Load Project menu, navigate to the **Run Machine Model** tab and you should see the following window. 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/runrfmodel.PNG" width="343" height="132" />

2. Click on `Model Selection`. The following window, containing the classifier names that were defined when you created the project, will pop up.

<p align="center">
  <img width="312" height="256" src="https://github.com/sgoldenlab/simba/blob/master/images/rfmodelsettings.PNG">
</p>

3. Click on `Browse File` and select the model (*.sav*) file associated with the classifier name. If you are following along from [Scenario 1](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario1.md), the *.sav* file will be saved in your earlier project, in the `project_folder\models\generated_models` directory. You can also select an *.sav* file located in any other 








