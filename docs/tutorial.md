# Generic SimBA Tutorial
This is a step by step tutorial to start using SimBA to create behavioral classifiers.

![alt-text-1](/images/simba.png "simba")

### Pipeline breakdown:
For processing datasets, the pipeline is split into a few sections. These sections are listed below along with their corresponding functions:

![alt-text-1](/images/Vis_build_2.JPG "simbaworkflow")


### Part 1: [Create a new project](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#part-1-create-a-new-project-1)
- [Generate project config](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config) ([create new classifiers](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#create-new-classifiers) or [import exisiting classifiers](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#import-existing-classifiers))
- [Import videos into project folder](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-2-import-videos-into-project-folder)
- [Import DLC Tracking Data](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-3-import-dlc-tracking-data) (if have any)
- [Extract Frames into project folder](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-4-extract-frames-into-project-folder)

### Part 2: [Load project](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-load-project-config)
- [Load the project.ini](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-load-project-config)
- [Set video parameters](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-3-set-video-parameters)
- [Outlier correction](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-4-outlier-correction)
- [Extract Features](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features)
- [Label Behavior](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-6-label-behavior)
- [Train Machine Model](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-7-train-machine-model)
- [Run Machine Model](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model)
- [Analyze Machine Results](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-9-analyze-machine-results)
- [Visualization](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-10-visualization)
- [Plot Graphs](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-plot-graphs)
- [Merge Frames](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-12-merge-frames)
- [Create Video](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-13-create-video)

## Part 1: Create a new project
This section describes how to create a new project for your tracking analysis.

![](/images/createproject2.PNG "createproject2")

### Step 1: Generate Project Config

In this step you create your main project folder with all the required sub-directories.

1. In the main SimBA window, click on `File` and and `Create a new project`. The following windows will pop up.

![](/images/createproject.PNG "createproject")

2. Navigate to the `[ Generate project config ]` tab. Under **General Settings**, specify a `Project Path` which is the directory that will contain your main project folder.

3. `Project Name` is the name of your project. *Keep in mind that the project name cannot contain spaces. Instead use underscore "_"* 

4. Under `SML Settings`, put in the number of predictive classifiers that you wish to create. For an example, if you had three behaviors in your video, put 3 in the entry box.

5. Click <img src="https://github.com/sgoldenlab/simba/blob/master/images/addclassifier.PNG" width="153" height="27" /> and it creates a row as shown in the following image. In each entry box, fill in the name of the behavior that you want to classify. 

<p align="center">
  <img width="385" height="106" src="https://github.com/sgoldenlab/simba/blob/master/images/classifier1.PNG">
</p>

6. `Animal Settings` is the number of animals and body parts that that the pose estimation tracking data contains. The default for **SimBA** is 2 animals and 16 body parts ( `2 animals, 16bp`). There are a few other - ** yet not validaded** - options, accessible in the dropdown menu.

7. Click on `Generate Project Config` to generate your project. The project folder will be located in the specified `Project Path`. 

### Step 2: Import videos into project folder
In this step, you can choose to import either one or multiple videos. The imported videos are used for visualizing predictions and standardizing distances across videos by calculating metric distances from pixel distances. 

![](/images/createproject.PNG "createproject")

#### To import multiple videos
1. Navigate to the `[ Import videos into project folder ]` tab.
2. Under the `Import multiple videos` heading, click on `Browse Folder` to select a folder that contains all the videos that you wish to import into your project.
3. Enter the file type of your videos. (e.g., *mp4*, *avi*, *mov*, etc) in the `Video type` entry box.
4. Click on `Import multiple videos`. 
>**Note**: If you have a lot of videos, it might take a few minutes before all the videos are imported.
#### To import a single video
1. Under the `Import single video` heading, click on `Browse File` to select your video.
2. Click on `Import a video`.

### Step 3: Import DLC Tracking Data
In this step, you will import your pose-estimation tracking data from DeepLabCut in csv file format.

![](/images/importcsv.PNG "importcsv")

#### To import multiple DLC csv files
1. Navigate to the `[ Import tracking data ]` tab. Under the `Import multiple csv files` heading, click on `Browse Folder` to select the folder that contains the csv files that you wish to import into your project.
2. Click on `Import csv to project folder`. 
#### To import a single csv file
1. Under the `Import single csv files` heading, and click on `Browse File` to select the csv file containing your pose estimation data.
2. Click on `Import single csv to project folder`.

### Step 4: Extract frames into project folder
This step will extract all the frames from every video that are imported into the project following **Step 2**. The frames are used for behavioral labeling when creating classifiers, and for visualizing the classification results. Once the step is completed, close the `Project Configuration` window.
>**Note**: Splitting videos up into frames can take time if you have a lot of high resolution videos at high frame rates. The frames can also take up a lot of hard drive space. 

![](/images/extractframeintop.PNG "extractframeintop")

## Part 2: Load project
This section describes how to load and work with created projects.

### Step 1: Load Project Config
In this step you will load the *project_config.ini* file that was created.
> **Note:** A project_config.ini should always be loaded before any other process.
1. In the main SimBA window, click on `File` and `Load project`. The following windows will pop up.

<p align="center">
  <img width="1255" height="380" src="https://github.com/sgoldenlab/simba/blob/master/images/loadproject.PNG">
</p>

2. Under the **Load Project.ini** tab, click on `Browse File`. Then, go to the directory that you created your project in and click on your *project folder*. Locate the *project_config.ini* file and select it. Once this step is completed, it should look like the following, and you should no longer see the text *No file selected*.

<p align="center">
  <img width="500" height="60" src="https://github.com/sgoldenlab/simba/blob/master/images/loadedprojectini.PNG">
</p>

In this image, you can see the `Desktop` is my selected working directory, `tutorial` is my project name, and the last two sections of the folder path is always going to be `project_folder/project_config.ini`.

### Step 2 (Optional) : Import more DLC Tracking Data or videos
In this step, you can choose to import more pose estimation data in csv file format and/or more videos. If this isn't relevant then you can skip this step.

![](/images/importdlc.PNG "importdlc")

1. Click on the `[ Further imports (data/video/frames) ]` tab. From here you can import more data or videos into the project folder. The imported .csv files will be placed in the `project_folder/csv/input` directory, and the imported videos will be placed in the `project_folder/videos` directory. 

2. Once the videos are imported, you can extract frames from the additional videos by clicking on `Extract frames` under the **Extract further frames into project folder** heading. 

3. If you already have existing frames of the videos in the project folder, you can import the folder that contains the frames into the project. Under the **Import frame folders** heading, click on `Browse Folder` to choose the folder thar contains the frames, and click on `Import frames`. The frames will be imported into the `project_folder/frames/input` folder. 

### Step 3: Set video parameters
In this step, you can customize the meta parameters for each of your videos (fps, resolution, metric distances) and provide additional custom video information (Animal ID, group etc). You also set the **pixels per millimeter** for your videos. You will be using a tool that requires the known distance between two points (e.g., the cage width or the cage height) in order to calculate **pixels per millimeter**. The real life distance between the two points is called `Distance in mm`.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/setvidparameter.PNG" width="391" height="96" />

1. Under **Set video parameters(distances,resolution,etc.)**, the entry box named `Distance in mm` is the known distance between two points in the videos in millimeter. If the known distance is the same in all the videos in the project, then enter the value *(e.g,: 245)* and click on `Auto populate Distance in mm in tables`. and it will auto-populate the table in the next step (see below). If you leave the `Distance in mm` entry box empty, the known distance will default to zero and you will fill in the value for each video individually. 

2. Click on `Set Video Parameters` and the following windows will pop up.
<p align="center">
  <img width="1037" height="259" src="https://github.com/sgoldenlab/simba/blob/master/images/videoinfo_table.PNG">
</p>

3. In the above example I imported four videos and their names are listed the leftmost `Video` column. I auto-populated the known distance to 10 millimeter in the previous step, and this is now displayed in the `Distance in mm` column. 

4. I can click on the values in the entry boxes and change them until I am satisfied. Then, I click on `Update distance_in_mm` and this will update the whole table.

5. Next, to get the `Pixels/mm` for the first video, click on `Video1` and the following window will pop up. The window that pops up displays the first frame of `Video1`.

<p align="center">
  <img width="300" height="400" src="https://github.com/sgoldenlab/simba/blob/master/images/getcoord1.PNG">
</p>

6. Now, double **left** click to select two points that defines the known distance in real life. In this case, I know that the two **pink connected dots** represent a distance of 10 millimeter in real life.
<p align="center">
  <img width="300" height="400" src="https://github.com/sgoldenlab/simba/blob/master/images/getcoord2.PNG">
</p>

7. If you misplaced one or both of the dots, you can double click on either of the dots to place them somewhere else in the image. Once you are done, hit `Esc`.

<p align="center">
  <img width="400" height="500" src="https://github.com/sgoldenlab/simba/blob/master/images/getcoord.gif">
</p>

8. If every step is done correctly, the `Pixels/mm` column in the table should populate with the number of pixels that represent one millimeter, 
<p align="center">
  <img width="700" height="350" src="https://github.com/sgoldenlab/simba/blob/master/images/videoinfo_table2.PNG">
</p>

9. Repeat the steps for every video in the table, and once it is done, click on `Save Data`. This will generate a csv file named **video_info.csv** in `/project_folder/log` folder that contains a table with your video meta data. 

10. You can also chose to add further columns to the meta data file (e.g., AnimalID or experimental group) by clicking on the `Add Column` button. This information will be saved in additional columns to your **video_info.csv** file.

### Step 4: Outlier Correction

Outlier correction is used to correct gross tracking inaccuracies by detecting outliers based on movements and locations of body parts in relation to the animal body length. For more details, please click [here](https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf)

<img src="https://github.com/sgoldenlab/simba/blob/master/images/outliercorrection.PNG" width="156" height="109" />

1. Click on `Settings` and the following window will pop up.

<p align="center">
  <img width="300" height="400" src="https://github.com/sgoldenlab/simba/blob/master/images/outliercorrection2.PNG">
</p>

2. Select the body parts for Animal 1 and Animal 2 that you want to use to calculate a reference value. The reference value will be the mean or median Euclidian distance in millimeters between the two body parts of the two animals in all frames.

3. Enter values for the `Movement criterion` and the `Location criterion`. 

- `Movement criterion`. A body part coordinate will be flagged and corrected as a "movement outlier" if the body part moves the *reference value \times the criterion value* across two sequential frames.

- `Location criterion`. A body part coordinate will be flagged and correct as a "location outlier" if the distance between the body part and at least two other body parts belonging to the same animal are longer than the *reference value \times the criterion value* within a frame.

Body parts flagged as movement or location outliers will be re-placed in their last reliable coordinate. 

4. Chose to calculate the median or mean Euclidian distance in millimeters between the two body parts and click on `Confirm Config`. 

5. Click to run the outlier correction. You can follow the progress in the main SimBA window. Once complete, two new csv log files will appear in the `/project_folder/log` folder. These two files contain the number of body parts corrected following the two outlier correction methods for each video in the project.  

### Step 5: Extract Features
Based on the coordinates of body parts in each frame - and the frame rate and the pixels per millimeter values - the feature extraction step calculates a larger set of features used for behavioral classification. Features are values such as metric distances between body parts, angles, areas, movement, paths, and their deviations and rank in individual frames and across rolling windows. This set of features will depend on the body-parts tracked during pose-estimation (which is defined when creating the project). Click [here](https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv) for an example list of features when tracking 2 mice and 16 body parts. 

1. Click on `Extract Features`.

### Step 6: Label Behavior
This step is used for label the behaviors in each frames of a video. This data will be concatenated with the features and used for creating behavioral classifiers. 

1. Click on `Select folder with frames`. In your project folder navigate to the `/project_folder/frames/input/` folder, and you should see folders that are named after your videos. These folders contain the extracted video frames. Select one of the folder and the following window should pop up.

<p align="center">
  <img width="720" height="720" src="https://github.com/sgoldenlab/simba/blob/master/images/labelbe.PNG">
</p>

2. Please click [here](/docs/labelling_aggression_tutorial.md) to learn how to use the behavior annotation interface.

3. Once finished, click on `Generate/Save` and it will generate a new *.csv* file in */csv/targets_inserted* folder. 

### Step 7: Train Machine Model
This step is used for training new machine models for behavioral classifications. 

>**Note:** If you import existing models, you can skip this step and go straight to **Step 8** to run machine models on new video data.

#### Train single model

1. Click on `Settings` and the following window will pop up. 

<p align="center">
  <img width="378" height="712" src="https://github.com/sgoldenlab/simba/blob/master/images/machinemodelsettings.PNG">
</p>

>**Note:** If you have a .csv file containing hyper-parameter meta data, you can import this file by clicking on `Browse File` and then click on `Load`. This will autofill all the hyper-parameter entry boxes and model evaluation settings. 

2. Under **Machine Model**, choose a machine model from the drop down menu: `RF` ,`GBC`,`Xboost`.

- `RF`: Random forest

- `GBC`: Gradient boost classifier

- `Xgboost`: eXtreme Gradient boost

3. Under the **Model** heading, use the dropdown menu to select the behavioral classifier you wish to define the hyper-parameters for.

4. Under **Hyperparameters**, select the hyper-parameter settings for your model. For more details, please click [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). Alternatively, import the recommended settings from a meta data file (see above, **Step 1**). 

- `RF N estimators`: Number of decision trees in the decision ensemble.

- `RF Max features`: Number of features to consider when looking for the best split. 

- `RF Criterion`: The metric used to measure the quality of each split, i.e "gini" or "entropy".

- `Train Test Size`: The ratio of the dataset withheld for testing the model (e.g., 0.20).

- `RF Min sample leaf`: The minimum number of samples required to be at a leaf node. 

- `Under sample setting`: "Random undersample" or "None". If "Random undersample", a random sample of the majority class will be used in the train set. The size of this sample will be taken as a ratio of the minority class and should be specified in the "under sample ratio" box below. For more information, click [here](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html).

- `Under sample ratio`: The ratio of samples of the majority class to the minority class in the training data set. Applied only if "Under sample setting" is set to "Random undersample". Ignored if "Under sample setting" is set to "None" or NaN. 

- `Over sample setting`: "SMOTE", "SMOTEEN" or "None". If "SMOTE" or "SMOTEEN", synthetic data will be generated in the minority class based on k-means to balance the two classes. For more details, click [here](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html). Alternatively, import recommended settings from a meta data file (see **Step 1**). 

- `Over sample ratio`: The desired ratio of the number of samples in the minority class over the number of samples in the majority class after over sampling.


5. Under **Model Evaluation Settings**.

- `Generate RF model meta data file`: Generates a .csv file listing the hyper-parameter settings used when creating the model. The generated meta file can be used to create further models by importing it in the **Load Settings** menu (see above, **Step 1**).

- `Generate Example Decision Tree`: Saves a visualization of a random decision tree in .pdf and .dot formats. Requires [graphviz](https://graphviz.gitlab.io/). For more information, click [here](https://chrisalbon.com/machine_learning/trees_and_forests/visualize_a_decision_tree/). 

- `Generate Classification Report`: Saves a classification report truth table in .png format. Depends on [yellowbrick](www.scikit-yb.org/). For more information, click [here](http://www.scikit-yb.org/zh/latest/api/classifier/classification_report.html).

- `Generate Features Importance Log`: Creates a .csv file that lists the importance's [(gini importances)](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) of all features for the classifier.

- `Generate Features Importance Bar Graph`: Creates a bar chart of the top N features based on gini importances. Specify N in the `N feature importance bars` entry box below.

- `N feature importance bars`: Integer defining the number of top features to be included in the bar graph (e.g., 15). 

- `Compute Feature Permutation Importance's`: Creates a .csv file listing the importance's (permutation importance's) of all features for the classifier. For more details, please click [here](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html)). **Note:** Calculating permutation importance's is computationally expensive and takes a long time. 

- `Generate Sklearn Learning Curve`: Creates a .csv file listing the f1 score at different test data sizes. For more details, please click [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)). This is useful for estimating the benefit of annotating further data. 

- `LearningCurve shuffle K splits`: Number of cross validations applied at each test data size in the learning curve. 

- `LearningCurve shuffle Data splits`: Number of test data sizes in the learning curve.  

- `Generate Precision Recall Curves`: Creates a .csv file listing precision at different recall values. This is useful for titration of the false positive vs. false negative classifications of the models.  

6. Click on the `Save settings into global environment` button to save your settings into the *project_config.ini* file and use the settings to train a single model. 

7. Alternatively, click on the `Save settings for specific model` button to save the settings for one model. To generate multiple models - for either multiple different behaviors and/or using multiple different hyper-parameters - re-define the Machine model settings and click on `Save settings for specific model` again. Each time the `Save settings for specific model` is clicked, a new config file is generated in the */project_folder/configs* folder. In the next step (see below), a model for each config file will be created if pressing the **Train multiple models, one for each saved settings** button.   

8. If training a single model, click on `Train Model`.

#### To train multiple models

1. Click on `Settings`.

2. Under **Machine Model**, choose the machine model from the drop down menu,`RF` ,`GBC`,`Xboost`.

3. Under **Model**, select the model you wish to train from the drop down menu.

4. Then, set the **Hyperparameters**.

5. Click the `Save settings for multiple models` button. This generates a _meta.csv file. Repeat the steps to generate multiple models.

6. Click on `Train Multiple Models`.

### Optional step before running machine model on new data
The user can validate each model *( saved in .sav format)* file. In this validation step the user specifies the path to a previously created model in .sav file format, and a .csv file containing the features extracted from a video. This process will (i) run the classifications on the video, and (ii) create a video with the predictions overlaid together with a gantt plot showing predicted behavioral bouts.  Click[here](https://youtu.be/UOLSj7DGKRo) for an example validation video.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/validatemodel.PNG" width="348" height="162" />

1. Under **Validate Model on Single Video** and `Select features file`, click on `Browse File`. Select a *.csv* files in the `project_folder/csv/output/features_extracted` directory.

2. Under `Select model file`, click on `Browse File` to select a model (*.sav file*).

3. Set the `Discrimination threshold` and `Minimum behavior bout length (ms)`:

- `Discrimination threshold`: The level of probability required to define that the frame belongs to the target class. Accepts a float value between 0.0-1.0. For example, if set to 0.50, then all frames with a probability of containing the behavior of 0.5 or above will be classified as containing the behavior. For more information on classification theshold, click [here](https://www.scikit-yb.org/en/latest/api/classifier/threshold.html).

- `Minimum behavior bout length (ms)`: The minimum length of a classified behavioral bout. **Example**: The random forest makes the following attack predictions for 9 consecutive frames in a 50 fps video: 1,1,1,1,0,1,1,1,1. This would mean, if we don't have a minimum bout length, that the animals fought for 80ms (4 frames), took a brake for 20ms (1 frame), then fought again for another 80ms (4 frames). You may want to classify this as a single 180ms attack bout rather than two separate 80ms attack bouts. With this setting you can do this. If the minimum behavior bout length is set to 20, any interruption in the behavior that is 20ms or shorter will be removed and the behavioral sequence above will be re-classified as: 1,1,1,1,1,1,1,1,1 - and instead classified as a single 180ms attack bout. 

4. Click on `Validate` to run the validation of the selected model. This will apply the selected model to the feature data and generate a video with behavioral predictions overlaid on the frames together with a gantt plot depicting predicted behavioral bouts. Click [here](https://youtu.be/UOLSj7DGKRo) for an example. 

### Step 8: Run Machine Model
This step runs behavioral classifiers on new data. 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/runrfmodel.PNG" width="343" height="132" />

1.  Under the **Run Machine Model** heading, click on `Model Selection`. The following window with the classifier names defined in the *project_config.ini* file will pop up.

<p align="center">
  <img width="312" height="256" src="https://github.com/sgoldenlab/simba/blob/master/images/rfmodelsettings.PNG">
</p>

2. Click on `Browse File` and select the model (*.sav*) file associated with each of the classifier names. 

3. Once all the models have been chosen, click on `Set Model` to save the paths. 

4. Fill in the `Discrimination threshold` and click on `Set` to save the settings.

- `Discrimination threshold`: The level of probability required to define that the frame belongs to the target class (see above). 

5. Fill in the `Minimum behavior bout length` and click on `Set` to save the settings.

- `Minimum behavior bout length (ms)`:  The minimum length of a classified behavioral bout(see above). 

6. Click on `Run RF Model` to run the machine model on the new data. 

### Step 9: Analyze Machine Results
Access this menu through the `Load project` menu and the `Run machine model` tab. This step performs summary analyses and presents descriptive statistics in .csv file format. There are three forms of summary analyses: `Analyze`, `Analyze distance/velocity`, and `Analyze severity`.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/analyzemachineresult.PNG" width="331" height="62" />

- `Analyze`: This button generates descriptive statistics for each predictive classifier in the project, including the total time, the number of frames, total number of ‘bouts’, mean and median bout interval, time to first occurrence, and mean and median interval between each bout. A date-time stamped output csv file with the data is saved in the `/project_folder/log` folder. 

- `Analyze distance/velocity`: This button generates descriptive statistics for mean and median movements and distances between animals. The date-time stamped output csv file with the data is saved in the `/project_folder/log` folder. 

- `Analyze severity`: Calculates the ‘severity’ of each frame classified as containing attack behavior based on a user-defined scale. **Example:** the user sets a 10-point scale. One frame is predicted to contain an attack, and the total body-part movements of both animals in that frame is in the top 10% percentile of movements in the entire video. In this frame, the attack will be scored as a 10 on the 10-point scale. A date-time stamped output .csv file containing the 'severity' data is saved in the `/project_folder/log` folder.

### Step 10: Visualization
These steps generate visualizations of features and machine learning classification results. This includes images and videos of the animals with prediction overlays, gantt plots, line plots, paths plots and data plots. In this step the different frames can also be merged into video mp4 format. 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/plotsklearn.PNG" width="1246" height="380" />

1. Under the **Sklearn visualization** heading, click on `Visualize classification results`.
This step grabs the frames of the videos in the project, and draws circles at the location of the tracked body parts, the convex hull of the animal, and prints the behavioral predictions on top of the frame. For an example, click [here](https://www.youtube.com/watch?v=7AVUWz71rG4&t=519s).

### Step 11: Plot graphs
The user can also create a range of plots: **gantt plot**, **Data plot**, **Path plot**, and **Distance plot**.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/plotgraphs.PNG" width="262" height="383" />

#### Gantt plot
Gantt plot generates gantt plots that display the length and frequencies of behavioral bouts for all the videos in the project.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/gantt_plot.gif" width="300" height="225" />

1. Under the **Gantt plot** heading, click on `Generate Gantt plot` and gantt plot frames will be generated in the `project_folder/frames/output/gantt_plots` folder.

#### Data plot
Generates 'live' data plot frames for all of the videos in the project that display current distances and velocities. 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/dataplot.gif" width="300" height="200" />

1. Under the **Data plot** heading, click on `Generate Data plot` and data plot frames will be generated in the `project_folder/frames/output/live_data_table` folder.

#### Path plot
Generates path plots displaying the current location of the animal trajectories, and location and severity of attack behavior, for all of the videos in the project.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/pathplot.gif" width="199" height="322" />

1. Under the **Path plot** heading, fill in the following user defined values.

- `Max Lines`: Integer specifying the max number of lines depicting the path of the animals. For example, if 100, the most recent 100 movements of animal 1 and animal 2 will be plotted as lines.

- `Severity Scale`: Integer specifying the scale on which to classify 'severity'. For example, if set to 10, all frames containing attack behavior will be classified from 1 to 10 (see above). 

- `Bodyparts`: String to specify the bodyparts  tracked in the path plot. For example, if Nose_1 and Centroid_2, the nose of animal 1 and the centroid of animal 2 will be represented in the path plot.

- `plot_severity`: Tick this box to include color-coded circles on the path plot that signify the location and severity of attack interactions.

2. Click on `Generate Path plot`, and path plot frames will be generated in the `project_folder/frames/output/path_plots` folder.

#### Distance plot
Generates distance line plots between two body parts for all of the videos in the project.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/distance_plot.gif" width="300" height="225" />

1. Fill in the `Body part 1` and `Body part 2`

- `Body part 1`: String that specifies the the bodypart of animal 1. Eg., Nose_1

- `Body part 2`: String that specifies the the bodypart of animal 1. Eg., Nose_2

2. Click on `Generate Distance plot`, and the distance plot frames will be generated in the `project_folder/frames/output/line_plot` folder.

### Step 12: Merge Frames
Merge all the generated plots from the previous step into single frames.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/mergeframes.PNG" width="121" height="62" />

<img src="https://github.com/sgoldenlab/simba/blob/master/images/mergeplot.gif" width="600" height="348" />

1. Under **Merge Frames**, click `Merge Frames` and frames with all the generated plots will be combined and saved in the `project_folder/frames/output/merged` folder.

### Step 13: Create Videos
This step is to generate a video from the merged frames.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/createvideoini.PNG" width="200" height="100" />

1. Enter the `Bitrate` and the `File format`  

- `Bitrate`: [Bitrate](https://en.wikipedia.org/wiki/Bit_rate) is the number of bits per second. The symbol is bit/s. It generally determines the size and quality of video and audio files: the higher the bitrate, the better the quality and the larger the file size. If unsure, try setting bitrate to 2400.

- `File format`: The format of the output video, it can be mp4, mov, flv, avi, etc...

> **Note**: Please enter the file format without the ".".

2. Click on `Create Video`. 


