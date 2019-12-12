# SimBA Tutorial
This is a step by step tutorial to start using SimBA.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/simba.png" width="294" height="115" />

### Pipeline breakdown:
For processing a tracking dataset, the pipeline is split into a few sections. These sections are listed below along with their corresponding functions:

<img src="https://github.com/sgoldenlab/simba/blob/master/images/simbaworkflow.PNG" width="989" height="532" />


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
- [Plot Sklearn Results](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-10-plot-sklearn-results)
- [Plot Graphs](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-plot-graphs)
- [Merge Frames](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-12-merge-frames)
- [Create Video](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-13-create-video)

## Part 1: Create a new project
This section creates a new project for your tracking analysis.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/createproject2.PNG" width="270" height="150" />

### Step 1: Generate Project Config

In this step, you will be generating your main project folder with all the sub-directories.

1. Go to `File` and click on `Create a new project` The following windows will pop up.
<p align="center">
  <img width="702" height="432" src="https://github.com/sgoldenlab/simba/blob/master/images/createproject.PNG">
</p>

2. Navigate to `[ Generate project config ]` tab. Under **General Settings**, `Project Path` is the main working directory that your project will be created. Click on `Browse Folder` and select your working directory

3. `Project Name` is the name for your project. *Keep in mind that the project name cannot contain any spaces, use underscore "_" instead* 

4. Under `SML Settings`, put in the number of predictive classifiers that you wished. For an example, if you had three behaviors in your video, put 3 in the entry box.

5. Click <img src="https://github.com/sgoldenlab/simba/blob/master/images/addclassifier.PNG" width="153" height="27" /> and it creates a row as shown in the following image. Fill in the box with a behavior classifier.

<p align="center">
  <img width="385" height="106" src="https://github.com/sgoldenlab/simba/blob/master/images/classifier1.PNG">
</p>

6. `Video Settings` is the metadata of your videos. Filled in the information based on your videos.

7. Click `Generate Project Config` to generate your project. The project folder will be located at the `Project Path` that you specified

### Step 2: Import Videos into project folder
In this step, you can choose to import either one or multiple videos.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/importvids.PNG" width="346" height="251" />

#### To import multiple videos
1. Navigate to the `[ Import videos into project folder ]` tab.
2. Under `Import multiple videos`, click on `Browse Folder` to select your folder that contains all the videos that you wished to be included in your project.
3. Enter the type of your video *mp4*, *avi*, *mov*,etc. in the `Video type` entry box.
4. Click `Import multiple videos` 
>**Note**: It might take awhile for all the videos to be imported.
#### To import a single video
1. Under `Import single video`, click on `Browse File` to select your video
2. Click `Import a video`

### Step 3: Import DLC Tracking Data
In this step, you will import your csv tracking data.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/importcsv.PNG" width="300" height="202" />

#### To import multiple csv files
1. Navigate to the `[ Import tracking data ]` tab. Under `Import multiple csv file`, click on `Browse Folder` to select your folder that contains all the csv files that you wished to be included in your project.
2. Click `Import csv to project folder` 
#### To import a single csv file
1. Under `Import single csv file`, click on `Browse File` to select your video
2. Click `Import single csv to project folder`

### Step 4: Extract frames into project folder
This step will extract all the frames from every video that are in the videos folder. Once all the steps are completed, close the `Project Configuration` window.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/extractframeintop.PNG" width="491" height="126" />

## Part 2: Load project
This section loads a project that you have created.

### Step 1: Load Project Config
In this step, you will load the *project_config.ini* file that was created.
> **Note:** project_config.ini should always be loaded before starting anything else.
1. Go to `File` and click on `Load project` The following windows will pop up.

<p align="center">
  <img width="1255" height="380" src="https://github.com/sgoldenlab/simba/blob/master/images/loadproject.PNG">
</p>

2. Under **Load Project.ini**, click on `Browse File`. Then, go to the directory that you created your project and click on your *project folder*. Then, click on *project_folder* and then *project_config.ini*. Once this step is completed, it should look like the following instead of *No file selected*.

<p align="center">
  <img width="500" height="60" src="https://github.com/sgoldenlab/simba/blob/master/images/loadedprojectini.PNG">
</p>

In this image, you can see, the `Dekstop` is my selected working directory, `tutorial` is my project name, and the last two sections is always going to be `project_folder/project_config.ini` 

### Step 2 (Optional) : Import more DLC Tracking Data or videos
In this step, you can choose to import more csv files or videos. If you don't you can ignore and skip this step.

1. Click on the `{ Further imports (data/video/frames) ]` tab and from there you can import more data or videos into the project folder. The .csv files imported will be located in `project_folder/csv/input` and the videos imported will be located in `project_folder/videos`.

2. Once the videos are imported, you can extract frames by clicking `Extract frames` under **Extract further frames into project folder**

3. If you have existing frames of the videos in the project folder, you can import the folder containing the frames into the project by clicking `Browse Folder` to choose the frame folder and click `Import frames`. The folder will be imported to `project_folder/frames/input`

### Step 3: Set video parameters
In this step, you can customize the parameters for each of your videos. You will also be setting the **pixels per milimeter** of for your videos. You will be using a tool that requires the distance of a point to another point in order to calculate the **pixels per milimeter**. The real life distance between two points is call `Distance in mm`.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/setvidparameter.PNG" width="476" height="97" />

1. Under **Set video parameters(distances,resolution,etc.)**, the `Distance in mm` here is the distance between two points in milimeter. You can enter the values *(eg: 10,20)* and click `Auto populate Distance in mm in tables` and it will populate the table that you are going to see in the next step. If you leave it empty, it will just be zero.

2. Click on `Set Video Parameters` and the following windows will pop up.
<p align="center">
  <img width="700" height="350" src="https://github.com/sgoldenlab/simba/blob/master/images/videoinfo_table.PNG">
</p>

3. As you can see, I imported four videos and they are in the `Video` column. I have set the `Distance_in_mm` to 10 in the previous step, else, it will be 0.

4. Now, I can click on the values in the box and change it until I am satisfied. Then, click `Update distance_in_mm`, this will actually update the whole table.

5. Next, to get the `Pixels/mm`, click on `Video1`,which will be the first video in the table and the following window will pop up. The windows that pop up is a frame from your first video in the table.

<p align="center">
  <img width="300" height="400" src="https://github.com/sgoldenlab/simba/blob/master/images/getcoord1.PNG">
</p>

6. Now, double **left** click to select two points that you know the distance in real life. In this case, I know the two **pink dot that connects** has a distance of 10mm in real life.
<p align="center">
  <img width="300" height="400" src="https://github.com/sgoldenlab/simba/blob/master/images/getcoord2.PNG">
</p>

7. If you misplace the dots, you can double click on either of them and redo the step. Once you are done, you can hit `Esc`.

<p align="center">
  <img width="400" height="500" src="https://github.com/sgoldenlab/simba/blob/master/images/getcoord.gif">
</p>

8. If every steps are done correctly, the values should populate in the `Pixels/mm` column in the table.
<p align="center">
  <img width="700" height="350" src="https://github.com/sgoldenlab/simba/blob/master/images/videoinfo_table2.PNG">
</p>
9. Repeat the steps for every videos and once it is done, click `Save Data`. This will generate a csv file named **video_info.csv** in `/project_folder/log`

### Step 4: Outlier Correction

Outlier correction is used to correct gross tracking inaccuracies by detecting outliers based on movements and locations of body-parts in relation to the animal body-length.( For more details, please click [here](https://github.com/sgoldenlab/social_tracker/blob/master/Outlier_correction.pdf)

<img src="https://github.com/sgoldenlab/simba/blob/master/images/outliercorrection.PNG" width="156" height="109" />

1. Click on `Settings` and the following window will pop up.

<p align="center">
  <img width="300" height="400" src="https://github.com/sgoldenlab/simba/blob/master/images/outliercorrection2.PNG">
</p>

2. Select the body parts for Animal 1 and Animal 2 that you want to use to calculate a reference value. The reference value will be the mean Euclidian distance in millimeters between the two body parts in all frames.

3. Enter values for `Location Criterion` and `Movement Criterion`. 

- `Movement Criterion`. A bodypart will be flagged and corrected as a "movement outlier" if the bodypart moves the *reference value p x criterion value k* across two sequencial frames.

- `Location Criterion`. A bodypart will be flagged and correct as a "location outlier" if the distance between the bodypart and atleast two other bodyparts belonging to the same animal are bigger than *reference value p x criterion value k* within a frame.

### Step 5: Extract Features
Based on the coordinates of bodyparts in each frames, frame rate, and pixels per millimeter values, the code constructs and exhaustive set of features. This includes metric distances between bodyparts, angles, areas, movement, paths, and the deviations and rank in individual frames and across rolling windows.

1. Click `Extract Features`

### Step 6: Label Behavior
This step is to label the behavior in each frames of a video.

1. Click on `Select folder with frames`. In your project folder go to `/project_folder/frames/input/`, there should be folders that are named after your videos that contain all the video frames. Select one of the folder and the following window should pop up.

<p align="center">
  <img width="720" height="720" src="https://github.com/sgoldenlab/simba/blob/master/images/labelbe.PNG">
</p>

2. Please click [here](/docs/labelling_aggression_tutorial.md) to learn how to use this labelling behavior interface.

3. Click `Generate/Save` and it will generate a *.csv* file in */csv/targets_inserted*

### Step 7: Train Machine Model
This step is to train the machine model.

>**Note:** If you import existing models, you can skip this step and go straight to **Step 8** to run machine model.

#### Train single model

1. Click on `Settings` and the following window will pop up. 

<p align="center">
  <img width="378" height="712" src="https://github.com/sgoldenlab/simba/blob/master/images/machinemodelsettings.PNG">
</p>

>**Note:** If you have a .csv of meta data to load, you can click `Browse File` and then click `Load`. This will autofill all the settings in this step.

2. Under **Machine Model**, choose the machine model from the drop down menu,`RF` ,`GBC`,`Xboost`.

- `RF`: Random forest

- `GBC`: Gradient boost classifier

- `Xgboost`: eXtreme Gradient boost

3. Under **Model**, select the model you wish to train from the drop down menu.

4. Under **Hyperparameters**, select the hyperparameter settings for your model.(for more details, please click [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html))

- `RF N estimators`: Number of decision trees in the decision ensemble

- `RF Max features`: Number of features to consider when looking for the best split. 

- `RF Criterion`: The metric used to measure the quality of each split, i.e "gini" or "entropy"

- `Train Test Size`: The ratio of the dataset withheld for testing the model (e.g., 0.20)

- `RF Min sample leaf`: The minimum number of samples required to be at a leaf node. 

- `Under sample ratio`: The ratio of samples of the majority class to the minority class in the training data set. Applied only if "Under sample setting" is set to "Random undersample"  

- `Under sample setting`: "Random undersample" or "None". If "Random undersample", a random sample of the majority class will be used in the train set. The size of this sample will be taken as a ratio of the minority class and should be specified in the "under sample ratio" box below.

- `Over sample ratio`: The desired ratio of the number of samples in the minority class over the number of samples in the majority class after resampling

- `Over sample setting`: "SMOTE", "SMOTEEN" or "None". If "SMOTE" or "SMOTEEN", synthetic data will be generated in the minority class based on k-means to balance the two classes. (for more details, please click [here](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html))


5. Under **Model Evaluation Settings**,

- `Generate RF model meta data file`: Generate a .csv file containing hyperparameter settings associated with the model. 

- `Generate Example Decision Tree`: Save a random decision tree in .pdf format. Depends on [graphviz](https://graphviz.gitlab.io/)

- `Generate Classification Report`: Save a classification report in .png format. Depends on [yellowbrick](www.scikit-yb.org/)

- `Generate Features Importance Log`: Create a .csv file listing the importances (gini importances) of all features for the classifier. 

- `Generate Features Importance Bar Graph`: Creates a bar chart of the top N features based on gini importances. 

- `N feature importance bars`: Integer definiting the number of top features to be included in the bar chart

- `Compute Feature Permutation Importances`: Creates a .csv file listing the importances (permutation importances) of all features for the classifier. (for more details, please click [here](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html))

- `Generate Sklearn Learning Curve`: Creates a .csv file listing the f1 score at different test data sizes. (for more details, please click [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html))

- `LearningCurve shuffle K splits`: Number of cross validations applied at each test data size.

- `LearningCurve shuffle Data splits`: Number of test data sizes. 

- `Generate Precision Recall Curves`: Creates a .csv file listing precision at different recall values. 

6. Click the `Save settings for single model` button to save your settings into the *config.ini* file.

7. Click on `Train Model` in the main window.

#### Train multiple models

1. Click on `Settings`.

2. Under **Machine Model**, choose the machine model from the drop down menu,`RF` ,`GBC`,`Xboost`.

3. Under **Model**, select the model you wish to train from the drop down menu.

4. Then, set the **Hyperparameters**.

5. Click the `Save settings for multiple models` button. This generates a _meta.csv file. Repeat step 1. to 5. to generate multiple models.

6. Click on `Train Multiple Models` in the main window.


### Step 8: Run Machine Model
Run machine model explaination goes here

<img src="https://github.com/sgoldenlab/simba/blob/master/images/runrfmodel.PNG" width="343" height="132" />

1.  Under **Run Machine Model**, click on `Model Selection`. The following window with the classifier defined in the *project.ini* file will pop up.

<p align="center">
  <img width="312" height="256" src="https://github.com/sgoldenlab/simba/blob/master/images/rfmodelsettings.PNG">
</p>

2. Click on `Browse File` and select the model (*.sav*) to run.

3. Once all the model is chosen, click on `Set Model`.

4. Fill in the `Discrimination threshold` and click on `Set` to save the settings.

- `Discrimination threshold`: The level of probability required to define that the frame belongs to the target class

5. Fill in the ` Mininum behavior bout length` and click on `Set` to save the settings.

- `Minimum behavior bout length`:  The shortest possible length a behavioural bout can be. For example, the random forest makes the following aggression predictions for 5 consecutive frames in a 50 fps video: 1,0,0,0,1.This would mean, if we don't have a minimum bout length, that the animals fought twice in a few milliseconds. In reality they fought once standing still for a few milliseconds in the middle of the fight. This setting correct the 0s to 1s.

6. Click on `Run RF Model` to run the machine model.

### Step 9: Analyze Machine Results
In this step, there are three main analysis, which are `Analyze`, `Analyze distance/velocity`, and `Analyze severity`.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/analyzemachineresult.PNG" width="331" height="62" />

- `Analyze`: Generates descriptive statistics for each predictive classifier in the project, including the total time, the number of frames, total number of ‘bouts’, mean and median bout interval, time to first occurrence, and mean and median interval between each bout. The datetime stamped output csv file is saved in the logs folder.

- `Analyze distance/velocity`: Generated descriptive statistics for mean and median movements and distances between animals. The datetime stamped output csv file is saved in the logs folder.

- `Analyze severity`: Calculates the ‘severity’ of each frame classified as containing attack based on a user-defined scale. .  For example, on a 10-point scale, if the total body-part movement distances of both animals were in the top 10% percentile of movements in the video and a frame was scored as aggression, the aggressive attack was scored as a 10.
The datetime stamped output csv file is saved in the logs folder.

### Step 10: Plot Sklearn Results
Plot Sklearn result creates the frames with the animals and the predictions overlayed and the body-part circles overlayed

<img src="https://github.com/sgoldenlab/simba/blob/master/images/plotsklearn.PNG" width="170" height="58" />

1. Under **Plot Sklearn Results**, click `Plot Sklearn Results`.

### Step 11: Plot Graphs
The user is able to choose what plot to generate, **Gantt plot**, **Data plot**, **Path plot**, and **Distance plot**

<img src="https://github.com/sgoldenlab/simba/blob/master/images/plotgraphs.PNG" width="262" height="383" />

#### Gantt plot
Gantt plot generates gantt plot frames for all of the videos in the project

<img src="https://github.com/sgoldenlab/simba/blob/master/images/gantt_plot.gif" width="300" height="225" />

1. Under **Gantt plot**, click `Generate Gantt plot` and gantt plot frames will be generated in the `project_folder/frames/output/gantt_plots` folder.

#### Data plot
Generates 'live' data plot frames for all of the videos in the project

<img src="https://github.com/sgoldenlab/simba/blob/master/images/dataplot.gif" width="300" height="200" />

1. Under **Data plot**, click `Generate Data plot` and data plot frames will be generated in the `project_folder/frames/output/live_data_table` folder.

#### Path plot
Generates path plot frames for all of the videos in the project

<img src="https://github.com/sgoldenlab/simba/blob/master/images/pathplot.gif" width="199" height="322" />

1. Under **Path plot**, fill in the entry box.

- `Max Lines`: Integer specifying the max number of lines depicting the path of the animals. For example, if 100, the most recent 100 movements of animal 1 and animal 2 will be plotted as lines.

- `Severity Scale`: Integer specifying the scale on which to classify 'severity'. For example, if set to 10, all frames containing attack will be classified from 1 to 10.

- `Bodyparts`: String to specify the bodyparts to use in the path plot. For example, if Nose_1 and Centroid_2, the nose of animal 1 and the centroid of animal 2 will be represented in the path plot.

- `plot_severity`: Tick this box to include color-coded circles on the path plot that signify the location and severity of attack interactions.

2. Click on `Generate Distance plot`, and path plot frames will be generated in the `project_folder/frames/output/path_plots` folder.

#### Distance plot
Generates distance frames between two body parts for all of the videos in the project

<img src="https://github.com/sgoldenlab/simba/blob/master/images/distance_plot.gif" width="300" height="225" />

1. Fill in the `Body part 1` and `Body part 2`

- `Body part 1`: String that specifies the the bodypart of animal 1. Eg., Nose_1

- `Body part 2`: String that specifies the the bodypart of animal 1. Eg., Nose_2

2. Click on `Generate Distance plot`, and the distance plot frames will be generated in the `project_folder/frames/output/line_plot` folder.

### Step 12: Merge Frames
Merge all the generated plots from the previous step into a frame.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/mergeframes.PNG" width="121" height="62" />

<img src="https://github.com/sgoldenlab/simba/blob/master/images/mergeplot.gif" width="600" height="348" />

1. Under **Merge Frames**, click `Merge Frames` and frames with all the graph that were plotted will be combined and generated in the `project_folder/frames/output/merged` folder.

### Step 13: Create Video
This step is to generate a video from the merged frames.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/createvideoini.PNG" width="200" height="100" />

1. Enter the `Bitrate` and the `File format`  

- `Bitrate`: [Bitrate](https://en.wikipedia.org/wiki/Bit_rate) is the number of bits per second. The symbol is bit/s. It generally determines the size and quality of video and audio files: the higher the bitrate, the better the quality and the larger the file size

- `File format`: The format of the output video, it can be mp4, mov, flv, avi, etc...

> **Note**: Please enter the file format without the "."

2. Click `Create Video`


