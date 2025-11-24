# SimBA Tutorial:

To faciliate the initial use of SimBA, we provide several use scenarios. We have created these scenarios around a hypothetical experiment that take a user from initial use (completely new start) all the way through analyzing a complete experiment and then adding additional experimental datasets to an initial project.

All scenarios assume that the videos have been [pre-processed](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md) and that [DLC behavioral tracking CSV dataframes](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md) have been created.

# **Hypothetical Experiment**:
Three days of resident-intruder testing between aggressive CD-1 mice and subordinante C57 intruders. Each day of testing has 10 pairs of mice, for a total of 30 videos recorded across 3 days. Recordings are 3 minutes in duration, in color, at 30fps.

# **Scenario 2**: Using a classifier on new experimental data...
In this scenario you have either are now ready to do one of two things. 

(i) You have generated a classifier yourself which performance you are happy with. For example, you have followed [Scenario 1](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario1.md), and generated the classifier for "Behavior that Will Get a Nature Paper (Behavior BtWGaNP)" and its working well. 

(ii) Or, you have received a behavioral classifier generated somewhere else, and now you want to use the classifier to score Behavior BtWGaNP on your experimental videos. For example, you have downloaded the classifier from our [OSF repository](https://osf.io/d69jt/).

## Part 1: 'Clean up your project', or create a new project. 

We will need start with a project directory tree that does not contain any other data than the data we want to analyze. If you are coming along from [Scenario 1](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario1.md), you will have a project tree already. However, this project tree contains the files used to create the BtWGaNP classifier: if you look in the subdirectories of the `project_folder/csv/input` directory, you will see the 19 CSV files we used to generate the project. If we continue using this project, SimBA will see these CSV files and analyze these files in addition to your Experimental data. Thus, one option is to manually remove these files from the subdirectories of our project (see the legacy tutorial for [Scenario 4](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario4.md#part-1-clean-up-your-project--or-alternatively-create-a-new-project) where we take this approach), or you could use the `Archive Processed files` function in the `Load Project` tab described in the new [Scenario 4 tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario4_new.md#part-1-clean-up-your-project--or-alternatively-create-a-new-project) and shown in the image below.  

![](https://github.com/sgoldenlab/simba/blob/master/images/InkedCreate_project_12_LI.jpg "create_project")

Another alternative is to [create a new](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project) project that only contains the data from our Experiment. In this Scenario, we will create a [new project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project). 

Go ahead and create a new project with your experimental data, follow the instructions for creating a new project in either of these tutorials: [1](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project-1), [2](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config). Instead of using your pilot data as indicated in the tutorial from [Scenario 1](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project-1), use the data and videos for the experiment you want to analyze.

**Important**: In the final *Step 4* of the [tutorial for creating a new project](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-1-create-a-new-project-1), we extract the frames from the imported videos. Having the frames is only necessery if you wish to visualize the predictive classifications generated in this current Scenario 2. If you do not want to visualize the machine predictions you can skip this step. However, we recommend that you at least visualize the machine predictions using one or a few videos to gauge its performance. How to visualize the machine predictions is described in [Part 5](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-machine-predictions) of this tutorial. 

## Part 2: Load project and process your tracking data

In [Part 1](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-1-clean-up-your-project-or-create-a-new-project) above, we created a project that contains your experimental data. To continue working with this project, we **must** load it. To load the project and process your experimental data, follow the instructions for **Step 1 to 5** in either the [Scenario 1 tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md) or [Part I of the generic tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config]). 

In this current Scenario 2, you can **ignore Steps 6-7 of the tutorials**, which deals with annotating data and creating classifiers. 

However, **Step 1-5** of the [Scenario 1 tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md), which we need to complete,  performs [outlier correction in the tracking](https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf) and [extracts features](https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv), which we will need to do before analyzing our experimental data.

## Part 3: Run the classifier on new data

At this point we have Experimental data, which has been corrected for outliers and with features extracted, and we want to predict behavior BtWGaNP in these videos. 

> [!NOTE]
> If you haven't generated the predictive classifier yourself, and you have instead downloaded the predictive classifier or been given the predictive calssifier by someone > else, we will also include information on how you can use that classifier to predict behaviors in your videos. 

1. In the Load Project menu, navigate to the **[Run Machine Model]** tab and you should see the `RUN MODELS` button. Clicking on thus button gives you the following pop-up.
   This pop-up displays a table, with one row for each of the classifiers in your project, ane we will go thorugh which each of the settings mean:

<img width="1477" height="509" alt="model_parameters_run" src="https://github.com/user-attachments/assets/5a15260d-5a66-4b08-834c-a5cb5dbcd16d" />

2. For each classifier, click on <kbd>`BROWSE FILE`</kbd>, and select the model (*.sav*) file associated with the classifier name. If you are following along from [Scenario 1](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario1.md), the *.sav* file will be saved in your earlier project, in the `project_folder\models\generated_models` directory or the `project_folder\models\validation\model_files` directory. You can also select an *.sav* file located in any other directory. For example, if you have downloaded a random forest model from [our OSF repository](https://osf.io/d69jt/), you can specify the path to that file here. 

3. Once the path has been selected, go ahead and modify the `DISCRIMINATION THRESHOLD` and `MINIMUM BOUT LENGTH` for each classifier separately. If you want to explore the optimal threshold for your classifier, go ahead and read [Scenario 1 - Critical validation step before running machine model on new data](https://github.com/sgoldenlab/simba/blob/simba_JJ_branch/docs/Scenario1.md#critical-validation-step-before-running-machine-model-on-new-data) on how to use the `Validate Model on Single Video` menu (this information is also repeated in brief in Step 4 below). The `THRESHOLD` entry box accepts a value between 0 and 1. The `MINIMUM BOUT LENTH` is a time value in milliseconds that represents the minimum length of a classified behavioral bout. To read more about the `Minimum Bout` - go ahead and read [Scenario 1 - Critical validation step before running machine model on new data](https://github.com/sgoldenlab/simba/blob/simba_JJ_branch/docs/Scenario1.md#critical-validation-step-before-running-machine-model-on-new-data)(this information is also repeated in brief in Step 4 below). 

> [!IMPORTANT]
> Each model expects a specific number of features. The number of calculated features is determined by the number of body-parts tracked in pose-estimation. *For example*: if the input dataset contains the coordinates of three body parts, then a fewer number of features can be calculated than if 8 body parts were tracked. This means that you will get an error if you run a random forest model *.sav* file that has been generated using 8 body parts on a new dataset that contains only 3 body parts.

> [!NOTE]
> `Discrimination threshold`: This value represents the level of probability required to define that the frame belongs to the target class and it accepts a float value between 0.0 and 1.0.  In other words, how certain does the computer have to be that behavior BtWGaNP occurs in a frame, in order for the frame to be classified as containing behavior BtWGaNP? For example, if the discrimination theshold is set to 0.50, then all frames with a probability of containing the behavior of 0.5 or above will be classified as containing the behavior. For more information on classification theshold, click [here](https://www.scikit-yb.org/en/latest/api/classifier/threshold.html).
>
> You can titrate the discrimination threshold to best fit your data. Decreasing the threshold will predict that the classified behavior is *more* frequent, while increasing the threshold will predict that the behaviors as *less* frequent.
> 
> `Minimum behavior bout length (ms)`:  This value represents the minimum length of a classified behavioral bout. **Example**: The random forest makes the following predictions for behavior BtWGaNP over 9 consecutive frames in a 50 fps video: 1,1,1,1,0,1,1,1,1. This would mean, if we don't have a minimum bout length, that the animals enganged in behavior BtWGaNP for 80ms (4 frames), took a break for 20ms (1 frame), then again enganged in behavior BtWGaNP for another 80ms (4 frames). You may want to classify this as a single 180ms behavior BtWGaNP bout, rather than two separate 80ms BtWGaNP bouts. If the minimum behavior bout length is set to 20, any interruption in the behavior that is 20ms or shorter will be removed and the example behavioral sequence above will be re-classified as: 1,1,1,1,1,1,1,1,1 - and instead classified as a single 180ms BtWGaNP bout. If your videos in your project are recorded at different frame rates then SimBA will account for this when correcting for the `Minimum behavior bout length`.

4. Once filled in, click th <kbd>RUN</kbd> button. You can see how many files that SimBA will analyze using these classifiers in the header encompassing the run button (these are the number of files in your `project_folder/csv/features_extracted` directory). You can follow the progress in the main SimBA terminal window. A message will be printed once all the behaviors have been predicted in the experimental videos. New CSV files, that contain the predictions together with the features and pose-estimation data, are saved in the `project_folder/csv/machine_results` directory. 


## Part 4:  Analyze Machine Results

Once the classifications have been generated, we may want to analyze descriptive statistics for the behavioral predictions. For example, we might like to know how much time the animals in each video enganged in behavior BtWGaNP, how long it took for each animal to start enganging in behavior BtWGaNP, how many bouts of behavior BtWGaNP did occur in each video, and what where the mean/median interval and bout length for behavior BtWGaNP. We may also want some descriptive statistics on the movements, distances and velocities of the animals. If applicable, we can also generate an index on how 'severe' behavior BtWGaNP was, and/or split the different classification and movement statistics into time-bins. To generate such descriptive statistics summaries, click on the `Run machine model` tab in the `Load project` menu. In the sub-menu `Analyze machine results`, you should see the following buttons:

![alt-text-1](/images/data_analysis_0523_1.png "data_log")

1. `ANALYZE MACHINE PREDICTIONS: AGGREGATES`: This button generates descriptive statistics for each predictive classifier in the project, including the total time, the number of frames, total number of ‘bouts’, mean and median bout interval, time to first occurrence, and mean and median interval between each bout. Clicking the button will display a pop up window with tick-boxes for the different metric options, and the user ticks the metrics that the output file should contain. The pop-up window should look like this:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/agg_descriptive_stats_clf.png" />
</p>


- In the `MEASUREMENTS` frame, select which descriptive statistics you wish to compute for each of the videos in your project. 

- In the `CLASSIFIERS` frame, select which classifiers you wish to compute the `MEASUREMENTS` for.
  
- We may want to create a summary CSV file that contains the exact start times, end times, and duration of each classified event for the behaviors selected in the `CLASSIFIERS` frame. To do this, check the `Detailed bout data` checkbutton. For an example of expected summary file, click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/detailed_bout_data_summary_20231011091832.csv).

- At times, we may want also want to output a reference to the video meta data (e.g., the number of frames or the length of the video) we can help us compute proportions. To add such data to the output, tick the relevant boxes in the `METADATA` frame.

- Be default, SimBA outputs a CSV file, where every unique classifier measurement in each video is represented by a row. Some prefers the data to be transposed so that every unique classifier measurement is represented by a column and each video as rows. If you prefer the latter output, then tick the `Transpose output` checkbox. 

Next, clicking `RUN` executes the selected desciptive statistics on all the files in `project_folder/csv/machine_results` directory. A date-time stamped output csv file with the data is saved within the `/project_folder/log` folder. Check the main SimBA terminal window for the exact output filename and file path. 


2. `ANALYZE DISTANCES / VELOCITY: AGGREGATES`: This button generates descriptive statistics for distances and velocities. Clicking the button will display a pop-up window where the user selects how many animal, and which body-parts, the user wants to use to calculate the distance and velocity metrics. The pop up window should look like this:

![alt-text-1](/images/data_analysis_0523_3.png "data_log")

Clicking the `Run` buttons calculates the descriptive statistics on all the CSV files in `project_folder/csv/machine_results` directory. A date-time stamped output csv file with the data is saved in the `/project_folder/log` folder. 

> Note: When clicking the `Body-part` dropdown in the `ANALYZE DISTANCES / VELOCITY: AGGREGATES` pop-up menu, you should see all the body-parts available in your project. You should also see options with the suffix **CENTER OF GRAVITY**, e.g., an option may be named `Animal 1 CENTER OF GRAVITY`. If you use this option, SimBA will estimate the centroid of the choosen animal and compute the moved distance and the velocity based on the estimated centroid. 

3. `ANALYZE MACHINE PREDICTIONS: TIME BINS`: Use this menu to compute descriptive statistics of classification within **user-defined time-bins**. This menu looks very similar to the menu used for aggregate machine classification computations, but has one additional entry-box at the bottom. In this bottom entry-box, enter the size of your time-bins in **seconds**. 

![alt-text-1](/images/data_analysis_0523_4.png "data_log")

>Note: (i) If no behavior was expressed in a certain time bin, then the fields representing that time bin is missing. (ii) If there was 1 behavior event within a time bin, then the `Mean event interval (s)` and `Median event interval (s)` fields are missing for that time-bin. 

4. `ANALYZE DISTANCES / VELOCITY: TIME-BINS`: This button generates descriptive statistics for movements, velocities, and distances between animals in **user-defined time-bins**. Clicking this button brings up a pop-up menu very similar to the `ANALYZE DISTANCES / VELOCITY: AGGREGATES`,  but has one additional entry-box at the bottom. In this bottom entry-box, enter the size of your time-bins in **seconds**. It also has a checkbox named `Create plots`. If the `Create plots` checkbox is ticked, SimBA will generate line plots, with one line plot per videos, representing the movement of your animals in the defined time-bins. 

![alt-text-1](/images/data_analysis_0523_5.png "data_log")

5. ``ANALYZE MACHINE PREDICTIONS: BY ROI``: If you have drawn [user-defined ROIs](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md), then we can compute how much time, and how many classified behavioral bout started and ended in each user-defined ROI. Clicking on the `Classifications by ROI` brings up the following pop-up:

![alt-text-1](/images/data_analysis_0523_6.png "clf_by_roi")

In this pop-up. Tick the checkboxes for which classified behaviors and ROIs you wish to analyze. Also tick the buttons for which measurements you want aggregate statistics for. In the `Select body-part` drop-down menu, select the body-part you shich to use as a proxy for the location of the behavior. Once filled in, click `Analyze classifications in each ROI`. An output data file will be saved in the `project_folder/logs` directory of your SimBA project.


6. ``Analyze machine predictions: BY SEVERITY``. This type of analysis is only relevant if your behavior can be graded on a scale ranging from mild (the behavior occurs in the presence of very little body part movements) to severe (the behavior occurs in the presence of a lot of body part movements). For instance, attacks could be graded this way, with 'mild' or 'moderate' attacks happening when the animals aren't moving as much as they are in other parts of the video, while 'severe' attacks occur when both animals are tussling at full force. This button and code calculates the ‘severity’ of each frame (or bout) classified as containing the behavior based on a user-defined scale. Clicking the severity button brings up the following menu. We go through the meaning of each setting below:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/severity_analysis_pop_up.png" />
</p>

**CLASSIFIER:** This drop-down shows the classifiers in your SimBA project. Select the classifier which you want to score the severity for. 

**BRACKETS:** Select the size of the severity scale. E.g., select 10 if you want to score your classifications on a 10-point scale.

**ANIMALS:** Select which animals body-parts you want to use to calculate the movement. E.g., select ALL ANIMALS to calculate the movement based on all animals and their body-parts.

**BRACKET TYPE:** If `QUANTIZE`, then SimBA  creates **N equally sized bins** (with N defined in the BRACKETS dropdown). If `QUANTILE`, 
SimBA forces an equal number of frames into each bin and creates **N unequally sized bins**. For more detailed, see the differences between [pandas.qcut](https://pandas.pydata.org/docs/reference/api/pandas.qcut.html) and [pandas.cut](https://pandas.pydata.org/docs/reference/api/pandas.cut.html).

**DATA TYPE:** When binning the severity, we can either obtain severity scores for each (i) individual classifified frame, or (ii) for each classified bout. Select `BOUTS` to get a severity score for each classified bout and `FRAMES` to get a severity score per classified frame. 

**MOVEMENT NORMALIZATION TYPE:** When creating the severity bins, we can either (i) use all the movement data represented by all files with classifications (all data within the project_folder/csv/machine_results directory). This selection will results in a single bin reference scale that are applied equally to all videos, or (ii) we can create the scales by referencing the movement only in each of the videos themselves. This selection will results in different severity scale bins for each of the different videos. 

**SAVE BRACKET DEFINITIONS:** If ticked, SimBA will save a CSV log file containing the bin definitions for each analysed video in the project (saved inside the project_folder/logs directory). Note: If `MOVEMENT NORMALIZATION TYPE` is set to `ALL VIDEOS` this log should show the same bin definitions for each video. 

**VISUALIZE**: If ticked, SimBA will generate visualization example clips.

**SHOW POSE-ESTIMATED LOCATIONS**: If ticked, SimBA will include pose-estimated body-part locations (as circles) in the video clips.

**VIDEO SPEED**: The FPS of the example clips relative to the original video FPS. E.g., if `1`, the clips will be saved at origginal speed. If `0.5`, the clips will be saved at hald the original speed.

**CLIP COUNT**: How many example clips we want to create. E.g., if `10`, SimBA will randomly sample `10` classified bouts to create 10 example video clips. If the selected clip count is higher then the number of total classified bouts, then SimBA will create clips for all bouts. 
Click on RUN SEVERITY ANALYSIS. You can follow progress in the main SimBA terminal. The results are saved in the `project_folder/logs/` directory of your SimBA project.

Congrats! You have now used machine models to classify behaviors in new data. To visualize the machine predictions by rendering images and videos with the behavioral predictions overlaid, and plots describing on-going behaviors and bouts, proceed to [*Part 5*](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-machine-predictions) of the current tutorial. 

## Part 5:  VISUALIZING RESULTS

In this part of the tutorial we will create visualizations of machine learning classifications and the features which you have generated. This includes images and videos of the animals with *prediction overlays, gantt plots, line plots, paths plots, heat maps and data plot etc.* These visualizations can help us understand the classifier(s), behaviors, and differences between experimental groups. 

To access the visualization functions, click the `[Visualizations]` tab.

### VISUALIZING CLASSIFICATIONS

On the left of the `Visualization` tab menu, there is a sub-menu with the heading `DATA VISUALIZATION` with a button named `VISUALIZE CLASSIFICATIONS`. Use this button to create videos with classification visualization overlays, similar to what is presented [HERE](https://youtu.be/lGzbS7OaVEg). Clicking this button brings up the below pop-up menu allowing customization of the videos and how they are created. We will go through each of the settings in the visualization options in turn:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/viz_clf_202510.png" height="700"/>
</p>

* **BODY-PART VISUALIZATION THRESHOLD** (0.0-1.0): In this entry-box, enter the **minimum** pose-estimation detection probability threshold required for the body-part to be included in the visualization. For example, enter `0.0` for **all** body-part predictions to be included in teh visualization. Enter `1.0` for only body-parts detected with 100% certainty to be visualized. 

* **STYLE SETTINGS**: By default, SimBA will **auto-compute** suitable visualization (i) font sizes, (ii) spacing between text rows, (iii) font thickness, and (iv) pose-estimation body-part location circles which depend on the resolution of your videos. If you do **not** want SimBA to auto-compute these attributes, go ahead and and **un-tick** the `Auto-compute font/key-point sizes checkbox, and fill in these values manually in each entry box. 

* **VISUALIZATION SETTINGS**:
  - **CPU CORES**: How many CPU cores to use to create the visualization(s). The more cores the faster the videos will be created, but the more the RAM memory will be taxed. Defaults to half of your available cores.
  - **USE GPU**: If `True`, parts of the video processing will use the GPU to speed the video creation up. Defaults to `False`.
  - **SHOW GANTT PLOT**: Chose to display a gantt plot to the right of the video displaying the bouts of detected behaviors. This can either be a static image (displaying the detected behaviors in the entire video) or a dynamic video showing all the behaviors detected up until the current time of the video. 
  - **SHOW TRACKING (POSE)**: If checked, the pose in the video is displayed as circles. 
  - **SHOW ANIMAL BOUNDING BOXES**: If checked, an axis-aligned bounding box is displayed for each animal encompassing the animals detected body-parts.
  - **SHOW ANIMAL NAMES(S)**: If checked, the animals names are printed next to each animal.
  - **CREATE VIDEO**: Tick the `Create video` checkbox to generate `.mp4` videos with classification result overlays.
  - **CREATE FRAMES**: Tick the `Create frames` checkbox to generate `.png` files with classification result overlays (NOTE: this will create one png file for each frame in each video. If you are concerned about storage, leave this checkbox unchecked). 
  - **INCLUDE TIMERS OVERLAY**: Tick the `Include timers overlay` checkbox to insert the cumulative time in seconds each classified behavior has occured in the top left corner of the video. 
  - **ROTATE VIDEO 90°**: Tick the `Rotate video 90°` checkbox to rotate the output video 90 degrees clockwise relative to the input video. 

* **RUN**:

  - **SINGLE VIDEO**: To create classification visualizations for a single video, select the video in the `Video` drop-down menu and click the `Create single video` button. You can follow the progress in the main SimBA terminal. The results will be stored in the `project_folder/frames/output/sklearn_results` directory of your SimBA project.
  - **MULTIPLE VIDEO**: To create classification visualizations for all videos in your project, click the `Create multiple videos` button. You can follow the progress in the main SimBA terminal. The results will be stored in the `project_folder/frames/output/sklearn_results` directory of your SimBA project.

### VISUALIZING GANTT CHARTS 

Clicking the `VUSIALIZE GANTT` button brings up a pop-up menu allowing us to customize gantt charts. Gantt charts are broken horizontal bar charts allowing us to insepct when and for how long each of our classified behaviors occur as in the gif below. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/clf_viz_2.png" />
</p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/gantt_plot.gif" width="600" height="450" />
</p>


* **STYLE SETTINGS**: Use this menu to specify the resultion of the Gantt plot videos and/or frames. Furthermore, use the `Font size` entry box to specify the size of the y- and x-axis label text sizes. Use the `Font rotation degree` entry-box to specify the rotation of the y-axis classifier names (set to `45` by default which is what is visualized in the gif above). 

* **VISUALIZATION SETTINGS**:
  - **Create video**: Tick the `Create video` checkbox to generate gantt plots `.mp4` videos.
  - **Create frames**: Tick the `Create frames` checkbox to generate gantt plots `.png` files (NOTE: this will create one png file for each frame in each video. If you are concerned about storage, leave this checkbox unchecked). 
  - **Create last frame**: Tick the `Create last frame` checkbox to generate a gantt plots `.png` file representing the entire video.
  - **Multiprocess videos (faster)**: Creating gantt videos and/or images can be computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this using multiprocessing over multiple cores on your computer. To use multi-processing, tick the `Multiprocess videos (faster)` checkbox. Once ticked, the `CPU cores` dropdown becomes enabled. This dropdown contains values between `2` and the number of cores available on your computer, with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your gantt charts. 

* **RUN**:
  - **SINGLE VIDEO**: To create gantt chart visualizations for a single video, select the video in the `Video` drop-down menu and click the `Create single video` button. You can follow the progress in the main SimBA terminal. The results will be stored in the `project_folder/frames/output/sklearn_results` directory of your SimBA project.
  - **MULTIPLE VIDEO**: To create gantt chart visualizations for all videos in your project, click the `Create multiple videos` button. You can follow the progress in the main SimBA terminal. The results will be stored in the `project_folder/frames/output/sklearn_results` directory of your SimBA project.

>*Note*: If you'd like to create a  gif from the gantt frames, you can do so by using the [SimBA tools menu](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md) and the [Generate gifs](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#generate-gifs) tool. 

### VISUALIZING CLASSIFICATION PROBABILITIES

SimBA can create line plots depicting the *classification probability* that a specific behavior is occuring in the current frame across the video.
On the left of the `Visualization` menu, a button named `VISUALIZE PROBABILITIES`. Clicking this button brings up the below sub-menu allowing users to customize the videos and how they are created.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/clf_viz_3.png" height="700"/>
</p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/prob_plot.gif" height="700"/>
</p>

* **STYLE SETTINGS**: Use this menu to specify the resultion of the probability plot videos and/or frames.
  - **RESOLUTION**: Use this dropdown to select the resolution (size) of the output video and/or output frames.
  - **LINE COLOR**: Use this dropdown to specify the color of the line in the charts.
  - **FONT SIZE**: In this entry-box, enter the font size of the y- and x-axis labels and tick labels. (e.g., `10`)
  - **LINE WIDTH**: In this entry-box, enter the thickness of the line in the chart (e.g., `6`).
  - **CIRCLE SIZE**: In this entry-box, enter the size of the circle representing the current frame probability value (e.g., `20`)

* **VISUALIZATION SETTINGS**:
  - **CLASSIFIER**: Use this drop down menu to select the classifier you which to create the line plot for.
  - **CREATE FRAMES**: Tick the `Create frames` checkbox to create probability plots `.png` files (NOTE: this will create one png file for each frame in each video. If you are concerned about storage, leave this checkbox unchecked). 
  - **CREATE VIDEOS**: Tick the `Create video` checkbox to create probability plots `.mp4` videos. 
  - **Multiprocess videos (faster)**: Creating probability videos and/or images can be computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this using multiprocessing over multiple cores on your computer. To use multi-processing, tick the `Multiprocess videos (faster)` checkbox. Once ticked, the `CPU cores` dropdown becomes enabled. This dropdown contains values between `2` and the number of cores available on your computer, with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your probability charts. 

* **RUN**:
  - **SINGLE VIDEO**: To create probability chart visualizations for a single video, select the video in the `Video` drop-down menu and click the `Create single video` button. You can follow the progress in the main SimBA terminal. The results will be stored in the `project_folder/frames/output/probability_plots` directory of your SimBA project.
  - **MULTIPLE VIDEO**: To create probability chart visualizations for all videos in your project, click the `Create multiple videos` button. You can follow the progress in the main SimBA terminal. The results will be stored in the `project_folder/frames/output/probability_plots` directory of your SimBA project.

>*Note*: If you'd like to create a gif from the probability_plots frames, you can do so by using the [SimBA tools menu](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md) and the [Generate gifs](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#generate-gifs) tool. 

### VISUALIZING PATH PLOTS

SimBA can create path plots depicting the location of the animal(s), their paths, as well the locations of the classified behaviors. In the [Visualizations] tab, click the [VISUALIZE PATHS] button, which brings up the below pop-up menu. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/create_path_plots_0823.png" height="700"/>
</p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/pathplot.gif" height="500"/>
</p>

* **STYLE SETTINGS**:
  - **AUTO-COMPUTE STYLES**: By default, SimBA will **auto-compute** suitable visualization styles which depend on the resolution of your videos. If you do **not** want SimBA to auto-compute these attributes, go ahead and and **un-tick** the `Auto-compute styles` checkbox, and fill in these values manually in each entry box. 
  - **MAX PRIOR LINES** (int): Number of milliseconds for which the movement path is diplayed. E.g., a value of `2000` will display the movement path for the most recent 2s. 
  - **LINE WIDTH** (int): The width of the lines representing the movement path. E.g., `6`.
  - **CIRCLE SIZE** (int): The size of the circle representing the animals current location. E.g., `20`.
  - **FONT SIZE**: The size of the font text of the animals name. `E.g., 3`.
  - **FONT THICKNESS**: The thickness (boldness) of the font text of the animals name. `E.g., 2`.
  - **BACKGROUND COLOR**: The background color of the path plots. E.g., `White`.

* **SEGMENTS**: Use this menu to create path plots for a specific segment of the videos rather than the entire videos.
  - **START TIME**: Fill in the time where the path plot video should start. E.g., if I want to discard the first 4 minutes of the video in the path plot, set `Start time` to 00:04:00.
   - **END TIME**: Fill in the time where the path plot video should end. E.g., if I want to end my path plot at the 8 minute mark, set `End time` to 00:08:00.

> Note: Say you have a 10 minute video, but you want the path plots to represent movements between minutes 4-8 (and thus discard the first 4 and last 2 minutes from the path plot) set the start time to 00:04:00 and end time to 00:08:00.


* **CHOOSE CLASSIFICATION VISUALIZATION**: Use this menu to specify if and how the location of classified events are printed on the path plots.
  - **INCLUDE CLASSIFICATION LOCATIONS**: Check this box to include the location of classified events in the path plot.
  
  - You should see a row for each classifier, and three drop-down menues for each classifier. In the example screengrab above, I have two classifiers (Classifier 1: *Attack*, Classifier 2: Sniffing). In the second drop-down, select which **color** the circles depicting the location of the classified events should have. In the third dropdown, select the **size** the circles depicting the location of the classified events should have.

  >Note: The classified event location will be inferred to be in the first animals body-part location

* **CHOOSE BODY-PARTS**: Use this menu to specify which body-parts of the animals will represent their location.
  - **# ANIMALS**: Use this drop-down to specify how many animals you want to visualize paths for.
  
  - You should see a row for each animal, and two drop-down menues per. In the example screengrab above, I have two animals. In the first drop-down, select the body-part which you want to represent the path. In the second drop-down, select which **color** the circles and lines depicting the location of the animal should have.
 
* **VISUALIZATION SETTINGS**:
  - **Create video**: Tick the `Create video` checkbox to generate `.mp4` video path plots.
  - **Create frames**: Tick the `Create frames` checkbox to generate `.png` files with path plots (NOTE: this will create one png file for each frame in each video. If you are concerned about storage, leave this checkbox unchecked). 
  - **Create last frame**: Create a single `.png` image representing the path plot at the end of each video.
  - **Include animal names**: Print the animal name at the current location of each animal in each frame. If unchecked, the animal name as text will not be shown.
   - **Multiprocess videos (faster)**: Creating heatmaps is computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this in part by using multiprocessing over the multiple cores on your computer. To use multi-processing, tick the `Multiprocess videos (faster)` checkbox. Once ticked, the `CPU cores` dropdown becomes enabled. This dropdown contains values between `2` and the number of cores available on your computer with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your visualizations. 

* **RUN**:
  - **SINGLE VIDEO**: Use this menu to create a *single* path visualization video. The `Video` drop-down will contain all the videos in your `project_folder/machine_results` directory. Choose which video you want to create a path visualization for. Once choosen, click the `Create single video` button. You can follow the progress in the main SimBA terminal window. Once complete, a new video and/or frames will be saved in the `project_folder/frames/output/path_plots` directory. 
  - **MULTIPLE VIDEO**: Use this menu to create a path visualization video for every video in your project. After clicking the `Create multiple videos` button. You can follow the progress in the main SimBA terminal window. Once complete, one new video and/or frames folder for every input video will be saved in the `project_folder/frames/output/path_plots` directory.

### VISUALIZING DISTANCE PLOTS

SimBA can create distance plots depicting the distance between different body-parts and/or animals across the videos. In the [Visualizations] tab, click the [VISUALIZE DISTANCES] button, which brings up the below pop-up menu. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/clf_viz_5.png" height="700"/>
</p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/distance_plot.gif" height="500"/>
</p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/distance_plot_termites.gif" height="500"/>
</p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/clf_viz_8.png" height="500"/>
</p>



* **STYLE SETTINGS**:
  - **RESOLUTION**: Use the drop-down to set the size of the output video(s) and/or frames.
  - **FONT SIZE** (int): The size of the text representing the y- and x-axis labels and graph title.
  - **LINE WIDTH** (int): The width of the lines representing the animal body-part distances.

* **CHOOSE DISTANCES**: 
  - **# DISTANCES**: Use the drop-down to specify how many distances (lines) you want to display in the distance plot. E.g., the two gifs above 1 and 4 distances, respectively. 
  - Once you have selected a number of lines, the table show be populated with as many rows as distances chosen, with **three** drop-down menus per row. Use the first two drop-down menus to select the two body-parts which distance in-between you want depicted in the output video and/or frames. Use the third right-most drop-down to select the color of that specific line.

* **VISUALIXATION SETTINGS**: 
  - **Create video**: Tick the `Create video` checkbox to generate `.mp4` videos with distance plots.
  - **Create frames**: Tick the `Create frames` checkbox to generate `.png` files with distance plots (NOTE: this will create one png file for each frame in each video. If you are concerned about storage, leave this checkbox unchecked). 
  - **Create last frame**: Create a single `.png` image representing the body-part distances at the end of each video (as in the bottom image above).

* **RUN**:
  - **SINGLE VIDEO**: Use this menu to create a *single* distance visualization video. The `Video` drop-down will contain all the videos in your `project_folder/machine_results` directory. Choose which video you want to create a distance visualization for. Once choosen, click the `Create single video` button. You can follow the progress in the main SimBA terminal window. Once complete, a new video and/or frames will be saved in the `project_folder/frames/output/line_plots` directory. 
  - **MULTIPLE VIDEO**: Use this menu to create a distance visualization video for every video in your project. After clicking the `Create multiple videos` button. You can follow the progress in the main SimBA terminal window. Once complete, one new video and/or frames folder for every input video will be saved in the `project_folder/frames/output/line_plots` directory.


### VISUALIZING CLASSIFICATION HEATMAPS

SimBA can create heatmap videos and/or images representing the location of classified events. For an idea of how classification heatmaps works, see [THIS VIDEO](https://youtu.be/O41x96kXUHE).

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/clf_viz_6.png" height="500"/>
</p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/clf_viz_7.png" height="500"/>
</p>

* **STYLE SETTINGS**:
  - **PALETTE**: Pick the heatmap color palette. For examples, [CLICK HERE](https://matplotlib.org/stable/gallery/color/colormap_reference.html)
  - **SHADING**: Pick the shading/smoothing. The left image above was created using *Gouraud*, the right using *Flat* shading.
  - **CLASSIFIER**: Pick the classifier to plot in the heatmap.
  - **BODY-PART**: Pick the body-part which represents the location the classified events.
  - **MAX TIME SCALE (S)**: Pick the time, in seconds, which represents the maximum color intensity in the heatmap. Choose `Auto-compute` to let SimBA find the max in the video.
  - **BIN SIZE (MM)**: Pick the size of each location in the image. For more information on bin sizes, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-5-miscellaneous-roi-tools)

* **VISUALIXATION SETTINGS**: 
  - **Create video**: Tick the `Create video` checkbox to generate `.mp4` videos heat maps.
  - **Create frames**: Tick the `Create frames` checkbox to generate `.png` files with heat map plots (NOTE: this will create one png file for each frame in each video. If you are concerned about storage, leave this checkbox unchecked). 
  - **Create last frame**: Create a single `.png` image representing the classification heat maps at the end of each video. 
   - **Multiprocess videos (faster)**: Creating heatmaps is computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this in part by using multiprocessing over the multiple cores on your computer. To use multi-processing, tick the `Multiprocess videos (faster)` checkbox. Once ticked, the `CPU cores` dropdown becomes enabled. This dropdown contains values between `2` and the number of cores available on your computer with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your visualizations. 

* **RUN**:
  - **SINGLE VIDEO**: Use this menu to create a *single* heatmap visualization. The `Video` drop-down will contain all the videos in your `project_folder/machine_results` directory. Choose which video you want to create a distance visualization for. Once choosen, click the `Create single video` button. You can follow the progress in the main SimBA terminal window. Once complete, a new video and/or frames will be saved in the `project_folder/frames/output/heatmaps_classifier_locations` directory. 
  - **MULTIPLE VIDEO**: Use this menu to create a heatmap visualization for every video in your project. After clicking the `Create multiple videos` button. You can follow the progress in the main SimBA terminal window. Once complete, one new video and/or frames folder for every input video will be saved in the `project_folder/frames/output/heatmaps_classifier_locations` directory.



### VISUALIZING DATA TABLES

In the `Visualization` sub-menu, use the second button named `VISUALIZE DATA PLOTS` to create a frames that display the velocities and movements of animals:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/data_table_ex1.gif" height="500"/>
</p>

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/data_table_ex2.gif" height="500"/>
</p>

* **STYLE SETTINGS**:
  - **RESOLUTION**: The size of the output video and/or frames in pixels.
  - **DECIMAL ACCURACY**: The number of floating points in the values displayed.
  - **BACKGROUND COLOR**: The background color of the data tables.
  - **HEADER COLOR**: The colors of the headers in the data table.
  - **FONT THICKNESS**: The thickness of the font in teh the table.

* **CHOOSE BODY-PARTS**: 

* **VISUALIXATION SETTINGS**: 
  - **Create video**: Tick the `Create video` checkbox to generate `.mp4` videos data plots.
  - **Create frames**: Tick the `Create frames` checkbox to generate `.png` files with data plots (NOTE: this will create one png file for each frame in each video. If you are concerned about storage, leave this checkbox unchecked). 

* **RUN**:
  - **SINGLE VIDEO**: Use this menu to create a *single* data table visualization. The `Video` drop-down will contain all the videos in your `project_folder/machine_results` directory. Choose which video you want to create a distance visualization for. Once choosen, click the `Create single video` button. You can follow the progress in the main SimBA terminal window. Once complete, a new video and/or frames will be saved in the `project_folder/frames/output/live_data_table` directory. 
  - **MULTIPLE VIDEO**: Use this menu to create a data table visualization for every video in your project. After clicking the `Create multiple videos` button. You can follow the progress in the main SimBA terminal window. Once complete, one new video and/or frames folder for every input video will be saved in the `project_folder/frames/output/live_data_table` directory.


### MERGING (CONCATENATING VIDEOS)

Next, we may want to merge (concatenate) several of the videos we have created in the prior steps into a single video file. To do this, click the `MERGE FRAMES` button in the [VISUALIZATIONS] tab, and you should see this pop up to the left:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/frm_merge.png" />
</p> 

Begin by selecting how many videos you want to concatenate together in the `VIDEOS #` drop-down menu and click `SELECT`. A table, with one row representing each of the videos, will show up titled `VIDEO PATHS`. Here, click the `BROWSE FILE` button and select the videos that you want to merge into a single video. 

Next, in the `JOIN TYPE` sub-menu, we need to select how to join the videos together, and we have 4 options:

* MOSAIC: Creates two rows with half of your choosen videos in each row. If you have an unequal number of videos you want to concatenate, then the bottom row will get an additional blank space. 
* VERTICAL: Creates a single column concatenation with the selected videos. 
* HORIZONTAL: Creates a single row concatenation with the selected videos. 
* MIXED MOSAIC: First creates two rows with half of your choosen videos in each row. The video selected in the `Video 1` path is concatenated to the left of the two rows. 

Finally, we need to choose the resolution of the videos in the `Resolution width` and the `Resolution height` drop-down videos. **If choosing the MOSAIC, , VERTICAL, or horizontal join type, this is the resolution of each panel video in the output video. If choosing MIXED MOSAIC, then this is the resolution of the smaller videos in the panel (to the right)**. 

After clicking `RUN`, you can follow the progress in the main SimBA terminal and the OS terminal. Once complete, a new output video with a date-time stamp in the filename is saved in the `project_folder/frames/output/merged` directory of your SimBA project.

<img src="https://github.com/sgoldenlab/simba/blob/master/images/mergeplot.gif" width="600" height="348" />

Go to [Scenario 3](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario3.md) to read about how to update a classifier with further annotated data.

Go to [Scenario 4](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario4.md) to read about how to analyze new experimental data with a previously started project.


##
Author [Simon N](https://github.com/sronilsson)
