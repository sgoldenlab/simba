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

>*Note*: If you haven't generated the predictive classifier yourself, and you have instead downloaded the predictive classifier or been given the predictive calssifier by someone else, we will also include information on how you can use that classifier to predict behaviors in your videos. 

1. In the Load Project menu, navigate to the **Run Machine Model** tab and you should see the following window. 

![](https://github.com/sgoldenlab/simba/blob/master/images/runrfmodel.PNG "rfmodelsettings1")

2. Click on `Model Selection`. The following window, containing the classifier names that were defined when you created the project, will pop up. The image below depicts a full suite of behavioral predictive classifiers relevant to aggression behavior, but yours should only show Behavior BtWGaNP. 

![](https://github.com/sgoldenlab/simba/blob/master/images/rfmodelsettings.PNG "rfmodelsettings")

3. Click on `Browse File` and select the model (*.sav*) file associated with the classifier name. If you are following along from [Scenario 1](https://github.com/sgoldenlab/simba/edit/master/docs/Scenario1.md), the *.sav* file will be saved in your earlier project, in the `project_folder\models\generated_models` directory or the `project_folder\models\validation\model_files` directory. You can also select an *.sav* file located in any other directory. For example, if you have downloaded a random forest model from [our OSF repository](https://osf.io/d69jt/), you can specify the path to that file here. 

Once the path has been selected, go ahead and modify the discrimination `Threshold` and `Minimum Bout` for each classifier separately. If you want to explore the optimal threshold for your classifier, go ahead and read [Scenario 1 - Critical validation step before running machine model on new data](https://github.com/sgoldenlab/simba/blob/simba_JJ_branch/docs/Scenario1.md#critical-validation-step-before-running-machine-model-on-new-data) on how to use the `Validate Model on Single Video` menu (this information is also repeated in brief in Step 4 below). The `Threshold` entry box accepts a value between 0 and 1. The `Minimum Bout` is a time value in milliseconds that represents the minimum length of a classified behavioral bout. To read more about the `Minimum Bout` - go ahead and read [Scenario 1 - Critical validation step before running machine model on new data](https://github.com/sgoldenlab/simba/blob/simba_JJ_branch/docs/Scenario1.md#critical-validation-step-before-running-machine-model-on-new-data)(this information is also repeated in brief in Step 4 below). 

>**Note**: In the real world you may want want to run mutiple classifiers on each video, one for each of the behaviors you are intrested in. **In such a scenario you have defined mutiple predictive calssifier names when you created the project in Step 1**. Each one will be displayed in the `Model Selection`, and you can specify a different path to a different *.sav* file for each of them. 

>**Important**: Each random forest expects a specific number of features. The number of calculated features is determined by the number of body-parts tracked in pose-estimation in DeepLabCut or DeepPoseKit. *For example*: if the input dataset contains the coordinates of three body parts, then a fewer number of features can be calculated than if 8 body parts were tracked. This means that you will get an error if you run a random forest model *.sav* file that has been generated using 8 body parts on a new dataset that contains only 3 body parts.

4. Fill in the `Discrimination threshold` and click on `Set` to save the settings.

- `Discrimination threshold`: This value represents the level of probability required to define that the frame belongs to the target class and it accepts a float value between 0.0 and 1.0.  In other words, how certain does the computer have to be that behavior BtWGaNP occurs in a frame, in order for the frame to be classified as containing behavior BtWGaNP? For example, if the discrimination theshold is set to 0.50, then all frames with a probability of containing the behavior of 0.5 or above will be classified as containing the behavior. For more information on classification theshold, click [here](https://www.scikit-yb.org/en/latest/api/classifier/threshold.html).

>*Note*: You can titrate the discrimination threshold to best fit your data. Decreasing the threshold will predict that the classified behavior is *more* frequent, while increasing the threshold will predict that the behaviors as *less* frequent.

5. Fill in the `Minimum behavior bout length` and click on `Set` to save the settings.

- `Minimum behavior bout length (ms)`:  This value represents the minimum length of a classified behavioral bout. **Example**: The random forest makes the following predictions for behavior BtWGaNP over 9 consecutive frames in a 50 fps video: 1,1,1,1,0,1,1,1,1. This would mean, if we don't have a minimum bout length, that the animals enganged in behavior BtWGaNP for 80ms (4 frames), took a brake for 20ms (1 frame), then again enganged in behavior BtWGaNP for another 80ms (4 frames). You may want to classify this as a single 180ms behavior BtWGaNP bout, rather than two separate 80ms BtWGaNP bouts. If the minimum behavior bout length is set to 20, any interruption in the behavior that is 20ms or shorter will be removed and the example behavioral sequence above will be re-classified as: 1,1,1,1,1,1,1,1,1 - and instead classified as a single 180ms BtWGaNP bout. If your videos in your project are recorded at different frame rates then SimBA will account for this when correcting for the `Minimum behavior bout length`. 

6. Click on `Run RF Model` to run the machine model on the new experimental data. You can follow the progress in the main SimBA terminal window. A message will be printed once all the behaviors have been predicted in the experimental videos. New CSV files, that contain the predictions together with the features and pose-estimation data, are saved in the `project_folder/csv/machine_results` directory. 

## Part 4:  Analyze Machine Results

Once the classifications have been generated, we may want to analyze descriptive statistics for the behavioral predictions. For example, we might like to know how much time the animals in each video enganged in behavior BtWGaNP, how long it took for each animal to start enganging in behavior BtWGaNP, how many bouts of behavior BtWGaNP did occur in each video, and what where the mean/median interval and bout length for behavior BtWGaNP. We may also want some descriptive statistics on the movements, distances and velocities of the animals. If applicable, we can also generate an index on how 'severe' behavior BtWGaNP was. To generate such descriptive statistics summaries, click on the `Run machine model` tab in the `Load project` menu. In the sub-menu `Analyze machine results`, you should see the following menu with three buttons:

<img src="https://github.com/sgoldenlab/simba/blob/master/images/analyzemachineresult.PNG" width="331" height="62" />

1. `Analyze`: This button generates descriptive statistics for each predictive classifier in the project, including the total time, the number of frames, total number of ‘bouts’, mean and median bout interval, time to first occurrence, and mean and median interval between each bout. Clicking the button will run the desciptive statistics on all the csv files in `project_folder/csv/machine_results` directory. A date-time stamped output csv file with the data is saved in the `/project_folder/log` folder. The analysis log file should look something like this (click on image to enlarge):

![alt-text-1](/images/data_log.JPG "data_log")

2. `Analyze distance/velocity`: This button generates descriptive statistics for mean and median movements and distances between animals. Clicking the button will run the desciptive statistics on all the csv files in `project_folder/csv/machine_results` directory. A date-time stamped output csv file with the data is saved in the `/project_folder/log` folder. The log file should look like this (click on image to enlarge):

![alt-text-1](/images/movement_log.JPG "movement_log")

3. `Analyze severity`: This button and analysis is only relevant if your behavior can be graded on a scale ranging from mild (the behavior occurs in the presence of very little body part movements) to severe (the behavior occurs in the presence of a lot of body part movements). For instance, attacks could be graded this way, with 'mild' or 'moderate' attacks happening when the animals aren't moving as much as they are in other parts of the video, while 'severe' attacks occur when both animals are tussling at full force.  This button and code calculates the ‘severity’ of each frame classified as containing the behavior based on a user-defined scale. This calculation requires that the user defines the 'scale' of severity. This can be filled in the entry box above the `Analyze severity` button. An example of the expected output can be downloaded [here](https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_severity.csv).  **Note:** This code currently only works for the predictive classifier 'Attack'.   

**Example:** The user sets a 10-point scale by entring the number 10 in the entry box above the `Analyze severity` button. One frame is predicted to contain an attack, and the total body-part movements of both animals in that frame is in the top 10% percentile of movements in the entire video. In this frame, the attack will be scored as a 10 on the 10-point scale. Clicking the button will run the 'severity' statistics on all the csv files in `project_folder/csv/machine_results` directory. A date-time stamped output csv file with the data is saved in the `/project_folder/log` folder.

Congrats! You have now used machine models to classify behaviors in new data. To visualize the machine predictions by rendering images and videos with the behavioral predictions overlaid, and plots describing on-going behaviors and bouts, proceed to [*Part 5*](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-machine-predictions) of the current tutorial. 

## Part 5:  Visualizing machine predictions

In this part of the tutorial we will create visualizations of features and machine learning classifications that you have generated. This includes images and videos of the animals with prediction overlays, gantt plots, line plots, paths plots, heat plots and data plot. In this step the different frames can also be merged into video mp4 format. Please note that generating images and videos of the animals with prediction overlays requires that the frames for the videos to have been generated (extracted) in **Part 1** of this tutorial. Also note that the rendering of images and videos can take up a lot of hardrive space, especially if your videos are recorded at high frame rates, are long duration, or high resolution. One suggestion, if you are short on harddrive space, is to only generate frames for a few select videos (more information below). We are currently working on updating SimBA to give the user the oppurtunity to work with compressed videos rather than frames during the `visualization` steps.   

1. In the `Load Project` menu, click on the `Visualization` tab and you will see the following menus. 

![alt-text-1](https://github.com/sgoldenlab/simba/blob/master/images/Visualization_menu_2.PNG "viz")

2. **Visualize predictions**. On the left of the `Visualization` menu, there is a sub-menu with the heading `Sklearn visualization`. This button grabs the frames of the videos in the project, and draws circles at the location of the tracked body parts, the convex hull of the animal, and prints the behavioral predictions on top of the frame together with the classified time spent enganging in the behavior.  

>*Note I*: SimBA uses a "one-click-interface" unless explicitly stated. For example, if you click on `Sklearn visualization` twice, the images will be generated twice. 

>*Note II*: The code will run through each CSV file in your `project_folder\csv\machine_results` directory, find a matching folder of frames in your `project_folder\frames\input` directory, and save new frames in the `project_folder\frames\output\sklearn_results` directory, contained within a new folder named after the video file. If you would like to generate visualizations for only a select CSV file, remove the files you want to omit from visualizing from the `project_folder\csv\machine_results` directory. For example, you can manually create a temporary `project_folder\csv\machine_results\temp` directory and place the files you do *not* want to visualize in this temporary folder. 

>*Note III*: If you do not want to create any further visualizations (e.g., gantt plots, path plots, data plots, or distance plots), you can stop at this point of the tutorial. Furthermore, if you want to create a video from the frames generated in the `project_folder\frames\output\sklearn_results` directory, you can do so by using the [SimBA tools menu](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md) and the [`Merge images to video` tool](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#merge-images-to-video). You could use the [`Merge images to video` tool](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#merge-images-to-video) This will generate a video like the following [example](https://www.youtube.com/watch?v=lGzbS7OaVEg&feature=youtu.be).

3. **Generate gantt plots**. In the `Visualization` menu, and the sub-menu `Visualizations`, use the first button named `Generate gantt plot` to create a gantt plot (a.k.a, horizontal bar chart or harmonogram) displaying the occurances, length, and frequencies of behavioural bouts of interest. These charts are similar to the ones generated by [popular propriatory software for behavioral analysis](https://www.noldus.com/the-observer-xt/data-analysis), and can look like this when generated through SimBA: 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/gantt_plot.gif" width="300" height="225" />

>*Note*: The code will run through each csv file in your `project_folder\csv\machine_results` directory, and generate one gantt frame for each frame of the video and save these frames in the `project_folder\frames\output\gantt_plots` directory, contained within a new folder named after the video file. If you would like to generate gantt charts for only a select csv file, remove the files you want to omit from visualizing  gantt charts from the `project_folder\csv\machine_results` directory. For example, you can manually create a temporary `project_folder\csv\machine_results\temp` directory and place the files you do **not** want to visualize in this temporary folder. If you'd like to create a video or gif from the gantt frames, you can do so by using the [SimBA tools menu](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md) and the [`Merge images to video`](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#merge-images-to-video) or [Generate gifs](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#generate-gifs) tools. 

4. **Generate data plot**. In the `Visualization` menu, and the sub-menu `Visualizations`, use the second button named `Generate data plot` to create a frames that display the velocities, movements, and distances between the animals:

<img src="https://github.com/sgoldenlab/simba/blob/master/images/dataplot.gif" width="300" height="200" />

The data plots currently displays the below metrics. These metrics can, currently, **not** be defined by the user.
  * Distance between the noses of the two animals
  * Distance between the centroids of the two animals
  * Current velocity of Animal 1
  * Current velocity of Animal 2
  * Mean velocity of Animal 1
  * Mean velocity of Animal 2
  * Total distance moved for Animal 1
  * Total distance moved for Animal 2
 
>*Note*: The code will run through each csv file in your `project_folder\csv\machine_results` directory, and generate one data frame for each frame of the video and save it in the `project_folder\frames\output\live_data_table` directory, contained within a new folder named after the video file. If you would like to generate data plots for only a select csv file, remove the files you want to omitt from visualizing  gantt charts from the `project_folder\csv\machine_results` directory. For example, you can manually create a temporary `project_folder\csv\machine_results\temp` directory and place the files you do **not** want to visualize in this temporary folder. If you'd like to create a video or gif from the data frames, you can do so by using the [SimBA tools menu](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md) and the [`Merge images to video`](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#merge-images-to-video) or [Generate gifs](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#generate-gifs) tools. 

4. **Generate path plot**. In the `Visualization` menu, and the sub-menu `Visualizations`, use the third menu named `Path plot` to generate graphs depicting the movement of the animals, the location of the animals, where the behaviors of interest occurs. The 'severity' and location of a behavior can also be visulized with a color-coded heat map. The output may look something like this:

<img src="https://github.com/sgoldenlab/simba/blob/master/images/pathplot.gif" width="199" height="322" />

The generation of the path plots requires several user-defined parameters:

- `Max Lines`: Integer specifying the max number of lines depicting the path of the animals. For example, if 100, the most recent 100 movements of animal 1 and animal 2 will be plotted as lines.

- `Severity Scale`: Integer specifying the scale on which to classify 'severity'. For example, if set to 10, all frames containing attack behavior will be classified from 1 to 10 (see above). 

- `Bodyparts`: String to specify the body parts tracked in the path plot. For example, if Nose_1 and Centroid_2, the nose of animal 1 and the centroid of animal 2 will be represented as the current location of the animals in the path plot.

If you are using the recommended [16 body-part, 2 mice setting for pose estimation tracking](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#pose-estimation-body-part-labelling) hover your computer mouse over the title of the entry box to see alternatives for `Bodyparts`.

- `plot_severity`: Tick this box to include color-coded circles on the path plot that signify the location and severity of attack interactions.

After specifying these values, click on `Generate Path plot`.

>*Note*: After clicking on `Generate Path plot`, the code will run through each csv file in your `project_folder\csv\machine_results` directory, and generate one path plot frame for each frame of the video and save it in the `project_folder\frames\output\path_plots` directory, contained within a new folder named after the video file. if you would like to generate path plots for only a select csv file, remove the files you want to omit from visualizing path plots for from the `project_folder\csv\machine_results` directory. For example, you can manually create a temporary `project_folder\csv\machine_results\temp` directory and place the files you do **not** want to visualize in this temporary folder. If you'd like to create a video or gif from the path frames, you can do so by using the [SimBA tools menu](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md) and the [`Merge images to video`](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#merge-images-to-video) or [Generate gifs](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#generate-gifs) tools. 

5. **Generate distance plot**. In the `Visualization` menu, and the sub-menu `Visualizations`, use the fourth sub-menu titled `Distance plot` to create frames that display the distances between the animals, like in this example gif:

<img src="https://github.com/sgoldenlab/simba/blob/master/images/distance_plot.gif" width="300" height="225" />

The generation of the distance plots requires two user-defined parameters:

- `Body part 1`: String that specifies the the bodypart of animal 1 (e.g., Nose_1)

- `Body part 2`: String that specifies the the bodypart of animal 1 (e.g., Nose_2)

If you are using the recommended [16 body-part, 2 mice setting for pose estimation tracking](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#pose-estimation-body-part-labelling) hover your computer mouse over the title of the entry box to see alternatives for `Body part 1` and `Body part 2`/.

Click on `Generate Distance plot`, and the distance plot frames will be generated in the `project_folder/frames/output/line_plot` folder.

>*Note*: After clicking on `Generate Distance plot`, the code will run through each csv file in your `project_folder\csv\machine_results` directory, and generate one distance plot frame for each frame of the video and save it in the `project_folder\frames\output\line_plot` directory, contained within a new folder named after the video file. if you would like to generate path plots for only a select csv file, remove the files you want to omitt from visualizing distance plots for from the `project_folder\csv\machine_results` directory. For example, you can manually create a temporary `project_folder\csv\machine_results\temp` directory and place the files you do **not** want to visualize in this temporary folder. If you'd like to create a video or gif from the distance frames, you can do so by using the [SimBA tools menu](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md) and the [`Merge images to video`](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#merge-images-to-video) or [Generate gifs](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#generate-gifs) tools. 


6. **Generate heatmap**. In the `Visualization` menu, and the sub-menu `Visualizations`, use the fifth sub-menu titled `Heatmap` to create frames displaying a heatmap and spatial information on where the classified behaviors occur according to user-defined scale. SimBA accepts six different palettes for heatmaps:

<p align="center">
  <img width="400" height="137" src="https://github.com/sgoldenlab/simba/blob/master/images/SimBA_pallettes.PNG">
</p>


| palette | gif |palette|gif |
| ------------- | ------------- |------------- |------------- |
| magma         | <img src="https://github.com/sgoldenlab/simba/blob/master/images/magma_heatmap.gif" width="300" height="350">  |inferno        |<img src="https://github.com/sgoldenlab/simba/blob/master/images/inferno_heatmap.gif" width="300" height="350">   |
| jet           | <img src="https://github.com/sgoldenlab/simba/blob/master/images/jet_heatmap.gif" width="300" height="350">  |plasma        |<img src="https://github.com/sgoldenlab/simba/blob/master/images/plasma_heatmap.gif" width="300" height="350">   |
| viridis       | <img src="https://github.com/sgoldenlab/simba/blob/master/images/viridis_heatmap.gif" width="300" height="350">   |gnuplot       |<img src="https://github.com/sgoldenlab/simba/blob/master/images/gnuplot_heatmap.gif" width="300" height="350">   |


To generate heatmaps, SimBA needs several user-defined variables:

- `Bin size(px)` : Pose-estimation coupled with supervised machine learning in SimBA gives information on the location of an event at the single pixel resolution, which is too-high of a resolution to be useful in heatmap generation. In this entry box, insert an integer value (e.g., 100) that dictates, in pixels, how big a location is. For example, if the user inserts *100*, and the video is filmed using 1000x1000 pixels, then SimBA will generate a heatmap based on 10x10 locations (each being 100x100 pixels large).   

- `# Scale increments` : How many color increments on the heatmap that should be generated. For example, if the user inputs *11*, then a 11-point scale will be created (as in the gifs above). 

- `Scale increment (s)` : How many seconds should constitute a color increment. For example, if the users specifies *0.1*, then a 100ms increase in the time of enganging in the bahvior in a specific location results in a color increment. 

- `Color Palette` : Which color pallette to use to plot the heatmap. See the gifs above for different output examples. 

- `Target` : Which target behavior to plot in the heatmap. 

Once filled in, click on `Generate heatmap`:

>*Note*: After clicking on `Generate heatmap`, the code will run through each csv file in your `project_folder\csv\machine_results` directory, and generate one heatmap frame for each frame of the video and save it in the `project_folder\frames\output\heatmap` directory, contained within a new folder named after the video file. if you would like to generate heatmaps for only a select csv file, remove the files you want to omitt from visualizing distance plots for from the `project_folder\csv\machine_results` directory. For example, you can manually create a temporary `project_folder\csv\machine_results\temp` directory and place the files you do **not** want to visualize in this temporary folder. If you'd like to create a video or gif from the heatmap frames, you can do so by using the [SimBA tools menu](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md) and the [`Merge images to video`](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#merge-images-to-video) or [Generate gifs](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#generate-gifs) tools. 

6. **Merge Frames**. If you have followed through all of **Part 5** of this tutorial, you should have generated several graphs of your machine classifications and extracted data (i.e., gantt plots, line plots, path plots, data plots, sklearn plots). These images are stored in different sub-directories in the `project_folder\frames\output` folder. Now you may want to merge all these frames into single frames, and later into a video, to more readily observe the behavior of interest and its different expression in experimental groups, like in the following video example:   

<img src="https://github.com/sgoldenlab/simba/blob/master/images/mergeplot.gif" width="600" height="348" />

To merge all the generated plots from the previous step into single frames, navigate to the following button in the `Visualization` menu and click it: 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/mergeframes.PNG" width="121" height="62" />

When clicking on `Merge Frames`, all the generated plots for each video will be combined and saved in the `project_folder/frames/output/merged` folder, in subfolders named after each video. 

>*Note*: After clicking on `Merge Frames`, the code will look in each folder contained in the `project_folder\frames\output` directory, for subfolders with matching names (e.g. `project_folder\frames\output\gantt_plots\Video1`, `project_folder\frames\output\line_plot\Video1`, `project_folder\frames\output\sklearn_results\Video1` etc...). If any of the folders are missing, or if any of the matching folder differs in the numer of frames contained within them, you will get an error. 

7. **Create Videos**. At this point, you may want to merge the frames contained within subfolders of the `project_folder/frames/output/merged` directory to video files, with one video for each of the subdirectories in the `project_folder/frames/output/merged` folder. In the `Visualization` menu, navigate to the following sub-menu:

<img src="https://github.com/sgoldenlab/simba/blob/master/images/createvideoini.PNG" width="200" height="100" />

To create a video from the frames, SimBA requires two user-defined parameters: two video file format, and the bitrate:

- `Bitrate`: Bitrate is the number of bits per second. It generally determines the size and quality of video and audio files: the higher the bitrate, the better the quality and the larger the file size. If unsure, try setting the bitrate to something small,like **2400**. To read more about bitrate, click [here](https://help.encoding.com/knowledge-base/article/understanding-bitrates-in-video-files/). 

- `File format`: Enter the format of the output video, it can be mp4, mov, flv, avi, mpeg. Please enter the file format without the ".". (e.g., enter *mp4*, not *.mp4*). 

The output video(s) from the merged frames will be stored in the `project_folder\frames\output\merged` directory. 

>*Note*: If the videos are very large, and you would like to down-sample the resolution of the videos to make them smaller, you can do so by using the [SimBA tools menu](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md) and the [Downsample video](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#downsample-video) tool. In the SimBA tools menu, you can also [crop](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#crop-video), and [trim](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#shorten-videos) specific videos as well as many more things.

Go to [Scenario 3](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario3.md) to read about how to update a classifier with further annotated data.

Go to [Scenario 4](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario4.md) to read about how to analyze new experimental data with a previously started project.
