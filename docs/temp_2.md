## Part 4:  Analyze Machine Results

Once the classifications have been generated, we may want to analyze descriptive statistics for the behavioral predictions. For example, we might like to know how much time the animals in each video enganged in behavior BtWGaNP, how long it took for each animal to start enganging in behavior BtWGaNP, how many bouts of behavior BtWGaNP did occur in each video, and what where the mean/median interval and bout length for behavior BtWGaNP. We may also want some descriptive statistics on the movements, distances and velocities of the animals. If applicable, we can also generate an index on how 'severe' behavior BtWGaNP was. To generate such descriptive statistics summaries, click on the `Run machine model` tab in the `Load project` menu. In the sub-menu `Analyze machine results`, you should see the following menu with four buttons:

![alt-text-1](/images/Analyze_1.PNG "data_log")

1. `Analyze machine predictions`: This button generates descriptive statistics for each predictive classifier in the project, including the total time, the number of frames, total number of ‘bouts’, mean and median bout interval, time to first occurrence, and mean and median interval between each bout. Clicking the button will display a pop up window with tick-boxes for the different metric options, and the user ticks the metrics that the output file should contain. The pop up window should look like this:

![alt-text-1](/images/Analyze_2.PNG "data_log")

Clicking on `Analyze` runs the selected desciptive statistics on all the CSV files in `project_folder/csv/machine_results` directory. A date-time stamped output csv file with the data is saved within the `/project_folder/log` folder. Check the main SimBA terminal window for the exact output filename and file path. For an example of the output for this analysis, click on [THIS LINK](https://github.com/sgoldenlab/simba/blob/master/docs/data_summary_example.csv). This example shows the expected summary statistics output when analysing 2 videos (`Together_1` and `Together_2`) and a single classifier (`Attack`) after ticking all of the boxes in the above pop-up menu. 

2. `Analyze distance/velocity`: This button generates descriptive statistics for mean and median movements and distances between animals. Clicking the button will display a pop-up window where the user selects how many animal, and which bosy-parts, the user wants to use to calculate the distance and velocity metrics. The pop up window should look like this:

![alt-text-1](/images/Analyze_3.PNG "data_log")

Clicking the `Run` buttons calculates the descriptive statistics on all the CSV files in `project_folder/csv/machine_results` directory. A date-time stamped output csv file with the data is saved in the `/project_folder/log` folder. The log file should look like this (click on image to enlarge):

![alt-text-1](/images/movement_log.JPG "movement_log")

3. `Time bins: machine predictions`: This button creates descriptive statistics of classification results within **user-defined time-bins**. SimBA will create a CSV file that contains data, **split into time bins**, for the following variables: 
* Total total number of classified ‘bouts’ (count)
* Total time in seconds of classified behavior 
* Time (within the time bin) of first behavior occurance
* Mean bout duration in seconds.
* Median bout duration in seconds.
* Mean interval between behavior bouts (within the time bin) in seconds
* Median interval between behavior bouts (within the time bin) in seconds

Clicking on `Time bins: machine predictions` brings up a pop-up menu where the user selects the size of each time bin in seconds. The pop up should look like this:

![alt-text-1](/images/Analyze_4.PNG "data_log")

Fill in the time bins in seconds. The data is saved in a timestaped CSV file in the in the `/project_folder/log` folder. For an example of the output, see [THIS](https://github.com/sgoldenlab/simba/blob/master/misc/Time_bins_ML_results_20220711185633.csv) file which includes the analysis of 2 videos and 2 classifiers. 

>Note: (i) If no behavior was expressed in a certain time bin, then the fields representing that time bin is missing. (i) If there was 1 behavior event within a time bin, then the `Mean event interval (s)` and `Median event interval (s)` fields are missing for that time-bin. 

4. `Time bins: Distance / velocity`: This button generates descriptive statistics for movements, velocities, and distances between animals in **user-defined time-bins**. Clicking this button brings up a pop-up menu where the user selects the size of each time bin in seconds (see screengrab avove). The data is saved in a timestaped CSV file in the in the `/project_folder/log` folder. Click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/Time_bins_movement_results_20220714053407.csv) for an example output file from this analyis. The first column (Video) denotes the video name analysed. The second column (Measurement) contains the name of the measure. The third column (Time bin #) denotes the time-bin number (the first time-bin is time-bin `0`). The last column denotes the value - i.e., in this example file, the animal named `Simon` moved 33.6cm in the first time-bin of the video names `Together_1`. 

5. `Classifications by ROI`: If you have drawn [user-defined ROIs](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md), then we can compute how much time, and how many classified behavioral bout started and ended in each user-defined ROI. Clicking on the `Classifications by ROI` brings up the following pop-up:

![alt-text-1](/images/clf_per_roi_01.png "clf_by_roi")

In this pop-up. Tick the checkboxes for which classified behaviors and ROIs you wish to analyze. Also tick the buttons for which measurements you want aggregate statistics for. In the `Select body-part` drop-down menu, select the body-part you shich to use as a proxy for the location of the behavior. Once filled in, click `Analyze classifications in each ROI`. A out-put data file will be saved in the `project_folder/logs` directory of your SimBA project.

6. `Analyze machine predictions: by severity`: This type of analysis is only relevant if your behavior can be graded on a scale ranging from mild (the behavior occurs in the presence of very little body part movements) to severe (the behavior occurs in the presence of a lot of body part movements). For instance, attacks could be graded this way, with 'mild' or 'moderate' attacks happening when the animals aren't moving as much as they are in other parts of the video, while 'severe' attacks occur when both animals are tussling at full force.  This button and code calculates the ‘severity’ of each frame classified as containing the behavior based on a user-defined scale. Clicking the severity button brings up the following menu: 

![alt-text-1](/images/severity_pop_up.png "severity_pop_up")

* **Classifier** dropdown: Select which classifier you want to calculate severity scores for.
* **Brackets** dropdown: Select the size of the severity scale. E.g., select **10** if you want to score your classifications on a 10-point scale.
* **Animals** dropdown: Select which animals body-parts you want to use to calculate the movement. E.g., select `ALL ANIMALS` to calculate the movement based on all animals and their body-parts. 
* **FRAME COUNT** checkbox: Check this box to get the results presented as **number of frames** in each severity bracket.
* **SECONDS** checkbox: Check this box to get the results presented as **number of seconds** in each severity bracket.

Click on RUN SEVERITY ANALYSIS. You can follow progress in the main SimBA terminal. The results are saved in the `project_folder/logs/` directory of your SimBA project. You can found an expected output of this analysis [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/severity_example_20230301090603.csv)

Congrats! You have now used machine models to classify behaviors in new data. To visualize the machine predictions by rendering images and videos with the behavioral predictions overlaid, and plots describing on-going behaviors and bouts, proceed to [*Part 5*](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-machine-predictions) of the current tutorial. 
