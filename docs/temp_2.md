## Part 4:  Analyze Machine Results

Once the classifications have been generated, we may want to analyze descriptive statistics for the behavioral predictions. For example, we might like to know how much time the animals in each video enganged in behavior BtWGaNP, how long it took for each animal to start enganging in behavior BtWGaNP, how many bouts of behavior BtWGaNP did occur in each video, and what where the mean/median interval and bout length for behavior BtWGaNP. We may also want some descriptive statistics on the movements, distances and velocities of the animals. If applicable, we can also generate an index on how 'severe' behavior BtWGaNP was, and/or split the different classification and movement statistics into time-bins. To generate such descriptive statistics summaries, click on the `Run machine model` tab in the `Load project` menu. In the sub-menu `Analyze machine results`, you should see the following buttons:

![alt-text-1](/images/data_analysis_0523_1.png "data_log")

1. `ANALYZE MACHINE PREDICTIONS: AGGREGATES`: This button generates descriptive statistics for each predictive classifier in the project, including the total time, the number of frames, total number of ‘bouts’, mean and median bout interval, time to first occurrence, and mean and median interval between each bout. Clicking the button will display a pop up window with tick-boxes for the different metric options, and the user ticks the metrics that the output file should contain. The pop up window should look like this:

![alt-text-1](/images/data_analysis_0523_2.png "data_log")

Clicking on `RUN` runs the selected desciptive statistics on all the files in `project_folder/csv/machine_results` directory. A date-time stamped output csv file with the data is saved within the `/project_folder/log` folder. Check the main SimBA terminal window for the exact output filename and file path. 

2. `ANALYZE MOVEMENT / VELOCITY: AGGREGATES`: This button generates descriptive statistics for distances and velocities. Clicking the button will display a pop-up window where the user selects how many animal, and which body-parts, the user wants to use to calculate the distance and velocity metrics. The pop up window should look like this:

![alt-text-1](/images/data_analysis_0523_3.png "data_log")

Clicking the `Run` buttons calculates the descriptive statistics on all the CSV files in `project_folder/csv/machine_results` directory. A date-time stamped output csv file with the data is saved in the `/project_folder/log` folder. 

> Note: When clicking the `Body-part` dropdown in the `ANALYZE MOVEMENT / VELOCITY: AGGREGATES` pop-up menu, you should see all the body-parts available in your project. You should also see options with the suffix **CENTER OF GRAVITY**, e.g., an option may be named `Animal 1 CENTER OF GRAVITY`. If you use this option, SimBA will estimate the centroid of the choosen animal and compute the moved distance and the velocity based on the estimated centroid. 

3. `ANALYZE MACHINE PREDICTIONS: TIME BINS`: Use this menu to compute descriptive statistics of classification within **user-defined time-bins**. This menu looks very similar to the menu used for aggregate machine classification computations, but has one additional entry-box at the bottom. In this bottom entry-box, enter the size of your time-bins in **seconds**. 

![alt-text-1](/images/data_analysis_0523_4.png "data_log")

>Note: (i) If no behavior was expressed in a certain time bin, then the fields representing that time bin is missing. (ii) If there was 1 behavior event within a time bin, then the `Mean event interval (s)` and `Median event interval (s)` fields are missing for that time-bin. 

4. `ANALYZE MOVEMENT / VELOCITY: TIME-BINS`: This button generates descriptive statistics for movements, velocities, and distances between animals in **user-defined time-bins**. Clicking this button brings up a pop-up menu very similar to the `ANALYZE MOVEMENT / VELOCITY: AGGREGATES`,  but has one additional entry-box at the bottom. In this bottom entry-box, enter the size of your time-bins in **seconds**. It also has a checkbox named `Create plots`. If the `Create plots` checkbox is ticked, SimBA will generate line plots, with one line plot per videos, representing the movement of your animals in the defined time-bins. 

![alt-text-1](/images/data_analysis_0523_4.png "data_log")

5. ``ANALYZE MACHINE PREDICTIONS: BY ROI``: If you have drawn [user-defined ROIs](https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md), then we can compute how much time, and how many classified behavioral bout started and ended in each user-defined ROI. Clicking on the `Classifications by ROI` brings up the following pop-up:

![alt-text-1](/images/data_analysis_0523_5.png "clf_by_roi")

In this pop-up. Tick the checkboxes for which classified behaviors and ROIs you wish to analyze. Also tick the buttons for which measurements you want aggregate statistics for. In the `Select body-part` drop-down menu, select the body-part you shich to use as a proxy for the location of the behavior. Once filled in, click `Analyze classifications in each ROI`. An output data file will be saved in the `project_folder/logs` directory of your SimBA project.

6. `Analyze machine predictions: by severity`: This type of analysis is only relevant if your behavior can be graded on a scale ranging from mild (the behavior occurs in the presence of very little body part movements) to severe (the behavior occurs in the presence of a lot of body part movements). For instance, attacks could be graded this way, with 'mild' or 'moderate' attacks happening when the animals aren't moving as much as they are in other parts of the video, while 'severe' attacks occur when both animals are tussling at full force.  This button and code calculates the ‘severity’ of each frame classified as containing the behavior based on a user-defined scale. Clicking the severity button brings up the following menu: 

![alt-text-1](/images/severity_pop_up.png "severity_pop_up")

* **Classifier** dropdown: Select which classifier you want to calculate severity scores for.
* **Brackets** dropdown: Select the size of the severity scale. E.g., select **10** if you want to score your classifications on a 10-point scale.
* **Animals** dropdown: Select which animals body-parts you want to use to calculate the movement. E.g., select `ALL ANIMALS` to calculate the movement based on all animals and their body-parts. 
* **FRAME COUNT** checkbox: Check this box to get the results presented as **number of frames** in each severity bracket.
* **SECONDS** checkbox: Check this box to get the results presented as **number of seconds** in each severity bracket.

Click on RUN SEVERITY ANALYSIS. You can follow progress in the main SimBA terminal. The results are saved in the `project_folder/logs/` directory of your SimBA project. You can found an expected output of this analysis [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/severity_example_20230301090603.csv)

Congrats! You have now used machine models to classify behaviors in new data. To visualize the machine predictions by rendering images and videos with the behavioral predictions overlaid, and plots describing on-going behaviors and bouts, proceed to [*Part 5*](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-machine-predictions) of the current tutorial. 
