# Post-classification Validation (detecting false-positives)

Post-classification validation generates video clips of each detected (classified) behavioral bout in the videos in your SimBA project. SimBA will create clips for each video that is represented by a (i) file inside the `project_folder/csv/machine_results` directory, and (ii) a video file inside the `project_folder/videos` directory. This tool can help us visualize any potential false-positives in the classifaction results.  

1. Load your SimBA project and click on the `CLASSIFIER VALIDATION CLIPS` button in [Visualizations] tab and you should see the following sub-menu:

![](/images/clf_validation_0423.png)

* The `SECONDS` entry-box: Sometimes it can be difficult to understand the context if we only see the classified behavior bout, and we want to introduce a little temporal "padding" pre- and post-behavioral bout to get a better understand of what is going on. In this entry-box, enter the **number of seconds** 
pre- and post-behavioral bout that should be included in the output video clips. 

* The `CLASSIFIER` drop-down menu: Select which behavioral classifications should be visualized in the output video clips.
 
* The `TEXT COLOR` drop-down menu: Select the color of the text overlay that best suits your videos. 

* The `HIGHLIGHT TEXT COLOR` drop-down menu: If you have introduced some temporal "padding" in the `SECONDS` entry-box, we may want to highlight the text overlay when the behavior is actually happening to make it more salient. In this dropdown, select the color you wish to use for the text overlay when the behavior is happening. If `NONE`, then the text color selected in the `TEXT COLOR` dropdown will be used through-out the validation videos. 

* The `VIDEO SPEED` drop-down menu: We may want to slow down (or speed up) the validation videos to better undersatnd what is going on. In this dropdown select how much you'd like to slow down (or speed up) your videos. If `1.0`, the videos will be created in their original speed. If `0.5`, then the videos will be created at half speed. 

* The `CREATE ONE CLIP PER BOUT` checkbox: If you want to store the results as a separete video file for each detected bout, tick the `One clip per bout` checkbox. For example, if SimBA has detected 50 behavioral bouts, ticking this checkbox will create 50 separate video files. 

* The `ONE CLIP PER VIDEO` checkbox: If you want to store the results from each separate video as a single output video file, tick the `One clip per video` checkbox. For example, if SimBA has detected 50 behavioral bouts in `Video 1`, a single video file representing `Video 1` with the 50 behavioral bouts will be created. 

## How to use it

1. Enter 1 or 2 in the `SECONDS` entry box. *Note: the larger the seconds, the longer the duration of the video.**

2. Select the target behavior from the `CLASSIFIER` dropdown box.

3. Tick the `CREATE ONE CLIP PER BOUT` and/or the `ONE CLIP PER VIDEO` checkboxes. 

4. Click the `RUN` button and the videos will be generated in `/project_folder/frames/output/classifier_validation`. The name of the video will be formated in the following manner: **CLASSIFIER NAME** + **VIDEO NAME** + **number of bouts** OR **all_events** + .mp4

![](/images/classifiervalidation.gif)

Author [Simon N](https://github.com/sronilsson), [JJ Choong](https://github.com/inoejj)
