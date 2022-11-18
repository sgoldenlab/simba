# Post-classification Validation (detecting false-positives)

Post-classification validation generates video clips of each detected (classified) behavioral bout in the videos in your SimBA project. SimBA will create clips for each video that is represented by a (i) file inside the `project_folder/csv/machine_results` directory, and (ii) a video file inside the `project_folder/videos` directory. This tool can help us visualize any potential false-positives in the classifaction results.  

1. Load your SimBA project and click on the [Classifier validation] tab, and you should see the following sub-menu:

![](/images/clf_validation_12.png)

* The `Seconds` entry-box: Sometimes it can be difficult to understand the context if we only see the classified behavior bout, and we want to introduce a little temporal "padding" pre- and post-behavioral bout to get a better understand of what is going on. In this entry-box, enter the **number of seconds** 
pre- and post-behavioral bout that should be included in the output video clips. 

* The `Target` drop-down menu: Select which behavioral classifications should be visualized in the output video clips. 

* The `One clip per bout` checkbox: If you want to store the results as a separete video file for each detected bout, tick the `One clip per bout` checkbox. For example, if SimBA has detected 50 behavioral bouts, ticking this checkbox will create 50 separate video files. 

* The `One clip per video` checkbox: If you want to store the results from each separate video as a single output video file, tick the `One clip per video` checkbox. For example, if SimBA has detected 50 behavioral bouts in `Video 1`, a single video file representing `Video 1` with the 50 behavioral bouts will be created. 

## How to use it

1. Enter 1 or 2 in the `Seconds` entry box. *Note: the larger the seconds, the longer the duration of the video.**

2. Select the target behavior from the `Target` dropdown box.

3. Tick the `One clip per bout` and/or the `One clip per video` checkboxes. 

4. Click `Validate` button and the videos will be generated in `/project_folder/frames/output/classifier_validation`. The name of the video will be formated in the following manner: **videoname** + **target behavior** + **number of bouts** + .mp4

![](/images/classifiervalidation.gif)

Author [Simon N](https://github.com/sronilsson), [JJ Choong](https://github.com/inoejj)
