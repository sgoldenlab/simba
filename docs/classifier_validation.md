# Post-classification Validation (detecting false-positives)

This tool can help us visualize any potential false-positives within the classifaction results. Post-classification validation process creates video clips of each detected (classified) behavioral bout in user-defined videos in your SimBA project. This method can be applied to any video within your SimBA project that is represented by (i) a file inside the `project_folder/csv/machine_results` directory, and (ii) a video file inside the `project_folder/videos` directory. See below for instructions on how to use this tool.

1. Load your SimBA project and click on the `CLASSIFIER VALIDATION CLIPS` button in [Visualizations] tab and you should see the following sub-menu:

![](/images/clf_validation_1223.png)

* The `SECONDS` entry-box: Sometimes it can be difficult to understand the context if we only see the classified behavior bout, and we want to introduce a little temporal "padding" pre- and post-behavioral bout to get a better understand of what is going on. In this entry-box, enter the **number of seconds** pre- and post-behavioral bout that should be included in the output video clips. 

* The `CLASSIFIER` drop-down menu: Select which behavioral classifications should be visualized in the output video clips. If you want to create video clips for multiple classifiers, then you need to run through this process multiple times: only one classifier can be visualized in each individual video clip.
 
* The `TEXT COLOR` drop-down menu: Select the color of the text overlay that best suits your videos. 

* The `HIGHLIGHT TEXT COLOR` drop-down menu: If you have introduced some temporal "padding" in the `SECONDS` entry-box, we may want to highlight the text overlay when the behavior is actually happening to make the classifications more salient and videos useful. In this dropdown, select the color you wish to use for the text overlay when the behavior is happening. If `NONE`, then the text color selected in the `TEXT COLOR` dropdown will be used through-out the validation videos. 

* The `VIDEO SPEED` drop-down menu: We may want to slow down (or speed up) the validation videos to better understand what is going on. In this dropdown, select how much you'd like to slow down (or speed up) your videos. If `1.0`, the videos will be created in their original speed. If `0.5`, then the videos will be created at half speed. If `2.0`, then the videos will be shown in twice the original speed.

* The `CREATE ONE CLIP PER BOUT` checkbox: If you want to store the results as a separete video file for each detected bout, tick the `One clip per bout` checkbox. For example, if SimBA has detected 50 behavioral bouts, ticking this checkbox will create 50 separate video files.  

* The `ONE CLIP PER VIDEO` checkbox: If you want to store the results from each separate video as a single output video file, tick the `One clip per video` checkbox. For example, if SimBA has detected 50 behavioral bouts in `Video 1`, a single video file representing `Video 1` with the 50 behavioral bouts will be created. The bouts will be separated by a short intermission that delineates each bout. 

* The `MULTIPROCESS VIDEOS` checkbox: Creating videos can be computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this using multiprocessing over multiple cores on your computer. To use multi-processing, tick the Multiprocess videos (faster) checkbox. Once ticked, the CPU cores dropdown becomes enabled. This dropdown contains values between 2 and the number of cores available on your computer with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your visualizations.

SINGLE VIDEO: To create post-classification validation visualizations for a single video, select the video in the Video drop-down menu and click the <kbd>Create single video button</kbd>. You can follow the progress in the main SimBA terminal. The results will be stored in the project_folder/frames/output/sklearn_results directory of your SimBA project. You can also click `BROWSE` and select a file from the `project_folder/csv/machine_results` directory manually, and click the <kbd>Create single video</kbd>. The results will be stored in the `project_folder/frames/output/classifier_validation` directory of your SimBA project.

MULTIPLE VIDEO: To create post-classification validation visualizations for all videos in your project, click the <kbd>Create multiple videos button</kbd>. Videos will be created for all of the files found within the `project_folder/csv/machine_results` directory. You can follow the progress in the main SimBA terminal. The results will be stored in the project_folder/frames/output/classifier_validation directory of your SimBA project.

![](/images/classifiervalidation.gif)

Author [Simon N](https://github.com/sronilsson), [JJ Choong](https://github.com/inoejj)
