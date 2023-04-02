### Step 8. Evaluating the model on new (out-of-sample) data.

If you have chosen to generate classification reports or other metrics of classifier performance, it is worth studying them to ensure that the model(s) performance is acceptable. However, a classifiers performance is perhaps most readily validated by visualizing its predictions and prediction probabilities on a new video, which have not been used for training or testing. This step is critical for (i) visualizing and choosing the ideal classification probability thresholds which captures all of your BtWGaNP behaviors (for more information on what classification threshold is - see Step 4 below), and (i) visual confirmation that model performance is sufficent for running it on experimental data.

You can validate each model *(saved in SAV format)* file. This should be done in a "gold-standard" video that has been fully manually annotated for your behavior of interest, but has not been included in the training dataset. If you followed the tutorial, you may remember that we stored away one CSV file away in a safe place earlier, a [file which we had extracted the features](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-5-extract-features) for but we did not use this file for training or testing of the classifier. Now is the time to use this file. 

In this validation step the user specifies the path to a previously created model in SAV file format, and the path to a CSV file [that contain the features extracted from a video (Step 5 above)](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-5-extract-features). The first process will run the predictions on the video, and visualize the probabilities in a user-interactable line chart, that together with a user-interactable visualization of the video can be used to gauge the ideal discrimination threshold. The second process will create a video with the predictions overlaid together with a gantt plot showing predicted behavioral bouts. Click [here](https://youtu.be/UOLSj7DGKRo) for an expected output validation video for predicting the behavior *copulation*. 

This process allows you to rapidly access the results of the Hyperparameters you have selected on a "gold-standard" behavioral video. If the predictions are not good, you can go back to tweak the appropriate parameters without first running through numerous other videos or adding and/or refining your annotations.

In this step, we will (i) run the classifier on new data, (ii) interactively inspect suitable discrimination thresholds, and (iii) create a video with the predictions overlaid ontop of the new data together with a gantt plot showing predicted behavioral bouts. Click [HERE](https://youtu.be/UOLSj7DGKRo) for an expected validation video. For this, navigate to the [Run machine model] tab and `VALIDATE MODEL ON SINGLE VIDEO menu:

<p align="center">
  <img width="511" height="232" src="https://github.com/sgoldenlab/simba/blob/master/images/validate_single_video_1.png">
</p>


**(1).**  In `SELECT FEATURE DATA FILE`, select the path to a path to a file containing features in the `project_folder/csv/features_extracted` directory created as describes in [STEP 5](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-5-extract-features) above. Note: Make sure this file is **not** a file where the behavior has been labeleld and used to create the classifier you are evaluating.

**(2).** In `SELECT MODEL FILE`, select the path to a path a classifier. In SimBA, classifier are saved in `.sav` file format and typically live in the `model` sub-directories within your SimBA project. 

**(3).** To run the model selected in step 2 on the feature data selected in step 1, click the `RUN` button. In the background, SimBA will generate a behavior probability score for each of the frames in the data and store it in the `project_folder/csv/validation` directory of your SimBA project. 

**(4).** Next, we want to interactively inspect the prediction probabilities of each frame and view them alongside the video. We do this to try an discern a prediction probability demarcation point where the model reliably splits behavior from non-behavior frames. In other words, we determine how sure the model has to be that a behavior occurs on a frame for it to classify a behavior to occur in a frame. To do this, click the `INTERACTIVE PROBABILITY PLOT` button. 

https://user-images.githubusercontent.com/34761092/229304497-8f0f4532-e613-4f96-bcda-dededca39dc6.mp4

In the left window in the example above, you can see the video of the analyzed file. Similar to the [annotation interface](https://github.com/sgoldenlab/simba/blob/master/docs/label_behavior.md), you can you the buttons to jump forwards and backwards between frames. Some keyboard shortcuts are showed on the right. There is also a button named `SHOW HIGHEST PROBABILITY FRAME`. Clicking on this button will show you the frame with the highest behavioral classification probability. 

In the right window we see the frame number on the x-axis and the classification probability on the y-axis. Clicking on any frame in the graph will display the associated video frame in the left window. The frame number, and the classification probability of the frame, is shown in the graph title. We look at the graph and determine a suitable behavioral probability threshold (the y-axis in the right graph) that separetes the non-behavior from the behavior frames. 

**(5).** Once we have decided on the probability threshold, we fill this value into the `DISCRIMINATION THRESHOLD (0.00-1.0):` entry box. For example, if set to 0.50, then all frames with a probability of containing the behavior of 0.5 or above will be classified as containing the behavior. For further information on classification theshold, click [here](https://www.scikit-yb.org/en/latest/api/classifier/threshold.html). In this Scenario. Go ahead and enter the classification threshold identified in the previous Steps.

**(6).** We can also set a `MINIMUM BOUT LENGTH (MS)` criterion. This value represents the minimum length of a classified behavioral bout. **Example**: The random forest makes the following predictions for behavior BtWGaNP over 9 consecutive frames in a 50 fps video: 1,1,1,1,0,1,1,1,1. This would mean, if we don't have a minimum bout length, that the animals enganged in behavior BtWGaNP for 80ms (4 frames), took a break for 20ms (1 frame), then again enganged in behavior BtWGaNP for another 80ms (4 frames). You may want to classify this as a single 180ms behavior BtWGaNP bout, rather than two separate 80ms BtWGaNP bouts. If the minimum behavior bout length is set to 20, any interruption in the behavior that is 20ms or shorter will be removed and the example behavioral sequence above will be re-classified as: 1,1,1,1,1,1,1,1,1 - and instead classified as a single 180ms BtWGaNP bout. 

**(7).** Next we want to go ahead and create a validation video and we click on `CREATE VALIDATION VIDEO` and the following pop-up should be shown which gives user controls how the video is created. If you want to use the deafult parameters, just go ahead and click `RUN`. 

<p align="center">
  <img src="https://github.com/sgoldenlab/simba/blob/master/images/validate_single_video_2.png">
</p>


- If you want SimBA to try to autompute the appropriate font sizes etc., keep the `AUTO COMPUTE STYLES` checked. Otherwise, un-check this box and fill in try out your own values. 

- `TRACKING OPTIONS`: Choose if you want to display the pose-estimation body-part locations and/or the animal names in the validation video.

-  `MULTI-PROCESSING SETTINGS`: Creating videos can be computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this using multiprocessing over multiple cores on your computer. To use multi-processing, tick the Multiprocess videos (faster) checkbox. Once ticked, the CPU cores dropdown becomes enabled. This dropdown contains values between 2 and the number of cores available on your computer, with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your video. SimBA will create a snippet of video on each core, and then jopin them together to a single video. 

-  `GANTT SETTINGS`: If you select  `Gantt chart: video` or `Gantt chart: final frame only (slightly faster)` in the `Create Gantt plot` drop-down menu, SimBA will create a validation video with an appended Gantt chart plot (see the final gif image in this tutorial below for an example). Creating Gantt charts take longer, and we suggest selecting `None` in the `Create Gantt plot` drop-down menu unles syou use multi-processing.

**(8).** Click the `RUN` button. You can follow the progress in the main operating system terminal. Once complete, you should see a video file representing the analyzed file inside the `project_folder/frames/output/validation` directory.


