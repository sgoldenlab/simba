## Part 5:  VISUALIZING RESULTS

In this part of the tutorial we will create visualizations of machine learning classifications and the features which you have generated. This includes images and videos of the animals with *prediction overlays, gantt plots, line plots, paths plots, heat maps and data plot etc.* These visualizations can help us understand the classifier(s), behaviors, and differences between experimental groups. 

To access the visualization functions, click the `[Visualizations]` tab.

### VISUALIZING CLASSIFICATIONS

On the left of the `Visualization` tab menu, there is a sub-menu with the heading `DATA VISUALIZATION` with a button named `VISUALIZE CLASSIFICATIONS`. Use this button to create videos with classification visualization overlays, similar to what is presented [HERE](https://youtu.be/lGzbS7OaVEg). Clicking this button brings up the below pop-up menu allowing customization of the videos and how they are created. We will go through each of the settings in the visualization options in turn:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/clf_viz_1.png" height="700"/>
</p>

* **BODY-PART VISUALIZATION THRESHOLD** (0.0-1.0): In this entry-box, enter the **minimum** pose-estimation detection probability threshold required for the body-part to be included in the visualization. For example, enter `0.0` for **all** body-part predictions to be included in teh visualization. Enter `1.0` for only body-parts detected with 100% certainty to be visualized. 

* **STYLE SETTINGS**: By default, SimBA will **auto-compute** suitable visualization (i) font sizes, (ii) spacing between text rows, (iii) font thickness, and (iv) pose-estimation body-part location circles which depend on the resolution of your videos. If you do **not** want SimBA to auto-compute these attributes, go ahead and and **un-tick** the `Auto-compute font/key-point sizes checkbox, and fill in these values manually in each entry box. 

* **VISUALIZATION SETTINGS**:
  - **Create video**: Tick the `Create video` checkbox to generate `.mp4` videos with classification result overlays.
  - **Create frames**: Tick the `Create frames` checkbox to generate `.png` files with classification result overlays (NOTE: this will create one png file for each frame in each video. If you are concerned about storage, leave this checkbox unchecked. 
  - **Include timers overlay**: Tick the `Include timers overlay` checkbox to insert the cumulative time in seconds each classified behavior has occured in the top left corner of the video. 
  - **Rotate video 90°**: Tick the `Rotate video 90°` checkbox to rotate the output video 90 degrees clockwise relative to the input video. 
  - **Multiprocess videos (faster)**: Creating videos can be computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this using multiprocessing over multiple cores on your computer. To use multi-processing, tick the `Multiprocess videos (faster)` checkbox. Once ticked, the `CPU cores` dropdown becomes enabled. This dropdown contains values between `2` and the number of cores available on your computer with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your visualizations. 

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


* **STYLE SETTINGS**: Use this menu to specify the resultion of the Gantt plot videos and/or frames. Use the `Font size` entry box to specify the size of the y- and x-axis label text sizes. Use the `Font rotation degree` entry-box to specify the rotation of the y-axis classifier names (set to `45` by default which is what is visualized in the gif above). 

* **VISUALIZATION SETTINGS**:
  - **Create video**: Tick the `Create video` checkbox to generate gantt plots `.mp4` videos.
  - **Create frames**: Tick the `Create frames` checkbox to generate gantt plots `.png` files (NOTE: this will create one png file for each frame in each 
  - **Multiprocess videos (faster)**: Creating gantt videos and/or images can be computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this using multiprocessing over multiple cores on your computer. To use multi-processing, tick the `Multiprocess videos (faster)` checkbox. Once ticked, the `CPU cores` dropdown becomes enabled. This dropdown contains values between `2` and the number of cores available on your computer, with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your gantt charts. 

* **RUN**:
  - **SINGLE VIDEO**: To create gantt chart visualizations for a single video, select the video in the `Video` drop-down menu and click the `Create single video` button. You can follow the progress in the main SimBA terminal. The results will be stored in the `project_folder/frames/output/sklearn_results` directory of your SimBA project.
  - **MULTIPLE VIDEO**: To create gantt chart visualizations for all videos in your project, click the `Create multiple videos` button. You can follow the progress in the main SimBA terminal. The results will be stored in the `project_folder/frames/output/sklearn_results` directory of your SimBA project.








