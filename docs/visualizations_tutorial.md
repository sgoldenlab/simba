## Part 5:  Visualizing machine predictions

In this part of the tutorial we will create visualizations of machine learning classifications and the features which you have generated. This includes images and videos of the animals with *prediction overlays, gantt plots, line plots, paths plots, heat maps and data plot etc.* These visualizations can help us understand the classifier(s), behaviors, and differences between experimental groups. 

To access the visualization functions, click the `[Visualizations]` tab.


### VISUALIZING CLASSIFICATIONS

On the left of the `Visualization` menu, there is a sub-menu with the heading `CLASSIFICATION VISUALIZATION` with a button named `VISUALIZE CLASSIFICATION SETTINGS`. Use this button to create videos with classification visualization overlays. Clicking this button brings up the below sub-menu allowing users to customize the videos and how they are created. We will go through each of the submenus and visualization options in turn:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/clf_viz_1.png" height="700"/>
</p>

* BODY-PART VISUALIZATION THRESHOLD (0.0-1.0): In this entry-box, enter the **minimum** pose-estimation detection probability threshold required for the body-part to be included in the visualization. For example, enter `0.0` for **all** body-part predictions to be included in teh visualization. Enter `1.0` for only body-parts detected with 100% certainty to be visualized. 

* STYLE SETTINGS: By default, SimBA will **auto-compute** suitable visualization (i) font sizes, (ii) spacing between text rows, (iii) font thickness, and (iv) pose-estimation body-part location circles which depend on the resolution of your videos. If you do **not** want SimBA to auto-compute these attributes, go ahead and and **un-tick** the `Auto-compute font/key-point sizes checkbox, and fill in these values manually in each entry box. 

* VISUALIZATION SETTINGS:
  - Create video: Tick the `Create video` checkbox to generate `.mp4` videos with classification result overlays.
  - Create frames: Tick the `Create frames` checkbox to generate `.png` files with classification result overlays (NOTE: this will create one png file for each frame in each video. If you are concerned about storage, leave this checkbox unchecked. 
  - Include timers overlay: Tick the `Include timers overlay` checkbox to insert the cumulative time in seconds each classified behavior has occured in the top left corner of the video. 
  - Rotate video 90°: Tick the `Rotate video 90°` checkbox to rotate the output video 90 degrees clockwise relative to the input video. 
  - Multiprocess videos (faster): Creating videos can be computationally costly, and creating many, long, videos can come with unacceptable run-times. We can solve this using multiprocessing over multiple cores on your computer. To use multi-processing, tick the `Multiprocess videos (faster)` checkbox. Once ticked, the `CPU cores` dropdown becomes enabled. This dropdown contains values between `2` and the number of cores available on your computer with fancier computers having higher CPU counts. In this dropdown, select the number of cores you want to use to create your visualizations. 



