# Validate model on single video

1. Click on `File` --> `Load project`, and the following window will pop up.

![](/images/loadprojectini.PNG)

2. Click `Browse File` and select the *project_config.ini* file and click `Load Project`.

3. Under **[Run machine model]** tab --> **validate Model on Single Video**, select your features file (.csv). It should be located in `project_folder/csv/features_extracted`.

![](/images/validatemodel_graph1.PNG)

**Note: If you check the 'Generate Gantt plot' box, the final validation video will include the gantt chart. However, to do this you must extract frames which drastically increases the amount of time needed to generate the video. We suggest leaving it unchecked unless you really want it.**

4. Select a model file (.sav).

5. Click on `Run Model`.

6. Once, it is completed, it should print *"Predictions generated."*, now you can click on `Generate plot`. A graph window and a frame window will pop up.

- `Graph window`: model prediction probability versus frame numbers will be plot. The graph is interactive, click on the graph and the frame window will display the selected frames.

- `Frame window`: Frames of the chosen video with controls.

![](/images/validategraph1.PNG)

7. Click on the points on the graph and picture displayed on the other window will jump to the corresponding frame. There will be a red line to show the points that you have clicked.

![](/images/validategraph2.PNG)

8. Once it jumps to the desired frame, you can navigate through the frames to determine if the behavior is present. This step is to find the optimal threshold to validate your model.

![](/images/validategraph.gif)

9. Once the threshold is determined, enter the threshold into the `Discrimination threshold` entry box and the desire minimum behavior bouth length into the `Minimum behavior bout lenght(ms)` entrybox.

- `Discrimination threshold`: The level of probability required to define that the frame belongs to the target class. Accepts a float value between 0.0-1.0. For example, if set to 0.50, then all frames with a probability of containing the behavior of 0.5 or above will be classified as containing the behavior. For more information on classification theshold, click [here](https://www.scikit-yb.org/en/latest/api/classifier/threshold.html)

- `Minimum behavior bout length (ms)`: The minimum length of a classified behavioral bout. **Example**: The random forest makes the following attack predictions for 9 consecutive frames in a 50 fps video: 1,1,1,1,0,1,1,1,1. This would mean, if we don't have a minimum bout length, that the animals fought for 80ms (4 frames), took a brake for 20ms (1 frame), then fought again for another 80ms (4 frames). You may want to classify this as a single 180ms attack bout rather than two separate 80ms attack bouts. With this setting you can do this. If the minimum behavior bout length is set to 20, any interruption in the behavior that is 20ms or shorter will be removed and the behavioral sequence above will be re-classified as: 1,1,1,1,1,1,1,1,1 - and instead classified as a single 180ms attack bout. 

10. Click `Validate` to validate your model.
