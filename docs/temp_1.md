### Step 8. Validating the model on new (out-of-sample) data.

The user may want to confirm that the classifications are accurate on new data, that has not been used to train/test the model. 


The user can validate each model *( saved in .sav format)*. In this validation step the user specifies the path to a previously created model in .sav file format, and a .csv file containing the features extracted from a video. This process will (i) run the classifications on the video, and (ii) create a video with the predictions overlaid together with a gantt plot showing predicted behavioral bouts.  Click[here](https://youtu.be/UOLSj7DGKRo) for an example validation video.

1. Click `Browse File` and select the *project_config.ini* file and click `Load Project`.

2. Under **[Run machine model]** tab --> **validate Model on Single Video**, select your features file (.csv). It should be located in `project_folder/csv/features_extracted`.

![](/images/validatemodel_graph.PNG)

3. Under `Select model file`, click on `Browse File` to select a model *(.sav file)*.

4. Click on `Run Model`.

5. Once, it is completed, it should print *"Predictions generated."*, now you can click on `Generate plot`. A graph window and a frame window will pop up.

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

10. Click `Validate` to validate your model. **Note that this step will take a long time as it will generate a lot of frames.**
