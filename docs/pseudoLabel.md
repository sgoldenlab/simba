# Pseudo Label
Before running pseudo label, the video has to go through [**Run machine model**](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model) and has .csv file in the `/project_folder/csv/output/machine_results` folder.
With this tool, user is able to look at the machine annotation and correct them.

![](/images/pseudolabel.PNG)

1. Load the *project_config.ini* file.

2. Under `[ Label behavior ]`, **Pseudo Labelling**, select the video folder.

3. Set the threshold for the behavior.

4. Click `Correct label`.

5. Note that the checkboxes will autopopulate base on the computer's prediction on the threshold set by the user.

6. Once it is completed, click on `Save csv` and the .csv file will be saved in `/project_folder/csv/output/target_inserted`.
