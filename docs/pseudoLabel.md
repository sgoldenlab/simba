# Pseudo Label
Before running pseudo label, the video has to go through **Run machine model** and has .csv file in the `/project_folder/csv/output/machine_results` folder.
With this tool, user is able to look at the machine annotation and correct them.

![](/images/pseudolabel.PNG)

1. Load the *project_config.ini* file.

2. Under `[ Label behavior ]`, **Pseudo Labelling**, select the video folder.

3. Set the threshold for the behavior

4. Click `Correct label`.
