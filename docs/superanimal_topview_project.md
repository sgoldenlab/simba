# SuperAnimal-Topview data in SimBA

### CREATE A SuperAnimal-Topview PROJECT IN SIMBA

Before working with SuperAnimal-Topview data in SimBA, make sure you have the latest version of [SimBA installed](https://github.com/sgoldenlab/simba/blob/master/docs/installation_new.md).

If you have SimBA installed on your system, you can make sure you have the latest version by typing `pip install simba-uw-tf-dev --upgrade`. 

To work with SuperAnimal-Topview  tracking data in SimBA, first create a SuperAnimal-Topview  project in SimBA. 

Follow [Scenario 1](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config) tutorial up-to the `Generate Project Config` section. 

Next, under `TYPE OF TRACKING` select `Multi tracking`. Under `BODY PART CONFIGURATION`, select **SuperAnimal-Topview** as in the screenshot below. 

<img width="693" height="782" alt="image" src="https://github.com/user-attachments/assets/a4505bf6-7ba3-4031-9857-bf3460fe0dc2" />

Once selected, click the <kbd>CREATE PROJECT CONFIG</kbd> button to create the project in your chosen directory. 

### IMPORT SuperAnimal-Topview VIDEOS TO YOU SIMBA PROJECT

After clicking on the <kbd>CREATE PROJECT CONFIG</kbd> button, head to the [`IMPORT VIDEOS`] tab to import the videos representing your SuperAnimal-Topview data into your new SuperAnimal-Topview SimBA project:
For more information on video import, see the [Scenario 1 documentation](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config). 

The import of videos is not a strict requirement, but recommended, for amongst others, visualization puposes and the ability to read video meta data relevant for data smoothing if selected (see next step).

<img width="743" height="400" alt="image" src="https://github.com/user-attachments/assets/03182aff-189a-4b97-a69b-3b87bbf58be8" />

### IMPORT SuperAnimal-Topview DATA TO YOU SIMBA PROJECT

Next, we need to import the SuperAnimal-Topview tracking data to the project. Click on the `[Import tracking data]` tab. Here, select `H5 (SuperAnimal-Topview)` from the `DATA TYPE` dropdown:

<img width="699" height="479" alt="image" src="https://github.com/user-attachments/assets/2f220b72-b46c-47eb-8bf7-5aa6b22a7ff4" />

(i) If the SuperAnimal-Topview data contains missing data, you can interpolate it using one of the options available in the `INTERPOLATION METHOD` dropdown. For more information on interpolation, see the 
interpolation section in the [Scenario 1 tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#to-import-multiple-dlc-csv-files).

(ii) If the SuperAnimal-Topview data is "jittery", we may want to smooth the tracking data using one of the options available in the `SMOOTHING` dropdown. For more information on interpolation, see the 
smoothing section in the [Scenario 1 tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#to-import-multiple-dlc-csv-files).

>[!NOTE] 
> If you are performing smoothing, it is required that you import the videos first (so SimBA can read the FPS frame-rate from the video file meta-data).

(iii) Next, select the number of animals beeing tracked in your SuperAnimal-Topview tracking data, and give them names.

(iv) Next, to import a directory of SuperAnimal-Topview H5 files, select a directory containing `.h5` files and hit the import button. Alternatively, to import a single SuperAnimal-Topview tracking data `.h5` file, go to the labelframe
`IMPORT SuperAnimal-Topview h5 file` and hit the import single file button. You can follow the progress in the main SImBA terminal. 

(v) If you are tracking multiple animals using SuperAnimal-Topview, then you will be assigned track identities manually, as described [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/Multi_animal_pose.md).

Once complete, close the `PROJECT CONFIGURATION` window and load the project as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-load-project-config)










