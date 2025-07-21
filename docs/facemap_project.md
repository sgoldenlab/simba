# FaceMap data in SimBA

### CREATE A FACEMAP PROJECT IN SIMBA

Before working with FaceMap data in SimBA, make sure you have the latest version of [SimBA installed](https://github.com/sgoldenlab/simba/blob/master/docs/installation_new.md).

If you have SimBA installed on your system, you can make sure you have the latest version by typing `pip install simba-uw-tf-dev --upgrade`. 

To work with FaceMap tracking data in SimBA, first create a FaceMap project in SimBA. 

Follow [Scenario 1](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config) tutorial up-to the `Generate Project Config` section. 

Next, under `TYPE OF TRACKING` select `Classic tracking`. Under `BODY PART CONFIGURATION`, select **FaceMap** as in the screenshot below. 

<img width="757" height="709" alt="image" src="https://github.com/user-attachments/assets/3fa24a37-348e-4f93-a31a-726e93d98f48" />

Once selected, click the <kbd>CREATE PROJECT CONFIG</kbd> button to create the project in your chosen directory. 

### IMPORT FACEMAP VIDEO TO YOU SIMBA PROJECT

After clicking on the <kbd>CREATE PROJECT CONFIG</kbd> button, head to the [`IMPORT VIDEOS`] tab to import the videos representing your FaceMap data into your new FaceMap SimBA project:
For more information on video import, see the [Scenario 1 documentation](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-generate-project-config). 

The import of videos is not a strict requirement, but recommended, for amongst others, visualization puposes and the ability to read video meta data relevant for data smoothing if selected (see next step).

<img width="743" height="400" alt="image" src="https://github.com/user-attachments/assets/03182aff-189a-4b97-a69b-3b87bbf58be8" />

### IMPORT FACEMAP DATA TO YOU SIMBA PROJECT

Next, we need to import the FaceMap tracking data to the project. Click on the `[Import tracking data]` tab. Here, select `H5 (FaceMap)` from the `DATA TYPE` dropdown:

<img width="743" height="400" alt="image" src="https://github.com/user-attachments/assets/3f898e78-2de2-4282-bdf7-9a2f3405b7a0" />

(i) If the FaceMap data contains missing data, you can interpolate it using one of the options available in the `INTERPOLATION METHOD` dropdown. For more information on interpolation, see the 
interpolation section in the [Scenario 1 tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#to-import-multiple-dlc-csv-files).

(ii) If the FaceMap data is "jittery", we may want to smooth the tracking data using one of the options available in the `SMOOTHING` dropdown. For more information on interpolation, see the 
smoothing section in the [Scenario 1 tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#to-import-multiple-dlc-csv-files).

>[!NOTE] 
> If you are performing smoothing, it is required that you import the videos first (so SimBA can read the FPS frame-rate from the video file meta-data).

(iii) Next, to import a directory of FaceMap H5 files, select a directory containing `.h5` files and hit the import button. Alternatively, to import a single FaceMap `.h5` file, go to the labelframe
`IMPORT FACEMAP h5 file` and hit the import single file button. You can follow the progress in the main SImBA terminal. 


Once complete, close the `PROJECT CONFIGURATION` window and load the project as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-1-load-project-config)

>[!IMPORTANT] 
> When working with FaceMap data in SimBA, there is no need to correct outliers (or no need click `skip outlier correction`). 











