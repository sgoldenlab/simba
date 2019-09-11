# Tutorial

### Pipeline breakdown:

For processing a tracking dataset, the pipelines is split into ? sections. These sections are listed below along with the functions that belong to each section:


### Part 1: Create a new project
- In here you set up your project name, directories, and videos settings
- Then, generate project config
- Import videos into project folder
- Import DLC Tracking Data (if have any)
- Extract Frames into project folder

### Part 2: Load project.ini file
- First you load the project.ini that was generated when you created your new project
- set video parameters and save it as csv
- Run outlier correction
- Extract Features
- Label Behavior
- Train Machine Model
- Run Machine Model
- Analyze Machine Results
- Plot Skleran Results
- Plot Graphs
- Merge Frames
- Create Video

## Part 1: Create a new project
This sections create a new project for your tracking analysis.

### Step 1: Generate Project Config
In this step, you will be generating your main project folder with all the sub-directories.

1. Go to `File` and click on `Create a new project` The following windows will pop up.
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/createproject.PNG" width="500" height="600" />

2. `Project Path` is the main working directory that your project will be created. Click on `Browse Folder` and select your working directory

3. `Project Name` is the name for your project. *Keep in mind that the project name cannot contain any spaces, use underscore "_" instead* 

4. Under `SML Settings`, put in the number of predictive classifiers that you wished. For an example, if you had three behaviors in your video, put 3 in the entry box.

5. Click <img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/addclassifier.PNG" width="150" height="50" /> and it creates a row as shown in the following image.
If you are generating a brand new model, just fill in the entry box and leave the path as it is (*No classifier is selected*). Repeat this step for every behaviors.
> **Note**: Make sure to click the button once and finished it before clicking it again. Do not click to add multiple rows before filling up the information.
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/classifier1.PNG" width="700" height="130" />

6. For existing model, you can import the models by clicking `Browse` and select your *.sav* file. At the end, it should look like the following image.
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/classifier2.PNG" width="700" height="160" />

7. `Video Settings` is the metadata of your videos. Filled in the information based on your videos.

8. Click `Generate Project Config` to generate your project. The project folder will be located at the `Project Path` that you specified

### Step 2: Import Videos into project folder
In this step, you can choose to import only one video or multiple videos.
#### To import multiple videos
1. Click on `Browse Folder` to select your folder that contains all the videos that you wished to be included in your project.
2. Enter the type of your video *mp4*, *avi*, *mov*,etc. in the `Video type` entry box.
3. Click `Import multiple videos` 
>**Note**: It might take awhile for all the videos to be imported.
#### To import a single video
1. Click on `Browse File` to select your video
2. Click `Import a video`

### Step 3: Import DLC Tracking Data
In this step, you will import your csv tracking data.
#### To import multiple csv files
1. Click on `Browse Folder` to select your folder that contains all the csv files that you wished to be included in your project.
2. Click `Import csv to project folder` 
#### To import a single csv file
1. Click on `Browse File` to select your video
2. Click `Import single csv to project folder`

### Step 4: Extract frames into project folder
This step will extract all the frames from every videos that are in the videos folder.

Once, all the steps are completed, close the `Project Configuration` window.

## Part 2: Load project
This sections loads a project that you have created.

### Step 1: Load Project Config
In this step, you will load the *project_config.ini* file that was created.
> **Note:** project_config.ini should always be loaded before starting anything else.
1. Go to `File` and click on `Load project` The following windows will pop up.
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/loadproject.PNG" width="600" height="600" />

2. Under **Load Project.ini**, click on `Browse File`. Then, go to the directory that you created your project and click on your *project folder*. Then, click on *project_folder* and then *project_config.ini*. Once this step is completed, it should look like the following instead of *No file selected*.
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/loadedprojectini.PNG" width="500" height="60" />

In this image, you can see, the `Dekstop` is my selected working directory, `tutorial` is my project name, and the last two sections is always going to be `project_folder/project_config.ini` 

### Step 2 (Optional) : Import more DLC Tracking Data or videos
In this step, you can choose to import more csv files or videos. If you don't you can ignore and skip this step.

### Step 3: Set video parameters
In this step, you can customize the parameters for each of your videos. You will also be setting the **pixels per milimeter** of for your videos. You will be using a tool that requires the distance of a point to another point in order to calculate the **pixels per milimeter**. The real life distance between two points is call `Distance in mm`.

1. Under **Set video parameters(distances,resolution,etc.)**, the `Distance in mm` here is the distance between two points in milimeter. You can enter the values *(eg: 10,20)* and click `Auto populate Distance in mm in tables` and it will populate the table that you are going to see in the next step. If you leave it empty, it will just be zero.

2. Click on `Set Video Parameters` and the following windows will pop up.
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/videoinfo_table.PNG" width="1000" height="500" />

3. As you can see, I imported four videos and they are in the `Video` column. I have set the `Distance_in_mm` to 10 in the previous step, else, it will be 0.

4. Now, I can click on the values in the box and change it until I am satisfied. Then, click `Update distance_in_mm`, this will actually update the whole table.

5. Next, to get the `Pixels/mm`, click on `Video1`,which will be the first video in the table and the following window will pop up.
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/getcoord1.PNG" width="300" height="400" />
The windows that pop up is a frame from your first video in the table.

6. Now, double **left** click to select two points that you know the distance in real life. 
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/getcoord2.PNG" width="300" height="400" />

In this case, I know the two **pink dot that connects** has a distance of 10mm in real life.

7. If you misplace the dots, you can double click on either of them and redo the step. Once you are done, you can hit `Esc`.
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/get_coord.gif" width="720" height="405" />

8. If every steps are done correctly, the values should populate in the `Pixels/mm` column in the table.
<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/videoinfo_table2.PNG" width="1000" height="500" />

9. Repeat the steps for every videos and once it is done, click `Save Data`. This will generate a csv file named **video_info.csv** in `/project_folder/log`
