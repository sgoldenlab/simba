# Create tracking model using DeepLabCut

### Pipeline breakdown
this is something

### Part 1: Create DLC Project

### Part 2: Load DLC Project
- Extract Frames
- Label Frames
- Generate Training Set
- Download weights and train model
- Evaluate Model
- Analyze Videos

## Part 1: Create DLC Project
This section create a new project for your DLC tracking analysis.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/createdlcmodel.png" width="400" height="200" />

### Step 1: Generate a new DLC project
blablabla

1. Go to `Tracking` --> `DeepLabCut` --> `Create DLC Model`. The following window will pop up.

<img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/createdlcmodel2.png" width="400" height="336" />

2. Enter the name of your project in `Project Name`

3. Enter your name in the `Experimenter Name`

> **Note:** Spaces are not allowed for your project name and experimenter name.

4. If you are only tracking one video, you can click on <img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/importsinglevideo.PNG" width="120" height="27" /> and the <img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/videofolder.PNG" width="274" height="27" /> in **green** color font should change to <img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/videopath.PNG" width="274" height="27" /> in **blue** font. Then click on `Browse File` to select the video.

5. If you are tracking multiple videos, click on <img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/importmultivideo.PNG" width="120" height="27" /> and the **green** color <img src="https://github.com/sgoldenlab/tkinter_test/blob/master/images/videofolder.PNG" width="274" height="27" /> will appear and click on `Browser Folder` to choose a folder with the videos. The default settings is always to **import multiple videos**.

6. Next, select the main directory that your project will be in. In `Working Directory` click `Browser Folder` and choose a directory.

7. Choose to apply Golden Aggression Config by checking the checkbox.

8. You can have a choice of copying all the videos to your DLC project folder by checking the `Copy Videos` checkbox.

9. Click `Create Project` to create your DLC project.


## Part 2: Load DLC Project






