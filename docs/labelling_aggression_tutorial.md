# Labelling Behavior

## 1. Loading project_config file 
`File > Load Project > Load Project.ini > Browse File` Select the config file (project_config.ini) for the current project. This step **must** be done before proceeding to the next step.

## 2. Opening labelling behavior window
In the 'Load project' window, under `Label Behavior` select `Select folder with frames`. This will prompt you to select a **folder** containing video frames (in png format). When the folder is selected a new window pops up displaying the first frame of the video. 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/labellingtable.PNG" width="752" height="802" />

> Note: At any time, users can refer to the main screen to see the values of the frame. In the picture above, **Name** is the current frame number.

## 3. Labelling 
Under the **Check Behaviors** heading is a list of checkboxes, one for each classifier that were specified when the project was [created](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#part-1-create-a-new-project-1). Users can check (or uncheck) the boxes if the frame displays the behaviors, and press `Save and Advance to the next frame` to save the behavioral data for that particular frame.


<img src="https://github.com/sgoldenlab/simba/blob/master/images/labelling_mainscreen.PNG" width="500" height="450" />

## Navigation
Underneath the displayed image, there are options for navigating through all the frames:

#### Using Mouse

`<<` jump to the first image in the folder 

`>>` jump to the last image in the folder

`<` go back to the previous frame 

`>` advance to the next frame 

`Jump Size` Users can select a range by which to advance or jump backwards, then pressing the `<<` or `>>` buttons to the right of the scale to display the corresponding frame.

`Jump to selected frame` Users also have the option to manually enter frame numbers into the entry box, then clicking this button to display the corresponding image.

`Frame Range` By selecting this box and entering a range of numbers in the adjacent entry boxes, saving and advancing to the next frame will save all marked behaviors from the first to the last selected frame inclusive. 

`Generate and Quit` This will compile and export the data to a .csv file located in the `project folder\csv\targets_inserted\` that contain pose estimation data, features data, and target columns.

#### Key Shortcuts 

Key shortcuts information provided on the right side of the window for ease of use. 
`For Video` is only applicable for navigating the video when opened.
`Key Presses` allows users to jump from frame to frame as well as save frames using the keyboard. 
> Important: If using Ctrl + S to save, do NOT hold down keys to save multiple frames.


#### Playing Video
`Open Current Video` Pressing this button will open the video that corresponds to the frames folder that is being analyzed. Refer to Key Shortcuts to pause/play and move forward or backwards in the video by a certain amount of frames.

`Show current video frame` This will display the current frame of the paused video on the labelling screen.

![](https://github.com/sgoldenlab/tkinter_test/blob/master/images/openingvideo.gif)
> Note: Video name must be the same as the folder name. The video must also be paused by pressing `p` before any other keyboard command (such as moving forward or backwards a set number of frames). 



