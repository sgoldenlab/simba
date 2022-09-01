# <p align="center"> SimBA behavioral Annotator GUI </p>


The SimBA behavioural annotator GUI is used to label (annotate) frames in behavioral videos that contain behaviors of interest. SimBA appends the behavioral annotations directly to the pose-estimation tracking data to build supervised machine learning predictive classifiers of behaviors of interest. Specifically, this GUI integrates two additional consoles with the main SimBA console: (1) a video player and (2) a frame viewer. The video player and frame viewer are synced, such that pausing or starting the video will advance the frame viewer to the identical frame that the video was paused or started at. The frame viewer is then used to annotate when behaviors of interest are present or absent within the given frame. Such annotations can be flexibily annotated from single to numerous frames using the annotation interface. Please see below for details for how to best use the annotator in your specific use case. 

Note that SimBA performs similar functions such as the open-source [JWatcher](http://www.jwatcher.ucla.edu/) or commercial [Noldus Observer](https://www.noldus.com/human-behavior-research/products/the-observer-xt) systems, with the exception that SimBA automates the backend integration of behavioral annotation with creating predictive classifiers. If you already have such annotations stored in alterantive file formats, like [JWatcher](http://www.jwatcher.ucla.edu/) or [Noldus Observer](https://www.noldus.com/human-behavior-research/products/the-observer-xt), they can be appended directly to the tracking data and no behavioral annotations needs to be done in SimBA. To append annotations created in alternative third-party software, check out [THIS TUTORIAL](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md). The [Crim13 dataset](http://www.vision.caltech.edu/Video_Datasets/CRIM13/CRIM13/Main.html) was annotated using [Piotr’s Matlab Toolbox](https://github.com/pdollar/toolbox) and we appended the annotations to the tracking data using a version of [this script](https://github.com/sgoldenlab/simba/blob/master/misc/Caltech_2_DLC.py). 

If you already have annotation videos created with these alternative tools, or any other behavioral annotator, and would like to use them to create predictive classifiers, please let us know as we would like to write scripts that could process these data for SimBA. If you have created such scripts yourself, please consider contributing them to the community! If you have any issues using the SimBA annotation interface, please open a [GitHub issue](https://github.com/sgoldenlab/simba/issues) or reach out to us on our [Gitter](https://gitter.im/SimBA-Resource/community) support channel. 

**We will provide support in either scenario**.

## Step 1. Loading project_config file 
In the main SimBA menu, click on `File > Load Project > Load Project.ini > Browse File` and select the config file (project_config.ini) representing your SimBA project. This step **must** be done before proceeding to the next step.

## Step 2. Opening the labelling behavior interface

Once your project is loaded, click on the [Label Behavior] tab and you should see the below four sub-menus (Note: I'm writing this document on a Mac, if you're running SimBA on a PC or Linux, the aestetics might be slightly different): 
<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_1.png" />
</p>

These four different sub-menus represent four different ways of annotating your videos. The differences between the different menus, and when to use them, are detailed below, but in brief: 

* (1) **LABEL BEHAVIOR**: When selecting a new video to annotate, SimBA assumes that the behavior is absent in any given frame unless indicated by the user. In other words, the default annotation is that the behavior(s) are **not** present. 
* (2) **PSEDU-LABELLING**: When selecting a new video to annotate, SimBA uses machine classifications the default annotation. Thus, any frame with a classification probability above the user-specified threshold will have **behavior present** as the default value.  
* (3) **ADVANCED LABEL BEHAVIOR**. When selecting a new video to annotate, SimBA **has no default annotatation for any frame**. In other words, the user is required annotate each frame as either behavior-absent or behavior-present. Only annotated frames will be used when creating the machine learning model(s). 
* (4) **IMPORT THIRD-PARTY BEHAVIOR ANNOTATIONS**. Use these menus to import annotations created in other tools (without performing annotations in SimBA. Click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md) to learn more about how to import annotations from third-party software. 

Regardless of which method you choose to use, by clicking on `Select video...` (or `Correct labels) you will access the same user interface. The only difference between the methods is how non user-annoated videos are going to be treated. 

In this tutorial, we will click on `Select video (create new video annotation)`. This will bring up a file selection dialog menu. We navigate to the `project_folder/videos` directory and select a video we wich to annotate. In this tutorial, I am selecting `BtWGANP.mp4` and click Open:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/docs/annotator_2.png" />
</p>

## Step 3. Using the labelling behavior interface

Once I've selected my video file, the annotation interface will pop open, looking something like this:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_3.png" />
</p>

In this image I have marked out 8 different parts of the window, which will we can use to accurately label the frames of the video as containing (or not containing) your behavior(s) of interest. We go over the 

(1) This is the title header of the window. It tells you which 













In the 'Load project' window, under `Label Behavior` click on `Select folder with frames`. This will prompt you to select a **folder** containing video frames (in png format). Following folder selection a new window will display the first frame of the video. If you have not extracted the frames for the videos that you want to label, they need to be created now. For information on how to extract video frames in SimBA, please check these parts of the tutorial: [1](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#extract-frames), [2](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-4-extract-frames-into-project-folder). 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/labellingtable.PNG" width="752" height="802" />

> Note: At any time, the user can refer to the main screen window to see the values of the frame. In the picture above, **Name** is the current frame number.

## 3. Labelling 
Under the **Check Behaviors** heading is a list of checkboxes, one for each classifier that were specified when the project was [created](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#part-1-create-a-new-project-1). Users can check (or uncheck) the boxes if the frame displays the behaviors. When the user press `Save and Advance to the next frame` the behavioral data for that particular frame is saved. 

<img src="https://github.com/sgoldenlab/simba/blob/master/images/labelling_mainscreen.PNG" width="500" height="450" />

## Navigation
Underneath the displayed image, there are options for navigating through all the frames:

#### Using Mouse

`<<` jump to the first image frame in the folder 

`>>` jump to the last image frame in the folder

`<` go back to the previous frame 

`>` advance to the next frame 

`Jump Size` You can select a range of frames to advance or jump backwards. When you then press the `<<` or `>>` buttons to the right of the scale, the frames advance or go back the Jump Size frame count indicated. 

`Jump to selected frame` You have the option to manually enter a frame number into this entry box, which will then display the corresponding image.

`Frame Range` By selecting this box and entering a range of numbers in the adjacent entry boxes, saving and advancing to the next frame will save all frames in the Frame Range, inclusive, as containing the marked behaviors. 

`Generate and Quit` This will compile and export the data to a .csv file located in the `project folder\csv\targets_inserted\` that contain the behavioral annotations. 

#### Keyboard Shortcuts 

Keyboard shortcut information is displayed on the right side of the window for ease of use. 
`For Video` shortcuts are only applicable for navigating the video when opened, and have no effect on the displayed frame. 
`Key Presses` allows users to jump from frame to frame as well as saving frame information by using the keyboard. 
> Important: If using Ctrl + S to save, do NOT hold down keys to save multiple frames.

#### Playing Video
`Open Current Video` Pressing this button will open the video that corresponds to the frame folder that is being analyzed. Refer to keyboard shortcuts to pause/play and move forward or backwards in the video by a certain amount of frames.

`Show current video frame` This will display the current frame of the paused video on the labelling screen.

![](https://github.com/sgoldenlab/tkinter_test/blob/master/images/openingvideo.gif)
> *Note*: Video name must be the same as the folder name. The video must also be paused by pressing `p` before any other keyboard command (such as moving forward or backwards a set number of frames). 
