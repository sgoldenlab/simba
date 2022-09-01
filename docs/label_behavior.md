# <p align="center"> SimBA behavioral Annotator GUI </p>


The SimBA behavioural annotator GUI is used to label (annotate) frames in behavioral videos that contain behaviors of interest. SimBA appends the behavioral annotations directly to the pose-estimation tracking data to build supervised machine learning predictive classifiers of behaviors of interest. Specifically, this GUI integrates two additional consoles with the main SimBA console: (1) a video player and (2) a frame viewer. The video player and frame viewer are synced, such that pausing or starting the video will advance the frame viewer to the identical frame that the video was paused or started at. The frame viewer is then used to annotate when behaviors of interest are present or absent within the given frame. Such annotations can be flexibily annotated from single to numerous frames using the annotation interface. Please see below for details for how to best use the annotator in your specific use case. 

Note that SimBA performs similar functions such as the open-source [JWatcher](http://www.jwatcher.ucla.edu/) or commercial [Noldus Observer](https://www.noldus.com/human-behavior-research/products/the-observer-xt) systems, with the exception that SimBA automates the backend integration of behavioral annotation with creating predictive classifiers. If you already have such annotations stored in alterantive file formats, like [JWatcher](http://www.jwatcher.ucla.edu/) or [Noldus Observer](https://www.noldus.com/human-behavior-research/products/the-observer-xt), they can be appended directly to the tracking data and no behavioral annotations needs to be done in SimBA. To append annotations created in alternative third-party software, check out [THIS TUTORIAL](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md). The [Crim13 dataset](http://www.vision.caltech.edu/Video_Datasets/CRIM13/CRIM13/Main.html) was annotated using [Piotr’s Matlab Toolbox](https://github.com/pdollar/toolbox) and we appended the annotations to the tracking data using a version of [this script](https://github.com/sgoldenlab/simba/blob/master/misc/Caltech_2_DLC.py). 

If you already have annotation videos created with these alternative tools, or any other behavioral annotator, and would like to use them to create predictive classifiers, please let us know as we would like to write scripts that could process these data for SimBA. If you have created such scripts yourself, please consider contributing them to the community! If you have any issues using the SimBA annotation interface, please open a [GitHub issue](https://github.com/sgoldenlab/simba/issues) or reach out to us on our [Gitter](https://gitter.im/SimBA-Resource/community) support channel. **We will provide support in either scenario**.

## Step 1. Loading project_config file 
In the main SimBA menu, click on `File > Load Project > Load Project.ini > Browse File` and select the config file (project_config.ini) representing your SimBA project. This step **must** be done before proceeding to the next step.

## Step 2. Opening the labelling behavior interface

Once your project is loaded, click on the [Label Behavior] tab and you should see the below four sub-menus (Note: I'm writing this document on a Mac, if you're running SimBA on a PC or Linux, the aestetics might be slightly different): 
<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_1.png" />
</p>

These four different sub-menus represent four different ways of annotating your videos. The differences between the different menus, and when to use them, are detailed below, but in brief: 

* (1) **LABEL BEHAVIOR**: When selecting a new video to annotate, SimBA assumes that the behavior is absent in any given frame unless indicated by the user. In other words, the default annotation is that the behavior(s) are **not** present. 
* (2) **PSEUDO-LABELLING**: When selecting a new video to annotate, SimBA uses machine classifications the default annotation. Thus, any frame with a classification probability above the user-specified threshold will have **behavior present** as the default value.  
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
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_3_new.png" />
</p>

In this image I have marked out 9 different parts of the window, which will we can use to accurately label the frames of the video as containing (or not containing) your behavior(s) of interest. We go over the 

(1) In the title header of the window, it will say which type (of the ones listed above) of annotations you are doing. I opened the annotation interface through the **LABEL BEHAVIOR** setting, hence it reads **ANNOTATING FROM SCRATCH**. If you are using **PSEUDO-LABELLING**, it will read **PSEUDO-LABELLING**, and if you opened **ADVANCED LABEL BEHAVIOR**, it will read **ADVANCED ANNOTATION**

**(2)** In box 2, there are buttons to navigate between the frames of your video. The middle entry box is telling you the `frame number` within the video that is currently beeing displayed. The inner buttons `>` and `<` will show you the proceeding and preceding frame, respectively. The outer buttons `>>` and `<<` will show tou the last and first frame of the video, respectively. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_1.gif" />
</p>

To navigate to a specific frame, change the value in the frame number entry box and click on `Jump to selected frame`. To jump a user-specified number of frames forwards or backwards in the video, drag the `Jump Size` bar to set the number of frames you wish to jump forwards or backwards. Then use the `<<` and `>>` buttons next to the `Jump size` bar to jump the selected number of frames in the video:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_2.gif" />
</p>


**(3)** This part of the window contains one check box for each classifier in your SimBA project. In my project, I only have one behavior - `Attack`. This is the first time I view this frame in the SimBA annotator GUI - and because I started the annotator GUI through the **LABEL BEHAVIOR** button - it's by default unchecked (behavior is not present). If the behavior of interest (Attack) is occuring in the frame I am viewing, then I go ahead and check the `Attack` checkbox. If You then navigate back or forwards several frames, and back to the frame where I checked the `Attack` checkbox, you can see that it remains checked for that particular frame. My choice have been recorded, and is being held in SimBA memory. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_3.gif" />
</p>


**(4)** There are times where I want to batch label a range of images as either containing or not containing by behavior(s) of interest and for that we will use the menus in sub-menu number 4. In the gif below, I tick the `Frame range` checkbox. I then fill in a start frame (57) and an end frame (257). I next tick the checkboxes for the behavior that is present in these frames. Finally, I click `Save Range` to store my selections in SimBA memory. The viewed frame will jump to the last frame in the selected range.  

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_4.gif" />
</p>


**(5)** At times, it can be difficult to see what is going on when viewing a still frame, and we will need to look at a sequence of frames in order to judge of the animal is doing the behaviors of interest or not. To view the video, click on the `Open video` button at the top right corner. At the bottom right of the video, the current time and current frame number is printed in yellow font. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_5.gif" />
</p>

**(6)** If you highlight this video (click on it), you can use keybord shortcuts to study it frame-by-frame. Once the video is highlighted. Use the keyboard shortcuts printed in the main frame of the annotation GUI to navigate between frames. After you have pressed `p` for pause, you can also close the video by clicking the window close button at the top left. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_4.png" />
</p>

**(7)** When you're viewing the video, you may see a frame that you want to view in the main SimBA annotator and view, label, or change labels for that partuclar frame or range of frames. To do this. First highlight the video and press the `p` button on your keyboard for pause. Next, click the `Show current video frame` button. This will diplay whatever frame is shown in the video player in the main SimBA annotator GUI. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_6.gif" />
</p>


**(8)** There are also several keyboard-shortcuts that will allow you to navigate the frame displayed in the main SimBA annotator GUI. These keyboard shortcuts allows you to perfrom some of the same functions as performed with the buttons in part **1**. They also allow you to save your annotations onto the computer disc and into your SimBA project (see below) which you are required to do in order to use your annotations for creating mchine learning models. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_5.png" />
</p>

**(8)** The last button in the SimBA annotator interface is labelled `Save Annotations` and it is **very important**. This buttons saves your annotations into your SimBA project  which you are required to do in order to use the annotation for creating mchine learning models. Clicking this buttons saves a data file inside your `project_folder/csv/targets_inserted` directory. This file will contain all of the body-part coordinates and features in seperate columns, plus a few columns at the end (one for each bahevior that you are annotating) with the headers that represent the behaviors. Hence, clicking this button in this tutoral, will generate a file inside the `project_folder/csv/targets_inserted` directory called `BtWGANP` where the last column is named `Attack`. This column will be filed with `1`s and `0`s: a `1` for every frame where I noted the behavior to be present, and a `0` for every frame where I note the behavior to be absent. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_6.png" />
</p>

## CONTINOUING PREVIOUSLY STARTED ANNOTATIONS

After clicking `Save Annotations` and closing the SimBA annotation GUI, you may want to come back and continoue annotating the same video. To do this, click the `Select video (continue existing video annotation)` button:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_7.png" />
</p>

When clicked, this bring up a file selection dialog menu. We navigate to the `project_folder/videos` directory and select a video we wich to continue annotating. SimBA will read the file name for the video you select e.g., (BtWGANP.mp4), and scan the `project_folder/csv/targets_inserted` directory within your SimBA project to find the file that represents your previous annotations. 

## NOTES ON THE DIFFERENT FORMS OF BEHAVIOR LABELLING (ADVANCED vs. PSUDO vs. 'STANDARD' LABELLING)


### 'STANDARD' LABELLING

If you use standard labelling (as in the tutorial above) all frames are by default labelled as behavior absent. Thus, if I open a video of 3k frames, and immidiatly click `Save Annotations`, a file will be saved inside the `project_folder/csv/targets_inserted` directory with 3k rows, where the final columns representing my annotations all are filled with `0s`. Using the 'STANDARD' labelling method, it is not *required* to view all frames, it is only necessery to label all the frames where you know that the behavior is present. 

However, it is very important to not feed your classifier errorous information. An assumption that behavior is absent in unviewed frames, and the behavior is actually present, can lead to terrible classifiers. If you are working with  infrequent behaviors, and have a good understanding of when they are occuring, then this option can nevertheless save you a lot of time. 


### 'PSUDO' LABELLING

If you use pseudo labelling (as in the tutorial above) all frames are, by default, labelled according to what your latest machine learning inference on that video (and your input threshold) said was occuring in that frame. Therefore, in order to perform psudo-labelling, the video has to have been scored with a classifier. That means that the video has to have been analyzed through the [Run machine model] header, and a file representing the video has to be present within the `project_folder/csv/machine_results` directory. To read more about the benefits and motives behind pseudo-labelling, check out [THIS TUTORIAL](https://github.com/sgoldenlab/simba/edit/master/docs/pseudoLabel.md). 


### 'ADVANCED' LABELLING

If you use ADVANCED labelling, all frames are, by default, labelled as `None`. That means that if you have not looked at a frame in the SimBA annotation GUI, and indicated what the frame contains (behavior or no behavior), then the information for that frame won't be saved when clicking the `Save Annotations`,  and the frame won't be used in any downstream machine learning classificatier creation tasks. For example, if you (i) open a video containing 3k frames, (ii) look at, and annotate, the first N frames, (iii) and click `Save Annotations`, and navigate to the `project_folder/csv/machine_results` directory and open the file representing the video you just annotated. If you open this file, you should see that it containing N rows. 


