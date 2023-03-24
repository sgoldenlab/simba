# <p align="center"> SimBA Behavioral Annotator GUI </p>


The SimBA behavioural annotator GUI is used to label (annotate) frames in videos that contain behaviors of interest. SimBA appends the behavioral annotations directly to the pose-estimation tracking data to build supervised machine learning predictive classifiers of behaviors of interest. Specifically, this GUI integrates two additional consoles with the main SimBA console: (1) a video player and (2) a frame viewer. The video player and frame viewer are synced, such that pausing or starting the video will advance the frame viewer to the identical frame that the video was paused or started at. The frame viewer is then used to annotate when behaviors of interest are present or absent within the given frame. Such annotations can be flexibily annotated from single to numerous frames using the annotation interface. Please see below for details for how to best use the annotator in your specific use case. 

Note that SimBA performs similar functions such as the open-source [JWatcher](http://www.jwatcher.ucla.edu/) or commercial [Noldus Observer](https://www.noldus.com/human-behavior-research/products/the-observer-xt) systems, with the exception that SimBA automates the backend integration of behavioral annotation with creating predictive classifiers. If you already have such annotations stored in alterantive file formats, like [JWatcher](http://www.jwatcher.ucla.edu/) or [Noldus Observer](https://www.noldus.com/human-behavior-research/products/the-observer-xt), they can be appended directly to the tracking data and no behavioral annotations needs to be done in SimBA. To append annotations created in alternative third-party software, check out [THIS TUTORIAL](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md). The [Crim13 dataset](http://www.vision.caltech.edu/Video_Datasets/CRIM13/CRIM13/Main.html) was annotated using [Piotr’s Matlab Toolbox](https://github.com/pdollar/toolbox) and we appended the annotations to the tracking data using a version of [this script](https://github.com/sgoldenlab/simba/blob/master/misc/Caltech_2_DLC.py). 

If you already have annotated videos created with these alternative tools, or any other behavioral annotator, and would like to use them to create predictive classifiers, please let us know as we would like to write scripts that could process these data for SimBA. If you have created such scripts yourself, please consider contributing them to the community! If you have any issues using the SimBA annotation interface, please open a [GitHub issue](https://github.com/sgoldenlab/simba/issues) or reach out to us on our [Gitter](https://gitter.im/SimBA-Resource/community) support channel. **We will provide support in either scenario**.

## Step 1. Loading project_config file 
In the main SimBA menu, click on `File > Load Project > Load Project.ini > Browse File` and select the config file (project_config.ini) representing your SimBA project. This step **must** be done before proceeding to the next step.

## Step 2. Opening the labelling behavior interface

Once your project is loaded, click on the [Label Behavior] tab and you should see the below four sub-menus (Note: I'm writing this document on a Mac, if you're running SimBA on a PC or Linux, the aestetics might be slightly different): 
<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_0.png" />
</p>

These four different sub-menus represent four different ways of getting human annotations appended to your videos. The differences between the different menus, and when to use them, are detailed below. In brief, the main difference between the methods is how **non** user-annoated frames are going to be treated.  

* (1) **LABEL BEHAVIOR**: When selecting a new video to annotate, SimBA assumes that the behavior is absent in any given frame unless indicated by the user. In other words, the default annotation is that the behavior(s) are **not** present in the frame. 
* (2) **PSEUDO-LABELLING**: When selecting a new video to annotate, SimBA uses prior machine classifications as the default annotation. Thus, any frame with a classification probability *above* the user-specified threshold will have **behavior present** as the default value. Conversely, any frame with a classification probability *below* or at the user-specified threshold will have **behavior absent** as the default value. Pseudo-labelling requires you to [first analyze the video using behavioral classifiers](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data).
* (3) **ADVANCED LABEL BEHAVIOR**. When selecting a new video to annotate using advanced labelling, SimBA **has no default annotatation for any frame**. In other words, the user is required annotate each frame as either behavior-absent or behavior-present. Only frames with behavior labelled as either present or absent will be used when creating the machine learning model(s). You can read more about advanced labelling [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/advanced_labelling.md).
* (4) **IMPORT THIRD-PARTY BEHAVIOR ANNOTATIONS**. Use these menus to import annotations created in other software tools (without performing any annotation work in SimBA. Click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md) to learn more about how to import annotations from third-party software. 

Regardless of which method you choose (standard `LABEL BEHAVIOR`, `PSEUDO-LABELLING`, or `ADVANCED LABEL BEHAVIOR`), the SimBA behavior annotation interface, its buttons and functions, are very similar. The procedure **is** identical when using standard `LABEL BEHAVIOR`, `PSEUDO-LABELLING`, while differing slighly when using the `ADVANCED LABEL BEHAVIOR` interface. If you are intrested in the using the `ADVANCED LABEL BEHAVIOR` tool, I advice to first get familiar with the standard `LABEL BEHAVIOR` interface, before proceeding to the [ADVANCED LABEL BEHAVIOR tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/advanced_labelling.md). 

In this tutorial, we will click on `Select video (create new video annotation)` in the `LABEL BEHAVIOR` sub-menu. This will bring up a file selection dialog menu. We navigate to the `project_folder/videos` directory and select a video we wich we want to annotate. In this tutorial, I am selecting `BtWGANP.mp4` and click `Open`:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/docs/annotator_2.png" />
</p>

## Step 3. Using the labelling behavior interface

Once I've selected my video file, the annotation interface will pop open, looking something like this:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_3_new.png" />
</p>

In this image I have marked 9 different parts of the window with boxes, which will we can use to scan and accurately label the frames of the video as containing (or not containing) your behavior(s) of interest:

**(1)** In the title header of the window, it will state which type of annotations you are currently doing (of the ones listed above). I opened the annotation interface through the **LABEL BEHAVIOR** setting and the `Select video (create new video annotation)` button, hence it reads **ANNOTATING FROM SCRATCH**. If you are using **PSEUDO-LABELLING**, it will read **PSEUDO-LABELLING**, and if you opened **ADVANCED LABEL BEHAVIOR**, it will read **ADVANCED ANNOTATION**.

**(2)** In box 2, there are buttons to navigate between different frames of your video. The middle entry box is telling you the `frame number` within the video that is currently beeing displayed. Clicking the inner buttons `>` and `<` will show you the proceeding and preceding frame, respectively. Clicking the outer buttons `>>` and `<<` will show you the last and first frame of the video, respectively. 

**Navigating to a different frame than the one being viwerd (i.e., a preceding or a proceeding frame) will save your annotation selections for the previously-viewed frame into SimBA memory (more information below).**

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_1.gif" />
</p>

If you want to navigate to a specific frame number, change the value in the frame number entry box and click on the `Jump to selected frame` button. To jump a user-specified number of frames forwards (or backwards), drag the `Jump Size` bar to set the number of frames you wish to jump forwards or backwards. Then click the `<<` and `>>` buttons next to the `Jump size` bar to jump the selected number of frames in the video:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_2.gif" />
</p>


**(3)** If you are using standard *LABEL BEHAVIOR*, or *PSEUDO-LABELLING*, this part of the window contains one check box for each classifier in your SimBA project. In my project, I only have one behavior - `Attack`. Furthermore, this is the first time I view this frame in the SimBA annotator GUI - and because I started the annotator GUI through the *LABEL BEHAVIOR* button - it's by default unchecked (behavior is not present). If the behavior of interest (Attack) is occuring in the frame I am viewing, then I go ahead and check the `Attack` checkbox. If I then navigate back or forwards several frames, and back to the frame where I checked the `Attack` checkbox, you can see that it remains checked for that particular frame. My choice that `Attack` is occuring in frame `57` has been recorded, and is being held in SimBA memory.

>Note: If you are using **ADVANCED LABEL BEHAVIOR** in SimBA, every classifier in your SimBA project will have **two** check-boxes (rather than one) in the SimBA annotator GUI - one checkbox to indicate that the behavior is present, and one check-box to indicate that the behavior is absent. For a further tutorial documenting how to use the advanced behavior annotator in SimBA, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/advanced_labelling.md). 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_3.gif" />
</p>


**(4)** There are times where we may want to batch label a range of images as either containing, or not containing, our behavior(s) of interest and for that we will use the functions in sub-menu number `4`. In the gif below, I tick the `Frame range` checkbox. I then fill in a start frame number (57) and an end frame number (257). I next tick the checkboxes for the behavior(s) that is present in these frames. Finally, I click `Save Range` to store my selections in SimBA memory. The viewed frame will jump to the last frame in the selected range when I click `Save Range`.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_4.gif" />
</p>


**(5)** At times, it may be difficult to see what is going on when viewing a still frame, and we will need to look at a sequence of frames in order to judge if the animal is doing the behaviors of interest or not. To view the video, click the `Open video` button at the top right corner and another window showing the video will pop open. At the bottom right of the new video window that pops open, the current time and current frame number is printed in yellow font. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_5.gif" />
</p>

**(6)** If you highlight this video window (i.e., click on it), you can use keybord shortcuts to study it frame-by-frame. Once the video is highlighted, press `p` on your keyboard to pause the video. Use the keyboard shortcuts printed in the main frame of the annotation GUI to navigate between frames. After you have pressed `p` for pause, you can also close the video by clicking the window close button at the top left. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_4.png" />
</p>

**(7)** When you're viewing the video, you may see a frame in the main SimBA annotator GIO and label, or change labels, for that partuclar frame or range of frames. To do this. First highlight the video (by clicking on the frame) and press the `p` button on your keyboard to pause the video. Next, click the `Show current video frame` button. This will diplay whatever frame is shown in the video player in the main SimBA annotator GUI frame-viewer. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_6.gif" />
</p>


**(8)** There are keyboard-shortcuts that will allow you to navigate the frame displayed in the main SimBA annotator GUI. These keyboard shortcuts allows you to perform some of the same functions as performed with the buttons documented in part **(1)** in this document. They also allow you to save your annotations into your SimBA project (see below), which you are **required to do** in order to use your annotations for creating machine learning models. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_shortcuts.png" />
</p>

Two of these shortcuts needs further description:

> Control-a: Hold down the `ctrl` keyboard button and press `a` to move to the next frame as well as keeping the classifier checkboxes the same as however they where checked in the prior frame. Good for quickly labelling successive frames with the same annotations. 
> Control-p: Hold down the `ctrl` keyboard button and press `p` to print a table telling you how many frames you have annotated so far and how you have anntated them, as in the image below.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/label_table.png" />
</p>

**(8)** The last button in the SimBA annotator interface is labelled `Save Annotations` and it is a **very important** button. This buttons saves your annotations into your SimBA project which you are required to do in order to use the annotation for creating mchine learning models. Clicking this buttons saves a data file inside your `project_folder/csv/targets_inserted` directory. This file will contain all of the body-part coordinates and features in seperate columns, plus a few additional columns at the end of the file (one for each behavior that you are annotating) with the headers that represent the behavior names. Hence, clicking this button in this tutoral, will generate a file inside the `project_folder/csv/targets_inserted` directory called `BtWGANP` where the last column is named `Attack`. This column will be filled with `1`s and `0`s - a `1` for every frame where I noted the behavior to be present, and a `0` for every frame where I note the behavior to be absent. 

>Note: I used the standard *LABEL BEHAVIOR* method in SimBA in this tutorial. This means that any frame that I did **not** view in the SimBA annotator GUI frame viewer will be saved as **behavior absent**. Hence, if I open the BtWGANP file inside the `project_folder/csv/targets_inserted` directory, there will be a `0` for every frame that I did not view in the SimBA annotator GUI frame viewer. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_6.png" />
</p>

## CONTINOUING PREVIOUSLY STARTED ANNOTATIONS

After clicking `Save Annotations` and closing the SimBA annotation GUI, you may want to come back and continue annotating the same video. To do this, click the `Select video (continue existing video annotation)` button:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_7.png" />
</p>

When clicked, this brings up a file selection dialog menu. We navigate to the `project_folder/videos` directory and select a video we wich to continue annotating. SimBA will read the file name for the video you select e.g., (BtWGANP.mp4), and scan the `project_folder/csv/targets_inserted` directory within your SimBA project to find the file that represents your previously stored annotations for this video. You can then preceed to annotate video frames as documented above. When continuing to annotate a previously started video, SimBA opens the GUI annotator interface at the frame number of the latest saved frame. 

## NOTES ON THE DIFFERENT FORMS OF BEHAVIOR LABELLING (ADVANCED vs. PSUDO vs. 'STANDARD' LABELLING)


### 'STANDARD' LABELLING

If you use standard labelling (as in the tutorial above) all frames are by default labelled as **behavior absent**. Thus, if I open a video of 3k frames, and **immediately** click `Save Annotations`, a file will be saved inside the `project_folder/csv/targets_inserted` directory with 3k rows, where the final columns representing my annotations **all** are filled with `0`. Using the standard `LABEL BEHAVIOR`  method, it is not *required* to view all frames, it is only necessery to label all the frames where you know that the behavior is present. 

It is very important to not feed your classifier errorous information. An assumption that behavior is absent in unviewed frames, where the behavior is actually present, can lead to classifiers that perform terribly. If you are working with infrequent behaviors, and have a good understanding of when they are occuring, then this option can nevertheless save you a lot of time. 


### 'PSUDO' LABELLING

If you use pseudo labelling all frames are, by default, labelled according to what your latest machine learning inference on that video (and your input threshold) said was occuring in that frame. Therefore, in order to perform psudo-labelling, the video has to have been scored with a classifier prior to starting the annotation of that video. That means that the video has to have been analyzed through the [Run machine model] header, and a file representing the video has to be present within the `project_folder/csv/machine_results` directory. To read more about the benefits and motives behind pseudo-labelling, and how to use it, check out [THIS TUTORIAL](https://github.com/sgoldenlab/simba/edit/master/docs/pseudoLabel.md). 

When closing the SimBA annotation GUI after annotating a video using the psedu-annotator in SimBA, you may want to come back and continue annotating the same video. To do this, click the `Select video (continue existing video annotation)` button in the standard `LABEL BEHAVIOR` sub-menu in the [Label behavior] tab. 


### 'ADVANCED' LABELLING

If you use ADVANCED labelling, all frames are, by default, labelled as `None`. That means that if you have not looked at a frame in the SimBA annotation GUI, and indicated what behaviors the frame contains (i.e., if the behvaior is present or absent), then the information for that specific frame **will not** be saved when clicking the `Save Annotations`. That means that the specific frame won't be used in any downstream machine learning classifier creation tasks. For example, if you (i) open a video containing 3k frames, (ii) look at, and annotate, the first N frames, (iii)  click `Save Annotations`, and navigate to the `project_folder/csv/machine_results` directory and open the file representing the video you just annotated. When you open this file, you should see that it only contains N rows representing the N rows you labelled in the SimBA annotator GUI.

This method (`ADVANCED LABELLING`) allows you the greatest control over the data that is used by the machine learning models to build predictive classifiers. This is the only method that allows you to control which frames of the specific video that goes into training and testing the classifiers. While 'PSUDO' and standard labelling use all of the frames of a specific video, `ADVANCED LABELLING` allows you to pick a specific subset of frames from a a specific video for creating machine learning models. To read more about how to use the `ADVANCED LABELLING` method in SimBA, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/advanced_labelling.md).


## MISCELLANEOUS LABELLING TOOLS IN SIMBA

* After we have created (or [imported](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md)) our annotations into SimBA, we may want to view them so confirm that they are accurate. For this, click on `Visualize annotations` in the `LABELLING TOOLS` submenu in the [Label behavior] tab and you should see the follwoing pop-up:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/labelling_901.png" />
</p>

Select which classifier annotatated frames you wish to extract by ticking the appropriate tick-boxes. If you need to down-sample your images, use the `Down-sample images` dropdown to select how many times the images should be reduced from the original. Once completed, click the `RUN` button. You can follow the prohres sin the main SimBA terminal. 

Once complete, the images annotated as **behavior present** will be stored within sub-directories of the `project_folder/frames/output/annotated_frames` folder in the hierarchy of `video name` -> `classifier behavior name`. The file names of the frames are the frame number of the annotated behaviour. E.g., a file at the path `project_folder/frames/output/annotated_frames/My_video/Attack/28.png` represents an **behavior-present** annotation for the behavior `Attack` at the 28th frame of video `My_video`.  



#
Author [Simon N](https://github.com/sronilsson)
