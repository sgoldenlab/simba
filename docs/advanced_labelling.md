# <p align="center"> SimBA behavioral Annotator GUI (ADVANCED LABELLING) </p>


The SimBA behavioural annotator GUI is used to label (annotate) frames in videos that contain behaviors of interest. SimBA allows three different ways of annotating behaviors as present or absent in individual frames. The  difference between these methods is how **non** user-annoated frames are treated

* (1) **LABEL BEHAVIOR**: When selecting a new video to annotate, SimBA assumes that the behavior is absent in any given frame unless indicated by the user. In other words, the default annotation is that the behavior(s) are **not** present. 
* (2) **PSEUDO-LABELLING**: When selecting a new video to annotate, SimBA uses machine classifications the default annotation. Thus, any frame with a classification probability above the user-specified threshold will have **behavior present** as the default value.  
* (3) **ADVANCED LABEL BEHAVIOR**. When selecting a new video to annotate, SimBA **has no default annotatation for any frame**. In other words, the user is required annotate each frame as either behavior-absent or behavior-present. Only annotated frames will be used when creating the machine learning model(s). 

This tutoral details how to use the `ADVANCED LABEL BEHAVIOR` annotator in SimBA. To use advanced labelling in SimBA, we recommend first reading the tutorial on how to use the [standard `LABEL BEHAVIOR` / `PSUDO-LABELLING`](https://github.com/sgoldenlab/simba/edit/master/docs/label_behavior.md). The [standard `LABEL BEHAVIOR` / `PSUDO-LABELLING`](https://github.com/sgoldenlab/simba/edit/master/docs/label_behavior.md) tutoral documents the functions of the buttons in the SimBA annotator GUI, which are identical when using advanced labelling in SimBA. 

> Note: For information on how to append annotations created in alternative third-party software outside of SimBA, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md)

## Step 1. Loading project_config file 
In the main SimBA menu, click on `File > Load Project > Load Project.ini > Browse File` and select the config file (project_config.ini) representing your SimBA project. This step **must** be done before proceeding to the next step.

## Step 2. Opening the labelling behavior interface

Once your project is loaded, click on the [Label Behavior] tab and you should see the below four sub-menus (Note: I'm writing this document on a Mac, if you're running SimBA on a PC or Linux, the aestetics might be slightly different): 
<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_0.png" />
</p>

These four different sub-menus represent four different ways of getting human annotations appended to your video data. In this tutorial, we will use the buttons in sub-menu number `3`. We will click on `Select video (create new video annotation)` in the `ADVANCED BEHAVIOR LABEL` submenu. This will bring up a file selection dialog menu. We navigate to the `project_folder/videos` directory and select a video we wich to annotate in SimBA. In this tutorial, I am selecting `Together_1.mp4` and click `Open`:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_10.png" />
</p>


## Step 3. Using the advanced labelling behavior interface

Once I've selected my video file, the annotation interface will pop open, looking something like the below image. The different buttons have identical functions to when using standard `LABEL BEHAVIOR` annotator or `PSEUDO-LABELLING`. For detailed instructions of their functions, [check out the standard label behavior tutorial](https://github.com/sgoldenlab/simba/edit/master/docs/label_behavior.md).

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_adv_1.png" />
</p>

In this tutorial I am annotating a video within a SimBA project that has two classifiers. We can see this in the `Check behavior` submenu, which is populated with two rows for the two classifiers (`Attack` and `Sniffing`) and two columns for annotating if the behavior happening, or not happening, in the frame (`PRESENT` and `ABSENT`). 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_adv_2.png" />
</p>

In the below gif, I am viewing frame number `28` of a video called `Together_1` (Note: you can see the name of the video currently beeing annotated by looking in the GUI annotator title window). In frame number `28`, I annotate `Sniffing` as beeing present, and `Attack` as being absent. My annotations are saved into SimBA working memory when I click to navigate to a different frame.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/advanced_annotator_6_fast.gif" />
</p>

As opposed to [`PSEDO-LABELLING` or standard `LABELING BEHAVIOR`](https://github.com/sgoldenlab/simba/edit/master/docs/label_behavior.md), within advanced labelling, it is possible to **omit** a frame from being annotated and omitted from ant downstream machine learning processes. This means that we score the behavior(s) as neither `PRESENT` or `ABSENT` in the frame (or a range of frames). For example, if we **do not** tick any behaviors as either absent of present (leave all classifier checkboxes unticked) for a frame (or range of frames), that means that the specific frame (or range of frames) will be omitted as annotated examples for any downstream processes involving  machine learning predictive classifiers. In the gif below, I use the label `frame range` tool in the annotator to label frames `50-100` as **containing no suitable examples** of `Sniffing` and `Attack` as being present or absent. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/advanced_annotator_7_fast.gif" />
</p>


> IMPORTANT: If you are working on annotating a video for multiple classifiers (just as in the current tutorial, where we are annotating both `Attack` and `Sniffing`), then we cannot annotate some behaviors as `PRESENT` or `ABSENT`, while we annotate other behaviors as *neither* `PRESENT` or `ABSENT` in the same frame. For example,  in the below gif I annotate `Sniffing` as present in frame number `35` and I don't tick `Attack` as either `PRESENT` or `ABSENT`. In this scenario I will be presented with an error message, and asked to either (i) annotate all behaviors as neither `PRESENT` or `ABSENT` (remove all ticks from all tick-boxes), or (ii) provide ticks in tick-boxes representing all behaviors. In this below gif example, I go for the latter solution, and tick `Attack` as absent. 


<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/advanced_annotator_8_fast.gif" />
</p>

Once you have completed your annotations, we need to save all your annotations into your SimBA project. We do this by clicking the `Save Annotations` button, and this is a very important button. This buttons saves your annotations into your SimBA project, which you are required to do in order to use the annotations for creating machine learning models. Clicking this buttons saves a data file inside your `project_folder/csv/targets_inserted` directory. This file will contain all of the body-part coordinates and features in seperate columns, plus a few additional columns at the end (one for each behavior that you are annotating) with the headers that represent the behavior names. Hence, clicking this button, in this tutoral, will generate a file inside the project_folder/csv/targets_inserted directory called `Together_1` (which is the name of the video I am annotating). The last two columns in this file will be named `Attack` and `Sniffing`. These two column will be filed with `1` and `0`s - a 1 for every frame where I noted the behavior to be present, and a 0 for every frame where I note the behavior to be absent. This file will contain as many rows as the number of frames where I annotated the behaviors as `PRESENT or `ABSENT`. 

## CONTINOUING PREVIOUSLY STARTED ANNOTATIONS

After clicking `Save Annotations` and closing the SimBA annotation GUI, you may want to come back and continue annotating the same video using the advanced SimBA annotation GUI. To do this, click the `Select video (continue existing video annotation)` button in the **ADVANCED LABEL BEHAVIOR** sub-menu:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/advanced_annotator_9.png" />
</p>

SimBA will then look for the file representing your video inside the `project_folder/csv/targets_inserted` directory. SimBA will also look for the file representing your video inside the `project_folder/csv/features_extracted` directory in order to de-code how many frames you have provided neither `PRESENT` or `ABSENT` annotations for. It will next bring up the SimBA advanced annotation GUI at the frame number where you last clicked `Save Annotations`. 



##
Author [Simon N](https://github.com/sronilsson)















