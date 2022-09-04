# <p align="center"> SimBA behavioral Annotator GUI (ADVANCED LABELLING) </p>


The SimBA behavioural annotator GUI is used to label (annotate) frames in videos that contain behaviors of interest. SimBA allows four different ways of annotating behaviors as present or absent in individual frames. The  difference between the methods is how **non** user-annoated frames are going to be treated. 

* (1) **LABEL BEHAVIOR**: When selecting a new video to annotate, SimBA assumes that the behavior is absent in any given frame unless indicated by the user. In other words, the default annotation is that the behavior(s) are **not** present. 
* (2) **PSEUDO-LABELLING**: When selecting a new video to annotate, SimBA uses machine classifications the default annotation. Thus, any frame with a classification probability above the user-specified threshold will have **behavior present** as the default value.  
* (3) **ADVANCED LABEL BEHAVIOR**. When selecting a new video to annotate, SimBA **has no default annotatation for any frame**. In other words, the user is required annotate each frame as either behavior-absent or behavior-present. Only annotated frames will be used when creating the machine learning model(s). 
* (4) **IMPORT THIRD-PARTY BEHAVIOR ANNOTATIONS**. Use these menus to import annotations created in other tools (without performing annotations in SimBA. Click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md) to learn more about how to import annotations from third-party software. 

This tutoral details how to use the `ADVANCED LABEL BEHAVIOR` annotator in SimBA. For information on how to use the standard `LABEL BEHAVIOR` annotator or `PSEUDO-LABELLING`, click [HERE](https://github.com/sgoldenlab/simba/edit/master/docs/label_behavior.md). For information on how to append annotations created in alternative third-party software, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md)

## Step 1. Loading project_config file 
In the main SimBA menu, click on `File > Load Project > Load Project.ini > Browse File` and select the config file (project_config.ini) representing your SimBA project. This step **must** be done before proceeding to the next step.

## Step 2. Opening the labelling behavior interface

Once your project is loaded, click on the [Label Behavior] tab and you should see the below four sub-menus (Note: I'm writing this document on a Mac, if you're running SimBA on a PC or Linux, the aestetics might be slightly different): 
<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_1.png" />
</p>

These four different sub-menus represent four different ways of annotating your videos. In this tutorial, we will use the buttons in sub-menu number `3`. We will click on `Select video (create new video annotation)` in the `ADVANCED BEHAVIOR LABEL` submenu. This will bring up a file selection dialog menu. We navigate to the `project_folder/videos` directory and select a video we wich to annotate. In this tutorial, I am selecting `BtWGANP.mp4` and click Open:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/docs/annotator_2.png" />
</p>


## Step 3. Using the advanced labelling behavior interface

Once I've selected my video file, the annotation interface will pop open, looking something like the below image. The different buttons have identical functions to when using standard `LABEL BEHAVIOR` annotator or `PSEUDO-LABELLING`. For detailed instructions of their functions, [check out the standard label behavior tutorial](https://github.com/sgoldenlab/simba/edit/master/docs/label_behavior.md)

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/annotator_adv_1.png" />
</p>

In this tutorial I am annotating a video within a SimBA project with two classifiers, 



