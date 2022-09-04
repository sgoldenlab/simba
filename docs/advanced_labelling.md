# <p align="center"> SimBA behavioral Annotator GUI (ADVANCED LABELLING) </p>


The SimBA behavioural annotator GUI is used to label (annotate) frames in videos that contain behaviors of interest. SimBA allows four different ways of annotating behaviors as present or absent in individual frames. The  difference between the methods is how **non** user-annoated frames are going to be treated. 

* (1) **LABEL BEHAVIOR**: When selecting a new video to annotate, SimBA assumes that the behavior is absent in any given frame unless indicated by the user. In other words, the default annotation is that the behavior(s) are **not** present. 
* (2) **PSEUDO-LABELLING**: When selecting a new video to annotate, SimBA uses machine classifications the default annotation. Thus, any frame with a classification probability above the user-specified threshold will have **behavior present** as the default value.  
* (3) **ADVANCED LABEL BEHAVIOR**. When selecting a new video to annotate, SimBA **has no default annotatation for any frame**. In other words, the user is required annotate each frame as either behavior-absent or behavior-present. Only annotated frames will be used when creating the machine learning model(s). 
* (4) **IMPORT THIRD-PARTY BEHAVIOR ANNOTATIONS**. Use these menus to import annotations created in other tools (without performing annotations in SimBA. Click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md) to learn more about how to import annotations from third-party software. 

This tutoral details how to use the `ADVANCED LABEL BEHAVIOR` annotator in SimBA. For information on how to use the standard `LABEL BEHAVIOR` annotator or `PSEUDO-LABELLING`, click [HERE](https://github.com/sgoldenlab/simba/edit/master/docs/label_behavior.md). For information on how to append annotations created in alternative third-party software, click [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md)


