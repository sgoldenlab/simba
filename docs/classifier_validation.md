# Classifier Validation

Classifier validation generates videos in the project that contains all the bouts of the target behavior that the machine predicts.

![](/images/classifiervalidation1.PNG)

- `Seconds` is the seconds to add at the starting of a bout and the end of the bout. Let's say there was a bout of **2 seconds of an attack**, entering 1 in the **Seconds** entry box will add 1 second before the 2 second attack and 1 second after.

- `Target` is the target behavior to implement into this step.

## How to use it

1. Enter 1 or 2 in the `Seconds` entry box. *Note: the larger the seconds, the longer the duration of the video.**

2. Select the target behavior from the `Target` dropdown box.

3. Click `Validate` button and the videos will be generated in `/project_folder/frames/output/classifier_validation`. The name of the video will be formated in the following manner: **videoname** + **target behavior** + **number of bouts** + .mp4

![](/images/classifiervalidation.gif)
