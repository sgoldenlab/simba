# Post-classification Validation (detecting false-positives)

Post-classification validation generates a video for each .CSV in the project that contains the concatenated clips of all the events of the target behavior that the predictive classifier identifies.

![](/images/classifiervalidation1.PNG)

- `Seconds` is the duration to add in seconds to the start of an event and to the end of the event. Let's say there was a event of **2 seconds of an attack**, entering 1 in the **Seconds** entry box will add 1 second before the 2 second attack and 1 second after.

- `Target` is the target behavior to implement into this step.

## How to use it

1. Enter 1 or 2 in the `Seconds` entry box. *Note: the larger the seconds, the longer the duration of the video.**

2. Select the target behavior from the `Target` dropdown box.

3. Click `Validate` button and the videos will be generated in `/project_folder/frames/output/classifier_validation`. The name of the video will be formated in the following manner: **videoname** + **target behavior** + **number of bouts** + .mp4

![](/images/classifiervalidation.gif)
