

# Tutorial for processing video in batch
The batch pre-processing tools allows users to change video parameters (clip/trim/crop etc.) for multiple videos independently. Once the required parameters has been set for all the videos, the user press `Execute` and the new videos will be generated. Videos are processed via **FFmpeg** Click [here](https://m.wikihow.com/Install-FFmpeg-on-Windows) to learn how to install FFmpeg on your computer. 

## Pipeline

The video parameters specified by the user will be processed in the following sequence. If the user leaves certain parameters unchanged in, they will be ignored in the pipeline.   

<img src=https://github.com/sgoldenlab/simba/blob/master/images/processvideo_flowdiagram.png width="800" height="310" />

## Step 1: Folder Selection

<img src=https://github.com/sgoldenlab/simba/blob/master/images/processvideo.PNG width="285" height="111" />

1. To get started, on the very top of the tabs in the window, click on `Process Videos` --> `Batch pre-process videos`. The windows in the image below will pop up

![](/images/batchprocessvideo1.PNG)

2. Under **Folder Selection**, `Video directory`, click on `Browse Folder` and navigate to a folder that contains videos to process and click 'Select Folder`.

![](/images/selectfolderwithvideos.PNG)

3. Under `Output Directory`, lick on `Browse Folder` and navigate to a folder*(usually an empty folder)* that you wished to saved your processed videos and click 'Select Folder`.

4. Click `Confirm` to confirm the directories selected.

>**Note**: Please make sure there is no spaces in your folder name or video name.

<img src=https://github.com/sgoldenlab/simba/blob/master/images/processvideo2.PNG width="450" height="200" />

## Step 2: Select parameters

1. A table with the videos in the folder will pop up.

<img src=https://github.com/sgoldenlab/simba/blob/master/images/batchprocessvideo.PNG width="1102" height="200" />

2. If you wish to crop your videos, click on the `Crop` button. *(The button will turn red once you clicked on it)* A frame from the video will pop up. Left click on the frame and drag it to crop the video. Once it is done, double tap "Enter" on your keyboard.

![](/images/cropvideoroi.gif)

3. If you wish to trim your videos, check on the `Shorten` box and enter the **Start Time** and  **End Time** in the HH:MM:SS format.

4. If you wish to downsample your video, check on the `Downsample` box and enter the **Width** and **Height**.

5. If you wish to Grayscale your video, check on the `Grayscale` box.

6. If you wish to superimpose frames onto your video, check on the `Add Frame #` box.

7. If you wish to apply CLAHE, check on the `CLAHE` box.

8. In the first row, there are `Select All` checkboxes to check all the checkboxes in the given column.

> **Note:** The `Select All` checkbox might be off position. We are working on it to fix it.

## Step 3: Execute

1. Once the parameters are confirmed, click `Execute`

2. The final processed videos will be in the `Output Directory` that you have selected.

<img src=https://github.com/sgoldenlab/simba/blob/master/images/processvideo4.PNG width="500" height="200" />

3. The **tmp** folder contains the step-by-step processsed videos.

4. The **process_archieve** folder contains the **.txt** file of the process that were ran. 

5. The **Output Directory** will have all the final processed video.

