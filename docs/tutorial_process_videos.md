

# Tutorial for processing videos in batch

It is often helpful, and sometimes necessary, to pre-process experimental videos prior to analysis. This is generally accomplished through the use of open-source approaches like **FFmpeg** or commercial software, but can be a time consuming and cumbersome when applying to numerous similar videos. To streamline this process, SimBA incorporates **FFmpeg** into a a batch-process GUI. 

This video pre-processing tool allows you to change multiple video parameters (clip/trim/crop, etc.) for multiple videos in a batch-process that is rapidly configured and then executed when ready. Once the required parameters has been set for all the videos, the user press `Execute` and the new videos will be generated according to the user input. Videos are processed using **FFmpeg**. Click [here](https://m.wikihow.com/Install-FFmpeg-on-Windows) to learn how to install FFmpeg on your computer. 

**Note: Processing numerous high-resolution or long-duration videos takes time. We strongly suggest that you not execute a batch until you are ready to commit computational time to the process.**

We suggest pre-processing videos in the following scenarios:

1) Red-light: If you have recorded in red-light conditions, we suggest converting to gray-scale and using CLAHE.
2) Reflections: If your recordings include reflections on the walls of your testing apparatus, we suggest cropping to remove the reflections.
3) Artifacts: If your recordings include frames where your hands (or other unintended features) are included in the video, at either the start or the end, we suggest trimming the videos to remove these frames.
4) Resolution: If your recordings are significantly higher resolution than needed, we suggest down-sampling to decrease processing time.

## Pipeline

The video parameters that you specify will be processed in the following sequence. If the user leaves certain parameters unchanged, then they are ignored in the pipeline.  

![alt-text-1](/images/processvideo_flowdiagram.png "processvideo_flowdiagram")

## Step 1: Folder Selection

![alt-text-1](/images/processvideo.PNG "processvideo")

1. To begin batch pre-processing, in the main SimBA window click on `Process Videos` --> `Batch pre-process videos`. The window shown below will display. 

![](/images/batchprocessvideo1.PNG)

2. Under **Folder Selection** heading and next to `Video directory`, click on `Browse Folder` and navigate to a folder that contains the videos that should be batch processed and click on 'Select Folder`. All vidoes that you would like to process must be present in this directory.

![](/images/selectfolderwithvideos.PNG)

3. Next to `Output Directory`, click on `Browse Folder` and navigate to a folder *(usually a new, empty, folder)* that should store the processed videos and click on 'Select Folder`.

4. Click to `Confirm` the two selected directories.

>**Note**: Please make sure there is no spaces in your folder names or video names, use underscores if needed.

![](/images/processvideo2.PNG)

## Step 2: Select parameters

1. Once you select `Confirm`, a table listing the videos in the `Video directory` folder will auto-generate and display. This sequence of videos will now be processed with the parameters you selected, started with the cropping fucntion.

![](/images/batchprocessvideo.PNG)

2. If you wish to crop your videos, click on the `Crop` button. A frame from the video will pop up. Left click on the frame and drag the rectangular bounding box to mark the region of the video you wish to keep. Once the rectangle is marked, double tap "Enter" on your keyboard. *(The relevant crop button will turn red once you have selected to crop a video)*

![](/images/cropvideoroi.gif)

3. If you wish to trim a specific video, check the `Shorten` box and enter the **Start Time** and  **End Time** in the HH:MM:SS format.

4. If you wish to downsample a specific video, check the `Downsample` box and enter the **Width** and **Height** in pixels.

5. If you wish to change a specific video to grayscale, check the `Grayscale` box.

6. If you wish to superimpose the specific frame numbers onto the frames of the video, check on the `Add Frame #` box. For an example output video with frame numbers overlaid, click [here](https://youtu.be/TMQmNr8Ssyg). 

7. If you wish to apply CLAHE, check on the `CLAHE` box. For more information on CLAHE, click [here](https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html)

8. In the first row of the table, there are `Select All` checkboxes. Use these checkboxes to apply a manipulation to all of the videos in the folder. 

> **Note:** We know that the `Select All` checkbox might be slightly off position in the table. We are working on a fix. 

## Step 3: Execute

1. Once all the parameters are set, click on `Execute`. 

2. The final output videos will be saved in the `Output Directory` that you selected in *Step 1*.

![alt-text-1](/images/processvideo4.PNG "processvideo4.PNG")

3. A subfolder in the `Output Directory` called **tmp** will contain the step-by-step processsed videos.

4. The **process_archieve** folder contains a **.txt** file that lists the processes that were run. 

5. The **Output Directory** will contain all the final processed videos.

### NEXT STEP: [Create tracking model](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md) 
