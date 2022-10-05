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

>**Note**: Please make sure there is no spaces in your folder names or video names. Instead use underscores if needed.

![](/images/processvideo2.PNG)

## Step 2: The batch processing interface.

1. Once you select `Confirm`, an interface will be displayed which will allow us to manipulate the attributes of each video, or batch change attributes of all videos in the directory. Below is a screengrab of this interface, which I have labelled into three different parts: **(1) QUICK SETTINGS, (2) VIDEOS, and (3) EXECUTE**. We will go through the functions of each one in turn. 

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/batch_preprocess_2.png" />
</p>

### QUICK SETTINGS

The quick setting menu allows us to batch specify new resolutions, new frames rates, or clipping times for all videos. In my toy example I have 10 videos, each which is 10s long (Note: you can see that they are 10s long in the middle **VIDEOS** table, by looking in the *End Time* column, shich SimBa has populated with the video meta-information data). 

Let's say I want to remove the first 5s from each of the videos, and to do this I can use the `Clip Videos Settings` sub-menu in QUICK SETTINGS. To do this, I set the `Start Time` to 00:00:05, and the `End Time` to 00:00:10 and click `Apply`, as in the gif below. Note that the `Start Time` of all videos listed in the VIDEOS table are updated:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/quick_clip.gif" />
</p>

Similarly, let say I want to downsample all my videos to a 1200x800 resolution. I then update the `Width` and `Height` values in the `Downsample Videos` sub-menu, and click `Apply`:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/quick_downsample.gif" />
</p>

### VIDEOS TABLE

The middle VIDEOS table list all the video files found inside your input directory defined in Step 1, with one video per row. Each video has a `Crop` button, and several entry boxes and radio buttons that allows us to specify which pre-processing functions we should apply to each video. In the header of the VIDEOS table, there are also radio buttons that allows us to tick all of the videos in the table. For example, if I want to apply the 00:00:05 to 00:00:10 clip trimming to all videos, I go ahead and click the `Clip all videos` radio button. If I want downsample all videos, I go ahead and click the `Downsample All Videos` radiobutton. If I want to `Clip all videos` *except* few videos. I go ahead and de-select the videos I want to omit from downsampling. The same applies for the FPS, greyscale, CLAHE and Frame count radio buttons:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/header_radiobtn.gif" />
</p>

Next, it might be that you want to crop some of the videos listes in the VIDEOS table. To do this, click on the `Crop` button associated with the video. In this scenario I want to crop Video 1, and click the `Crop` button. Once clicked, the first frame of `Video 1` pops open. To draw the region of the video to keep, click and hold the left mouse button at the top left corner of your rectangular region and drag the mouse to the bottom right corner of the rectanglar region. If you're unhappy with your rectangle, start to draw the rectangle again by holding the left mouse button at the top left corner of your, new, revised, rectangle. The previous rectangle will be automatically discarded. When you are happy with your region, press the keyboard SPACE or ESC button to save your rectangle. Notice that the `Crop` button associated with Video 1 turns red after I've defined the cropped region.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/crop_fast.gif" />
</p>










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



#
Author [Simon N](https://github.com/sronilsson), [JJ Choong](https://github.com/inoejj)
