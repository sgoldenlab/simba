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

The quick setting menu allows us to batch specify new resolutions, new frames rates, or clipping times for all videos. In my toy example I have 10 videos, each which is 10s long (Note: you can see that they are 10s long in the middle **VIDEOS** table, by looking in the *End Time* column, which SimBA has populated with the video meta-information data). 

Let's say I want to remove the first 5s from each of the videos, and to do this I can use the `Clip Videos Settings` sub-menu in QUICK SETTINGS. To do this, I set the `Start Time` to 00:00:05, and the `End Time` to 00:00:10 and click `Apply`, as in the gif below. Note that the `Start Time` of all videos listed in the VIDEOS table are updated:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/quick_clip.gif" />
</p>

Similarly, let say I want to downsample all my videos to a 1200x800 resolution. I then update the `Width` and `Height` values in the `Downsample Videos` sub-menu, and click `Apply`:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/quick_downsample.gif" />
</p>

As of 12/2025, the `QUICK SETTINGS` menu includes a dropdown named `USE GPU`. If this is selected as TRUE, then SimBA will run the video editing using FFMpeg *GPU* codecs instead of FFMpeg *CPU* codecs. Using the GPU potentially means a lot faster processing. For an indication of the potential time-savings when this checkbox is checked, see [THIS](https://github.com/sgoldenlab/simba/blob/master/docs/gpu_vs_cpu_video_processing_runtimes.md) table. This function requires an NVIDEA GPU, and this dropdown will be grayed out if SimBA does not detect a GPU on your computer. 

<img width="1318" height="540" alt="image" src="https://github.com/user-attachments/assets/5dfa7423-33e7-40ed-b25c-3a2f541eb135" />

As of 12/2025, SimBA includes a `OUTPUT VIDEO QUALITY` frame inside the `QUICK SETTINGS` frame, as in the screensgrab below, which allows you to control the quality of the output video. Higher values creates better quality videos with larger file-sizes. Lower values creates lower quality videos with smaller file-sizes. If you want to set different output qualities for different videos, see each row in the `VIDEOS TABLE` below. 


<img width="1428" height="595" alt="image" src="https://github.com/user-attachments/assets/c9205a3a-c0d8-4792-b16e-90eace748597" />

### VIDEOS TABLE

The middle VIDEOS table list all the video files found inside your input directory defined in Step 1, with one video per row. Each video has a `Crop` button, and several entry boxes and radio buttons that allows us to specify which pre-processing functions we should apply to each video. In the header of the VIDEOS table, there are also radio buttons that allows us to tick all of the videos in the table. For example, if I want to apply the 00:00:05 to 00:00:10 clip trimming to all videos, I go ahead and click the `Clip all videos` radio button. If I want downsample all videos, I go ahead and click the `Downsample All Videos` radiobutton. If I want to `Clip all videos` *except* few videos. I go ahead and de-select the videos I want to omit from downsampling. The same applies for the FPS, greyscale, CLAHE and Frame count radio buttons:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/header_radiobtn.gif" />
</p>

Next, it might be that you want to crop some of the videos listes in the VIDEOS table. To do this, click on the `Crop` button associated with the video. In this scenario I want to crop Video 1, and click the `Crop` button. Once clicked, the first frame of `Video 1` pops open. To draw the region of the video to keep, click and hold the left mouse button at the top left corner of your rectangular region and drag the mouse to the bottom right corner of the rectanglar region. If you're unhappy with your rectangle, start to draw the rectangle again by holding the left mouse button at the top left corner of your, new, revised, rectangle. The previous rectangle will be automatically discarded. When you are happy with your region, press the keyboard SPACE or ESC button to save your rectangle. Notice that the `Crop` button associated with Video 1 turns red after I've defined the cropped region.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/crop_fast.gif" />
</p>

The final column in the `VIDEO TABLE` is named `Quality %` and allows you to control balance of quality versus file-size of each of the videos individually (to quick set all of the videos to a particular quality, see the quick settings menu above). I recommend using a video quality setting of about 60-70%. It can feel tempting to say 100% - however a very high setting (90-100%) is typically associated with files that are **very large** with limited or no discernable quality improvements from a lower quality setting.  

>[!NOTE]
>The video file size and video quality will be determined by a combination of the selected quality **AND** if you are producing the videos with or without the GPU codecs as decribed in the `QUICK SETTINGS` [above](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md#quick-settings).

<img width="346" height="456" alt="image" src="https://github.com/user-attachments/assets/6728c1d9-2695-4eb6-8638-47cb674e4fd1" />


Chose with types of manipulation you which to perform on each video. Once done, head to the **EXECUTE** section. To learn more about each individual manipulation, check out their descriptions in the [SimBA TOOLS Guide](https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md).

### EXECUTE

The Ecexute section contains three buttons: (i) RESET ALL, (ii) RESET CROP, and (iii) EXECUTE. 

**RESET ALL**: The RESET ALL button puts all the choices back to how they were when opening the batch processing interface:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/reset_settings.gif" />
</p>

**RESET CROP**: The RESET CROP button removes all crop settings only (once clicking the `RESET CROP` button, you should see any red `Crop` button associated the videos go bck to their original color). 

**EXECUTE**: The `EXECUTE` button initates your chosen manipulations on each video according the settings in the VIDEO TABLE. The results are stored in the `Output Directory` defined during Step 1 above. Together with the output files, there is a `.json` file that is also saved in the output directory. This `.json` file contains the information on the manipulations performed on the videos in the VIDEO TABLE. For an exampple of this .json file, click [HERE](https://github.com/sgoldenlab/simba/blob/master/misc/batch_process_log.json). 

> Note: If you have a lot of videos (>100s), and are performing a lot of manipulations, then batch pre-processing videos may take some time, and it might be best to make it an over-nighter. 

If you have any questions, bug reports or feature requests, please let us know by opening a new github issue or contact us through gitter and, we will fix it together!

Author [Simon N](https://github.com/sronilsson)
