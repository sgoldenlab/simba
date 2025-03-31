############
Blob tracking
############

.. raw:: html

    <video width="100%" controls>
        <source src="https://github.com/user-attachments/assets/f8a18ba8-76b4-473b-b001-2331192b72c1" type="video/webm">
    </video>

INTRODUCTION
===========

Many of us use pose-estimation to track animal body-parts. However, I have noticed that in many scenarios this is not required and simpler, faster, background subtraction methods as those used by commercial tools (like `NOLDUS <https://noldus.com/ethovision-xt>`_ or `AnyMaze <https://www.any-maze.com/>`_ or other open-source tools (like `eZTrack <https://github.com/denisecailab/ezTrack>`_) will suffice. These scenarios where background subtraction is sufficient is when we are tracking a single animal, and the body-parts we are interested in are along the perimeters of the animal or body parts that can be heuristically inferred by the perimeter and movement of the animal (e.g., snout, nose, centroid, lateral sides). We don't have to train supervised models to detect these body-parts, meaning no deep learning, no annotations, and the tracking can be done in minutes for many body-parts.

.. important::

   Analysis Speed
   -------------

   The speed of tracking is not only determined by your available hardware (i.e., you have a GPU, how many CPU cores you have, and how much RAM memory you have available). It is also determined by the resolution of your videos and the frame-rate of the videos.

   For example, if your video resolutions are large (1000 x 1000), but you can visually detect the animals even at smaller resolutions (640 x 640, or even 480 x 480), then consider downsampling your videos before performing blob detection.

   Moreover, if you have collected your videos at a higher frame-rate (say 30 fps), but you really don't need to know where the animal is every 33ms of the video, then reduce the FPS of your videos to say 10 to get a location reading every 100ms.

   These preprocessing steps can greatly improve the speed processing. You can perform these pre-processing steps in SimBA. See the SimBA video pre-processing documentation `here <https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md>`_ or `here <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md>`_ for how to change the resolution and frame-rate of your videos.

   What Are Good Background Videos
   -----------------------------

   It is possible to select the same video as those being analyzed as the background reference. However, this may:

   1. Slow down processing
   2. Fail in situations where the animal is freezing/staying still in a single location throughout the video which will cause the code to mistake the animal as background

   Best is to use shorter snippets of the original videos, where the light levels are the same as during the actual video, as the background video. These snippets should have the same resolution as the original video, but is ideally much shorter, and can have reduced FPS.

Quick Settings
============

You can use these options to set all the videos to the same values. To do this, select the values you want to use in the dropdown menu, and hit the appropriate :kbd:`APPLY` button. This will update the relevant column in the ``VIDEOS`` table.

**THRESHOLD**
    How different the animal has to be from the background to be detected. Higher values will result in less likelihood to wrongly assume that parts the animal belong to the background, and increase the likelihood of wrongly assume that parts background belong to the animal.

.. raw:: html

    <video width="100%" controls>
        <source src="https://github.com/user-attachments/assets/b04b8f13-f0c7-489a-a3cd-a07b8a39dc68" type="video/webm">
    </video>

**SMOOTHING TIME (S)**
    If set to None, no temporal smooth of the animal tracking points are performed. If not None, then SimBA performs Savitzky-Golay smoothing across the chosen sliding temporal window.

.. raw:: html

    <video width="100%" controls>
        <source src="https://github.com/user-attachments/assets/3418e032-9dcc-4b81-8c56-d0daaae36c99" type="video/webm">
    </video>

**BUFFER SIZE (PIXELS)**
    If set to None, the animals detected key-points will be placed right along the hull perimeter. We may want to "buffer" the animals shape a little, to capture a larger area as the animal. Set how many pixels that you wish to buffer the animals geometry with.

.. raw:: html

    <video width="100%" controls>
        <source src="https://github.com/user-attachments/assets/62cca260-5ab9-41ca-9cae-5369b3dc194c" type="video/webm">
    </video>

**CLOSING KERNEL SIZE (PX)**
    Controls animal shape refinement. Larger values will merge separate parts if the images detected as the foreground together into a single entity. The larger the value, the further apart the separate parts of the foreground is allowed to be and still be merged into a single entity.

.. raw:: html

    <video width="100%" controls>
        <source src="https://github.com/user-attachments/assets/a86a0d7b-35c6-4da7-b44c-4856d71fd861" type="video/webm">
    </video>

**OPENING KERNEL SIZE (PX)**
    Controls removal of noise. Larger values will remove parts smaller parts of the detected foreground. This can be helpful to get rid of small noise related to the movement of bedding or light across the video.

.. raw:: html

    <video width="100%" controls>
        <source src="https://github.com/user-attachments/assets/91b8b70a-a442-454c-b957-31ba5253d042" type="video/webm">
    </video>

Run-time Settings
===============

**BACKGROUND DIRECTORY**
    Choose a directory that contains background video examples of the videos. Once selected, hit the :kbd:`APPLY` button. SimBA will automatically pair each background video with the videos in the table. Before clicking :kbd:`APPLY`, make sure the chosen directory contains videos with the same file names as the videos in the table.

.. raw:: html

    <video width="100%" controls>
        <source src="https://github.com/user-attachments/assets/2fcb2aae-0c27-472e-8d29-b80e7306a6ad" type="video/webm">
    </video>

.. note::
   The larger the background videos are (higher resolution, higher FPS, longer time), the longer the processing will be. To speed up processing, it is best to have a short representative background video reference for each video to be processed. E.g., it could be videos representing the first 20-30s of the videos in the table, or copies of the videos in the table where the FPS has been much reduced.

**GPU**
    Toggle "USE GPU" if available to accelerate some aspects of processing. This option is automatically disabled if an NVIDIA GPU is not available on your machine.

**CPU COUNT**
    Choose the number of CPU cores you wish to use. Default is the maximum available on your machine. Higher values will result in faster processing but will require more RAM. If you are hitting memory related errors, you can try to decrease the ``CPU COUNT`` value.

**VERTICE COUNT**
    Controls animal shape precision. For example, if set to ``30``, then SimBA will output a CSV file containing the position 30 body-part keypoints along the animals outer bounds for every frame in the video.

**SAVE BACKGROUND VIDEOS**
    If True, then black-and-white background subtracted videos (where the foreground is white and background is black), which SimBA uses to detect the animals location, will be saved in your chosen output directory. If False, these temporarily stored videos are removed. Default is True.

**CLOSE ITERATIONS**
    If ``CLOSING KERNEL SIZE (PX)`` isn't ``None``, SimBA will smooth out the animals geometry N times. Higher iteration number will result in smoother, more "blob" like animals. Default: 3.

**DUPLICATE INCLUSION ZONES**
    If we have drawn ``INCLUSION ZONES`` on a video, we can duplicate those inclusion zones to all other videos in the project. To duplicate a videos inclusion zones to all videos in the project, choose the video in the dropdown and hit the :kbd:`APPLY` button.

.. raw:: html

    <video width="100%" controls>
        <source src="https://github.com/user-attachments/assets/bfcd54f8-c610-4ad1-970c-1176873950b5" type="video/webm">
    </video>

Video Table
==========

This table lists all video files found inside your defined video directory, with one row per video. For each row, there are various settings, allowing you control over the precise methods for how the animals location are detected in each video.

.. note::
   If all videos have been recorded in a standardized way, you will likely get away with using the ``QUICK SETTINGS`` frame above to bulk set the methods for all videos at once.

**BACKGROUND REFERENCE**
    Select the path to a video file serving as the background reference video for the specific file.

.. note::
   Filling out the background videos individually row-by-row can be tedious. It is recommended to use the ``BACKGROUND DIRECTORY`` in the ``RUN-TIME SETTINGS`` frame above to set all of the reference video paths at once.

**THRESHOLD**
    How different the animal has to be from the background to be detected. Higher values will result in fewer pixels being detected as the animal while more pixels being assigned to the background.

**INCLUSION ZONES**
    If we are having movement in the videos that are **not** performed by the animal (e.g., experimenters moving around along the perimeters of the video, light intensities changes outside of the arena), we can tell the code about which parts of the arena the animal can be detected.

.. raw:: html

    <video width="100%" controls>
        <source src="https://github.com/user-attachments/assets/9e498c6d-cf2c-4084-ba42-2582305421bd" type="video/webm">
    </video>

.. note::
   1. To save time, consider specifying the inclusion zones on only one video, and duplicating these inclusion zones on the rest of the videos using the ``DUPLICATE INCLUSION ZONES`` dropdown menu in the ``RUN-TIME SETTING`` menu above.
   2. Only use inclusion zones if you see problems with the tracking. If you have confirmed visually that there are no problems with the tracking (which for me has been overwhelmingly the case) then no inclusion zones are required.

**SMOOTHING TIME**
    If set to None, no temporal smooth of the animal tracking points are performed. If not None, then SimBA performs Savitzky-Golay smoothing of the detected animal geometry using the selected smoothing time.

**BUFFER SIZE**
    If set to None, the animals detected key-points will be placed right along the hull perimeter. We may want to "buffer" the animals shape a little, to capture a larger area as the animal. Set how many pixels that you wish to buffer the animals geometry with.

**CLOSE KERNEL SIZE**
    Controls animal shape refinement. Larger values will merge separate parts if the images detected as the foreground together into a single entity. The larger the value, the further apart the separate parts of the foreground is allowed to be and still be merged into a single entity.

.. note::
   In elevated plus maze scenarios, the borders of the open arms are black (same color as the animal) causing the tracking to fail and split the animal into two parts as the animal gets mistaken for the background during a head dip. This is solved by smoothing (or "closing") the animals geometry.

**OPEN KERNEL SIZE**
    Controls removal of noise. Larger values will remove parts smaller parts of the detected foreground. This can be helpful if get rid of small noise related to the movement of bedding or light across the video.

.. note::
   Only relevant if the light and/or background changes slightly across videos. If there isn't any background noise, or the background noise is small enough not to be confused with the larger animal, set the OPEN KERNEL SIZE to ``None``.

**QUICK CHECK**
    We can do a "quick check" if the animal can be discerned from the background reliably, using the chosen settings for each video. Hitting the :kbd:`QUICK CHECK` button will remove the background and display the video frame by frame. If you can reliably see the animal in white, and the background in black, you are good to go!

.. raw:: html

    <video width="100%" controls>
        <source src="https://github.com/user-attachments/assets/b83dfef6-746e-4fb7-ab50-fec64ad803be" type="video/webm">
    </video>

Execute
=======

The ``EXECUTE`` frame contains two buttons:

**REMOVE INCLUSION ZONES**
    Hitting this button will remove all inclusion zones drawn on all videos.

**RUN**
    Once all has been set, click the :kbd:`RUN` button. You can follow the progress in the main SimBA terminal.

Expected Output
=============

In the selected output directory, there will be:

1. A single CSV file for each input video containing:

   - One row for every frame in the input video
   - Columns for each chosen vertex (``vertice_0_x``, ``vertice_0_y`` ... ``vertice_N_x``, ``vertice_N_y``)
   - Six columns representing positions (``center_x``, ``center_y``, ``nose_x``, ``nose_y``, ``tail_x``, ``tail_y``, ``left_x``, ``left_y``, ``right_x``, ``right_y``)

2. Background subtracted MP4 files (if ``SAVE BACKGROUND VIDEOS`` is True)

3. A ``blob_definitions.pickle`` file containing the settings used to process each video file