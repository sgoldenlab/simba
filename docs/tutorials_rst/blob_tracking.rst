Blob (contour) tracking
============

.. video:: https://github.com/user-attachments/assets/f8a18ba8-76b4-473b-b001-2331192b72c1
   :width: 800px
   :align: center
   :autoplay:
   :loop:
   :muted:

.. video:: https://github.com/sgoldenlab/simba/blob/master/images/EPM_2_V2_cropped.mp4
   :width: 800px
   :align: center
   :autoplay:
   :loop:
   :muted:

Introduction
-----------

Many of us use pose-estimation to track animal body-parts.

I have noticed that in many scenarios this is not required and simpler, faster, background subtraction methods as those used by commercial tools (like `NOLDUS <https://noldus.com/ethovision-xt>`_ or `AnyMaze <https://www.any-maze.com/>`_ or other open-source tools (like `eZTrack <https://github.com/denisecailab/ezTrack>`_) suffice.

The scenarios where background subtraction is sufficient is when we are tracking a single animal, and the body-parts we are interested are either:

1. Along the perimeters of the animal hull
2. Body parts that can be heuristically inferred by their distance from the animal hull and movement of the animal

These body-parts are typically things like snout, nose, centroid, lateral sides, tail fin etc.

In these scenarios, we really don't have to annotate body-parts and train supervised models to detect animals. Instead, we can use background subtraction and get the results much faster.

A Note on Background Subtraction Analysis Speed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The speed of tracking through background subtraction is not only determined by your available hardware (i.e., you have a GPU, how many CPU cores you have, and how much RAM memory you have available).

It is also determined by the resolution of your videos and the frame-rate of the videos. The higher the resolution, and the higher the frame-rate, the longer it will take.

.. video:: https://github.com/user-attachments/assets/b04b8f13-f0c7-489a-a3cd-a07b8a39dc68
   :width: 800px
   :align: center
   :autoplay:
   :loop:
   :muted:

For example, if your video resolutions are large (1000 x 1000), but you can visually detect the animals even at smaller resolutions (640 x 640, or even 480 x 480), then consider down sampling your videos before performing blob detection.

Also, if you have collected your videos at a relatively higher frame-rate (say 30 fps), but you really don't need to know where the animal is every 33ms of the video, then reduce the FPS of your videos to say 10 to get a location reading every 100ms.

These pre-processing steps can greatly improve the speed of animal tracking through background subtraction.

You can perform these pre-processing steps in SimBA. See the SimBA video pre-processing documentation `HERE <https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md>`_ or `HERE <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md>`_ for how to change the resolution and frame-rate of your videos.

A Note on Good Background Reference Videos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When performing animal tracking through background subtraction, for every video we analyze, we need a background reference video example. Hence, for every video we want to analyze, we want to have two video copies:

1. One which is the "background reference"
2. One which is the video we are interested in tracking the movement of the animal

.. video:: https://github.com/user-attachments/assets/3418e032-9dcc-4b81-8c56-d0daaae36c99
   :width: 800px
   :align: center
   :autoplay:
   :loop:
   :muted:


It is possible to select the same video as that being analyzed as the background reference video. However, note that this:

* May slow down processing of the video if the video is large (high resolution and/or frame rate), as the background has to be computed from a very large reference video
* May fail in situations where the animal is freezing/staying still in a single location throughout the video and reference video

Best is to use shorter snippets of the original videos as the background videos, where the animal is not present. These snippets reference videos lighting as the original videos but is ideally much shorter, and can have reduced FPS.

Again, you can perform these pre-processing steps in SimBA. See the SimBA video pre-processing documentation `HERE <https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md>`_ or `HERE <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md>`_ for how to change the resolution and frame-rate of your videos.

Starting Blob Tracking
------------------

After starting SimBA click on ``Process Videos`` in the main toolbar, and go to ``Blob tracking`` -> ``Perform blob tracking`` as in the screengrab below:

.. image:: img/blob_tracking/blob_tracking_0.webp
   :width: 800
   :align: center
   :alt: Blob tracking menu option in SimBA

Next, you should see the following pop-up menu. Next to the ``INPUT VIDEO DIRECTORY`` entry-box, click :kbd:`BROWSE` and select the directory which contains your videos. Next to the ``SAVE DATA DIRECTORY`` entry-box, click :kbd:`BROWSE` and select the directory where you want to save your tracking data.

.. image:: img/blob_tracking/blob_tracking_1.webp
   :width: 800
   :align: center
   :alt: Input and output directory selection in SimBA

.. note::
   For the ``SAVE DATA DIRECTORY``, select an empty directory.

Next, hit the :kbd:`RUN` button, and the below pop-up should show, which lists all the videos in the chosen ``INPUT VIDEO DIRECTORY``. We will go through each section of this pop-up below.

.. image:: img/blob_tracking/blob_tracking_2.webp
   :width: 800
   :align: center
   :alt: Video list and settings interface in SimBA

The Blob Tracking Menu
------------------

Quick Settings
~~~~~~~~~~~~

At the top left, you will see a frame named ``QUICK SETTINGS``.

You can use these options to set all the processing options, for all the videos listed in the table, to the same values.

To do this, select the values you want to use in the appropriate dropdown menu, and hit the :kbd:`APPLY` button next to the dropdown menu. This will update the relevant column in the ``VIDEOS`` table.

We will go through what they all mean below.

**THRESHOLD**: A value between 1 and 100 which represents how different the animal has to be from the background to be detected. Higher values will result in less likelihood to wrongly assume that parts the animal belong to the background but also increase the likelihood of wrongly assume that parts background belong to the animal.

Generally, a value between 20-70 should do the trick.

.. video:: https://github.com/user-attachments/assets/62cca260-5ab9-41ca-9cae-5369b3dc194c
   :width: 800px
   :align: center
   :autoplay:
   :loop:
   :muted:

**SMOOTHING TIME (S)**: If this dropdown is set to None, **no** temporal smooth of the animal tracking points is performed.

If set to, for example, 0.5, then SimBA performs Savitzky-Golay smoothing across the chosen, sliding, temporal window (500ms, or 0.5s in this case).

.. video:: https://github.com/user-attachments/assets/a86a0d7b-35c6-4da7-b44c-4856d71fd861
   :width: 800px
   :align: center
   :autoplay:
   :loop:
   :muted:

**BUFFER SIZE (PIXELS)**: If set to None, then animals detected key-points will be placed right along the hull perimeter.

We may want to buffer the animals shape a little, to capture a larger area as belonging to the animal. If so, set how many pixels that you wish to buffer the animals geometry with.

See further information below, or this visual example of expected results from when applying different buffer sizes:

.. video:: https://github.com/user-attachments/assets/91b8b70a-a442-454c-b957-31ba5253d042
   :width: 800px
   :align: center
   :autoplay:
   :loop:
   :muted:

**GAP FILL FILTER SIZE (%)**: Also known as *CLOSING KERNEL SIZE*. This value controls and refines the detected animal shape.

Larger values will merge separate parts of the image that are detected as the animal into a single entity. The larger the value, the further apart the separate parts of the animal is allowed to be and still be merged into a single entity.

This setting can be helpful if:

* There are smaller parts of the background/flooring/environment/arena that are of the same color as the animal
* The animal color is in part same or similar as the background
* We want to make the animal more "blob" like

Generally, a value between 0-3 should do the trick.

For further details on ``GAP FILL FILTER SIZE``, see below, or see this visual example on expected results from different closing kernel sizes:

.. video:: https://github.com/user-attachments/assets/2fcb2aae-0c27-472e-8d29-b80e7306a6ad
   :width: 800px
   :align: center
   :autoplay:
   :loop:
   :muted:

**NOISE REMOVAL FILTER SIZE (%)**: Also known as *OPENING KERNEL SIZE*. This controls and removes smaller parts of the background that has been mistakenly detected as the animal.

This can be helpful if get rid of small noise related to the movement of bedding or light across the video.

For further details on ``NOISE REMOVAL FILTER SIZE``, see below, or this visual example on expected results when removing the noise associated with bedding material using the noise removal filter:

.. video:: https://github.com/user-attachments/assets/9e498c6d-cf2c-4084-ba42-2582305421bd
   :width: 800px
   :align: center
   :autoplay:
   :loop:
   :muted:

Run-time Settings
~~~~~~~~~~~~~~

This frame contains several options regarding how we want to process the data in its entirety.

**BACKGROUND DIRECTORY**: Use this menu to batch select the paths to the background reference videos for all the videos in the table.

Note: for this to work, you need to select a ``BACKGROUND DIRECTORY`` that contains the same video filenames as the chosen directory ``INPUT VIDEO DIRECTORY``. In other words, the chosen directory needs to have a copy of each video file listed in the video table.

Choose a directory by clicking :kbd:`BROWSE` that contains background video examples of the videos. Once selected, hit the :kbd:`APPLY` button.

SimBA will automatically pair each background video with the videos in the table.

.. video:: https://github.com/user-attachments/assets/bfcd54f8-76e-4fb7-ab50-fec64ad803be
   :width: 800px
   :align: center
   :autoplay:
   :loop:
   :muted:

.. note::
   As discussed above, the larger the background videos are (higher resolution, higher FPS, longer time), the longer the detection processing will be.
   To speed up processing, it is best to have a short representative background video reference for each video to be processed. This could be videos representing the first 20-30s of the videos in the table, or copies of the videos in the table where the FPS has been much reduced.
   Again, you perform these pre-processing steps in SimBA. See the SimBA video pre-processing documentation `HERE <https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md>`_ or `HERE <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md>`_ for how to change the resolution and frame-rate of your videos.

**GPU**: Toggle "USE GPU" if available to accelerate some aspects of processing. This option is automatically disabled if an NVIDIA GPU is not available on your machine.

**CPU COUNT**: Choose the number of CPU cores you wish to use. The default is half of the cores available on your machine.

Higher values will result in faster processing but will require more RAM. If you are hitting memory related errors, you can try to decrease the ``CPU COUNT`` value.

**VERTICE COUNT**: Controls the detected animal shape precision.

For example, if set to ``30``, then SimBA will output a CSV file containing the position 30 body-part key-points along the animals outer bounds for every frame in the video. In other words, every output tracking file will contain 30 ``x`` and 30 ``y`` columns representing the positions of 30 "body-parts" in every frame of the video.

If you selected ``30`` in this dropdown, an example output CSV file will look like `THIS <https://github.com/sgoldenlab/simba/blob/master/misc/example_vertice_cnt_30.csv>`_. If you selected ``60`` in this dropdown, an example output CSV file will look like `THIS <https://github.com/sgoldenlab/simba/blob/master/misc/example_vertice_cnt_60.csv>`_.

**SAVE BACKGROUND VIDEOS**: If True, then black-and-white background subtracted videos (where the animal (foreground) is white and background is black), which SimBA uses to detect the animals location, will be saved in your chosen ``SAVE DATA DIRECTORY`` directory.

If False, these temporarily stored videos will be deleted. The default is True.

**GAP FILLING ITERATIONS**: Also known as *CLOSE KERNEL ITERATIONS*.

If ``GAP FILLING FILTER SIZE (%)`` is **not** ``None``, SimBA will apply smoothing to the detected animal geometry for a specified number of iterations (N).

Higher iteration number will result in smoother, more "blob" like animals. Default is 3, and I have had success with values between 1-10.

**NOISE REMOVAL ITERATIONS**: Also known as *OPEN KERNEL ITERATIONS*.

If ``NOISE REMOVAL FILTER SIZE (%)`` is **not** ``None``, SimBA will remove noise a specified number of iterations (N).

Higher iteration number will result in less noise, potentially at less precision for accurately detecting the animal. Default is 3, and I have had success with values between 1-10.

**DUPLICATE INCLUSION ZONES**: We can draw regions-of-interest inclusion zones on videos to heuristically restrict where SimBA can detect the animal (for more information, see below).

Here, if we have drawn ``INCLUSION ZONES`` on a single video, we can duplicate that inclusion zone to all other videos listed in the table.

To duplicate a video's inclusion zones to all videos in the project, choose the video in the dropdown and hit the :kbd:`APPLY` button.

Once duplicated on all videos, you can go ahead and modify the inclusion zones on each video.

This is typically a faster way to get inclusion zones drawn, than manually drawing ROI inclusion zones on each video from scratch, as the example in the below video.

Note: to date, I haven't found much use for ``INCLUSION ZONES``, and have been able to find the animals reliably without it.

Video Table
~~~~~~~~~

This table lists all video files found inside your defined video directory, with one row per video found inside your chosen ``INPUT VIDEO DIRECTORY``.

For each row, there is a bunch of settings, allowing you precise control for how the animals location are detected in each video.

However, I recommend using the ``QUICK SETTINGS`` and ``RUN-TIME SETTINGS`` frames above to batch set these values on all videos with the recording conditions remain stable across videos.

.. note::
   If all videos have been recorded in a standardized way, you will likely get away with using the ``QUICK SETTINGS`` frame above to bulk set the methods for all videos at once.

**BACKGROUND REFERENCE**: Select the path to a video file serving as the background reference video for the specific file.

.. note::
   Filling out the background videos individually row-by-row can be tedious, again it is much recommended to use the ``BACKGROUND DIRECTORY`` in the ``RUN-TIME SETTINGS`` frame above to set all the reference video paths at once.

**THRESHOLD**: How different the animal has to be from the background to be detected.

Higher values will result in fewer pixels being detected as the animal while more pixels being assigned to the background, while lower values will result in more pixels being assigned to the animal and fewer pixels being assigned to the background.

**INCLUSION ZONES**: If we are having movement in the videos that are **not** performed by the animal (e.g., experimenters moving around along the perimeters of the video throughout, or the light intensities changes abruptly or frequently outside the arena),
we can tell the code about which parts of the arena the animal can be detected and remove any detection outside of your defined zones.

To do this, click the :kbd:`SET INCLUSION ZONES` button for a video, and use the ROI drawing interface to specify which areas of the image the animals can be detected.

For a full tutorial for how to use this interface, see `THIS <https://github.com/sgoldenlab/simba/blob/master/docs/roi_tutorial_new_2025.md>`_ documentation. As an example, in the video below, I define a polygon called ``MAZE`` for the first video, and save it, making sure that the animal will only be detected inside the ``MAZE`` region of interest.

.. note::
   1. To save time, consider specifying the inclusion zones on only one video, and duplicating these inclusion zones on the rest of the videos using the ``DUPLICATE INCLUSION ZONES`` dropdown menu in the ``RUN-TIME SETTING`` menu above.
   2. Only use inclusion zones if you see problems with the tracking. If you have confirmed visually that there are no problems with the tracking (which for me has been overwhelmingly the case) then no inclusion zones are required. We can use ``QUICK CHECK`` (described below) to get an idea of how the tracking looks like.

**SMOOTHING TIME**: If set to None, no temporal smooth of the animal tracking points are performed. If not None, then SimBA performs Savitzky-Golay smoothing of the detected animal geometry using the selected smoothing time.

**BUFFER SIZE**: If set to None, the animals detected key-points will be placed right along the hull perimeter. We may want to "buffer" the animals shape a little, to capture a larger area as the animal. Set how many pixels that you wish to buffer the animals geometry with. For a visual example about what "buffering" means, see below video:

**GAP FILL SIZE (%)**: Controls animal shape refinement. Larger values will merge separate parts if the images detected as the foreground together into a single entity.

The larger the value, the further apart the separate parts of the foreground is allowed to be and still be merged into a single entity. See below video of an animal when increasingly larger kernel sizes are chosen:

.. note::
   In the example above, from an elevated plus maze, the borders of the open arms are black (same color as the animal) causing the tracking to fail and split the animal into two parts of the animal gets mistaken for the background as the animal performs a head dip. This is solved by smoothing (or "closing") the animals geometry.

**NOISE FILL SIZE (%)**: Controls removal of noise. Larger values will remove parts smaller parts of the detected foreground.

This can be helpful to get rid of small noise related to the movement of bedding or light across the video like the below video:

.. note::
   Only relevant if the light and or background changes slightly across videos. If there isn't any background noise, or the background noise is small enough not to be confused with the larger animal, set the NOISE FILL SIZE to ``None``.

**QUICK CHECK**: We can perform a "quick check", on each video separately, to get an idea of if the animal in each video be discriminated reliably from the background using the chosen settings for each video.

Hitting the :kbd:`QUICK CHECK` button will remove the background, and show a pop-up window with the video which you scan scroll through, frame-by-frame.

If you can reliably see the animal in white, and the background in black, you are good to go!

Execute
~~~~~~

The ``EXECUTE`` frame is duplicated and located both at the top right of the blob tracking interface, and at the bottom.

This frame contains two buttons:

**REMOVE INCLUSION ZONES**: Hitting this button will remove all inclusion zones drawn on all videos.

**RUN**: Once all has been set, and we want to run the blob tracking, click the :kbd:`RUN` button.

You can follow the progress in the main SimBA terminal and in the OS terminal used to boot up SimBA.

Expected Output
~~~~~~~~~~~~

In the selected output directory, there will be a single CSV file for each of the input video. Each of these CSV files will contain one row for every frame in the input video, and one column for each of the chosen number N of vertices (named ``vertice_0_x``, ``vertice_0_y`` ... ``vertice_N_x``, ``vertice_N_y``).

This file will also contain six columns representing the anterior, posterior, center, left and right positions of the animal geometry, each with an ``x`` and a ``y`` coordinate.

Moreover, if you had set ``SAVE BACKGROUND VIDEOS`` to True in ``RUN-TIME SETTINGS`` frame, this directory will also contain the background subtracted mp4 file for each processed videos.

Lastly, this directory will contain a ``.pickle`` file, named ``blob_definitions.pickle``, which contains the settings which you used to process each of the video files, that will look something like `THIS <https://github.com/sgoldenlab/simba/blob/master/misc/blob_definitions_example.json>`_.


Next steps
~~~~~~~~~~~~

TODO: Visualize the tracking.