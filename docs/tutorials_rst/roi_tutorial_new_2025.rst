:Authors: - sronilsson

Regions of Interest (ROIs) in SimBA - NEW VERSION
==========================================================

The SimBA region of interest (ROI) interface allows users to draw ROIs
on videos. ROI data can be used to calculate basic descriptive
statistics based on animals movements and locations such as:

-  How much time the animals have spent in different ROIs.
-  How many times the animals have entered different ROIs.
-  The movement distances / velocity animals have in the different ROIs.
-  Calculate how many times and for how long animals have engaged in different classified behaviors in each ROI. etc...

Moreover, the ROI data can be used to build potentially valuable,
additional, features for random forest predictive classifiers. Such
features can be used to generate a machine model that classify behaviors
that depend on the spatial location of body parts in relation to the
ROIs.

.. WARNING::
   If spatial locations are irrelevant for the behaviour
   being classified, then such features should *not* be included in the
   machine model generation as they just only introduce noise.

Before analyzing ROIs in SimBA
---------------------------------------------------

To analyze ROI data in SimBA (for descriptive statistics, machine
learning features, or both descriptive statistics and machine learning
features) the tracking data **first** has to be processed the **up-to
and including the Outlier correction step described in** `Part 2 - Step
4 - Correcting
outliers <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction>`__.
Thus, before proceeding to calculate ROI based measures, you should have
one CSV file for each of the videos in your project located within the
``project_folder\csv\outlier_corrected_movement_location`` sub-directory
of your SimBA project.

Specifically, for working with ROIs in SimBA, begin by (i) `Importing
your videos to your
project <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-import-videos-into-project-folder>`__,
(ii) `Import the tracking data and relevant videos to your
project <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-import-dlc-tracking-data>`__,
(iii) `Set the video
parameters <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters>`__,
and lastly (iv) `Correct
outliers <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction>`__
(or click to indicate that you want to *Skip outlier correction* as
detailed in the Correct outliers tutorial)

.. NOTE::
   **A short explanation on what is meant by “using ROI data as
   features for random forest classifiers”** When ROIs have been drawn
   in SimBA, then we can calculate several metrics that goes beyond the
   coordinates of the different body-parts from pose-estimation. For
   example - for each frame of the video - we can calculate the distance
   between the different body-parts and the different ROIs, or if the
   animal is inside or outside the ROIs. In some instances (depending on
   which body parts are tracked), we can also calculate if the animal is
   `directing towards the
   ROIs <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data>`__
   (see below for more information). These measures can be used to
   calculate several further features such as the percentage of the
   session, up to and including the any frame, the animal has spent
   within a specific ROI. These and other ROI-based features could be
   useful additions for classifying behaviors in certain scenarios.

Part 1. Defining ROIs in SimBA
---------------------------------------------------

1. In the main SimBA console window, begin loading your project by
   clicking on ``File`` and ``Load project``. In the **[Load Project]**
   tab, click on ``Browse File`` and select the ``project_config.ini``
   that belongs to your project.

2. Navigate to the **ROI** tab, which should look like the image below
   (we will be using the menu highted by the red rectangle and red
   arrow):

.. image:: img/roi/roi_tutorial_1.webp
  :width: 800
  :align: center

.. NOTE::
   If you have ROIs on some or all the videos in your project,
   and want to delete your ROI work and start from scratch, use the
   ``Delete all ROIs`` button directly below the ``Define ROIs`` button.
   More info at `the
   end <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md#delete-all-roi-definitions-in-your-simba-project>`__
   of this tutorial.

3. Next, we click on the ``Define ROIs`` button, and the following table
   will pop open.

.. image:: img/roi/roi_tutorial_2.webp
  :width: 600
  :align: center

This table contain one row for each of the videos in the project (each
video inside the \`project_folder/videos directory in the SImBA procet).
Each video in the project has three buttons associated with it: **Draw,
Reset,** and **Apply to all**. The functions associated with each button
is described in detail below. But in brief:

-  The \ **Draw**\  button allows you to start to draw ROI shapes on the
   specific video. If drawings already exist for the specific video,
   then the \ **Draw**\  buttons opens an interface where you can move,
   re-define, or add ROI shapes onto the video.

-  The \ **Reset**\  button deletes any ROIs made on the specific video
   and allows you to restart the ROI drawings from scratch by next
   clicking on the \ **Draw**\  button.

-  The \ **Apply to all**\  buttons copies the ROI drawing made on the
   specific video and replicates them on all other videos in the
   project. If a drawing has been applied to all videos videos in the
   project by clicking on the \ **Apply to all**\  button, then the
   shapes for any specific video can be moved or re-defined (and new ROI
   shapes can be added) by clicking on the \ **Draw**\  button.

4. To begin to draw your ROI shapes, click on the \ **Draw**\  button
   for the first video in the table. Once clicked, two windows will pop
   up that look like this:

.. image:: img/roi/roi_tutorial_3.webp
  :width: 600
  :align: center


The right window (titled **Define shape**) will display the first frame
of the video. The left window (titled **REGION OF INTEREST (ROI)
SETTINGS**) will contain information, buttons and entry-boxes for
creating and manpulating your ROI shapes, and we will go through each of
their function in detail below.

.. NOTE::
   The aesthetics of the menus might look slightly different on
   your computer (this tutorial was written on a Microsoft Windows
   computer). The functions, however, are the same regardless of
   operating system.

THE **REGION OF INTEREST (ROI) SETTINGS** WINDOW
---------------------------------------------------

VIDEO AND FRAME INFORMATION
---------------------------

The first top part of the **REGION OF INTEREST (ROI) SETTINGS** menu is
titled *VIDEO AND FRAME INFORMATION* and is useful for general
troubleshooting. This menu displays the name of the current video, the
format of the current video, its frame rate, and the frame number and
the timestamp of the frame that is being displayed in the right **Define
shape** window.

CHANGE IMAGE
---------------------------------------------------

Occationally, the very first frame of the video isn’t suitable for
defining your ROIs and you’d like to use a different frame while
drawing. Alternatively, you might want to check how your ROIs look in a
different frame of the video. To manipulate the frame being displayed in
the **Define shape** window, use the buttons in the **CHANGE IMAGE**
menus (see the video below):

-  Click on ``+1s`` to display a frame one second *later* in the video
   relative to the frame currently being displayed.

-  Click on ``-1s`` to display a frame one second *earlier* in the video
   relative to the frame currently displayed.

-  If you need move a custom distance forward or backwards in the video,
   then enter the number of seconds you want to move forward or backward
   in the ``CUSTOM SECONDS`` entry box, and either click on the FORWARD
   or REVERSE buttons.

-  If you want to display the first frame of the video, click on the
   FIRST FRAME. If you want to display the last frame of the video,
   click on the LAST FRAME.

.. video:: img/roi/roi_tutorial_4.webm
   :width: 800
   :align: center
   :autoplay:
   :loop:
   :muted:

SET NEW SHAPE
-------------

In the next menu, titles **SET NEW SHAPE**, we define which shape type
we want to draw. SimBA supports three shape types - rectangles, circles,
and polygon. Select the shape type you want to draw by clicking the
appropriate button. The selected ROI shape type will be highlighted in
red font.

SHAPE ATTRIBUTES
----------------

Next, once you have selected the ``Shape type``, you can pick a few of
its attributes (or go ahead with the default values). Users drawing ROIs
in SimBA are often working in a wide variety of video and monitor
resolutions and are sometimes drawing relatively complex geometries
involving many shapes. The options in this menu can help you keep shapes
visible, distinguable and aligned while drawing. SimBA allowes the user
to set three different *shape attributes*:

-  **Shape thickness**: This dropdown menu controls the thickness of the
   lines in the ROIs (see the top of the image below). If you select a
   higher value in the ``Shape thickness`` dropdown menu, then the lines
   of your ROI will be thicker.

-  **Ear tag size**: Each shape that you draw will have *ear tags* (more
   info below). These tags can be clicked on to move shapes, align
   shapes, or manipulate the dimensions of the shapes. In this dropdown
   menu, select the size that the ear-tags of your ROI should have (see
   the bottom of the image below). If you select a higher value in the
   ``Ear tag size`` dropdown, then the ear-tags of the ROI will be
   bigger.

-  **Shape color**: Each shape that you draw will have a color. From the
   dropdown, select the color that your ROI should have.

.. NOTE::
   If you want to change these shape attributes later, after
   completing your drawing, you can - more info below):

.. image:: img/roi/roi_new_2.png
  :width: 800
  :align: center

SHAPE NAME
----------

Each shape in SimBA has to have a unique name. This name cannot be
shared with another ROI name for the same video. In the ``SHAPE NAME``
entry box, enter the name of your shape as a string (e.g.,
``bottom left corner``, or ``center`` etc..).

DRAW
----

Once you have defined your shape, it is time to draw it. The methods for
drawing the three different shape types (``Rectangle``, ``Circle`` and
``Polygon``) is slightly different from each other and detailed below.
However, regardless of the shape type you are currently drawing, begin
by clicking on DRAW button.

DRAW RECTANGLE
~~~~~~~~~~~~~~

To draw a rectangle, click and hold the left mouse button at the top
left corner of your rectangle and drag the mouse to the bottom right
corner of the rectangle. If you’re unhappy with your rectangle, you can
start to draw the rectangle again by holding the left mouse button at
the top left corner of your, new, revised, rectangle. The previous
rectangle will be automatically discarded. When you are happy with your
rectangle, **press the keyboard ``ESCAPE`` key** to save your rectangle.

.. video:: img/roi/draw_rectangle.webm
   :width: 800
   :align: center
   :autoplay:
   :loop:
   :muted:

DRAW CIRCLE
~~~~~~~~~~~

Begin by **clicking and holding down** the left mouse button at the
center of the circle. Next, drag the mouse, **while holding down the
left mouse button**, towards the outer bounds of the circle. Once the
circle looks good, **without letting go of the left left mouse button**,
hit the **keyboard ``ESCAPE`` key**. If the circle looks off while you
are drawing it: let go of the left mouse button, the circle will
dissapear, and you can start again by pressing and holding down the left
mouse button at the center of the circle.

.. video:: img/roi/draw_circle.webm
   :width: 800
   :align: center
   :autoplay:
   :loop:
   :muted:

DRAW POLYGON
~~~~~~~~~~~~

Left mouse click on **at least three different locations** in the image
that defines the outer bounds of your polygon. You should see filled
circles, representing the polygon vertices, and and lines connecting the
vertices, appear where you click. Once you are happy with your polygon,
hit the **keyboard ``ESCAPE`` key** and the polygon will appear in full.

.. video:: img/roi/draw_polygon.webm
   :width: 800
   :align: center
   :autoplay:
   :loop:
   :muted:

SHAPE MANIPULATIONS
-------------------

SimBA allows several forms of shape manipulations that are described in
detail below, this includes:

-  `Deleting
   ROIs <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md#deleting-rois>`__
   - allows you to delete all ROIs, or single user-defined ROIs.

-  `Duplicating
   ROIs <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md#duplicating-rois>`__
   - allows you to duplicate already-drawn ROIs.

-  `Change ROI
   location <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md#changing-roi-locations>`__
   - allows you to move ROIs to different locations.

-  `Change the shape of
   ROIs <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md#changing-the-shape-of-the-roi>`__
   - allows you to change the width and/or hight of a rectangle, radius
   of a circle, or the locations of the outer bounds of a polygon.

-  `Drawing shapes of fixed metric
   sizes <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md#change-roi-attributes>`__
   - allows you to change the name, color or other attributes of an
   already created ROI.

DELETING ROIs
~~~~~~~~~~~~~

-  To delete all drawn ROIs, click the DELETE ALL button in the ``DRAW``
   sub-menu:

.. video:: img/roi/delete_all.webm
   :width: 800
   :align: center
   :autoplay:
   :loop:
   :muted:

-  To delete a specific ROI, first use the ``ROI`` drop-down menu in the
   ``DRAW`` sub-menu to select the ROI you wish to delete. Next, click
   on the DELETE SELECTED ROI button:

.. video:: img/roi/delete_selected.webm
   :width: 800
   :align: center
   :autoplay:
   :loop:
   :muted:

DUPLICATE ROIs
~~~~~~~~~~~~~~

1) To duplicate an already-draw ROI, first use the ``ROI`` drop-down
   menu in the ``DRAW`` sub-menu to select the ROI you wish to
   duplicate. Next, click on the DUPLICATE SELECTED ROI button. A new
   ROI, with the same dimensions and attributes as the ROI selected in
   the ``ROI`` drop-down menu, should appear in the frame near the
   original ROI:

.. video:: img/roi/draw_rectangle.webm
   :width: 800
   :align: center
   :autoplay:
   :loop:
   :muted:


2. The duplicated ROI will inherit the name of the original ROI with the
   ``_duplicated`` suffix appended to the name. Thus, if you look in the
   ``ROI`` dropdown menu, you should see a new ROI with a name in this
   format: ``MY_ROI_NAME_duplicated``. To change this name, and/or any
   other attribute belonging to this ROI, see the `Change ROI
   attributes <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md#change-roi-attributes>`__
   section below.

.. image:: img/roi/roi_tutorial_5.webp
  :width: 800
  :align: center

CHANGING ROI LOCATIONS
~~~~~~~~~~~~~~~~~~~~~~

1. To change the location of an ROI, begin by clicking the MOVE SHAPE
   button in the ``SHAPE INTERACTION`` sub-menu to enter into **MOVE
   MODE**. Once clicked, the “EAR TAGS” of each shape will be displayed
   in the drawing window. Rectangles will have 9 ear tags, circles have
   2 ear-tags, and polygons have as many ear-tags as there are
   user-defined outer bounds (plus a **center** ear tag).

2. Next, **click and hold the left mouse button** on the **center** ear
   tag of the ROI that you wish to move. You should see the entire ROI
   shape changing its color to **grey**, marking that the entire has
   been selected (if this grey color is unsutable, not to worry, I will
   show you how to change it later on).

3. Next, **while holding the mouse left button**, drag the mouse cursor
   to the new location the where you want your ROI to be located. Once
   in location, **let go of the left mouse button**. You should see your
   ROI, in its defined color, being displayed in the new location. If
   you want to change a second ROI location, go ahead and drag the
   **center** tag for that ROI.

4. Once the ROIs have been moved into their correct location. We need to
   exit the **MOVE MODE**. Select the ``DEFINE SHAPE`` window (e.g., by
   clicking the top bar), and hit the **keyboard ``ESCAPE`` key**.

.. video:: img/roi/move_shapes.webm
   :width: 800
   :align: center
   :autoplay:
   :loop:
   :muted:

CHANGING ROI SHAPES
~~~~~~~~~~~~~~~~~~~

1. To change the shape of an ROI, begin by clicking the MOVE SHAPE
   button in the ``SHAPE INTERACTION`` sub-menu to enter into **MOVE
   MODE**. Once clicked, the “EAR TAGS” of each shape will be displayed
   in the drawing window. Rectangles will have 9 ear tags, circles have
   2 ear-tags, and polygons have as many ear-tags as there are
   user-defined outer bounds (plus a **center** ear tag).

2. Next, **click and hold the left mouse button** on the edge ear tag
   which you want to manipulate. Once you \*\ **click and hold the left
   mouse button** on the ear-tag, the part of the ROI shape you are
   manipulating changes its color to grey (if this grey color is
   unsutable, not to worry, I will show you how to change it later on).
   Different ROI ear-tags will help you to manipulate different parts of
   the ROI. For example:

   **Rectangles**:

   -  Clicking and holding the left middle ear-tag of a rectangle allows
      you to control the location of the left border or the rectangle.

   -  Clicking and holding the top-left corner ear-tag of a rectangle
      allows you to control the top border and left border of the
      rectangle.

   -  Clicking and holding the bottom middle ear-tag of a rectangle
      allows you to control the bottom border or the rectangle.

   **Circles**:

   -  Clicking and holding the left border ear-tag of a circle allows
      you to control the radius of the circle.

   **Circles**:

   -  Clicking and holding the outer bounds of the polygon allows you to
      control the location of the outer bound in the polygon (i.e.,
      control the two lines connected to the clicked-on ear-tag).

3. Next, after selecting an ear-tag, **while holding the mouse left
   button**, drag the mouse cursor to the new location the where you
   want the edge you are manipulating to be located. Once the edge is in
   the location you want, **let go of the left mouse button**, and the
   new ROI shape should be displayed.

4. Finally, once the ROIs have been moved into their correct location.
   We need to exit the **MOVE MODE**. Select the ``DEFINE SHAPE`` window
   (e.g., by clicking the top bar), and hit the **keyboard ``ESCAPE``
   key**.

.. video:: img/roi/change_shapes.webm
   :width: 800
   :align: center
   :autoplay:
   :loop:
   :muted:

CHANGE ROI ATTRIBUTES
~~~~~~~~~~~~~~~~~~~~~

1. To change the attributes of an already-created ROI, click on the
   CHANGE ROI button and a pop-up menu will appear. This pop-up menu has
   a drop-down menu, titled ``CHANGE ROI``, which allows you to select
   the ROI that you want to change.

.. image:: img/roi/change_roi_popup.webp
  :width: 800
  :align: center

2. To change the name of the ROI, enter a new name in the
   ``NEW SHAPE NAME`` entry box. TO change the color, thickness, or
   ear-tag size of the shape, use the respective dropdown windows. Once
   complete, hit the SAVE ATTRIBUTES button.

.. video:: img/roi/change_attributes.webm
   :width: 800
   :align: center
   :autoplay:
   :loop:
   :muted:


SHOW SHAPE SIZE INFORMATION.
----------------------------

Sometimes we need some metrics representing the sizes of the ROIs we
have drawn. We can display this by clicking on the SHOW SHAPE INFO
button located to the right in the ``SHAPE INTERACTION`` submenu.

1. By clicking the ``Show shape info.`` button, some numbers are
   displayed inside our ROIs:

(i). One number is displayed inside each of your drawn **rectangles,
representing the area cm2 of the rectangle** (based on your pixel per
millimeter conversion factor in your
``project_folder/logs/video_info.csv``)

(ii)  One number is displayed inside each of your drawn **circles,
      representing the cm radius your circle.** (based on your pixel per
      millimeter conversion factor in your
      ``project_folder/logs/video_info.csv``)

(iii) One number is displayed inside each of your drawn **polygons,
      representing the area cm2 of your polygon.** (based on your pixel
      per millimeter conversion factor in your
      ``project_folder/logs/video_info.csv``)

2. When the SHOW SHAPE INFO button is clicked, the text of the button
   toggles to HIDE SHAPE INFO. Click the button again to hide the shape
   size information.

.. video:: img/roi/show_shape_size.webm
   :width: 800
   :align: center
   :autoplay:
   :loop:
   :muted:

DRAW SHAPES OF USER-DEFINED METRIC SIZES
----------------------------------------

To draw shapes of specific metric sizes (i.e., specified size in
millimeters), first open the ``File (ROI)`` drop-down menu at the head
of the **REGION OF INTEREST (ROI) SETTINGS** menu and click the
``Draw ROIs of pre-defined sizes`` option:

.. image:: img/roi/show_shape_size.webm
  :width: 800
  :align: center

Once clicked, you should see the following pop-up opening allowing you
to draw a range of different shapes with specified sizes. If you need a
shape-type that is missing, consider letting us know through
`Gitter <https://app.gitter.im/#/room/#SimBA-Resource_community:gitter.im>`__
or by opening en issue on
`GithHub <https://github.com/sgoldenlab/simba/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=>`__.

.. figure::
   https://github.com/user-attachments/assets/c76037f3-f15c-430f-b552-fd945c15862b
   :alt: predefined_sizes_2

   predefined_sizes_2

In the ``SETTINGS`` frame, fill in the name of your ROI, and choose its
color, thickness, and ear tag size.

DRAW A RECTANGLE OF SET SIZE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To draw a RECTANGLE of set size, fill in its metric width and height in
millimeter and click ``ADD RECTANGLE`` as in the video below:

.. image:: img/roi/predefined_sizes_rectangle.webm
  :width: 800
  :align: center

DRAW A CIRCLE OF A SPECIFIC RADIUS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To draw a CIRCLE with a specific radius, fill in its metric radius in
millimeter and click ``ADD CIRCLE`` as in the video below:

.. image:: img/roi/predefined_sizes_circle.webm
  :width: 800
  :align: center

DRAW A HEXAGON SPECIFIC RADIUS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To draw a HEXAGON with a specific radius, fill in its metric radius in
millimeter and click ``ADD HEXAGON`` as in the video below:

.. image:: img/roi/predefined_sizes_hexagon.webm
  :width: 800
  :align: center

DRAW A HALF CIRCLE POINTING NORTH
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To draw a HALF CIRCLE with a specific radius with its chunky part
pointing UP, fill in its metric radius in millimeter, choose ``NORTH``
in the ``direction drop-down menu``, and click ``ADD HALF CIRCLE`` as in
the video below:

.. image:: img/roi/predefined_sizes_half_circle_west.webm
  :width: 800
  :align: center

DRAW A HALF CIRCLE POINTING WEST
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To draw a HALF CIRCLE with a specific radius with its chunky part
pointing LEFT, fill in its metric radius in millimeter, choose ``WEST``
in the ``direction drop-down menu``, and click ``ADD HALF CIRCLE`` as in
the video below:

.. image:: img/roi/predefined_sizes_half_circle_north.webm
  :width: 800
  :align: center

PREFERENCES
----------

The ``File`` drop-down in the **REGION OF INTEREST (ROI) SETTINGS**
header also contains a *Preferences* option. Clicking this, will
bring-up a pop-up window that allows you to control some aspects of how
the SimBA ROI interface behaves.

.. image:: img/roi/preferences_toolbar.webp
  :width: 800
  :align: center

1) **ROI SELECT COLOR**: This controls the color of teh selected edges
   while in **MOVE MODE** as in the video below.

2) **DUPLICATION JUMP SIZE**: When ROIs are duplicated, each edge of the
   duplicated copy are placed, by defaults +20 by +20 pixels from the
   original ROI. Use this dropdown to select an alternative
   “*duplication-jump-size*”.

3) **LINE TYPE**: Most often you wouldn’t need to worry about this. It
   controls how lines are visualized on the frame through the
   `OpenCV <https://hawk-tech-blog.com/python-opencv-line-type/>`__
   library. In most cases its safe to keep this at ``-1``.

.. image:: img/roi/change_highlight_clr.webm
  :width: 800
  :align: center


APPLY SHAPES FROM ANOTHER VIDEO
-------------------------------

Sometimes we have created ROIs in one video, saved them, and opened up a
second video to start drawing new ROIs on this second video. Now we may
want to replicate the ROIs on the first video on the second video, and
we can do this with the ``Apply shapes from another video`` sub-menu.

1. To duplicate the ROI shapes already defined in a different video on
   the current video, navigate to the ``Select video`` dropdown menu in
   the ``Apply shapes from another video`` menu. This dropdown menu will
   show the videos in your SimBA project that has defined ROIs.

2. In this dropdown menu, select the video which has the ROIs you wish
   to replicate. Once selected, click ``Apply``. The ROIs from the video
   in the ``Select video`` dropdown menu will appear on the frame.

.. image:: img/roi/apply_video_from_different.webm
  :width: 800
  :align: center

Delete all ROI definitions in your SimBA project
------------------------------------------------

Somtimes we may want to delete all ROI definitions in a SimBA project
and start from scratch.

1. To delete all the ROI definitions in the SimBA projects, click on the
   ``Delete all ROI definitions`` under the [ROI] tab in the load
   project menu, and the following menu should pop open:

.. image:: img/roi/ROI_delete.png
  :width: 800
  :align: center

2. Click ``Yes`` to delete all ROI definitions in a SimBA project.

.. NOTE::
   Your ROIs are saved inside your SImBA project, at the
   location ``project_folder/logs/measures/ROIdefinitions.h5``. To
   delete the ROIs in the SImBA project, you could also manually delete
   this file.

SAVE ROIs
------------------------------------------------

Once all the ROI drawings on the video looks good. remember to hit the `SAVE` button at the bottom of the **REGION OF INTEREST (ROI) SETTINGS** window. After clicking the save button, you are good to close the ROI **REGION OF INTEREST (ROI) SETTINGS** window and **DEFINE SHAPES** window.


STANDARDIZE ROI SIZE ACROSS VIDEOS
------------------------------------------------

There may be situations where you have manually drawn ROIs on a bunch of videos where the camera location has shifted slightly across recordings.
You may also want to ROIs to have the same metric sizes across recordings, but due to this shift in camera locations, some videos may have different
pixel per millimeter conversion factors.

In these situations, you may want to "standardize" the ROI metric sizes relative to some baseline measurement. For example, say that this baseline video has a pixel per millimeter of `10`.
Say there are a further two videos in the project with ROIs, and these videos has pixels per millimeter of ``9`` and ``11``.
At runtime, the area of the rectangles, circles and polygons in the two additional videos get their ROI areas increased/decreased with 10% while the baseline video ROIs are unchanged.
To do this, click the ``File`` heading in the ``PROJECT VIDEOS: ROI TABLE`` window. This shows a pop-up with a single dropdown window, as in the screengrabs
below:


.. image:: img/roi/roi_tutorial_metric.webp
  :width: 800
  :align: center


Select the video that you want to act as the "baseline" reference video which the ROI sizes should be corrected against. Once selected, click the
RUN button. All ROIs in the project will be standardized using the ROIs in the baseline video as reference.

NEXT STEPS
----------

Once your ROI definitions are all defined, close the ``ROI table``,
``Regions of Interest Settings`` and ``Define Shape`` windows and head
back to the [ROI] tab in the load project menu.

-  SimBA saves your drawn ROI definitions (the ROI locations, colors,
   shapes, sizes with their associated video file names etc) in a single
   compressed ``.H5`` file in that you can find at
   ``project_folder/logs/measures/ROI_definitions.h5``. If you want to
   extract this H5 information, to a human-readable CSV format, use
   `THIS <https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#extract-roi-definitions-to-human-readable-format>`__
   tool.

-  If you want to analyze descriptive statistics of movements in
   relation to your defined ROIs, use the ``Analyze ROI data`` button as
   detailed in ``Step 2`` in `THIS
   TUTORIAL <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-2-analyzing-roi-data>`__.

-  If you want to create machine learning features using your ROI
   definitions, use the ``Append ROI data to features`` in the
   ``Extract features`` tab as detailed in `THIS
   TUTORIAL <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data>`__

If you have any questions, bug reports or feature requests, please let
us know by opening a new `github
issue <https://github.com/sgoldenlab/simba/issues>`__ or contact us
through `gitter <https://gitter.im/SimBA-Resource/community>`__!

Author `Simon N <https://github.com/sronilsson>`__
