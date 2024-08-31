User-defined pose-configurations in SimBA
=========================================

SimBA comes pre-packaged with the ability to import tracking data based
on `8 different body-part pose-estimation tracking
combinations <https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#pose-estimation-body-part-labelling>`__.
Where possible, we strongly recommend using SimBA and pose-estimation
packages (like DeepLabCut or DeepPoseKit) with a `16-body-part, 2
animal, pose-configuration
setting <https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#pose-estimation-body-part-labelling>`__.

However, SimBA can handle *any* combination of tracked body-parts, as
long as users create their own pose configuration setting in SimBA. In
this tutorial we describe how users define their own body part
configurations in SimBA.

.. important::
   When users create projects with user-defined body part
   configurations, SimBA calculates fewer and less precise features for the
   machine models and this may negatively affect the prediction
   performance. Thus - *if users have the option* - we strongly encourage
   that the `16-body-part, 2 animal, pose-configuration
   setting <https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md#pose-estimation-body-part-labelling>`__
   is used. Click
   `HERE <https://github.com/sgoldenlab/simba/blob/master/misc/features_user_defined_pose_config.csv>`__
   for a rough list of the features that SimBA calculates based on
   user-defined pose configurations. For comparison, click
   `HERE <https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv>`__
   for the list of features that SimBA calculates based on the
   16-body-part, 2 animal, pose-configuration setting.

Create a new user-defined pose-configuration
------------------------------------------------------

1. In the main SimBA console window, click on ``File``, and
   ``Create a new project``. The window that pops open is descibed
   in-depth the `Scenario 1 - Create
   Project <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-1-generate-project-config>`__
   tutorial. If you are creating a new user-defined pose-configuration
   setting, **do not** begin by specifying a project path, project name,
   or any SML settings. Instead, skip these menus, leave these entry
   boxes blank, and click on the button in the ``Animal Settings`` menu
   next to the text ``# config``. You should see the following default
   menu pop open:

.. image:: img/create_user_defined_pose_configuration/menu_1.png
  :width: 800
  :align: center

Click on ``Create pose config`` to begin to define a new user-defined
pose configuration.

2. After clicking on ``Create pose config``, the following menu pops
   open:

.. image:: img/create_user_defined_pose_configuration/menu_2.png
  :width: 800
  :align: center

In the *first* entry box, give your pose-configuration a name. In this
tutorial we will call our user-defined pose configuration
*BtWGaNP_pose*. Please avoid spaces in the pose config name. In the
*second* entry box, enter the number of animals you wish to track. This
could be 1 or 2 or mote. In the *third* entry box, enter the number of
body-parts your tracking data will contain. For example, if the dataset
contains 8 tracked body-parts on two different animals, I will enter the
integer **16**.

Next, select an image that is representative of your tracking
environment and contains a clear view of all the tracked bodyparts by
clicking on ``Browse File``. This image will be used to create a
reference image of your pose configuration settings. After you have
selected an image, click on ``Confirm``.

3. After clicking confirm a table should pop open. The table contains
   one row for each body-part specified in the ``# of Bodyparts`` entry
   box. If the user has entered ``1`` in the ``# of Animals`` entry box,
   this table will contain a single columns (as in the left image
   below). If the user has entered a number greater than 1 in the
   ``# of Animals`` entry box, then this table will contain two columns
   (as in the right image below).

Please name the bodyparts in the left-most column (titled *Bodyparts’
name*) by filling in their names. **Do not use spaces in the bodypart
names**.

If you have two or more animals, you will need to fill in the second
column (titled *Animal ID number*) with an integer which tells SimBA
which animal each of the individual body-part belong too. In this
example image above, we are creating a body-part pose-configuration with
7 bodyparts for 2 different animals, and I have entered ``1`` and ``2``
in the *Animal ID number* column to specify which animal the different
body-parts belong too. When done, click on ``Save Pose Configs`` to
proceed.

3. After clicking on ``Save Pose Configs``, a window named
   ``Define pose`` pops open that shows the frame that was chosen in the
   ``Image Path`` entry box. This image also displays some instructions
   at the top - like the image here below to the left. Follow the
   instructions at the top of the window to label all of your defined
   body parts, like the gif below:

.. image:: img/create_user_defined_pose_configuration/interface_1.gif
  :width: 1000
  :align: center

.. note::
   DOUBLE LEFT MOUSE CLICK TO PLACE BODY PART. PRESS ESC TO ADVANCE
   TO THE NEXT BODY PART

The window will close once all the body-parts have been marked. Double
left mouse-click to assign a body-part location. Press ``Esc`` to move
to the next body-part after a body-part as been assigned.

4. Your newly created body part configuration should now be accessable
   though the ``Animal Settings`` menu next to the text ``# config``:

.. image:: img/create_user_defined_pose_configuration/menu_3.png
  :width: 800
  :align: center

.. note::
   When a project has been created in SimBA, the
   pose-configuration selected when creating the project is stored
   within a CSV file within the project folder, and this file is
   referenced to at various stages of the workflow. You can check this
   file out by navigating to the
   ``project_folder\logs\measures\pose_configs\bp_names\project_bp_names.csv``
   file.

Removing / archiving user-defined pose-configurations in SimBA
------------------------------------------------------

The user may want to remove user-defined body-part configurations from
the ``# config`` list. To do this, locate the
``Reset user-defined pose configs`` button in the
``Project configuration`` tab:

.. image:: img/create_user_defined_pose_configuration/menu_4.png
  :width: 800
  :align: center

1. When clicked on, a warning message will pop open, asking if you are
   sure if you want to reset your pose-configurations. Click on ``Yes``
   to proceed.

.. image:: img/create_user_defined_pose_configuration/warning.png
  :width: 800
  :align: center

2. The next time you open the ``Project configuration`` tab, your
   user-defined pose-configurations will no longer be visable in the
   ``# config`` list.

.. note::
   The user defined configs are never deleted even though they
   are removed from the ``# config`` list. The user-defined
   pose-configurations are archived in the they can be retrieved. Once
   removed , they are stored in the
   ``SimBA\pose_configurations_archive`` folder.

Optional: Creating DeepLabCut projects using user-defined pose-configurations in SimBA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a project with a specific pose-configuration has been created in
SimBA, you may also want to use this pose-configuration and body part
naming convention in your DeepLabCut projects created through SimBA.

1. You newly created pose-configuration setting in a new DeepLabCut
   project through SimBA, click the following box next to the text
   ``Bp config file`` in the ``Create DLC model`` menu. For more
   information on how to use DeepLabCut in SimBA, click
   `HERE <https://github.com/sgoldenlab/simba/blob/simba_JJ_branch/docs/Tutorial_DLC.md>`__.

.. image:: img/create_user_defined_pose_configuration/menu_5.png
  :width: 800
  :align: center

Once clicked on, first navigate to your SimBA project. Your SimBA
body-part configuration is saved in a csv file within your SimBA
project. Go ahead and navigate to
``project_folder/logs/measures/pose_configs/bp_names\project_bp_names.csv``
and select this file.

When you click on ``Create project`` in the DeepLabCut
``Create Project`` menu, your DeepLabCut project and DeepLabCut project
*yaml* file will now be based on your SimBA-configured body-parts.

Author `Simon N <https://github.com/sronilsson>`__
