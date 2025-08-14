Install SimBA
==================

Requirements
------------

- `Python 3.6 <https://www.python.org/downloads/release/python-360/>`__

.. note::
  SimBA is validated using python 3.6, and the developers maintain SimBA mainly using 3.6. However, you can use python 3.10 if necessery.
  If you encounter bugs using 3.10, please each out to us by opening an `issue <https://github.com/sgoldenlab/simba/issues>`__ and sending us
  a message on `Gitter <https://app.gitter.im/#/room/#SimBA-Resource_community>`__.
- `FFmpeg <https://www.ffmpeg.org/>`__
.. note::
  See e.g., installation instructions for `Windows <https://m.wikihow.com/Install-FFmpeg-on-Windows>`__, `MacOS/Linux <https://www.ffmpeg.org/download.html>`__. FFMpeg is used
  in SimBA for video pre-processing and video editing, and visualization tools. FFMpeg is not a strict requirement, but is strongly recommended.

.. hint::

   - See this video for a visual walkthrough of installation using `conda <install_conda_video.html>`_.
   - See this video for a visual walkthrough of installation using `Anaconda Navigator Installation Video <install_anaconda_navigator_video.html>`_.
   - See this video for a visual walkthrough of installation using `python venv <install_venv_video.html>`_.

Install options:
----------------

- `Install SimBA in main Python
  environment <https://github.com/sgoldenlab/simba/blob/master/docs/installation_new.md#option-2-install-simba-in-main-python>`__
- `Install SimBA using
  conda <https://github.com/sgoldenlab/simba/blob/master/docs/installation_new.md#option-1-install-simba-using-conda-recommended>`__
- `Install SimBA using Anaconda
  Navigator <https://github.com/sgoldenlab/simba/blob/master/docs/anaconda_2025.md>`__
- `Install SimBA using Python
  venv <https://github.com/sgoldenlab/simba/blob/master/docs/installation_new.md#option-3-install-simba-using-python-venv>`__

Option 1: Install SimBA using conda (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Click
`HERE <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`__
for instructions for how to install conda.

1. Once conda is installed, create a new python3.6 environment using:

::

   conda create -n my_simba_env python=3.6 anaconda -y

…. or using python3.10, if really needed:

::

   conda create -n my_simba_env python=3.10 anaconda -y

2. Enter the conda ``my_simba_env`` environment created in Step 1 by
   typing:

::

   conda activate my_simba_env

3. Install SimBA in the ``my_simba_env`` environment by typing:

::

   pip install simba-uw-tf-dev

or, if you are in **python 3.10**, and hitting errors, try:

::

   pip install simba-uw-tf-dev --ignore-installed

4) Now launch SimBA by opening a command prompt and typing:

::

   simba

.. and hit the Enter key.

   [!NOTE] SimBA may take a little time to launch depending in your
   computer, but you should eventually see
   `THIS <https://github.com/sgoldenlab/simba/blob/master/simba/assets/img/splash_2024.mp4>`__
   splash screen followed by
   `THIS <https://github.com/sgoldenlab/simba/blob/master/images/main_gui_frm.webp>`__
   main GUI window.

..

   [!TIP] You can also use the Anaconda Navigator GUI interface to get
   the SimBA installation done. This methid creates conda environments
   through a graphical interface rather than through the command line.
   You can read about how to install SimBA through the Anaconda
   Navigator
   `HERE <https://github.com/sgoldenlab/simba/blob/master/docs/anaconda_installation.md>`__.

Option 2: Install SimBA in main python.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1). After installing python, open a command prompt and type the
following command:

::

   pip install simba-uw-tf-dev

..

   Note: If you are seeing error messages related to some dependency
   conflicts, then you need to either downgrade your pypi package or
   instruct SimBA to ignore these dependency conflicts - either works.
   To find more information on how to do this, click
   `HERE <https://github.com/sgoldenlab/simba/blob/master/docs/FAQ.md#when-i-install-or-update-simba-i-see-a-bunch-or-messages-in-the-console-telling-there-has-been-some-dependency-conflicts-the-messages-may-look-a-little-like-this>`__.

2) Now launch SimBA by typing:

::

   simba

.. and hit the Enter key. Note: SimBA may take a little time to launch
dependning in your computer CPU, but you should eventually see a splash
screen and the main GUI windows below

Option 3: Install SimBA using python venv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

I honestly haven’t used this method much. But these are the steps that I
have had reported to me that users had to go through:

1. Open bash or command prompt and run the following commands on current
   working directory

::

   python -m venv venv

or (to make sure your virtuall environment is python 3.6 if you have
multiple python versions in your machine)

::

   py -3.6 -m venv venv

2. Then activate virtual environment by

::

   venv\Scripts\activate

3. Make sure you are using the latest pip version and setup tools

::

   python -m pip install --upgrade pip
   pip uninstall setuptools
   pip install setuptools

4. Install simba

::

   pip install simba-uw-tf-dev

5. Fix some package version

::

   pip3 uninstall pyparsing
   pip3 install pyparsing==2.4.7

6. Now you can launch simba in the terminal with.

::

   simba

Option 4: Install SimBA using Anaconda Navigator.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See
`THESE <https://github.com/sgoldenlab/simba/blob/master/docs/anaconda_2025.md>`__
instructions.

.. _requirements-1:

Requirements
~~~~~~~~~~~~

For a list of the SimBA dependencies, and the packages that gets
installed when running ``pip install simba-uw-tf-dev``, see
`THIS <https://github.com/sgoldenlab/simba/blob/master/requirements.txt>`__
file.

Author `Simon N <https://github.com/sronilsson>`__ sronilsson@gmail.com
