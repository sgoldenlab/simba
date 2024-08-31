SimBA Installation
==================

Requirements
---------------------------

1. `Python 3.6 <https://www.python.org/downloads/release/python-360/>`__
   **<– VALIDATED WITH 3.6.0**
2. `FFmpeg <https://m.wikihow.com/Install-FFmpeg-on-Windows>`__

Installing SimBA Option 1
---------------------------

Install SimBAxTF-development version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open bash or command prompt and run the following commands on current
working directory

.. code:: bash

   pip install simba-uw-tf-dev

.. note::
   If you are seeing error messages related to some dependency
   conflicts, then you need to either downgrade your pypi package or
   instruct SimBA to ignore these dependency conflicts - either works.
   To find more information on how to do this, click
   `HERE <https://github.com/sgoldenlab/simba/blob/master/docs/FAQ.md#when-i-install-or-update-simba-i-see-a-bunch-or-messages-in-the-console-telling-there-has-been-some-dependency-conflicts-the-messages-may-look-a-little-like-this>`__

How to launch SimBA ( installed using pip install simba-uw-tf-dev)
---------------------------

1. Open up command prompt anywhere.

2. In the command prompt type

.. code:: bash
   simba

3. Hit ``Enter``.

.. note::
   If you installed SimBA on a virtual environment (anaconda),
   after installation, you may have to run run ``conda install shapely``
   for SimBA to work.

Installing SimBA using Anaconda
---------------------------

Click `here </docs/anaconda_installation.md>`__ for a detail step by
step guide on how to install using anaconda.

1. Open up terminal of your environment

2. In the terminal type

``pip install simba-uw-tf-dev``

3. It will error out when running simba. To fix it, first uninstall
   shapely.

``pip uninstall shapely``

4. Then, install shapely with conda command:

``conda install -c conda-forge shapely``

Installing on MacOS
---------------------------

Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  XCode installed
-  Homebrew
-  ffmpeg
-  Python 3.6
-  Anaconda

Installation process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create an environment for simba using anaconda terminal.

2. In the terminal type, ``pip install simba-uw-tf-dev``

3. Then, ``conda install -c anaconda python.app``

4. Then, ``conda install matplotlib``

5. Then, ``conda uninstall shapely``

6. Then, ``conda install -c conda-forge shapely``

7. Then, ``pip install shap``

8. Lastly, ``pip install h5py``

9. In the terminal, type in ``simba`` to test if it works.

Python dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For detailed and up-to-date dependency version, see `setup.py <https://github.com/sgoldenlab/simba/blob/master/setup.py>`_

+-----------------------------------+-----------------------------------+
| package                           | ver.                              |
+===================================+===================================+
| `Pillow <https://gi               | 5.4.1                             |
| thub.com/python-pillow/Pillow>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `deeplabcut <https://             | 2.0.9                             |
| github.com/AlexEMG/DeepLabCut>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `eli5 <https:/                    | 0.10.1                            |
| /github.com/TeamHG-Memex/eli5>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `imblearn <https://github.        | 0.5.0                             |
| com/scikit-learn-contrib/imbalanc |                                   |
| ed-learn/tree/master/imblearn>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `imutils <https:                  | 0.5.2                             |
| //github.com/jrosebr1/imutils>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `matplotlib <https://git          | 3.0.3                             |
| hub.com/matplotlib/matplotlib>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Shapely <https://shapely.readth  | 1.6.4.post2                       |
| edocs.io/en/latest/index.html>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `deepposekit <https://gi          | 0.3.5                             |
| thub.com/jgraving/DeepPoseKit>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `dtreeviz <http                   | 0.8.1                             |
| s://github.com/parrt/dtreeviz>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `opencv_python <https://gi        | 3.4.5.20                          |
| thub.com/skvark/opencv-python>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `numpy <h                         | 1.18.1                            |
| ttps://github.com/numpy/numpy>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `imgaug <https://img              | 0.4.0                             |
| aug.readthedocs.io/en/latest/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `pandas <https:/                  | 0.25.3                            |
| /github.com/pandas-dev/pandas>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `scikit_ima                       | 0.14.2                            |
| ge <https://scikit-image.org/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `scipy <h                         | 1.1.0                             |
| ttps://github.com/scipy/scipy>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `seaborn <https                   | 0.9.0                             |
| ://github.com/mwaskom/seaborn>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `sklearn <https://github.         | 1.1.0                             |
| com/scikit-learn/scikit-learn>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `scikit-learn <https://github.    | 0.22.1                            |
| com/scikit-learn/scikit-learn>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `tensorflow_gpu <https://git      | 0.14.1                            |
| hub.com/tensorflow/tensorflow>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `scikit-learn <https://github.    | 0.22.1                            |
| com/scikit-learn/scikit-learn>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `tqdm                             | 4.30.0                            |
| <https://github.com/tqdm/tqdm>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `yellowbrick <https://github.com  | 0.9.1                             |
| /DistrictDataLabs/yellowbrick>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `xgboost <ht                      | 0.9                               |
| tps://github.com/dmlc/xgboost>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `tabul                            | 0.8.3                             |
| ate <https://bitbucket.org/astani |                                   |
| n/python-tabulate/src/master/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `tabl                             | ≥ 3.5.1                           |
| es <https://www.pytables.org/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `dash <ht                         | 1.14.0                            |
| tps://github.com/plotly/dash/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `dash color                       | 0.0.1                             |
| picker <https://github.co         |                                   |
| m/vivekvs1/dash-color-picker/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `dash                             | 0.5.0                             |
| daqs <htt                         |                                   |
| ps://dash.plotly.com/dash-daq>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `h5py                             | 2.9.0                             |
| <https://github.com/h5py/h5py>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `numba <h                         | 0.48.0                            |
| ttps://github.com/numba/numba>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `numexpr <http                    | 2.6.9                             |
| s://github.com/pydata/numexpr>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `plot                             | 4.9.0                             |
| ly <https://github.com/plotly>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `statsmodels <https://githu       | 0.9.0                             |
| b.com/statsmodels/statsmodels>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `cefpython3 <https://g            | 66.0                              |
| ithub.com/cztomczak/cefpython>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `pyarr                            | 0.17.1                            |
| ow <https://github.com/apache/arr |                                   |
| ow/tree/master/python/pyarrow>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `shap <http                       | 0.35.0                            |
| s://github.com/slundberg/shap>`__ |                                   |
+-----------------------------------+-----------------------------------+

Author `Simon N <https://github.com/sronilsson>`__, `JJ
Choong <https://github.com/inoejj>`__
